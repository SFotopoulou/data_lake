"""
catalog_shard_ingest – two-phase map-reduce catalog ingest (``dl-ingest-from-shard``).

Why
---
The single-writer append path in ``ingest_catalog`` rewrites each Npix tile for
every source file that touches it.  At norder=4 a tractor brick sits inside a
~3.7° HEALPix pixel shared by hundreds of bricks → ~20 s/file → 2 000 CPU-hours
for 400k files.

Design
------
**Phase map** (embarrassingly parallel, one Slurm array task per RA folder):

    FITS → decode_catalog_file_to_batches → immutable per-Npix part files
    <shard-root>/<survey>/Norder=N/Dir=D/Npix=P/part-<stem>.parquet

Each part is written once and never modified.  Resume is free: skip if the
part file already exists.  A JSONL checkpoint records completed source paths.

**Phase reduce** (single job, many CPUs):

    scan shard tree → group by Npix → concat all parts → one lake tile per Npix
    catalogs/<survey>/Norder=N/Dir=D/Npix=N.parquet

Workers own disjoint Npix sets → no tile locking.  Dedup happens once here.
After all tiles are written, ``finalize_catalog_survey`` refreshes metadata.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import (
    CatalogParquetOptions,
    DuplicateIdMode,
    _write_catalog_parquet_tile,
    decode_catalog_file_to_batches,
    finalize_catalog_survey,
    healpix_dir,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CHECKPOINT_FILENAME = "map_checkpoint.jsonl"


def _shard_part_path(
    shard_root: Path,
    survey: str,
    norder: int,
    npix: int,
    stem: str,
) -> Path:
    """Return the canonical path for an immutable shard part file."""
    hp_dir = healpix_dir(norder, npix)
    return shard_root / survey / hp_dir / f"Npix={npix}" / f"part-{stem}.parquet"


def _checkpoint_path(shard_root: Path, survey: str) -> Path:
    return shard_root / survey / _CHECKPOINT_FILENAME


def _load_checkpoint(checkpoint: Path) -> set[str]:
    """Return the set of source-path strings that completed previously."""
    if not checkpoint.exists():
        return set()
    done: set[str] = set()
    try:
        for line in checkpoint.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            p = obj.get("path", "")
            if p:
                done.add(p)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read checkpoint %s: %s", checkpoint, exc)
    return done


def _append_checkpoint(checkpoint: Path, source_path: str) -> None:
    """Append one completed path to the JSONL checkpoint (atomic line append)."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps({"path": source_path, "ts": time.time()}) + "\n"
    with open(checkpoint, "a") as fh:
        fh.write(record)


# ---------------------------------------------------------------------------
# Map phase – worker
# ---------------------------------------------------------------------------


def _map_one_file(
    source_path: Path,
    shard_root: Path,
    survey: str,
    ra_col: str,
    dec_col: str,
    norder: int,
    link_id_col: str | None,
    columns: tuple[str, ...] | None,
    allow_incomplete_link_id: bool,
) -> tuple[str, int, int]:
    """Decode one FITS file and write immutable Npix shard parts.

    Returns ``(source_path_str, n_tiles_written, n_rows)`` on success.
    Raises on decode/write error.
    """
    stem = Path(source_path).stem
    batches, _sid_mode, n_rows = decode_catalog_file_to_batches(
        source_path,
        ra_col=ra_col,
        dec_col=dec_col,
        norder=norder,
        link_id_col=link_id_col,
        columns=list(columns) if columns else None,
        allow_incomplete_link_id=allow_incomplete_link_id,
    )
    options = CatalogParquetOptions()
    n_written = 0
    for npix, tile_table in batches:
        part_path = _shard_part_path(shard_root, survey, norder, npix, stem)
        if part_path.exists():
            # idempotent resume: never overwrite a completed part
            continue
        part_path.parent.mkdir(parents=True, exist_ok=True)
        _write_catalog_parquet_tile(tile_table, part_path, options)
        n_written += 1
    return str(source_path), n_written, n_rows


# ---------------------------------------------------------------------------
# Map phase – orchestrator
# ---------------------------------------------------------------------------


def run_map_phase(
    source_paths: list[Path],
    shard_root: Path,
    survey: str,
    ra_col: str,
    dec_col: str,
    norder: int,
    *,
    link_id_col: str | None = None,
    columns: Sequence[str] | None = None,
    allow_incomplete_link_id: bool = False,
    n_workers: int = 1,
) -> None:
    """Map FITS files → immutable Npix part files inside ``shard_root``.

    Skips files already recorded in the JSONL checkpoint (resume-safe).
    """
    checkpoint = _checkpoint_path(shard_root, survey)
    done = _load_checkpoint(checkpoint)

    pending = [p for p in source_paths if str(p.resolve()) not in done]
    log.info(
        "Map phase: %d total files, %d already done, %d to process.",
        len(source_paths),
        len(source_paths) - len(pending),
        len(pending),
    )

    cols_tuple: tuple[str, ...] | None = tuple(columns) if columns else None
    kwargs = dict(
        shard_root=shard_root,
        survey=survey,
        ra_col=ra_col,
        dec_col=dec_col,
        norder=norder,
        link_id_col=link_id_col,
        columns=cols_tuple,
        allow_incomplete_link_id=allow_incomplete_link_id,
    )

    t0 = time.monotonic()
    n_ok = 0
    n_err = 0

    if n_workers <= 1:
        for src in pending:
            try:
                _, n_tiles, n_rows = _map_one_file(src, **kwargs)  # type: ignore[arg-type]
                _append_checkpoint(checkpoint, str(src.resolve()))
                n_ok += 1
                log.debug("OK  %s  tiles=%d rows=%d", src.name, n_tiles, n_rows)
            except Exception as exc:  # noqa: BLE001
                n_err += 1
                log.error("FAIL %s: %s", src, exc)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_map_one_file, src, **kwargs): src  # type: ignore[arg-type]
                for src in pending
            }
            for fut in as_completed(futures):
                src = futures[fut]
                try:
                    _, n_tiles, n_rows = fut.result()
                    _append_checkpoint(checkpoint, str(src.resolve()))
                    n_ok += 1
                    log.debug("OK  %s  tiles=%d rows=%d", src.name, n_tiles, n_rows)
                except Exception as exc:  # noqa: BLE001
                    n_err += 1
                    log.error("FAIL %s: %s", src, exc)

    elapsed = time.monotonic() - t0
    log.info(
        "Map phase done: %d ok, %d errors in %.1f s (%.2f s/file).",
        n_ok,
        n_err,
        elapsed,
        elapsed / max(n_ok + n_err, 1),
    )
    if n_err:
        raise RuntimeError(f"Map phase finished with {n_err} error(s). See log above.")


# ---------------------------------------------------------------------------
# Reduce phase – worker
# ---------------------------------------------------------------------------


def _dedup_table(
    table: pa.Table,
    on_duplicate_id: DuplicateIdMode,
) -> pa.Table:
    """Remove duplicate ``_source_id`` rows according to *on_duplicate_id* policy."""
    if "_source_id" not in table.schema.names:
        return table

    import pyarrow.compute as pc

    sid_col = table.column("_source_id")
    # Build index: last occurrence wins (we reverse, dedup, reverse back)
    # We sort by _source_id then pick first or last depending on policy.
    n = len(table)
    if n == 0:
        return table

    # Indices of all rows
    order = pa.array(range(n), type=pa.int64())

    if on_duplicate_id == "last":
        # Keep last occurrence: iterate reversed, keep first seen
        seen: set[object] = set()
        keep: list[int] = []
        for i in reversed(range(n)):
            sid = sid_col[i].as_py()
            if sid not in seen:
                seen.add(sid)
                keep.append(i)
        keep.reverse()
        return table.take(pa.array(keep, type=pa.int64()))

    if on_duplicate_id == "skip":
        # Keep first occurrence
        seen = set()
        keep = []
        for i in range(n):
            sid = sid_col[i].as_py()
            if sid not in seen:
                seen.add(sid)
                keep.append(i)
        return table.take(pa.array(keep, type=pa.int64()))

    # "error"
    sids = sid_col.to_pylist()
    seen_set: set[object] = set()
    for sid in sids:
        if sid in seen_set:
            raise ValueError(
                f"Duplicate _source_id {sid!r} found in reduce for this Npix tile. "
                "Use --on-duplicate-id skip or last to resolve."
            )
        seen_set.add(sid)
    return table


def _reduce_one_npix(
    npix: int,
    part_paths: list[Path],
    catalog_root: Path,
    norder: int,
    on_duplicate_id: DuplicateIdMode,
) -> tuple[int, int]:
    """Concatenate all part files for one Npix and write the final lake tile.

    Returns ``(npix, n_rows)``.
    """
    tables = [pq.read_table(str(p)) for p in part_paths]
    combined = pa.concat_tables(tables, promote_options="default")
    combined = _dedup_table(combined, on_duplicate_id)

    hp_dir = healpix_dir(norder, npix)
    out_path = catalog_root / hp_dir / f"Npix={npix}.parquet"
    options = CatalogParquetOptions()
    _write_catalog_parquet_tile(combined, out_path, options)
    return npix, len(combined)


# ---------------------------------------------------------------------------
# Reduce phase – orchestrator
# ---------------------------------------------------------------------------


def _iter_shard_parts(
    shard_root: Path,
    survey: str,
    norder: int,
) -> Iterator[tuple[int, list[Path]]]:
    """Yield ``(npix, [part_paths])`` for every Npix that has at least one part."""
    survey_root = shard_root / survey
    if not survey_root.exists():
        return
    for npix_dir in sorted(survey_root.rglob(f"Npix=*")):
        if not npix_dir.is_dir():
            continue
        try:
            npix = int(npix_dir.name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        parts = sorted(npix_dir.glob("part-*.parquet"))
        if parts:
            yield npix, parts


def run_reduce_phase(
    shard_root: Path,
    survey: str,
    catalog_root: Path,
    norder: int,
    *,
    on_duplicate_id: DuplicateIdMode = "skip",
    n_workers: int = 1,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> None:
    """Concat shard parts → one lake tile per Npix, then finalize metadata."""
    npix_groups = list(_iter_shard_parts(shard_root, survey, norder))
    if not npix_groups:
        log.warning("No shard parts found under %s for survey %r.", shard_root, survey)
        return

    log.info("Reduce phase: %d Npix pixels to merge.", len(npix_groups))
    t0 = time.monotonic()
    n_ok = 0
    n_err = 0

    if n_workers <= 1:
        for npix, parts in npix_groups:
            try:
                _, n_rows = _reduce_one_npix(
                    npix, parts, catalog_root, norder, on_duplicate_id
                )
                n_ok += 1
                log.debug("Npix=%d  %d rows from %d parts", npix, n_rows, len(parts))
            except Exception as exc:  # noqa: BLE001
                n_err += 1
                log.error("FAIL Npix=%d: %s", npix, exc)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _reduce_one_npix,
                    npix,
                    parts,
                    catalog_root,
                    norder,
                    on_duplicate_id,
                ): npix
                for npix, parts in npix_groups
            }
            for fut in as_completed(futures):
                npix = futures[fut]
                try:
                    _, n_rows = fut.result()
                    n_ok += 1
                    log.debug("Npix=%d done (%d rows)", npix, n_rows)
                except Exception as exc:  # noqa: BLE001
                    n_err += 1
                    log.error("FAIL Npix=%d: %s", npix, exc)

    elapsed = time.monotonic() - t0
    log.info(
        "Reduce concat done: %d ok, %d errors in %.1f s.",
        n_ok,
        n_err,
        elapsed,
    )

    if n_err:
        raise RuntimeError(
            f"Reduce phase finished with {n_err} tile error(s). See log above."
        )

    log.info("Running finalize_catalog_survey …")
    finalize_catalog_survey(
        catalog_root=catalog_root,
        survey_name=survey,
        norder=norder,
        ra_col=ra_col,
        dec_col=dec_col,
    )
    log.info("Reduce phase complete.")


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------


def _collect_fits_files(
    input_dir: Path | None,
    file_list: Path | None,
) -> list[Path]:
    """Return sorted list of FITS paths from either --input-dir or --file-list."""
    if input_dir is not None and file_list is not None:
        raise ValueError("Specify --input-dir or --file-list, not both.")
    if input_dir is not None:
        # Non-recursive: one RA folder
        paths = sorted(input_dir.glob("*.fits")) + sorted(input_dir.glob("*.fits.gz"))
        if not paths:
            log.warning("No *.fits files found in %s", input_dir)
        return paths
    if file_list is not None:
        paths = []
        for line in file_list.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(Path(line))
        return paths
    raise ValueError("Either --input-dir or --file-list is required for --phase map.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    @click.command("dl-ingest-from-shard")
    @click.option(
        "--phase",
        type=click.Choice(["map", "reduce"], case_sensitive=False),
        required=True,
        help="map: FITS → shard parts.  reduce: shard parts → lake tiles.",
    )
    # Map-phase inputs
    @click.option(
        "--input-dir",
        "input_dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=None,
        help="(map) Directory of *.fits files (non-recursive, one RA folder).",
    )
    @click.option(
        "--file-list",
        "file_list",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="(map) Text file with one FITS path per line.",
    )
    # Shard / survey
    @click.option(
        "--shard-root",
        "shard_root",
        type=click.Path(path_type=Path),
        required=True,
        help="Root directory for immutable shard part files.",
    )
    @click.option(
        "--survey",
        required=True,
        help="Survey name (e.g. Legacy_DR11_north). Used as sub-directory and lake survey name.",
    )
    # Sky columns
    @click.option("--ra-col", default="ra", show_default=True)
    @click.option("--dec-col", default="dec", show_default=True)
    @click.option(
        "--link-id-col",
        "link_id_col",
        default=None,
        help="ID column (e.g. ls_id_dr11) or composite (TARGETID,SURVEY,PROGRAM).",
    )
    @click.option(
        "--norder",
        type=int,
        default=7,
        show_default=True,
        help="HEALPix order for partitioning.",
    )
    @click.option(
        "--columns",
        default=None,
        help="Comma-separated columns to keep (default: all).",
    )
    # Reduce options
    @click.option(
        "--on-duplicate-id",
        "on_duplicate_id",
        type=click.Choice(["skip", "error", "last"], case_sensitive=False),
        default="skip",
        show_default=True,
        help="(reduce) How to handle duplicate _source_id values within one tile.",
    )
    @click.option(
        "--lake-root",
        "lake_root",
        type=click.Path(path_type=Path),
        default=None,
        help="(reduce) Lake root directory. Defaults to shard-root/../lake.",
    )
    # Parallelism
    @click.option(
        "--n-workers",
        "n_workers",
        type=int,
        default=1,
        show_default=True,
        help="Worker processes (map: decode parallelism; reduce: tile concat parallelism).",
    )
    # Resume / misc
    @click.option(
        "--allow-incomplete-link-id",
        "allow_incomplete_link_id",
        is_flag=True,
        default=False,
        help="Leave _source_id null when composite link-id parts are missing.",
    )
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        phase: str,
        input_dir: Path | None,
        file_list: Path | None,
        shard_root: Path,
        survey: str,
        ra_col: str,
        dec_col: str,
        link_id_col: str | None,
        norder: int,
        columns: str | None,
        on_duplicate_id: str,
        lake_root: Path | None,
        n_workers: int,
        allow_incomplete_link_id: bool,
        verbose: bool,
    ) -> None:
        """Two-phase map-reduce catalog ingest for large FITS collections.

        \b
        PHASE MAP  – run one Slurm array task per RA folder:
            dl-ingest-from-shard --phase map --input-dir /data/legacy/north/000 \\
                --survey Legacy_DR11_north --shard-root $SCRATCH/shards \\
                --ra-col ra --dec-col dec --link-id-col ls_id_dr11 \\
                --norder 7 --n-workers 8

        \b
        PHASE REDUCE – run once after the array completes:
            dl-ingest-from-shard --phase reduce \\
                --survey Legacy_DR11_north --shard-root $SCRATCH/shards \\
                --lake-root /data/lake --norder 7 --n-workers 32
        """
        import logging

        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            stream=sys.stderr,
        )

        cols: Sequence[str] | None = (
            [c.strip() for c in columns.split(",") if c.strip()]
            if columns
            else None
        )
        on_dup: DuplicateIdMode = on_duplicate_id.lower()  # type: ignore[assignment]

        if phase == "map":
            source_paths = _collect_fits_files(input_dir, file_list)
            if not source_paths:
                log.error("No input files found. Check --input-dir / --file-list.")
                sys.exit(1)
            run_map_phase(
                source_paths=source_paths,
                shard_root=shard_root,
                survey=survey,
                ra_col=ra_col,
                dec_col=dec_col,
                norder=norder,
                link_id_col=link_id_col,
                columns=cols,
                allow_incomplete_link_id=allow_incomplete_link_id,
                n_workers=n_workers,
            )
        else:  # reduce
            if lake_root is None:
                lake_root = shard_root.parent / "lake"
                log.info("--lake-root not set; defaulting to %s", lake_root)
            catalog_root = lake_root / "catalogs" / survey
            run_reduce_phase(
                shard_root=shard_root,
                survey=survey,
                catalog_root=catalog_root,
                norder=norder,
                on_duplicate_id=on_dup,
                n_workers=n_workers,
                ra_col=ra_col,
                dec_col=dec_col,
            )

except ImportError:
    # click not installed: module still importable for engine functions
    cli = None  # type: ignore[assignment]
