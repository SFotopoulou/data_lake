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

import errno
import json
import logging
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import (
    CatalogParquetOptions,
    DuplicateIdMode,
    TileMode,
    _id_column_for_dedup,
    _is_valid_parquet_tile,
    _merge_tile_tables,
    _read_catalog_tile,
    _write_catalog_parquet_tile,
    check_fits_hdu_provenance,
    decode_catalog_file_to_batches,
    describe_fits_hdu,
    finalize_catalog_survey,
    healpix_dir,
    record_fits_hdu,
)
from data_lake.io.fits_read import catalog_fits_read_policy, parse_hdu_spec

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CHECKPOINT_FILENAME = "map_checkpoint.jsonl"
_SHARD_INFO_FILENAME = "shard_info.json"


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
    n_bad = 0
    # Many array tasks append concurrently; a torn line must not hide the rest.
    for line in checkpoint.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            p = json.loads(line).get("path", "")
        except (json.JSONDecodeError, AttributeError):
            n_bad += 1
            continue
        if p:
            done.add(p)
    if n_bad:
        log.warning("Ignored %d malformed line(s) in checkpoint %s", n_bad, checkpoint)
    return done


def _append_checkpoint(checkpoint: Path, source_path: str) -> None:
    """Append one completed path to the JSONL checkpoint (atomic line append)."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps({"path": source_path, "ts": time.time()}) + "\n"
    with open(checkpoint, "a") as fh:
        fh.write(record)


def _append_failure(failures_log: Path | None, entry: dict) -> None:
    """Append one failure record to the JSONL *failures_log* (no-op when None)."""
    if failures_log is None:
        return
    failures_log.parent.mkdir(parents=True, exist_ok=True)
    with failures_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({**entry, "ts": time.time()}, ensure_ascii=False) + "\n")


def _failure_entry(exc: BaseException, **ids: object) -> dict:
    return {
        **ids,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": "".join(traceback.format_exception(exc)),
    }


# ---------------------------------------------------------------------------
# Shard / catalog consistency guards
# ---------------------------------------------------------------------------


def _shard_info_path(shard_root: Path, survey: str) -> Path:
    return shard_root / survey / _SHARD_INFO_FILENAME


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path) as fh:
        return json.load(fh)


def _check_shard_norder(shard_root: Path, survey: str, norder: int) -> dict | None:
    """Abort when *norder* differs from the order the shard tree was mapped at."""
    info = _read_json(_shard_info_path(shard_root, survey))
    if info is not None and int(info["norder"]) != norder:
        raise ValueError(
            f"--norder {norder} does not match shard tree "
            f"{_shard_info_path(shard_root, survey)} (norder={info['norder']}). "
            "Use the same --norder for every map task and for reduce."
        )
    return info


#: shard_info keys where a later map task only warns on mismatch (others abort).
_WARN_ONLY_SHARD_KEYS = frozenset({"fits_hdu"})


def _warn_shard_fits_hdu(shard_root: Path, survey: str, fits_hdu: str | None) -> None:
    """Warn when this map run selects another FITS HDU than the shard tree used."""
    info = _read_json(_shard_info_path(shard_root, survey))
    if info is None:
        return
    recorded = parse_hdu_spec(info.get("fits_hdu"))
    if recorded != parse_hdu_spec(fits_hdu):
        log.warning(
            "Shard tree %s was mapped with --hdu %s; this run uses --hdu %s. "
            "Rows from different FITS extensions will be mixed in one survey.",
            _shard_info_path(shard_root, survey),
            describe_fits_hdu(recorded), describe_fits_hdu(parse_hdu_spec(fits_hdu)),
        )


def _record_shard_info(shard_root: Path, survey: str, info: dict) -> None:
    """Write ``shard_info.json`` once; abort if a later map task disagrees.

    Keys in :data:`_WARN_ONLY_SHARD_KEYS` are compared by the caller (warning only).
    """
    path = _shard_info_path(shard_root, survey)
    existing = _read_json(path)
    if existing is not None:
        diffs = {
            k: (existing.get(k), v) for k, v in info.items()
            if k not in _WARN_ONLY_SHARD_KEYS and existing.get(k) != v
        }
        if diffs:
            raise ValueError(
                f"Map settings differ from existing shard tree {path}: "
                + ", ".join(f"{k}: shards={a!r} now={b!r}" for k, (a, b) in diffs.items())
                + ". Use a new --shard-root or --survey."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Array tasks may race here; per-PID tmp + replace keeps the file whole.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(info, indent=2))
    os.replace(tmp, path)


def _check_against_catalog(shard_info: dict, catalog_root: Path) -> None:
    """Abort when the existing lake survey uses another order or link-id spec."""
    info = _read_json(catalog_root / "catalog_info.json")
    if info is None:
        return
    problems: list[str] = []
    lake_order = info.get("hats_order")
    if lake_order is not None and int(lake_order) != int(shard_info["norder"]):
        problems.append(f"hats_order: lake={lake_order} shards={shard_info['norder']}")
    lake_mode = info.get("link_id_mode")
    shard_mode = shard_info.get("link_id_mode")
    if lake_mode and shard_mode and lake_mode != shard_mode:
        problems.append(f"link_id_mode: lake={lake_mode!r} shards={shard_mode!r}")
    if problems:
        raise ValueError(
            f"Shards are incompatible with existing survey at {catalog_root}: "
            + "; ".join(problems)
            + ". Reduce into a new --survey, or re-map with matching settings."
        )


# ---------------------------------------------------------------------------
# Staging: write on fast local disk, publish to the final (large) area
# ---------------------------------------------------------------------------

# Headroom factor applied to the estimated tile size before staging it.
_STAGING_HEADROOM = 1.5


def _staged_path(staging_dir: Path, final_root: Path, final_path: Path) -> Path:
    """Mirror *final_path* (relative to *final_root*) under *staging_dir*."""
    return staging_dir / final_path.relative_to(final_root)


def _staging_has_room(staging_dir: Path, needed_bytes: int) -> bool:
    staging_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(staging_dir).free
    return free >= needed_bytes * _STAGING_HEADROOM


def _publish(staged: Path, final: Path) -> None:
    """Move a finished file from staging to its final location atomically.

    Same filesystem → ``os.replace``.  Cross-device → copy to ``<final>.tmp``
    then ``os.replace`` so readers never see a partial file.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(staged, final)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    tmp = final.with_name(final.name + ".tmp")
    try:
        shutil.copyfile(staged, tmp)
        os.replace(tmp, final)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    staged.unlink(missing_ok=True)


def _write_tile_staged(
    table: pa.Table,
    final_path: Path,
    final_root: Path,
    options: CatalogParquetOptions,
    staging_dir: Path | None,
    *,
    estimated_bytes: int,
) -> None:
    """Write *table* to *final_path*, via *staging_dir* when it has room."""
    if staging_dir is None or not _staging_has_room(staging_dir, estimated_bytes):
        if staging_dir is not None:
            log.warning(
                "Staging %s low on space for %s (~%.1f MiB); writing directly.",
                staging_dir, final_path.name, estimated_bytes / 2**20,
            )
        _write_catalog_parquet_tile(table, final_path, options)
        return
    staged = _staged_path(staging_dir, final_root, final_path)
    try:
        _write_catalog_parquet_tile(table, staged, options)
        _publish(staged, final_path)
    finally:
        staged.unlink(missing_ok=True)


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
    staging_dir: Path | None = None,
    fits_hdu: str | None = None,
) -> tuple[str, int, int, str]:
    """Decode one FITS file and write immutable Npix shard parts.

    Returns ``(source_path_str, n_tiles_written, n_rows, link_id_mode)`` on
    success.  Raises on decode/write error.
    """
    stem = Path(source_path).stem
    batches, sid_mode, n_rows = decode_catalog_file_to_batches(
        source_path,
        ra_col=ra_col,
        dec_col=dec_col,
        norder=norder,
        link_id_col=link_id_col,
        columns=list(columns) if columns else None,
        allow_incomplete_link_id=allow_incomplete_link_id,
        fits_read_policy=catalog_fits_read_policy(None, fits_hdu),
    )
    options = CatalogParquetOptions()
    n_written = 0
    for npix, tile_table in batches:
        part_path = _shard_part_path(shard_root, survey, norder, npix, stem)
        if part_path.exists():
            # idempotent resume: never overwrite a completed part
            continue
        _write_tile_staged(
            tile_table,
            part_path,
            shard_root,
            options,
            staging_dir,
            estimated_bytes=int(tile_table.nbytes),
        )
        n_written += 1
    return str(source_path), n_written, n_rows, sid_mode


# ---------------------------------------------------------------------------
# Map phase – orchestrator
# ---------------------------------------------------------------------------


def _run_tasks(
    fn: Callable[..., Any],
    items: Iterable[tuple[Any, tuple]],
    n_workers: int,
    *,
    max_in_flight: int | None = None,
) -> Iterator[tuple[Any, Any, BaseException | None]]:
    """Yield ``(key, result, exc)`` as ``fn(*args)`` tasks finish.

    At most *max_in_flight* (default ``2 * n_workers``) tasks are queued at
    once.  Closing the generator early cancels work not yet started.
    """
    if n_workers <= 1:
        for key, args in items:
            try:
                res = fn(*args)
            except Exception as exc:  # noqa: BLE001
                yield key, None, exc
            else:
                yield key, res, None
        return

    limit = max(max_in_flight or 2 * n_workers, n_workers)
    todo = iter(items)
    pool = ProcessPoolExecutor(max_workers=n_workers)
    in_flight: dict[Future, Any] = {}

    def _submit_next() -> None:
        for key, args in todo:
            in_flight[pool.submit(fn, *args)] = key
            return

    try:
        for _ in range(limit):
            _submit_next()
        while in_flight:
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in finished:
                key = in_flight.pop(fut)
                _submit_next()
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    yield key, None, exc
                else:
                    yield key, res, None
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _progress(total: int, desc: str, unit: str, enabled: bool):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, disable=not enabled, unit=unit, desc=desc)


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
    max_in_flight: int | None = None,
    staging_dir: Path | None = None,
    failures_log: Path | None = None,
    checkpoint_path: Path | None = None,
    skip_completed: bool = True,
    show_progress: bool = False,
    fits_hdu: str | None = None,
) -> dict[str, Any]:
    """Map FITS files → immutable Npix part files inside ``shard_root``.

    Skips files recorded in the JSONL checkpoint (default
    ``<shard_root>/<survey>/map_checkpoint.jsonl``) unless *skip_completed*
    is False; existing part files are never rewritten either way.
    With *staging_dir*, each part is written there first and then moved to
    ``shard_root``.  Per-file errors are appended to *failures_log* (JSONL).
    *fits_hdu* selects the FITS table extension(s) of every file.

    Returns a summary with ``n_files``, ``n_files_skipped``, ``n_files_ok``,
    ``n_files_failed`` and ``elapsed_s``.  Raises ``ValueError`` when the
    settings disagree with the existing shard tree.
    """
    fits_hdu = parse_hdu_spec(fits_hdu)
    _check_shard_norder(shard_root, survey, norder)
    _warn_shard_fits_hdu(shard_root, survey, fits_hdu)
    checkpoint = checkpoint_path or _checkpoint_path(shard_root, survey)
    done = _load_checkpoint(checkpoint) if skip_completed else set()

    pending = [p for p in source_paths if str(p.resolve()) not in done]
    log.info(
        "Map phase: %d total files, %d already done, %d to process.",
        len(source_paths),
        len(source_paths) - len(pending),
        len(pending),
    )

    worker = partial(
        _map_one_file,
        shard_root=shard_root,
        survey=survey,
        ra_col=ra_col,
        dec_col=dec_col,
        norder=norder,
        link_id_col=link_id_col,
        columns=tuple(columns) if columns else None,
        allow_incomplete_link_id=allow_incomplete_link_id,
        staging_dir=staging_dir,
        fits_hdu=fits_hdu,
    )

    t0 = time.monotonic()
    n_ok = 0
    n_err = 0
    info_recorded = False

    def _on_success(src: Path, n_tiles: int, n_rows: int, sid_mode: str) -> None:
        nonlocal n_ok, info_recorded
        if not info_recorded:
            _record_shard_info(
                shard_root,
                survey,
                {
                    "norder": norder,
                    "link_id_mode": sid_mode,
                    "ra_column": ra_col,
                    "dec_column": dec_col,
                    "allow_incomplete_link_id": allow_incomplete_link_id,
                    "fits_hdu": fits_hdu,
                },
            )
            info_recorded = True
        _append_checkpoint(checkpoint, str(src.resolve()))
        n_ok += 1
        log.debug("OK  %s  tiles=%d rows=%d", src.name, n_tiles, n_rows)

    def _on_failure(src: Path, exc: BaseException) -> None:
        nonlocal n_err
        n_err += 1
        log.error("FAIL %s: %s", src, exc)
        _append_failure(failures_log, _failure_entry(exc, path=str(src)))

    pbar = _progress(len(pending), "map", "file", show_progress)
    tasks = _run_tasks(
        worker, ((src, (src,)) for src in pending), n_workers, max_in_flight=max_in_flight,
    )
    try:
        for src, res, exc in tasks:
            if exc is not None:
                _on_failure(src, exc)
            else:
                _, n_tiles, n_rows, sid_mode = res
                _on_success(src, n_tiles, n_rows, sid_mode)
            if pbar is not None:
                pbar.update(1)
    finally:
        tasks.close()
        if pbar is not None:
            pbar.close()

    elapsed = time.monotonic() - t0
    log.info(
        "Map phase done: %d ok, %d errors in %.1f s (%.2f s/file).",
        n_ok,
        n_err,
        elapsed,
        elapsed / max(n_ok + n_err, 1),
    )
    if n_err:
        where = f" Failures: {failures_log}." if failures_log else ""
        log.error("Map phase finished with %d error(s).%s", n_err, where)
    return {
        "n_files": len(source_paths),
        "n_files_skipped": len(source_paths) - len(pending),
        "n_files_ok": n_ok,
        "n_files_failed": n_err,
        "elapsed_s": elapsed,
    }


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
    staging_dir: Path | None = None,
    tile_mode: TileMode = "append",
    parquet_options: CatalogParquetOptions | None = None,
) -> tuple[int, int, str]:
    """Concatenate all part files for one Npix and write the final lake tile.

    An existing lake tile is handled per *tile_mode*: ``append`` merges it
    with the shard rows (existing rows win on ``--on-duplicate-id skip``),
    ``skip`` leaves it untouched, ``overwrite`` replaces it.

    Returns ``(npix, n_rows, action)`` with action one of ``written``,
    ``merged``, ``unchanged``, ``skipped``.
    """
    hp_dir = healpix_dir(norder, npix)
    out_path = catalog_root / hp_dir / f"Npix={npix}.parquet"
    exists = out_path.exists()
    if exists and tile_mode == "skip":
        return npix, 0, "skipped"

    # ParquetFile avoids Hive-partition inference from ``Norder=/Npix=`` paths.
    tables = [pq.ParquetFile(str(p)).read() for p in part_paths]
    combined = pa.concat_tables(tables, promote_options="default")
    del tables
    combined = _dedup_table(combined, on_duplicate_id)
    estimated = sum(p.stat().st_size for p in part_paths)

    action = "written"
    if exists and tile_mode == "append":
        if _is_valid_parquet_tile(out_path):
            existing = _read_catalog_tile(out_path)
            n_before = existing.num_rows
            estimated += out_path.stat().st_size
            combined = _merge_tile_tables(
                existing,
                combined,
                sid_col=_id_column_for_dedup(existing, None),
                on_duplicate_id=on_duplicate_id,
            )
            del existing
            if combined.num_rows == n_before and on_duplicate_id != "last":
                return npix, n_before, "unchanged"
            action = "merged"
        else:
            log.warning(
                "Existing tile %s is corrupt (likely aborted write); replacing.",
                out_path,
            )

    _write_tile_staged(
        combined,
        out_path,
        catalog_root,
        parquet_options or CatalogParquetOptions(),
        staging_dir,
        estimated_bytes=estimated,
    )
    return npix, len(combined), action


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
    staging_dir: Path | None = None,
    tile_mode: TileMode = "append",
    failures_log: Path | None = None,
    compact: bool = False,
    max_in_flight: int | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Concat shard parts → one lake tile per Npix, then finalize metadata.

    Raises ``ValueError`` before writing anything if the shards were mapped
    at another ``norder`` or the existing lake survey uses another order /
    link-id spec.  Per-tile errors are appended to *failures_log* (JSONL);
    finalize is skipped when any tile failed.  With *staging_dir*, each tile
    is written there first and then moved to *catalog_root*.

    Returns a summary with ``n_tiles``, ``n_tiles_failed``, ``actions``
    (counts per written/merged/unchanged/skipped), ``finalized`` and
    ``elapsed_s``.
    """
    shard_info = _check_shard_norder(shard_root, survey, norder)
    if shard_info is None:
        log.warning(
            "No %s under %s; cannot verify link-id mode against the lake.",
            _SHARD_INFO_FILENAME, shard_root / survey,
        )
        shard_info = {"norder": norder}
    _check_against_catalog(shard_info, catalog_root)
    new_survey = check_fits_hdu_provenance(catalog_root, shard_info.get("fits_hdu"))

    t0 = time.monotonic()
    actions: Counter[str] = Counter()
    n_err = 0
    npix_groups = list(_iter_shard_parts(shard_root, survey, norder))

    def _summary(finalized: bool) -> dict[str, Any]:
        return {
            "n_tiles": len(npix_groups),
            "n_tiles_failed": n_err,
            "actions": dict(actions),
            "finalized": finalized,
            "elapsed_s": time.monotonic() - t0,
        }

    if not npix_groups:
        log.warning("No shard parts found under %s for survey %r.", shard_root, survey)
        return _summary(False)

    log.info(
        "Reduce phase: %d Npix pixels to merge (tile-mode=%s).",
        len(npix_groups), tile_mode,
    )
    options = CatalogParquetOptions.compact() if compact else CatalogParquetOptions()
    args = (catalog_root, norder, on_duplicate_id, staging_dir, tile_mode, options)

    def _on_failure(npix: int, parts: list[Path], exc: BaseException) -> None:
        nonlocal n_err
        n_err += 1
        log.error("FAIL Npix=%d: %s", npix, exc)
        _append_failure(
            failures_log,
            _failure_entry(exc, npix=npix, parts=[str(p) for p in parts]),
        )

    pbar = _progress(len(npix_groups), "reduce", "tile", show_progress)
    tasks = _run_tasks(
        _reduce_one_npix,
        (((npix, parts), (npix, parts, *args)) for npix, parts in npix_groups),
        n_workers,
        max_in_flight=max_in_flight,
    )
    try:
        for (npix, parts), res, exc in tasks:
            if exc is not None:
                _on_failure(npix, parts, exc)
            else:
                _, n_rows, action = res
                actions[action] += 1
                log.debug("Npix=%d %s (%d rows, %d parts)", npix, action, n_rows, len(parts))
            if pbar is not None:
                pbar.update(1)
    finally:
        tasks.close()
        if pbar is not None:
            pbar.close()

    log.info(
        "Reduce concat done in %.1f s: %s, %d errors.",
        time.monotonic() - t0,
        ", ".join(f"{n} {a}" for a, n in sorted(actions.items())) or "nothing written",
        n_err,
    )

    if n_err:
        where = f" Failures: {failures_log}." if failures_log else ""
        log.error(
            "Reduce phase finished with %d tile error(s); finalize skipped. "
            "Fix and re-run reduce (existing tiles are merged, not duplicated).%s",
            n_err, where,
        )
        return _summary(False)

    log.info("Running finalize_catalog_survey …")
    finalize_catalog_survey(
        catalog_root=catalog_root,
        survey_name=survey,
        norder=norder,
        ra_col=shard_info.get("ra_column", ra_col),
        dec_col=shard_info.get("dec_column", dec_col),
        link_id_mode=shard_info.get("link_id_mode"),
        allow_incomplete_link_id=shard_info.get("allow_incomplete_link_id"),
    )
    if new_survey:
        record_fits_hdu(catalog_root, shard_info.get("fits_hdu"))
    log.info("Reduce phase complete.")
    return _summary(True)


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
        from data_lake.ingest.checkpoint_sidecars import paths_from_file_list_file

        return paths_from_file_list_file(file_list)
    raise ValueError("Either --input-dir or --file-list is required for --phase map.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    from data_lake.cli_utils import (
        apply_fits_memmap_env,
        config_option,
        configure_cli_logging,
        fits_memmap_option,
        hdu_option,
        ingest_token_option,
        load_optional_config,
        logging_options,
        pick,
        progress_option,
        require_ingest_permission,
        require_output_root,
        resolve_log_level,
        validate_quiet_verbose,
        warn_high_parallelism_on_slow_storage,
    )
    from data_lake.ingest.catalog_cli_options import (
        allow_incomplete_link_id_option,
        tile_mode_option,
    )

    @click.command("dl-ingest-from-shard")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @ingest_token_option
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
        help="(map) Text file with one FITS path per line (relative to the list).",
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
        "survey_name",
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
        help=(
            "(map, required) Survey object ID column (e.g. ls_id_dr11) or "
            "composite (objid,brickname)."
        ),
    )
    @click.option(
        "--norder",
        type=int,
        default=None,
        help=(
            "HEALPix order. Default: the existing shard tree's order, else config "
            "hats_order, else 5."
        ),
    )
    @click.option(
        "--columns",
        default=None,
        help="(map) Comma-separated columns to keep (default: all).",
    )
    @allow_incomplete_link_id_option
    @fits_memmap_option
    @hdu_option
    # Reduce options
    @click.option(
        "--on-duplicate-id",
        "on_duplicate_id",
        type=click.Choice(["skip", "error", "last"], case_sensitive=False),
        default="skip",
        show_default=True,
        help=(
            "(reduce) Duplicate _source_id policy, within the shards and against an "
            "existing lake tile: skip keeps the first/existing row, last keeps the "
            "newest, error aborts."
        ),
    )
    @tile_mode_option(
        default="append",
        help=(
            "(reduce) Existing lake tile: append merges it with the shard rows "
            "(one rewrite per tile), skip leaves it untouched, overwrite replaces it."
        ),
    )
    @click.option(
        "--compact", is_flag=True,
        help="(reduce) Smaller Parquet tiles (ZSTD-9, no stats/dictionary).",
    )
    @click.option(
        "--lake-root",
        "lake_root",
        type=click.Path(path_type=Path),
        default=None,
        hidden=True,
        help="Deprecated alias for the OUTPUT_ROOT argument.",
    )
    # Parallelism
    @click.option(
        "--n-workers", default=4, show_default=True, type=int,
        help="Worker processes (map: FITS decode; reduce: per-tile concat).",
    )
    @click.option(
        "--max-in-flight",
        default=None,
        type=int,
        help="Max tasks queued to workers at once (default: 2 × n_workers).",
    )
    @click.option(
        "--staging-dir",
        "staging_dir",
        type=click.Path(file_okay=False, path_type=Path),
        default=None,
        envvar="DATA_LAKE_STAGING_DIR",
        show_envvar=True,
        help=(
            "Fast local scratch (e.g. $TMPDIR on node SSD). Each part/tile is "
            "written here, then moved to --shard-root (map) or OUTPUT_ROOT "
            "(reduce). Holds only ~n_workers files at a time; falls back to "
            "direct writes when free space is short."
        ),
    )
    # Resume / logging
    @click.option(
        "--checkpoint",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="(map) JSONL checkpoint. Default: <shard-root>/<survey>/map_checkpoint.jsonl.",
    )
    @click.option(
        "--no-skip-completed",
        is_flag=True,
        help="(map) Ignore the checkpoint; existing part files are still never rewritten.",
    )
    @click.option(
        "--failures-log",
        "failures_log",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help=(
            "Append-only JSONL of failures (map: one line per FITS path; "
            "reduce: one line per Npix with its part files)."
        ),
    )
    @progress_option
    @click.option(
        "--log-file",
        "log_file",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="File for INFO+ logs (useful with --quiet for an audit trail).",
    )
    @logging_options
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        ingest_token: str | None,
        phase: str,
        input_dir: Path | None,
        file_list: Path | None,
        shard_root: Path,
        survey_name: str,
        ra_col: str,
        dec_col: str,
        link_id_col: str | None,
        norder: int | None,
        columns: str | None,
        allow_incomplete_link_id: bool,
        fits_memmap: str,
        fits_hdu: str | None,
        on_duplicate_id: str,
        tile_mode: str,
        compact: bool,
        lake_root: Path | None,
        n_workers: int,
        max_in_flight: int | None,
        staging_dir: Path | None,
        checkpoint: Path | None,
        no_skip_completed: bool,
        failures_log: Path | None,
        show_progress: bool,
        log_file: Path | None,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Two-phase map-reduce catalog ingest for large FITS collections.

        OUTPUT_ROOT is the lake root (default: lake.root from --config);
        reduce writes tiles under OUTPUT_ROOT/catalogs/<survey>/.

        \b
        PHASE MAP  – one Slurm array task per RA folder:
            dl-ingest-from-shard --phase map --input-dir /data/legacy/north/000 \\
                --survey Legacy_DR11_north --shard-root $SCRATCH/shards \\
                --ra-col ra --dec-col dec --link-id-col ls_id_dr11 \\
                --norder 7 --n-workers 8

        \b
        PHASE REDUCE – once, after the array completes:
            dl-ingest-from-shard /data/lake --phase reduce \\
                --survey Legacy_DR11_north --shard-root $SCRATCH/shards \\
                --n-workers 32
        """
        validate_quiet_verbose(quiet, verbose)
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(quiet=quiet, verbose=verbose,
                                    config_level=cfg.ingest.log_level if cfg else None),
            log_file=log_file,
            quiet=quiet,
        )
        require_ingest_permission(cfg, ingest_token)

        shard_info = _read_json(_shard_info_path(shard_root, survey_name)) or {}
        n = pick(
            norder,
            shard_info.get("norder"),
            pick(None, cfg.partitioning.hats_order if cfg else None, 5),
        )
        cols = [c.strip() for c in columns.split(",") if c.strip()] if columns else None

        try:
            if phase == "map":
                if not link_id_col:
                    raise click.UsageError("--link-id-col is required for --phase map.")
                source_paths = _collect_fits_files(input_dir, file_list)
                if not source_paths:
                    raise click.UsageError(
                        "No input files found. Check --input-dir / --file-list."
                    )
                apply_fits_memmap_env(fits_memmap)
                warn_high_parallelism_on_slow_storage(n_workers, shard_root)
                result = run_map_phase(
                    source_paths=source_paths,
                    shard_root=shard_root,
                    survey=survey_name,
                    ra_col=ra_col,
                    dec_col=dec_col,
                    norder=n,
                    link_id_col=link_id_col,
                    columns=cols,
                    allow_incomplete_link_id=allow_incomplete_link_id,
                    n_workers=n_workers,
                    max_in_flight=max_in_flight,
                    staging_dir=staging_dir,
                    failures_log=failures_log,
                    checkpoint_path=checkpoint,
                    skip_completed=not no_skip_completed,
                    show_progress=show_progress,
                    fits_hdu=fits_hdu,
                )
                sys.exit(0 if result["n_files_failed"] == 0 else 1)

            if lake_root is not None:
                if output_root is not None and output_root != lake_root:
                    raise click.UsageError(
                        "OUTPUT_ROOT and --lake-root disagree; pass only OUTPUT_ROOT."
                    )
                log.warning("--lake-root is deprecated; pass the lake root as OUTPUT_ROOT.")
                output_root = lake_root
            lake = require_output_root(output_root, cfg, kind="catalogs")
            warn_high_parallelism_on_slow_storage(n_workers, lake)
            result = run_reduce_phase(
                shard_root=shard_root,
                survey=survey_name,
                catalog_root=lake / "catalogs" / survey_name,
                norder=n,
                on_duplicate_id=on_duplicate_id.lower(),  # type: ignore[arg-type]
                n_workers=n_workers,
                ra_col=ra_col,
                dec_col=dec_col,
                staging_dir=staging_dir,
                tile_mode=tile_mode.lower(),  # type: ignore[arg-type]
                failures_log=failures_log,
                compact=compact,
                max_in_flight=max_in_flight,
                show_progress=show_progress,
            )
            sys.exit(0 if result["n_tiles_failed"] == 0 else 1)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

except ImportError:
    # click not installed: module still importable for engine functions
    cli = None  # type: ignore[assignment]
