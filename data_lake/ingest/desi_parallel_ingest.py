"""
desi_parallel_ingest – multi-process ingest of many DESI coadd FITS files.

Architecture
------------
::

    file list (~10k coadds)
            │
            ├──▶ ProcessPoolExecutor  (N workers)
            │       per file:
            │         desispec.io.read_spectra
            │         desispec.coaddition.coadd_cameras
            │         vectorised healpy assignment
            │         group records by HEALPix tile
            │         return numpy payload per (file, tile)
            │
            └──▶ main-process writer  (single thread)
                      futures.as_completed →
                          open / create tile zarr (cached per npix)
                          append flux/ivar/mask/source_id/meta batch
                          track {source_id: (npix, local_idx)} for catalog patch
                      atomic checkpoint after every successful file
                      append-only failures.jsonl on errors

Why this split
--------------
* Decode + camera-coadd is CPU-bound and largely under the GIL, so threads
  don't help; ``ProcessPoolExecutor`` gives near-linear speedup with cores.
* Only one process ever touches the Zarr store → no append-corruption risk
  and no inter-process locking needed.  The writer is fast (a few ms per
  batch), so it keeps up with N workers easily.

Anti-patterns that do *not* work
--------------------------------
* ``parallel dl-ingest-spectra ...`` (multiple processes appending to the
  same tile zarr concurrently) – ``LocalStore`` has no cross-process locks
  and shards will silently corrupt.
* ``ThreadPoolExecutor`` – ``read_spectra`` is mostly Python; expect ~1.5×.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from data_lake.ingest.fits_to_parquet import assign_healpix, healpix_dir
from data_lake.ingest.fits_to_spectra_zarr import (
    _META_DTYPE,
    _meta_to_bytes,
    _open_or_create_spectrum_tile,
    _read_desi_with_desispec,
    _write_spectrum_info,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker payload types (picklable)
# ---------------------------------------------------------------------------


@dataclass
class TileBatch:
    """One (file × tile) batch of spectra ready for an append-only write."""
    npix: int
    flux: np.ndarray      # (n, N_pix) float32
    ivar: np.ndarray      # (n, N_pix) float32
    mask: np.ndarray      # (n, N_pix) uint8
    source_ids: np.ndarray  # (n,)      int64
    meta_bytes: bytes     # concatenated _META_DTYPE rows for n sources


@dataclass
class WorkerResult:
    """Per-file outcome shipped from worker back to writer."""
    path: str
    ok: bool
    batches: list[TileBatch] = field(default_factory=list)
    wavelength: np.ndarray | None = None     # (N_pix,) float64, shared grid
    wcs_attrs: dict | None = None
    n_pix: int = 0
    n_spectra: int = 0
    error: str | None = None
    tb: str | None = None
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Worker function (top-level → picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _decode_one_coadd(path_str: str, norder: int) -> WorkerResult:
    """Decode one DESI coadd FITS file into per-tile numpy batches.

    Runs entirely in a worker process.  All heavy lifting (FITS decompress,
    ``coadd_cameras`` IVAR-weighted combine, healpy assignment, meta byte
    packing) happens here so the writer is essentially free.
    """
    t0 = time.perf_counter()
    path = Path(path_str)

    records, wcs_attrs, _res_diags, _res_offsets = _read_desi_with_desispec(
        path, with_resolution=False,
    )
    if not records:
        return WorkerResult(path=path_str, ok=True, elapsed_s=time.perf_counter() - t0)

    n = len(records)
    n_pix = int(records[0].flux.shape[0])
    wavelength = np.asarray(records[0].wavelength, dtype=np.float64).copy()

    ras = np.fromiter((r.ra for r in records), dtype=np.float64, count=n)
    decs = np.fromiter((r.dec for r in records), dtype=np.float64, count=n)
    npix_arr = assign_healpix(ras, decs, norder)

    sort_idx = np.argsort(npix_arr, kind="stable")
    npix_sorted = npix_arr[sort_idx]

    flux_stack = np.stack([records[i].flux for i in sort_idx]).astype(np.float32, copy=False)
    ivar_stack = np.stack([records[i].ivar for i in sort_idx]).astype(np.float32, copy=False)
    mask_stack = np.stack([records[i].mask for i in sort_idx]).astype(np.uint8, copy=False)
    sids = np.fromiter(
        (records[i].source_id for i in sort_idx), dtype=np.int64, count=n,
    )

    unique_pix, group_starts = np.unique(npix_sorted, return_index=True)
    group_starts = np.append(group_starts, n)

    batches: list[TileBatch] = []
    for g, pix in enumerate(unique_pix):
        s, e = int(group_starts[g]), int(group_starts[g + 1])
        meta_bytes = b"".join(
            _meta_to_bytes(records[sort_idx[i]].meta) for i in range(s, e)
        )
        batches.append(TileBatch(
            npix=int(pix),
            flux=flux_stack[s:e],
            ivar=ivar_stack[s:e],
            mask=mask_stack[s:e],
            source_ids=sids[s:e],
            meta_bytes=meta_bytes,
        ))

    return WorkerResult(
        path=path_str,
        ok=True,
        batches=batches,
        wavelength=wavelength,
        wcs_attrs=wcs_attrs,
        n_pix=n_pix,
        n_spectra=n,
        elapsed_s=time.perf_counter() - t0,
    )


def _decode_one_coadd_safe(path_str: str, norder: int) -> WorkerResult:
    """Picklable wrapper: never raises, returns a failure-marked result instead."""
    try:
        return _decode_one_coadd(path_str, norder)
    except Exception as exc:
        return WorkerResult(
            path=path_str,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            tb=traceback.format_exc(),
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _canonical_fits_path(p: Path | str) -> str:
    """Absolute, resolved path string used for checkpoint ↔ pending matching."""
    return str(Path(p).expanduser().resolve())


def _paths_from_file_list_file(file_list_path: Path) -> list[Path]:
    """Paths from a text file list; relative lines are resolved vs the list file's parent."""
    fl = file_list_path.resolve()
    base = fl.parent
    paths: list[Path] = []
    for line in fl.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        paths.append(p if p.is_absolute() else (base / p))
    return paths


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically by writing to a tempfile then renaming."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _load_checkpoint(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
        raw = data.get("completed", [])
        out: set[str] = set()
        for item in raw:
            if not item or not isinstance(item, str):
                continue
            try:
                out.add(_canonical_fits_path(item))
            except OSError:
                # Broken symlinks / odd paths: keep literal so we do not drop entries
                out.add(item)
        return out
    except Exception:
        log.warning("Could not parse checkpoint %s; starting fresh.", path)
        return set()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def ingest_spectra_parallel(
    file_paths: Sequence[Path | str],
    output_root: Path | str,
    survey_name: str,
    *,
    n_workers: int,
    norder: int = 5,
    checkpoint_path: Path | str | None = None,
    failures_log: Path | str | None = None,
    show_progress: bool = True,
    decoder: Callable[[str, int], WorkerResult] = _decode_one_coadd_safe,
    executor_factory: Callable[[int], Executor] | None = None,
) -> dict:
    """Ingest many DESI coadd FITS files in parallel into per-tile Zarr stacks.

    Parameters
    ----------
    file_paths:
        Iterable of FITS file paths to ingest.
    output_root:
        Data lake root (the function appends ``spectra/<survey_name>/``).
    survey_name:
        Survey identifier (must match the directory under ``spectra/``).
    n_workers:
        Number of decoder processes.  No default by design — callers should
        pick this consciously (typical: ``cpu_count - 1``).
    norder:
        HEALPix partitioning order (default 5).
    checkpoint_path:
        Optional JSON checkpoint file; completed file paths are persisted
        after every successful file so a restart skips them.  Pass ``None``
        to disable checkpointing.
    failures_log:
        Optional append-only JSONL file recording per-file failures with
        traceback.  Pass ``None`` to disable.
    show_progress:
        Show a tqdm progress bar on the writer side.
    decoder:
        Worker decode function; defaults to ``_decode_one_coadd_safe``.
        Injectable for unit tests (must accept ``(path_str, norder)``).
    executor_factory:
        Callable taking ``n_workers`` and returning an ``Executor``.  Defaults
        to ``ProcessPoolExecutor``; tests can pass a thread-pool factory.

    Returns
    -------
    dict with keys:
        ``n_files_requested``    – before checkpoint filtering
        ``n_files_processed``    – attempts in this run
        ``n_files_skipped``      – already in checkpoint
        ``n_files_succeeded``    – ok=True results
        ``n_files_failed``       – ok=False results
        ``n_spectra``            – total spectra appended to tiles
        ``n_tiles``              – distinct HEALPix tiles written/touched
        ``index_map``            – dict[source_id, local_index_in_tile]
        ``failures``             – list[dict] of per-file errors
        ``elapsed_s``            – total wall-clock seconds

    Notes
    -----
    The single-thread writer is the strict serialisation point.  All Zarr
    state lives in this process; workers only return numpy arrays.  Pickle
    protocol 5 (default in Python 3.11) carries the arrays as out-of-band
    buffers so transfer is near-zero-copy.
    """
    try:
        from tqdm.auto import tqdm
        from tqdm.contrib.logging import logging_redirect_tqdm
    except ImportError:
        from contextlib import nullcontext

        def tqdm(x, **_kw):
            return x

        def logging_redirect_tqdm(*_a, **_kw):
            return nullcontext()

    output_root = Path(output_root)
    survey_root = output_root / "spectra" / survey_name
    survey_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    failures_log = Path(failures_log) if failures_log else None

    completed: set[str] = _load_checkpoint(checkpoint_path)
    requested = [_canonical_fits_path(p) for p in file_paths]
    n_requested = len(requested)

    pending = [p for p in requested if p not in completed]
    n_skipped = n_requested - len(pending)
    if n_skipped:
        log.info("Checkpoint has %d/%d files; resuming with %d pending.",
                 n_skipped, n_requested, len(pending))

    if not pending:
        return {
            "n_files_requested": n_requested,
            "n_files_processed": 0,
            "n_files_skipped": n_skipped,
            "n_files_succeeded": 0,
            "n_files_failed": 0,
            "n_spectra": 0,
            "n_tiles": 0,
            "index_map": {},
            "failures": [],
            "elapsed_s": 0.0,
        }

    if executor_factory is None:
        from data_lake.cli_utils import configure_warning_filters as _wf

        def _default_executor(nw: int) -> Executor:
            # initializer runs once in each worker process — filter state
            # does not propagate from the parent, so set it up there too.
            return ProcessPoolExecutor(max_workers=nw, initializer=_wf)

        executor_factory = _default_executor

    # --- Writer state ---
    tile_groups: dict[int, "object"] = {}  # npix -> zarr.Group
    n_pix_known: int | None = None
    wcs_attrs_known: dict | None = None
    failures: list[dict] = []
    index_map: dict[int, int] = {}
    n_spectra_written = 0
    n_files_ok = 0
    n_files_fail = 0
    t_start = time.perf_counter()

    if failures_log is not None:
        failures_log.parent.mkdir(parents=True, exist_ok=True)

    with executor_factory(n_workers) as pool:
        futures: dict[Future, str] = {
            pool.submit(decoder, p, norder): p for p in pending
        }
        with logging_redirect_tqdm():
            with tqdm(
                total=len(futures),
                disable=not show_progress,
                unit="file",
                desc="ingest",
            ) as pbar:
                for fut in as_completed(futures):
                    path_str = futures[fut]
                    try:
                        res: WorkerResult = fut.result()
                    except Exception as exc:
                        res = WorkerResult(
                            path=path_str, ok=False,
                            error=f"executor: {type(exc).__name__}: {exc}",
                            tb=traceback.format_exc(),
                        )

                    pbar.update(1)

                    if not res.ok:
                        n_files_fail += 1
                        fail_entry = {
                            "path": res.path,
                            "error": res.error,
                            "traceback": res.tb,
                        }
                        failures.append(fail_entry)
                        log.warning("FAIL %s: %s", res.path, res.error)
                        if failures_log is not None:
                            with failures_log.open("a") as fh:
                                fh.write(json.dumps(fail_entry) + "\n")
                                fh.flush()
                        continue

                    if not res.batches:
                        n_files_ok += 1
                        completed.add(res.path)
                        if checkpoint_path is not None:
                            _atomic_write_json(
                                checkpoint_path, {"completed": sorted(completed)}
                            )
                        continue

                    # First successful file fixes the wavelength grid & WCS
                    if n_pix_known is None:
                        n_pix_known = res.n_pix
                        wcs_attrs_known = res.wcs_attrs
                    elif res.n_pix != n_pix_known:
                        fail_entry = {
                            "path": res.path,
                            "error": (
                                f"n_pix={res.n_pix} differs from established "
                                f"grid n_pix={n_pix_known}; file rejected."
                            ),
                            "traceback": None,
                        }
                        failures.append(fail_entry)
                        n_files_fail += 1
                        if failures_log is not None:
                            with failures_log.open("a") as fh:
                                fh.write(json.dumps(fail_entry) + "\n")
                        continue

                    for b in res.batches:
                        tile_dir = survey_root / healpix_dir(norder, b.npix)
                        tile_dir.mkdir(parents=True, exist_ok=True)
                        tile_path = tile_dir / f"Npix={b.npix}.zarr"
                        if b.npix not in tile_groups:
                            tile_groups[b.npix] = _open_or_create_spectrum_tile(
                                tile_path, n_pix_known, "shared",
                                np.dtype(np.uint8), wcs_attrs_known,
                            )
                        root = tile_groups[b.npix]
                        start_idx = root["flux"].shape[0]

                        root["flux"].append(b.flux)
                        root["ivar"].append(b.ivar)
                        root["mask"].append(b.mask)
                        root["source_id"].append(b.source_ids)
                        meta_arr = np.frombuffer(
                            b.meta_bytes,
                            dtype="|V" + str(_META_DTYPE.itemsize),
                        )
                        root["meta"].append(meta_arr)

                        if start_idx == 0 and res.wavelength is not None:
                            root["wavelength"][:] = res.wavelength.astype(np.float64)

                        for i, sid in enumerate(b.source_ids.tolist()):
                            index_map[int(sid)] = start_idx + i
                        n_spectra_written += int(b.source_ids.size)

                    n_files_ok += 1
                    completed.add(res.path)
                    if checkpoint_path is not None:
                        _atomic_write_json(
                            checkpoint_path, {"completed": sorted(completed)}
                        )

    if n_pix_known is not None and wcs_attrs_known is not None:
        _write_spectrum_info(
            survey_root, survey_name, norder, n_pix_known,
            "shared", "uint8", wcs_attrs_known,
        )

    elapsed = time.perf_counter() - t_start
    log.info(
        "Parallel ingest: %d/%d files ok (%d failed, %d skipped from checkpoint), "
        "%d spectra → %d tiles in %.1fs",
        n_files_ok, len(pending), n_files_fail, n_skipped,
        n_spectra_written, len(tile_groups), elapsed,
    )

    return {
        "n_files_requested": n_requested,
        "n_files_processed": len(pending),
        "n_files_skipped": n_skipped,
        "n_files_succeeded": n_files_ok,
        "n_files_failed": n_files_fail,
        "n_spectra": n_spectra_written,
        "n_tiles": len(tile_groups),
        "index_map": index_map,
        "failures": failures,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    from ..cli_utils import (
        config_option,
        configure_warning_filters,
        load_optional_config,
        pick,
        require_output_root,
    )

    @click.command("dl-ingest-spectra-batch")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option("--survey", "survey_name", required=True, help="Short survey name.")
    @click.option(
        "--file-list", "file_list", type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Plain-text file with one coadd FITS path per line "
             "(mutually exclusive with --coadd-root).  Relative paths are "
             "resolved against this file's directory, not the process cwd.",
    )
    @click.option(
        "--coadd-root", type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=None,
        help="Directory to recursively search for coadd files "
             "(mutually exclusive with --file-list).",
    )
    @click.option(
        "--coadd-glob", default="coadd-*.fits", show_default=True,
        help="Glob pattern under --coadd-root.",
    )
    @click.option(
        "--n-workers", required=True, type=int,
        help="Number of worker processes (typically cpu_count - 1).",
    )
    @click.option("--norder", default=None, type=int,
                  help="HEALPix order (overrides config; default 5).")
    @click.option(
        "--checkpoint", "checkpoint_path",
        type=click.Path(path_type=Path), default=None,
        help="JSON checkpoint of completed files.  "
             "Default: <output>/spectra/<survey>/.ingest_checkpoint.json",
    )
    @click.option(
        "--failures-log", "failures_log",
        type=click.Path(path_type=Path), default=None,
        help="JSONL log for per-file failures.  "
             "Default: <output>/spectra/<survey>/.ingest_failures.jsonl",
    )
    @click.option(
        "--update-catalog/--no-update-catalog", default=True, show_default=True,
        help="Patch _spectrum_index in the Parquet catalog at end "
             "(skipped silently if no catalog exists for this survey).",
    )
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        survey_name: str,
        file_list: Path | None,
        coadd_root: Path | None,
        coadd_glob: str,
        n_workers: int,
        norder: int | None,
        checkpoint_path: Path | None,
        failures_log: Path | None,
        update_catalog: bool,
        verbose: bool,
    ) -> None:
        """Parallel ingest of many DESI coadd FITS files.

        Decode + camera-coadd run in N worker processes; a single main-thread
        writer is the only process that touches the Zarr store.  Resumable
        via a JSON checkpoint of completed file paths.

        OUTPUT_ROOT is optional when a lake config is available (via --config
        or $DATA_LAKE_CONFIG); it defaults to ``<lake.root>``.
        """
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="[%(asctime)s] %(name)-26s %(levelname)-7s %(message)s",
            datefmt="%H:%M:%S",
        )
        configure_warning_filters()

        if (file_list is None) == (coadd_root is None):
            raise click.UsageError(
                "Pass exactly one of --file-list or --coadd-root."
            )
        if n_workers < 1:
            raise click.UsageError("--n-workers must be >= 1")

        cfg = load_optional_config(config_path)
        resolved_output = require_output_root(output_root, cfg, kind="spectra")
        resolved_norder = pick(
            norder, cfg.partitioning.hats_order if cfg else None, 5,
        )

        # Build file list (relative paths are anchored to the list file, not cwd)
        if file_list is not None:
            paths = _paths_from_file_list_file(file_list)
        else:
            assert coadd_root is not None
            paths = sorted(coadd_root.rglob(coadd_glob))
        if not paths:
            raise click.UsageError(
                "No FITS files found from the given source.  "
                "Check --file-list contents or --coadd-root/--coadd-glob."
            )
        log.info("Ingesting %d coadd files into survey=%r with %d workers.",
                 len(paths), survey_name, n_workers)

        survey_root = resolved_output / "spectra" / survey_name
        if checkpoint_path is None:
            checkpoint_path = survey_root / ".ingest_checkpoint.json"
        if failures_log is None:
            failures_log = survey_root / ".ingest_failures.jsonl"

        result = ingest_spectra_parallel(
            file_paths=paths,
            output_root=resolved_output,
            survey_name=survey_name,
            n_workers=n_workers,
            norder=resolved_norder,
            checkpoint_path=checkpoint_path,
            failures_log=failures_log,
        )

        click.echo(
            f"\nDone: {result['n_files_succeeded']}/{result['n_files_processed']} files "
            f"succeeded ({result['n_files_failed']} failed, "
            f"{result['n_files_skipped']} skipped from checkpoint).  "
            f"Wrote {result['n_spectra']} spectra to {result['n_tiles']} tile(s) "
            f"in {result['elapsed_s']:.1f}s."
        )
        if result["n_files_failed"]:
            click.echo(f"See failures log: {failures_log}")

        # Optional catalog patch
        if update_catalog and result["index_map"]:
            try:
                from data_lake.ingest.update_catalog_indices import update_index_column
                n_modified = update_index_column(
                    lake_root=resolved_output,
                    survey_name=survey_name,
                    source_id_to_index=result["index_map"],
                    kind="spectrum",
                    norder=resolved_norder,
                )
                click.echo(f"Patched _spectrum_index in {n_modified} catalog tile(s).")
            except FileNotFoundError:
                log.info(
                    "No catalog found for survey %r — skipping _spectrum_index patch.",
                    survey_name,
                )

except ImportError:
    cli = None  # type: ignore[assignment]
