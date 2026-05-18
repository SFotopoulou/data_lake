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
                          inflight journal (per-tile row snapshot) → append
                          track {source_id: (npix, local_idx)} for catalog patch
                      checkpoint after append (then clear inflight)
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
    wait,
    FIRST_COMPLETED,
)
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

from data_lake.ingest.checkpoint_sidecars import paths_from_file_list_file as _paths_from_file_list_file
from data_lake.ingest.fits_to_parquet import assign_healpix, healpix_dir
from data_lake.ingest.fits_to_spectra_zarr import (
    _META_DTYPE,
    _meta_to_bytes,
    _open_or_create_spectrum_tile,
    _read_desi_with_desispec,
    _write_spectrum_info,
)

log = logging.getLogger(__name__)

ZarrDuplicateMode = Literal["append", "error", "skip"]


def _filter_tile_batch(
    batch: "TileBatch",
    existing_source_ids: np.ndarray,
    on_duplicate: ZarrDuplicateMode,
) -> "TileBatch | None":
    """Apply ``--on-duplicate`` policy before appending one parallel worker batch."""
    from data_lake.ingest.duplicate_policy import zarr_row_keep_mask

    keep = zarr_row_keep_mask(batch.source_ids, existing_source_ids, on_duplicate)
    if not keep.any():
        return None
    if keep.all():
        return batch
    idx = np.nonzero(keep)[0]
    itemsize = _META_DTYPE.itemsize
    meta_parts = [
        batch.meta_bytes[int(i) * itemsize : int(i) * itemsize + itemsize]
        for i in idx.tolist()
    ]
    return TileBatch(
        npix=batch.npix,
        flux=batch.flux[idx],
        ivar=batch.ivar[idx],
        mask=batch.mask[idx],
        source_ids=batch.source_ids[idx],
        meta_bytes=b"".join(meta_parts),
    )


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
    from data_lake.cli_utils import apply_parallel_worker_logging_after_heavy_imports

    apply_parallel_worker_logging_after_heavy_imports()

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


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically by writing to a tempfile then renaming."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


@dataclass
class _OpenTile:
    """One HEALPix tile Zarr group held open by the parallel writer."""
    root: Any
    existing_ids: set[int] | None = None


def _close_spectrum_tile_group(root: Any) -> None:
    """Release Zarr file handles for a tile (best-effort)."""
    store = getattr(root, "store", None)
    if store is None:
        return
    close = getattr(store, "close", None)
    if callable(close):
        close()


def _load_tile_source_id_set(root: Any) -> set[int]:
    """Read ``source_id`` into a set once when a tile is opened for duplicate checks."""
    n = int(root["source_id"].shape[0])
    if n == 0:
        return set()
    return set(np.asarray(root["source_id"][:], dtype=np.int64).tolist())


class _TileGroupCache:
    """LRU cache of open tile Zarr groups for the single-thread writer.

    Without eviction the writer keeps every tile it has ever touched open for
    the whole run (file descriptors + metadata).  For all-sky DESI ingest that
    can mean thousands of open stores and multi-GB of cached ``source_id`` reads
    when ``--on-duplicate skip``.
    """

    def __init__(
        self,
        *,
        survey_root: Path,
        norder: int,
        n_pix_known: int,
        wcs_attrs_known: dict,
        mask_dtype: np.dtype,
        on_duplicate: ZarrDuplicateMode,
        max_open: int,
    ) -> None:
        self._survey_root = survey_root
        self._norder = norder
        self._n_pix_known = n_pix_known
        self._wcs_attrs_known = wcs_attrs_known
        self._mask_dtype = mask_dtype
        self._track_ids = on_duplicate != "append"
        self._max_open = max_open
        self._tiles: OrderedDict[int, _OpenTile] = OrderedDict()
        self.tiles_touched: set[int] = set()

    def get(self, npix: int) -> _OpenTile:
        if npix in self._tiles:
            self._tiles.move_to_end(npix)
            return self._tiles[npix]

        tile_dir = self._survey_root / healpix_dir(self._norder, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path = tile_dir / f"Npix={npix}.zarr"
        root = _open_or_create_spectrum_tile(
            tile_path,
            self._n_pix_known,
            "shared",
            self._mask_dtype,
            self._wcs_attrs_known,
        )
        existing_ids = (
            _load_tile_source_id_set(root) if self._track_ids else None
        )
        entry = _OpenTile(root=root, existing_ids=existing_ids)
        self._tiles[npix] = entry
        self.tiles_touched.add(npix)
        self._evict_if_needed()
        return entry

    def note_appended(self, npix: int, source_ids: np.ndarray) -> None:
        entry = self._tiles.get(npix)
        if entry is None or entry.existing_ids is None:
            return
        entry.existing_ids.update(int(s) for s in source_ids.tolist())

    def close_all(self) -> None:
        for entry in self._tiles.values():
            _close_spectrum_tile_group(entry.root)
        self._tiles.clear()

    def _evict_if_needed(self) -> None:
        while self._max_open > 0 and len(self._tiles) > self._max_open:
            _npix, oldest = self._tiles.popitem(last=False)
            _close_spectrum_tile_group(oldest.root)


def _configure_file_logging(log_file: Path, *, verbose: bool) -> None:
    """Send all logging to ``log_file`` and detach the root logger from stderr.

    The progress bar lives on stderr; routing logs to a file keeps the bar
    uncorrupted by interleaved log lines.  Any pre-existing handlers are
    replaced so repeated invocations (tests, REPL) stay consistent.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] %(name)-26s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=[handler],
        force=True,
    )


def _checkpoint_jsonl_path(checkpoint_path: Path) -> Path:
    """Append-only completion log (one path per line) for large batch runs."""
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".jsonl")


def _append_checkpoint_path(checkpoint_path: Path, path_done: str) -> None:
    """Record one completed FITS path without rewriting a giant JSON array."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with _checkpoint_jsonl_path(checkpoint_path).open("a", encoding="utf-8") as fh:
        fh.write(path_done + "\n")


def _load_checkpoint(path: Path | None) -> set[str]:
    if path is None:
        return set()
    out: set[str] = set()
    if path.exists():
        try:
            data = json.loads(path.read_text())
            for item in data.get("completed", []):
                if not item or not isinstance(item, str):
                    continue
                try:
                    out.add(_canonical_fits_path(item))
                except OSError:
                    out.add(item)
        except Exception:
            log.warning("Could not parse checkpoint %s; ignoring JSON body.", path)
    jsonl = _checkpoint_jsonl_path(path)
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.add(_canonical_fits_path(line))
            except OSError:
                out.add(line)
    return out


# Row-aligned arrays in a spectrum tile (2-D source axis + wavelength metadata).
_SPECTRUM_ROW_ALIGNED = ("flux", "ivar", "mask", "source_id", "meta")


def _truncate_spectrum_tile_row_arrays(root: Any, n_rows: int) -> None:
    """Shrink every per-source array so its first dimension equals ``n_rows``."""
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative")
    for name in _SPECTRUM_ROW_ALIGNED:
        arr = root[name]
        shape = list(arr.shape)
        cur = int(shape[0])
        if cur == n_rows:
            continue
        if n_rows > cur:
            raise ValueError(
                f"Cannot truncate {name!r} to {n_rows} rows (current {cur})."
            )
        shape[0] = n_rows
        arr.resize(tuple(shape))


def _clear_parallel_inflight(path: Path) -> None:
    """Mark no in-flight file commit (atomic JSON write)."""
    _atomic_write_json(path, {"commit": None})


def _recover_stale_parallel_commit(
    survey_root: Path,
    norder: int,
    inflight_path: Path,
    completed: set[str],
) -> None:
    """If a previous run died mid-file, truncate Zarr rows then clear the journal.

    A commit journal entry records per-tile row counts *before* appending a
    coadd.  If that coadd's path is already listed in ``completed``, the
    append phase finished and the checkpoint was persisted (clear journal only).
    Otherwise we rewind each listed tile to the saved row counts so the next
    run can re-ingest the whole file without duplicates.
    """
    if not inflight_path.exists():
        return
    try:
        data = json.loads(inflight_path.read_text())
    except Exception as exc:
        log.warning(
            "Could not parse inflight journal %s (%s); removing file.",
            inflight_path, exc,
        )
        try:
            inflight_path.unlink()
        except OSError:
            pass
        return

    commit = data.get("commit")
    if not commit or not isinstance(commit, dict):
        return

    path_raw = commit.get("path")
    if not path_raw or not isinstance(path_raw, str):
        _clear_parallel_inflight(inflight_path)
        return

    try:
        path_canon = _canonical_fits_path(path_raw)
    except OSError:
        path_canon = path_raw

    if path_canon in completed:
        _clear_parallel_inflight(inflight_path)
        return

    tiles = commit.get("tiles")
    if not isinstance(tiles, dict):
        _clear_parallel_inflight(inflight_path)
        return

    stored_norder = int(commit.get("norder", norder))
    if stored_norder != int(norder):
        log.warning(
            "Inflight journal norder=%s differs from this run (%s); "
            "recovery uses this run's norder for tile paths.",
            stored_norder, norder,
        )

    import zarr

    for npix_str, start_rows in tiles.items():
        try:
            npix = int(npix_str)
            nrow = int(start_rows)
        except (TypeError, ValueError):
            continue
        tile_dir = survey_root / healpix_dir(norder, npix)
        tile_path = tile_dir / f"Npix={npix}.zarr"
        if not (tile_path / "zarr.json").exists():
            continue
        store = zarr.storage.LocalStore(str(tile_path))
        root = zarr.open_group(store=store, mode="a", zarr_format=3)
        try:
            _truncate_spectrum_tile_row_arrays(root, nrow)
        except Exception as exc:
            log.warning(
                "Inflight recovery: failed to truncate Npix=%s to %s rows: %s",
                npix, nrow, exc,
            )

    _clear_parallel_inflight(inflight_path)


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
    worker_log_file: Path | str | None = None,
    worker_verbose: bool = False,
    inflight_path: Path | str | None = None,
    on_duplicate_source_id: ZarrDuplicateMode = "append",
    track_index_map: bool = False,
    max_in_flight: int | None = None,
    max_open_tiles: int = 64,
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
    worker_log_file:
        When using the default process pool, library loggers in child
        processes (e.g. ``desispec``) append here; same file as the main CLI
        log is typical.  Ignored when ``executor_factory`` is overridden.
    worker_verbose:
        If true, worker-side library log level follows DEBUG when combined with
        ``worker_log_file``; passed with ``worker_log_file`` via subprocess env.
    inflight_path:
        JSON journal used for crash-safe per-file commits (default
        ``<survey_root>/.ingest_inflight.json``).  Pass ``None`` for that default.
    on_duplicate_source_id:
        ``append`` (default), ``error``, or ``skip`` when a ``TARGETID`` is already
        in a tile's Zarr (same as ``dl-ingest-spectra --on-duplicate``).
    track_index_map:
        When True, accumulate ``{source_id: local_index}`` in RAM for the return
        value and optional catalog patch.  Default False — for millions of spectra
        this dict can exhaust memory (use :func:`update_index_column_from_zarr_tiles`
        after ingest instead).
    max_in_flight:
        Max decoder futures in flight (default ``n_workers + 2``).  Lower if the
        process or terminal is killed under memory pressure (e.g. systemd-oomd).
    max_open_tiles:
        Max HEALPix tile Zarr groups kept open in the writer (default 64).
        Use ``0`` for unlimited (not recommended on 10k+ coadd runs).  Evicted
        tiles are closed and re-opened on the next write; duplicate-ID sets are
        reloaded when ``on_duplicate`` is not ``append``.

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

    **Crash safety:** before appending a coadd, the writer records each touched
    tile's current row count in ``inflight_path`` (atomic JSON).  After all
    appends for that file succeed, it updates the checkpoint (if enabled) and
    then clears the inflight journal.  On startup, if the journal names a file
    that is not yet in the checkpoint, each listed tile is truncated back to
    the saved row counts so the coadd can be re-ingested without duplicate rows.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(x, **_kw):
            return x

    output_root = Path(output_root)
    survey_root = output_root / "spectra" / survey_name
    survey_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    failures_log = Path(failures_log) if failures_log else None

    completed: set[str] = _load_checkpoint(checkpoint_path)
    resolved_inflight = (
        Path(inflight_path).resolve()
        if inflight_path is not None
        else survey_root / ".ingest_inflight.json"
    )
    _recover_stale_parallel_commit(survey_root, norder, resolved_inflight, completed)

    requested = [_canonical_fits_path(p) for p in file_paths]
    n_requested = len(requested)

    # Sorted submission improves average read locality on spinning disks when
    # workers pull the next path from the queue in order.
    pending = sorted(p for p in requested if p not in completed)
    n_skipped = n_requested - len(pending)
    if n_skipped:
        log.info("Checkpoint has %d/%d files; resuming with %d pending.",
                 n_skipped, n_requested, len(pending))

    if max_in_flight is None:
        max_in_flight = n_workers + 2
    if len(pending) > 500 and track_index_map:
        log.warning(
            "track_index_map=True with %d pending files can use many GB of RAM "
            "for the returned index map; prefer track_index_map=False and "
            "update_index_column_from_zarr_tiles() after ingest.",
            len(pending),
        )

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

    uses_parallel_subprocess_workers = executor_factory is None
    if executor_factory is None:
        from data_lake.cli_utils import init_parallel_ingest_subprocess as _init

        def _default_executor(nw: int) -> Executor:
            # initializer runs once in each worker process — filter state
            # does not propagate from the parent, so set it up there too.
            return ProcessPoolExecutor(max_workers=nw, initializer=_init)

        executor_factory = _default_executor

    # --- Writer state ---
    tile_cache: _TileGroupCache | None = None
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

    _env_prev_parallel: dict[str, str | None] = {}
    if uses_parallel_subprocess_workers:
        from data_lake.cli_utils import (
            PARALLEL_WORKER_LOG_FILE_ENV,
            PARALLEL_WORKER_VERBOSE_ENV,
        )

        _env_prev_parallel[PARALLEL_WORKER_LOG_FILE_ENV] = os.environ.get(
            PARALLEL_WORKER_LOG_FILE_ENV
        )
        _env_prev_parallel[PARALLEL_WORKER_VERBOSE_ENV] = os.environ.get(
            PARALLEL_WORKER_VERBOSE_ENV
        )
        os.environ[PARALLEL_WORKER_LOG_FILE_ENV] = (
            str(Path(worker_log_file).resolve()) if worker_log_file else ""
        )
        os.environ[PARALLEL_WORKER_VERBOSE_ENV] = "1" if worker_verbose else "0"

    path_iter = iter(pending)
    in_flight: dict[Future, str] = {}

    def _submit_more(pool: Executor) -> None:
        while len(in_flight) < max_in_flight:
            try:
                p = next(path_iter)
            except StopIteration:
                break
            fut = pool.submit(decoder, p, norder)
            in_flight[fut] = p

    def _process_result(path_str: str, res: WorkerResult) -> None:
        nonlocal n_files_ok, n_files_fail, n_spectra_written, n_pix_known, wcs_attrs_known, tile_cache

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
            return

        if not res.batches:
            n_files_ok += 1
            completed.add(res.path)
            if checkpoint_path is not None:
                _append_checkpoint_path(checkpoint_path, res.path)
            _clear_parallel_inflight(resolved_inflight)
            return

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
            return

        if tile_cache is None:
            tile_cache = _TileGroupCache(
                survey_root=survey_root,
                norder=norder,
                n_pix_known=n_pix_known,
                wcs_attrs_known=wcs_attrs_known or {},
                mask_dtype=np.dtype(np.uint8),
                on_duplicate=on_duplicate_source_id,
                max_open=max_open_tiles,
            )

        snap: dict[int, int] = {}
        for b in res.batches:
            root = tile_cache.get(b.npix).root
            snap[b.npix] = int(root["flux"].shape[0])

        _atomic_write_json(
            resolved_inflight,
            {
                "commit": {
                    "path": res.path,
                    "tiles": {str(k): v for k, v in snap.items()},
                    "norder": int(norder),
                },
            },
        )

        for b in res.batches:
            open_tile = tile_cache.get(b.npix)
            root = open_tile.root
            existing = open_tile.existing_ids
            if existing is None:
                existing_for_filter: set[int] | np.ndarray = np.array([], dtype=np.int64)
            else:
                existing_for_filter = existing
            filtered = _filter_tile_batch(
                b, existing_for_filter, on_duplicate_source_id,
            )
            if filtered is None:
                continue
            b = filtered
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
            tile_cache.note_appended(b.npix, b.source_ids)

            if start_idx == 0 and res.wavelength is not None:
                root["wavelength"][:] = res.wavelength.astype(np.float64)

            if track_index_map:
                for i, sid in enumerate(b.source_ids.tolist()):
                    index_map[int(sid)] = start_idx + i
            n_spectra_written += int(b.source_ids.size)

        n_files_ok += 1
        completed.add(res.path)
        if checkpoint_path is not None:
            _append_checkpoint_path(checkpoint_path, res.path)
        _clear_parallel_inflight(resolved_inflight)

    try:
        with executor_factory(n_workers) as pool:
            _submit_more(pool)
            with tqdm(
                total=len(pending),
                disable=not show_progress,
                unit="file",
                desc="ingest",
            ) as pbar:
                while in_flight:
                    done_set, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in done_set:
                        path_str = in_flight.pop(fut)
                        try:
                            res = fut.result()
                        except Exception as exc:
                            res = WorkerResult(
                                path=path_str, ok=False,
                                error=f"executor: {type(exc).__name__}: {exc}",
                                tb=traceback.format_exc(),
                            )
                        _process_result(path_str, res)
                        pbar.update(1)
                        _submit_more(pool)

    finally:
        if tile_cache is not None:
            tile_cache.close_all()
        if uses_parallel_subprocess_workers:
            for _k, _v in _env_prev_parallel.items():
                if _v is None:
                    os.environ.pop(_k, None)
                else:
                    os.environ[_k] = _v

    n_tiles_touched = len(tile_cache.tiles_touched) if tile_cache is not None else 0

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
        n_spectra_written, n_tiles_touched, elapsed,
    )

    return {
        "n_files_requested": n_requested,
        "n_files_processed": len(pending),
        "n_files_skipped": n_skipped,
        "n_files_succeeded": n_files_ok,
        "n_files_failed": n_files_fail,
        "n_spectra": n_spectra_written,
        "n_tiles": n_tiles_touched,
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
        ingest_token_option,
        load_optional_config,
        pick,
        require_ingest_permission,
        require_output_root,
    )

    @click.command("dl-ingest-spectra-batch")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @ingest_token_option
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
    @click.option(
        "--max-in-flight", "max_in_flight", default=None, type=int,
        help="Max coadd decodes queued ahead of the writer "
             "(default: n_workers + 2).  Lower to reduce peak RAM.",
    )
    @click.option(
        "--max-open-tiles", "max_open_tiles", default=64, show_default=True,
        type=int,
        help="Max HEALPix tile Zarr groups open in the writer; "
             "0 = unlimited (not recommended for large runs).",
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
        "--log-file", "log_file_path",
        type=click.Path(path_type=Path), default=None,
        help="File to receive INFO/DEBUG logs (terminal stays clean for the "
             "progress bar).  "
             "Default: <output>/spectra/<survey>/.ingest.log",
    )
    @click.option(
        "--on-duplicate",
        type=click.Choice(["append", "error", "skip"]),
        default="append",
        show_default=True,
        help="If TARGETID already exists in a tile Zarr: append, raise, or skip.",
    )
    @click.option(
        "--update-catalog/--no-update-catalog", default=True, show_default=True,
        help="Patch _spectrum_index in the Parquet catalog at end "
             "(skipped silently if no catalog exists for this survey).",
    )
    @click.option("-v", "--verbose", is_flag=True,
                  help="Use DEBUG level in the log file (no terminal effect).")
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        ingest_token: str | None,
        survey_name: str,
        file_list: Path | None,
        coadd_root: Path | None,
        coadd_glob: str,
        n_workers: int,
        max_in_flight: int | None,
        max_open_tiles: int,
        norder: int | None,
        checkpoint_path: Path | None,
        failures_log: Path | None,
        log_file_path: Path | None,
        on_duplicate: str,
        update_catalog: bool,
        verbose: bool,
    ) -> None:
        """Parallel ingest of many DESI coadd FITS files.

        Decode + camera-coadd run in N worker processes; a single main-thread
        writer is the only process that touches the Zarr store.  Resumable
        via a JSON checkpoint of completed file paths.

        OUTPUT_ROOT is optional when a lake config is available (via --config
        or $DATA_LAKE_CONFIG); it defaults to ``<lake.root>``.

        Logging is silent on the terminal — all INFO/DEBUG records go to the
        log file so the tqdm progress bar is the only thing on stderr.
        """
        if (file_list is None) == (coadd_root is None):
            raise click.UsageError(
                "Pass exactly one of --file-list or --coadd-root."
            )
        if n_workers < 1:
            raise click.UsageError("--n-workers must be >= 1")
        if max_in_flight is not None and max_in_flight < 1:
            raise click.UsageError("--max-in-flight must be >= 1")
        if max_open_tiles < 0:
            raise click.UsageError("--max-open-tiles must be >= 0")

        cfg = load_optional_config(config_path)
        require_ingest_permission(cfg, ingest_token)
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

        survey_root = resolved_output / "spectra" / survey_name
        if checkpoint_path is None:
            checkpoint_path = survey_root / ".ingest_checkpoint.json"
        if failures_log is None:
            failures_log = survey_root / ".ingest_failures.jsonl"
        if log_file_path is None:
            log_file_path = survey_root / ".ingest.log"

        _configure_file_logging(log_file_path, verbose=verbose)
        configure_warning_filters()

        click.echo(
            f"Ingesting {len(paths)} coadd files into survey={survey_name!r} "
            f"with {n_workers} workers.  Logs → {log_file_path}"
        )
        log.info("Ingesting %d coadd files into survey=%r with %d workers.",
                 len(paths), survey_name, n_workers)

        result = ingest_spectra_parallel(
            file_paths=paths,
            output_root=resolved_output,
            survey_name=survey_name,
            n_workers=n_workers,
            norder=resolved_norder,
            checkpoint_path=checkpoint_path,
            failures_log=failures_log,
            worker_log_file=log_file_path,
            worker_verbose=verbose,
            on_duplicate_source_id=on_duplicate,  # type: ignore[arg-type]
            max_in_flight=max_in_flight,
            max_open_tiles=max_open_tiles,
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

        # Optional catalog patch (scan Zarr tiles; avoids a multi-GB in-memory index map)
        if update_catalog:
            try:
                from data_lake.ingest.update_catalog_indices import (
                    update_index_column_from_zarr_tiles,
                )
                n_modified = update_index_column_from_zarr_tiles(
                    lake_root=resolved_output,
                    survey_name=survey_name,
                    kind="spectrum",
                    norder=resolved_norder,
                )
                click.echo(f"Patched _spectrum_index in {n_modified} catalog tile(s).")
            except FileNotFoundError as exc:
                log.info(
                    "Skipping _spectrum_index patch (%s).",
                    exc,
                )

except ImportError:
    cli = None  # type: ignore[assignment]
