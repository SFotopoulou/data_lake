"""
spectra_parallel_ingest – parallel file-list spectrum ingest (non-DESI formats).

Workers decode one FITS file at a time; a single main-thread writer appends to
``Npix=*.zarr`` tiles.  Use ``dl-ingest-spectra-batch-desi-coadds`` for DESI coadds.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections import OrderedDict, deque
from concurrent.futures import Executor, FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

from data_lake.ingest.desi_parallel_ingest import (
    TileBatch,
    WorkerResult,
    ZarrDuplicateMode,
    _atomic_write_json,
    _canonical_fits_path,
    _clear_parallel_inflight,
    _close_spectrum_tile_group,
    _load_tile_source_id_set,
    _recover_stale_parallel_commit,
    _truncate_spectrum_tile_row_arrays,
)
from data_lake.ingest.fits_to_parquet import healpix_dir
from data_lake.ingest.fits_to_spectra_zarr import (
    SpectrumDecodeConfig,
    _META_DTYPE,
    _open_or_create_spectrum_tile,
    _write_spectrum_info,
    append_tile_batch_to_zarr,
    decode_spectrum_file_safe,
)

log = logging.getLogger(__name__)


@dataclass
class _OpenSpectrumTile:
    root: Any
    tile_path: Path
    existing_ids: set[int] | None = None


class _SpectrumTileCache:
    """LRU cache of open spectrum tile groups for the parallel writer."""

    def __init__(
        self,
        *,
        survey_root: Path,
        norder: int,
        mask_dtype: np.dtype,
        wcs_attrs: dict,
        wavelength_mode: str,
        length_policy: str,
        on_duplicate: ZarrDuplicateMode,
        max_open: int,
        n_diag: int | None = None,
        res_offsets: np.ndarray | None = None,
    ) -> None:
        self._survey_root = survey_root
        self._norder = norder
        self._mask_dtype = mask_dtype
        self._wcs_attrs = wcs_attrs
        self._wavelength_mode = wavelength_mode
        self._length_policy = length_policy
        self._on_duplicate = on_duplicate
        self._max_open = max_open
        self._n_diag = n_diag
        self._res_offsets = res_offsets
        self._track_ids = on_duplicate != "append"
        self._tiles: OrderedDict[int, _OpenSpectrumTile] = OrderedDict()
        self.tiles_touched: set[int] = set()

    def get(self, npix: int, *, create_n_pix: int) -> _OpenSpectrumTile:
        if npix in self._tiles:
            self._tiles.move_to_end(npix)
            entry = self._tiles[npix]
            cur_w = int(entry.root["flux"].shape[1])
            if create_n_pix > cur_w:
                from data_lake.ingest.fits_to_spectra_zarr import widen_spectrum_tile

                if self._length_policy == "pad":
                    entry.root = widen_spectrum_tile(
                        entry.tile_path,
                        create_n_pix,
                        wavelength_mode=str(
                            entry.root.attrs.get("wavelength_mode", self._wavelength_mode)
                        ),
                        mask_dtype=self._mask_dtype,
                        wcs_attrs=self._wcs_attrs,
                        n_diag=self._n_diag,
                        resolution_offsets=self._res_offsets,
                    )
                else:
                    raise ValueError(
                        f"Tile Npix={npix} n_pix={cur_w} < incoming {create_n_pix}; "
                        f"use --on-length-mismatch pad"
                    )
            return entry

        tile_dir = self._survey_root / healpix_dir(self._norder, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path = tile_dir / f"Npix={npix}.zarr"
        tile_exists = tile_path.exists() and (tile_path / "zarr.json").exists()
        if tile_exists:
            import zarr

            store = zarr.storage.LocalStore(str(tile_path))
            root = zarr.open_group(store=store, mode="a", zarr_format=3)
            cur_w = int(root["flux"].shape[1])
            if create_n_pix > cur_w and self._length_policy == "pad":
                from data_lake.ingest.fits_to_spectra_zarr import widen_spectrum_tile

                root = widen_spectrum_tile(
                    tile_path,
                    create_n_pix,
                    wavelength_mode=str(
                        root.attrs.get("wavelength_mode", self._wavelength_mode)
                    ),
                    mask_dtype=self._mask_dtype,
                    wcs_attrs=self._wcs_attrs,
                    n_diag=self._n_diag,
                    resolution_offsets=self._res_offsets,
                )
        else:
            root = _open_or_create_spectrum_tile(
                tile_path,
                create_n_pix,
                self._wavelength_mode,
                self._mask_dtype,
                self._wcs_attrs,
                n_diag=self._n_diag,
                resolution_offsets=self._res_offsets,
            )

        existing_ids = (
            _load_tile_source_id_set(root) if self._track_ids else None
        )
        entry = _OpenSpectrumTile(
            root=root, tile_path=tile_path, existing_ids=existing_ids,
        )
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


def ingest_spectra_files_parallel(
    file_paths: Sequence[Path | str],
    *,
    output_root: Path | str,
    survey_name: str,
    n_workers: int,
    decode_config: SpectrumDecodeConfig,
    norder: int | None = None,
    checkpoint_path: Path | str | None = None,
    failures_log: Path | str | None = None,
    inflight_path: Path | str | None = None,
    show_progress: bool = True,
    skip_completed: bool = True,
    max_in_flight: int | None = None,
    max_open_tiles: int = 64,
    on_duplicate_source_id: ZarrDuplicateMode = "skip",
    executor_factory: Callable[[int], Executor] | None = None,
    decoder: Callable[[str, SpectrumDecodeConfig], WorkerResult] | None = None,
    heartbeat: "Any | None" = None,
    files_per_worker: int = 1,
) -> dict:
    """Parallel decode of spectrum FITS files with a single-thread Zarr writer."""
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if decode_config.with_resolution:
        raise ValueError(
            "--with-resolution is not supported with parallel spectrum file-list ingest."
        )
    if decode_config.fmt == "desi_coadd":
        raise ValueError(
            "Parallel file-list ingest does not support desi_coadd; "
            "use dl-ingest-spectra-batch-desi-coadds."
        )
    if files_per_worker < 1:
        raise ValueError("files_per_worker must be >= 1")

    output_root = Path(output_root)
    resolved_norder = int(decode_config.norder if norder is None else norder)
    survey_root = output_root / "spectra" / survey_name
    survey_root.mkdir(parents=True, exist_ok=True)

    mask_dtype = np.dtype(decode_config.mask_dtype)
    decoder = decoder or decode_spectrum_file_safe

    from data_lake.ingest.file_list_ingest import _append_checkpoint, _load_completed

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    failures_log = Path(failures_log) if failures_log else None
    completed: set[str] = (
        _load_completed(checkpoint_path) if skip_completed and checkpoint_path else set()
    )

    resolved_inflight = (
        Path(inflight_path).resolve()
        if inflight_path is not None
        else survey_root / ".ingest_inflight.json"
    )
    _recover_stale_parallel_commit(
        survey_root, resolved_norder, resolved_inflight, completed,
    )

    requested = [_canonical_fits_path(p) for p in file_paths]
    pending = sorted(p for p in requested if p not in completed)
    n_skipped_ckpt = len(requested) - len(pending)

    if not pending:
        return {
            "n_files_requested": len(requested),
            "n_files_processed": 0,
            "n_files_skipped": n_skipped_ckpt,
            "n_files_succeeded": 0,
            "n_files_failed": 0,
            "n_spectra": 0,
            "n_tiles": 0,
            "failures": [],
            "elapsed_s": 0.0,
        }

    if max_in_flight is None:
        max_in_flight = n_workers

    if executor_factory is None:
        from data_lake.cli_utils import init_parallel_ingest_subprocess as _init

        def _default_executor(nw: int) -> Executor:
            return ProcessPoolExecutor(max_workers=nw, initializer=_init)

        executor_factory = _default_executor

    failures: list[dict] = []
    n_files_ok = n_files_fail = 0
    n_spectra_written = 0
    tile_cache: _SpectrumTileCache | None = None
    wcs_attrs_known: dict | None = None
    wavelength_mode_run: str = decode_config.wavelength_mode
    n_pix_for_info = 0

    if failures_log is not None:
        failures_log.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    work_queue: deque[str] = deque(pending)
    in_flight: dict[Future, list[str]] = {}
    pool_recoveries = 0
    max_pool_recoveries = max(len(pending) * 8, 512)

    from data_lake.ingest.parallel_file_batch import decode_path_batch

    def _pop_path_batch() -> list[str]:
        batch: list[str] = []
        while work_queue and len(batch) < files_per_worker:
            batch.append(work_queue.popleft())
        return batch

    pool_cm = executor_factory(n_workers)
    pool: Executor = pool_cm.__enter__()

    def _recycle_pool_after_broken() -> None:
        nonlocal pool, pool_cm, pool_recoveries
        pool_recoveries += 1
        if pool_recoveries > max_pool_recoveries:
            raise RuntimeError(
                f"Process pool broke more than {max_pool_recoveries} times; aborting."
            ) from None
        log.warning("Process pool broke; recreating (recovery #%d).", pool_recoveries)
        try:
            pool_cm.__exit__(None, None, None)
        except Exception:
            log.exception("While shutting down broken process pool")
        pool_cm = executor_factory(n_workers)
        pool = pool_cm.__enter__()
        in_flight.clear()

    def _requeue_in_flight_paths() -> None:
        for _fut, batch_paths in list(in_flight.items()):
            for path_str in reversed(batch_paths):
                work_queue.appendleft(path_str)

    def _process_result(res: WorkerResult) -> None:
        nonlocal n_files_ok, n_files_fail, n_spectra_written, tile_cache
        nonlocal wcs_attrs_known, wavelength_mode_run, n_pix_for_info

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
                with failures_log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(fail_entry, ensure_ascii=False) + "\n")
            if heartbeat is not None:
                heartbeat.update(done=1, failed=1)
            return

        if not res.batches:
            n_files_ok += 1
            if checkpoint_path is not None:
                _append_checkpoint(checkpoint_path, res.path)
            _clear_parallel_inflight(resolved_inflight)
            return

        if wcs_attrs_known is None:
            wcs_attrs_known = res.wcs_attrs or {}
        if res.n_pix > n_pix_for_info:
            n_pix_for_info = res.n_pix

        if tile_cache is None:
            tile_cache = _SpectrumTileCache(
                survey_root=survey_root,
                norder=resolved_norder,
                mask_dtype=mask_dtype,
                wcs_attrs=wcs_attrs_known,
                wavelength_mode=decode_config.wavelength_mode,
                length_policy=decode_config.on_length_mismatch,
                on_duplicate=on_duplicate_source_id,
                max_open=max_open_tiles,
            )

        snap: dict[int, int] = {}
        for b in res.batches:
            create_n_pix = int(b.flux.shape[1]) if b.flux.size else res.n_pix
            open_tile = tile_cache.get(b.npix, create_n_pix=create_n_pix)
            snap[b.npix] = int(open_tile.root["flux"].shape[0])

        _atomic_write_json(
            resolved_inflight,
            {
                "commit": {
                    "path": res.path,
                    "tiles": {str(k): v for k, v in snap.items()},
                    "norder": int(resolved_norder),
                },
            },
        )

        file_wavelength_mode = decode_config.wavelength_mode
        if res.wavelength is not None:
            file_wavelength_mode = "shared"
        elif any(b.wavelength_rows is not None for b in res.batches):
            file_wavelength_mode = "per_source"

        for b in res.batches:
            create_n_pix = int(b.flux.shape[1]) if b.flux.size else res.n_pix
            open_tile = tile_cache.get(b.npix, create_n_pix=create_n_pix)
            n_appended = append_tile_batch_to_zarr(
                open_tile.tile_path,
                open_tile.root,
                b,
                file_n_pix=res.n_pix,
                wavelength_mode_effective=file_wavelength_mode,
                length_policy=decode_config.on_length_mismatch,
                mask_dtype=mask_dtype,
                wcs_attrs=wcs_attrs_known or {},
                on_duplicate=on_duplicate_source_id,
                shared_wavelength=res.wavelength,
                existing_ids=open_tile.existing_ids,
            )
            if n_appended:
                tile_cache.note_appended(b.npix, b.source_ids)
                n_spectra_written += n_appended

        n_files_ok += 1
        if heartbeat is not None:
            n_in_file = sum(b.flux.shape[0] for b in res.batches if b.flux.ndim >= 1)
            heartbeat.update(done=1, spectra=n_in_file)
        if checkpoint_path is not None:
            _append_checkpoint(checkpoint_path, res.path)
        _clear_parallel_inflight(resolved_inflight)

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    pbar = (
        tqdm(total=len(pending), disable=not show_progress, unit="file", desc="spectra")
        if tqdm is not None
        else None
    )
    try:
        while work_queue or in_flight:
            submit_broken = False
            while len(in_flight) < max_in_flight and work_queue:
                batch = _pop_path_batch()
                if not batch:
                    break
                try:
                    if len(batch) == 1:
                        fut = pool.submit(decoder, batch[0], decode_config)
                    else:
                        fut = pool.submit(
                            decode_path_batch, decoder, batch, decode_config,
                        )
                    in_flight[fut] = batch
                except BrokenProcessPool:
                    for p in reversed(batch):
                        work_queue.appendleft(p)
                    _requeue_in_flight_paths()
                    _recycle_pool_after_broken()
                    submit_broken = True
                    break
            if submit_broken:
                continue
            if not in_flight:
                break

            done_set, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            result_broken = False
            for fut in done_set:
                batch_paths = in_flight.pop(fut)
                try:
                    raw = fut.result()
                    results = raw if isinstance(raw, list) else [raw]
                except BrokenProcessPool:
                    for p in reversed(batch_paths):
                        work_queue.appendleft(p)
                    _requeue_in_flight_paths()
                    _recycle_pool_after_broken()
                    result_broken = True
                    break
                except Exception as exc:
                    results = [
                        WorkerResult(
                            path=p,
                            ok=False,
                            error=f"executor: {type(exc).__name__}: {exc}",
                            tb=traceback.format_exc(),
                        )
                        for p in batch_paths
                    ]
                for res in results:
                    _process_result(res)
                if pbar is not None:
                    pbar.update(len(batch_paths))
            if result_broken:
                continue
    finally:
        if pbar is not None:
            pbar.close()
        try:
            pool_cm.__exit__(None, None, None)
        except Exception:
            log.exception("While shutting down process pool")
        if tile_cache is not None:
            tile_cache.close_all()

    if tile_cache is not None and tile_cache.tiles_touched and wcs_attrs_known is not None:
        _write_spectrum_info(
            survey_root,
            survey_name,
            resolved_norder,
            max(n_pix_for_info, 1),
            wavelength_mode_run,
            str(mask_dtype),
            wcs_attrs_known,
            on_duplicate_source_id=on_duplicate_source_id,
            total_rows=n_spectra_written,
        )

    elapsed = time.perf_counter() - t_start
    n_tiles = len(tile_cache.tiles_touched) if tile_cache else 0

    return {
        "n_files_requested": len(requested),
        "n_files_processed": len(pending),
        "n_files_skipped": n_skipped_ckpt,
        "n_files_succeeded": n_files_ok,
        "n_files_failed": n_files_fail,
        "n_spectra": n_spectra_written,
        "n_tiles": n_tiles,
        "failures": failures,
        "elapsed_s": elapsed,
    }
