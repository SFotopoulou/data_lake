"""
cutout_parallel_ingest – parallel file-list cutout ingest.

Workers decode FITS cutouts; a single main-thread writer appends to Zarr tiles.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections import OrderedDict, deque
from concurrent.futures import Executor, FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

from data_lake.ingest.fits_to_parquet import healpix_dir
from data_lake.ingest.fits_to_zarr import (
    CutoutRecord,
    _extract_records_from_hdul,
    _filter_tile_records_duplicates,
    _open_or_create_tile_store,
    _wcs_params_to_structured,
)
from data_lake.ingest.zarr_ids import zarr_join_array
from data_lake.io.fits_read import default_fits_read_policy, open_fits

log = logging.getLogger(__name__)

CutoutDuplicateMode = Literal["append", "error", "skip"]


@dataclass(frozen=True)
class CutoutDecodeConfig:
    norder: int
    ra_col: str
    dec_col: str
    link_id_col: str | None
    image_hdu_index: int
    band_axis: int | None
    band_names: tuple[str, ...] | None
    dtype: str
    on_duplicate: CutoutDuplicateMode
    fits_memmap: str = "auto"


@dataclass
class CutoutTileBatch:
    npix: int
    images: np.ndarray
    source_ids: np.ndarray
    wcs_bytes: np.ndarray


@dataclass
class CutoutWorkerResult:
    path: str
    ok: bool
    batches: list[CutoutTileBatch] = field(default_factory=list)
    error: str | None = None
    tb: str | None = None
    elapsed_s: float = 0.0


def _decode_cutout_file(path_str: str, config: CutoutDecodeConfig) -> CutoutWorkerResult:
    from data_lake.cli_utils import apply_parallel_worker_logging_after_heavy_imports
    from data_lake.ingest.fits_to_parquet import assign_healpix

    apply_parallel_worker_logging_after_heavy_imports()
    t0 = time.perf_counter()
    dtype = np.dtype(config.dtype)
    with open_fits(path_str, default_fits_read_policy(config.fits_memmap)) as hdul:
        records = _extract_records_from_hdul(
            hdul,
            config.ra_col,
            config.dec_col,
            config.image_hdu_index,
            config.band_axis,
            dtype,
            link_id_col=config.link_id_col,
        )
    if not records:
        return CutoutWorkerResult(path=path_str, ok=True, elapsed_s=time.perf_counter() - t0)

    tile_groups: dict[int, list[CutoutRecord]] = {}
    for rec in records:
        pix = int(assign_healpix(np.array([rec.ra]), np.array([rec.dec]), config.norder)[0])
        tile_groups.setdefault(pix, []).append(rec)

    batches: list[CutoutTileBatch] = []
    for npix, tile_records in tile_groups.items():
        images = np.stack([r.image for r in tile_records], axis=0).astype(dtype)
        sids = np.array([r.source_id for r in tile_records], dtype=np.int64)
        wcs_raw = np.concatenate(
            [_wcs_params_to_structured(r.wcs_params) for r in tile_records], axis=0,
        )
        wcs_bytes = wcs_raw.view("|V" + str(wcs_raw.dtype.itemsize))
        batches.append(CutoutTileBatch(npix=npix, images=images, source_ids=sids, wcs_bytes=wcs_bytes))

    return CutoutWorkerResult(
        path=path_str,
        ok=True,
        batches=batches,
        elapsed_s=time.perf_counter() - t0,
    )


def _decode_cutout_file_safe(path_str: str, config: CutoutDecodeConfig) -> CutoutWorkerResult:
    try:
        return _decode_cutout_file(path_str, config)
    except Exception as exc:
        return CutoutWorkerResult(
            path=path_str,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            tb=traceback.format_exc(),
        )


def _decode_cutout_batch_safe(
    paths: list[str],
    config: CutoutDecodeConfig,
) -> list[CutoutWorkerResult]:
    from data_lake.ingest.parallel_file_batch import decode_path_batch

    return decode_path_batch(_decode_cutout_file_safe, paths, config)


@dataclass
class _OpenCutoutTile:
    root: Any
    tile_path: Path
    existing_ids: set[int] | None = None


class _CutoutTileCache:
    def __init__(
        self,
        *,
        survey_root: Path,
        norder: int,
        dtype: np.dtype,
        on_duplicate: CutoutDuplicateMode,
        max_open: int,
    ) -> None:
        self._survey_root = survey_root
        self._norder = norder
        self._dtype = dtype
        self._on_duplicate = on_duplicate
        self._max_open = max_open
        self._track_ids = on_duplicate != "append"
        self._tiles: OrderedDict[int, _OpenCutoutTile] = OrderedDict()
        self.tiles_touched: set[int] = set()

    def append_batch(self, batch: CutoutTileBatch) -> None:
        npix = int(batch.npix)
        tile_dir = self._survey_root / healpix_dir(self._norder, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path = tile_dir / f"Npix={npix}.zarr"
        if npix not in self._tiles:
            n_b, h, w = batch.images.shape[1:]
            root = _open_or_create_tile_store(tile_path, n_b, h, w, self._dtype)
            existing: set[int] | None = None
            if self._track_ids:
                sid_arr = zarr_join_array(root)
                existing = set(np.asarray(sid_arr[:]).tolist()) if sid_arr.shape[0] else set()
            self._tiles[npix] = _OpenCutoutTile(root=root, tile_path=tile_path, existing_ids=existing)
            if len(self._tiles) > self._max_open:
                old_npix, old = self._tiles.popitem(last=False)
                log.debug("Closed cutout tile Npix=%d (LRU)", old_npix)
        entry = self._tiles[npix]
        self._tiles.move_to_end(npix)
        root = entry.root
        images_arr = root["images"]
        sid_arr = zarr_join_array(root)
        wcs_arr = root["wcs"]
        existing = entry.existing_ids or set()
        # Rebuild pseudo-records for duplicate filter
        from data_lake.ingest.fits_to_zarr import CutoutRecord

        pseudo = [
            CutoutRecord(
                source_id=int(s),
                ra=0.0,
                dec=0.0,
                image=batch.images[i],
                wcs_params={},
            )
            for i, s in enumerate(batch.source_ids.tolist())
        ]
        kept = _filter_tile_records_duplicates(pseudo, existing, self._on_duplicate)
        if not kept:
            return
        idxs = [pseudo.index(r) for r in kept]
        imgs = batch.images[idxs]
        sids = batch.source_ids[idxs]
        wcs_b = batch.wcs_bytes[idxs]
        images_arr.append(imgs)
        sid_arr.append(sids)
        wcs_arr.append(wcs_b)
        if entry.existing_ids is not None:
            entry.existing_ids.update(int(s) for s in sids.tolist())
        self.tiles_touched.add(npix)


def ingest_cutouts_parallel(
    file_paths: Sequence[Path | str],
    *,
    output_root: Path | str,
    survey_name: str,
    n_workers: int,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    link_id_col: str | None = None,
    image_hdu_index: int = 0,
    band_axis: int | None = None,
    band_names: Sequence[str] | None = None,
    norder: int = 5,
    dtype: np.dtype | type = np.float32,
    on_duplicate: CutoutDuplicateMode = "skip",
    checkpoint_path: Path | str | None = None,
    failures_log: Path | str | None = None,
    show_progress: bool = True,
    skip_completed: bool = True,
    max_in_flight: int | None = None,
    max_open_tiles: int = 64,
    files_per_worker: int = 1,
    partition_by_dir: bool = False,
    fits_memmap: str = "auto",
) -> dict:
    """Parallel cutout ingest with a single Zarr writer thread."""
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if files_per_worker < 1:
        raise ValueError("files_per_worker must be >= 1")

    from data_lake.cli_utils import warn_high_parallelism_on_slow_storage
    from data_lake.ingest.file_list_ingest import _append_checkpoint, _load_completed
    from data_lake.ingest.parallel_file_batch import paths_grouped_by_directory

    output_root = Path(output_root)
    survey_root = output_root / "cutouts" / survey_name
    survey_root.mkdir(parents=True, exist_ok=True)
    warn_high_parallelism_on_slow_storage(n_workers, output_root)

    decode_cfg = CutoutDecodeConfig(
        norder=norder,
        ra_col=ra_col,
        dec_col=dec_col,
        link_id_col=link_id_col,
        image_hdu_index=image_hdu_index,
        band_axis=band_axis,
        band_names=tuple(band_names) if band_names else None,
        dtype=str(np.dtype(dtype)),
        on_duplicate=on_duplicate,
        fits_memmap=fits_memmap,
    )

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    failures_log = Path(failures_log) if failures_log else None
    completed: set[str] = (
        _load_completed(checkpoint_path) if skip_completed and checkpoint_path else set()
    )
    pending = sorted(str(Path(p).resolve()) for p in file_paths if str(Path(p).resolve()) not in completed)
    if partition_by_dir:
        pending = paths_grouped_by_directory(pending)

    if max_in_flight is None:
        max_in_flight = n_workers

    failures: list[dict] = []
    n_ok = n_fail = 0
    tile_cache: _CutoutTileCache | None = None
    t0 = time.perf_counter()
    work_queue: deque[str] = deque(pending)
    in_flight: dict[Future, list[str]] = {}

    from data_lake.cli_utils import init_parallel_ingest_subprocess

    def _pop_batch() -> list[str]:
        batch: list[str] = []
        while work_queue and len(batch) < files_per_worker:
            batch.append(work_queue.popleft())
        return batch

    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_parallel_ingest_subprocess) as pool:
        while work_queue or in_flight:
            while len(in_flight) < max_in_flight and work_queue:
                batch = _pop_batch()
                if batch:
                    in_flight[pool.submit(_decode_cutout_batch_safe, batch, decode_cfg)] = batch
            if not in_flight:
                break
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                batch_paths = in_flight.pop(fut)
                try:
                    results = fut.result()
                except Exception as exc:
                    for path_str in batch_paths:
                        n_fail += 1
                        failures.append({"path": path_str, "error": str(exc)})
                    continue
                if not isinstance(results, list):
                    results = [results]
                for res in results:
                    if not res.ok:
                        n_fail += 1
                        failures.append({"path": res.path, "error": res.error})
                        continue
                    if tile_cache is None:
                        tile_cache = _CutoutTileCache(
                            survey_root=survey_root,
                            norder=norder,
                            dtype=np.dtype(dtype),
                            on_duplicate=on_duplicate,
                            max_open=max_open_tiles,
                        )
                    for batch in res.batches:
                        tile_cache.append_batch(batch)
                    n_ok += 1
                    if checkpoint_path is not None:
                        _append_checkpoint(checkpoint_path, res.path)

    return {
        "n_files_succeeded": n_ok,
        "n_files_failed": n_fail,
        "failures": failures,
        "elapsed_s": time.perf_counter() - t0,
    }
