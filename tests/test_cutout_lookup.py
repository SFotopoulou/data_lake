"""Tests for bulk cutout source_id lookup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_lake.io.cutouts import CutoutAccessor

from synthetic_lake_helpers import NORDER, SURVEY


def _make_cutout_tile(lake_root: Path, npix: int, source_ids: list[int]) -> None:
    import zarr

    from data_lake.ingest.fits_to_parquet import healpix_dir
    from data_lake.ingest.zarr_ids import create_zarr_join_array

    n = len(source_ids)
    tile_dir = lake_root / "cutouts" / SURVEY / healpix_dir(NORDER, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    path = tile_dir / f"Npix={npix}.zarr"
    root = zarr.open_group(store=zarr.storage.LocalStore(str(path)), mode="w", zarr_format=3)
    root.create_array("images", shape=(n, 1, 4, 4), chunks=(1, 1, 4, 4), dtype=np.float32)
    root["images"][:] = np.arange(n * 16, dtype=np.float32).reshape(n, 1, 4, 4)
    create_zarr_join_array(root, shape=(n,), chunks=(n,), dtype=np.int64)
    from data_lake.ingest.zarr_ids import zarr_join_array

    zarr_join_array(root)[:] = np.array(source_ids, dtype=np.int64)
    root.create_array("wcs", shape=(n,), dtype="S100")
    root["wcs"][:] = [b""] * n
    (lake_root / "cutouts" / SURVEY / "cutout_info.json").write_text(
        f'{{"hats_order": {NORDER}, "band_names": ["r"]}}'
    )


@pytest.fixture
def cutout_lake(tmp_path: Path) -> Path:
    _make_cutout_tile(tmp_path, 100, [501, 502])
    _make_cutout_tile(tmp_path, 200, [601])
    return tmp_path


class TestCutoutGetBatchBulkLookup:
    def test_get_batch_resolves_all_ids(self, cutout_lake: Path) -> None:
        acc = CutoutAccessor(cutout_lake, SURVEY)
        imgs = acc.get_batch([501, 502, 601])
        assert imgs.shape == (3, 1, 4, 4)

    def test_build_source_id_lookup(self, cutout_lake: Path) -> None:
        acc = CutoutAccessor(cutout_lake, SURVEY)
        lookup = acc._build_source_id_lookup(
            np.array([501, 601, 999], dtype=np.int64), show_progress=False,
        )
        assert lookup[501][0] == 100
        assert lookup[601][0] == 200
        assert 999 not in lookup
