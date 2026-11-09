"""Tests for survey-B tile LRU and vectorized catalog ID coercion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from data_lake.ingest.fits_to_parquet import (
    LAKE_JOIN_ID_COLUMN,
    normalize_object_id,
    storage_int64_from_integer,
)
from data_lake.io.crossmatch import (
    _catalog_ids_to_int64,
    build_crossmatch,
    load_crossmatch_table,
)
from data_lake.io.crossmatch_b_cache import (
    CrossmatchBTileCache,
    CrossmatchBTileCacheConfig,
    filter_sky_bbox,
)
from tests.test_crossmatch import _write_catalog_tile


class TestCatalogIdsToInt64:
    def test_polars_int64(self) -> None:
        s = pl.Series("id", [1, 2, 3], dtype=pl.Int64)
        out = _catalog_ids_to_int64(s)
        np.testing.assert_array_equal(out, np.array([1, 2, 3], dtype=np.int64))

    def test_polars_uint64_bit_pattern(self) -> None:
        big = 2**63 + 7
        s = pl.Series("id", [big], dtype=pl.UInt64)
        out = _catalog_ids_to_int64(s)
        assert out.shape == (1,)
        assert int(out[0]) == storage_int64_from_integer(big)
        assert int(out[0]) == normalize_object_id(big)

    def test_numpy_int64(self) -> None:
        arr = np.array([10, 20], dtype=np.int64)
        out = _catalog_ids_to_int64(arr)
        np.testing.assert_array_equal(out, arr)

    def test_decimal_strings_via_list(self) -> None:
        out = _catalog_ids_to_int64(["101", "202"])
        np.testing.assert_array_equal(out, np.array([101, 202], dtype=np.int64))

    def test_nulls_raise(self) -> None:
        s = pl.Series("id", [1, None], dtype=pl.Int64)
        with pytest.raises(ValueError, match="null"):
            _catalog_ids_to_int64(s)

    def test_float_raises(self) -> None:
        with pytest.raises(ValueError, match="floating"):
            _catalog_ids_to_int64(np.array([1.0, 2.0]))


class TestFilterSkyBBox:
    def test_simple_box(self) -> None:
        df = pl.DataFrame({
            "ra": [10.0, 20.0, 30.0],
            "dec": [0.0, 0.0, 5.0],
        })
        out = filter_sky_bbox(
            df, ra_col="ra", dec_col="dec",
            ra_min=15.0, ra_max=25.0, dec_min=-1.0, dec_max=1.0,
        )
        assert out["ra"].to_list() == [20.0]

    def test_ra_wrap(self) -> None:
        df = pl.DataFrame({
            "ra": [359.0, 1.0, 180.0],
            "dec": [0.0, 0.0, 0.0],
        })
        # Wrap box: ra_min > ra_max
        out = filter_sky_bbox(
            df, ra_col="ra", dec_col="dec",
            ra_min=358.0, ra_max=2.0, dec_min=-1.0, dec_max=1.0,
        )
        assert set(out["ra"].to_list()) == {359.0, 1.0}


class TestCrossmatchBTileCache:
    def test_second_lookup_is_hit(self) -> None:
        import polars as pl

        df = pl.DataFrame({
            "id": [1, 2],
            "ra": [10.0, 11.0],
            "dec": [0.0, 0.1],
            "_healpix_norder5": [7, 7],
        })
        acc = MagicMock()
        acc.sources_in_healpix_pixels.return_value = df
        acc._empty_result.return_value = pl.DataFrame()

        cache = CrossmatchBTileCache(CrossmatchBTileCacheConfig(max_tiles=8))
        cols = ["id", "ra", "dec", "_healpix_norder5"]
        a = cache.get_tiles(acc, [7], cols, hp_col="_healpix_norder5")
        b = cache.get_tiles(acc, [7], cols, hp_col="_healpix_norder5")
        assert a.height == 2
        assert b.height == 2
        assert cache.misses == 1
        assert cache.hits == 1
        assert acc.sources_in_healpix_pixels.call_count == 1

    def test_disabled_always_fetches(self) -> None:
        df = pl.DataFrame({
            "id": [1],
            "ra": [10.0],
            "dec": [0.0],
            "_healpix_norder5": [3],
        })
        acc = MagicMock()
        acc.sources_in_healpix_pixels.return_value = df
        cache = CrossmatchBTileCache(CrossmatchBTileCacheConfig(enabled=False))
        cols = ["id", "ra", "dec", "_healpix_norder5"]
        cache.get_tiles(acc, [3], cols, hp_col="_healpix_norder5")
        cache.get_tiles(acc, [3], cols, hp_col="_healpix_norder5")
        assert acc.sources_in_healpix_pixels.call_count == 2


class TestBuildCrossmatchBCacheParity:
    def test_cache_on_vs_off_same_matches(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        # Two nearby A tiles that share covering B sky
        ra0, dec0 = 30.0, 5.0
        npix_a0 = int(assign_healpix_one(ra0, dec0, norder))
        npix_a1 = int(assign_healpix_one(ra0 + 0.5, dec0 + 0.5, norder))
        npix_b = int(assign_healpix_one(ra0, dec0, norder))

        _write_catalog_tile(
            lake, "A", norder=norder, npix=npix_a0,
            source_ids=[101], ra=[ra0], dec=[dec0],
        )
        # Append second A tile without clobbering catalog_info
        from data_lake.ingest.fits_to_parquet import healpix_dir
        import json
        import pyarrow as pa
        import pyarrow.parquet as pq

        tile_dir = lake / "catalogs" / "A" / healpix_dir(norder, npix_a1)
        tile_dir.mkdir(parents=True, exist_ok=True)
        hp_col = f"_healpix_norder{norder}"
        pq.write_table(
            pa.table({
                LAKE_JOIN_ID_COLUMN: pa.array([102], type=pa.int64()),
                "ra": pa.array([ra0 + 0.5], type=pa.float64()),
                "dec": pa.array([dec0 + 0.5], type=pa.float64()),
                hp_col: pa.array([npix_a1], type=pa.int64()),
                "_cutout_index": pa.array([-1], type=pa.int64()),
                "_spectrum_index": pa.array([-1], type=pa.int64()),
            }),
            tile_dir / f"Npix={npix_a1}.parquet",
        )
        info_path = lake / "catalogs" / "A" / "catalog_info.json"
        info = json.loads(info_path.read_text())
        info["total_rows"] = 2
        info_path.write_text(json.dumps(info))

        _write_catalog_tile(
            lake, "B", norder=norder, npix=npix_b,
            source_ids=[201, 202],
            ra=[ra0 + 0.0001, ra0 + 0.5001],
            dec=[dec0 + 0.0001, dec0 + 0.5001],
        )

        off = build_crossmatch(
            lake, "A", "B", radius_arcsec=2.0, show_progress=False, b_cache=False,
        )
        # Overwrite to rebuild with cache
        on = build_crossmatch(
            lake, "A", "B", radius_arcsec=2.0, show_progress=False,
            overwrite=True, b_cache=True,
        )
        assert on.n_match_rows == off.n_match_rows
        assert on.n_match_rows >= 1

        tbl = load_crossmatch_table(on.output_root)
        assert tbl.num_rows == on.n_match_rows


def assign_healpix_one(ra: float, dec: float, norder: int) -> int:
    from data_lake.ingest.fits_to_parquet import assign_healpix

    return int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
