"""Tests for partner catalog tile LRU cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_lake.discovery.partner_tile_cache import PartnerTileCache, PartnerTileCacheConfig
from data_lake.io.catalog import CatalogAccessor
from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, assign_healpix, healpix_dir
import json
import pyarrow as pa
import pyarrow.parquet as pq


def _write_catalog_tile(lake, survey, *, norder, npix, source_ids, ra, dec, extra=None):
    tile_dir = lake / "catalogs" / survey / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    hp_col = f"_healpix_norder{norder}"
    cols = {
        LAKE_JOIN_ID_COLUMN: pa.array(source_ids, type=pa.int64()),
        "ra": pa.array(ra, type=pa.float64()),
        "dec": pa.array(dec, type=pa.float64()),
        hp_col: pa.array([npix] * len(source_ids), type=pa.int64()),
    }
    if extra:
        for k, v in extra.items():
            cols[k] = pa.array(v)
    pq.write_table(pa.table(cols), tile_dir / f"Npix={npix}.parquet")
    (lake / "catalogs" / survey / "catalog_info.json").write_text(
        json.dumps({
            "hats_order": norder,
            "ra_column": "ra",
            "dec_column": "dec",
            "link_id_mode": "sequential",
            "link_id_column": LAKE_JOIN_ID_COLUMN,
            "total_rows": len(source_ids),
        })
    )


class TestPartnerTileCache:
    def test_duplicate_lookup_skips_disk_read(self, tmp_path: Path, monkeypatch) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(
            lake, "DESI_DR1", norder=norder, npix=npix,
            source_ids=[101, 102], ra=[ra, ra + 0.001], dec=[dec, dec + 0.001],
            extra={"z": [0.5, 1.2]},
        )
        calls: list[int] = []
        orig = CatalogAccessor.get_sources_by_id_in_healpix_pixels

        def counting(self, source_ids, npixels, columns=None, fmt="polars", **kwargs):
            calls.append(1)
            return orig(self, source_ids, npixels, columns=columns, fmt=fmt, **kwargs)

        monkeypatch.setattr(
            CatalogAccessor, "get_sources_by_id_in_healpix_pixels", counting,
        )
        cache = PartnerTileCache(PartnerTileCacheConfig(max_bytes=10 * 1024 * 1024, max_tiles=8))
        with CatalogAccessor(lake, "DESI_DR1") as acc:
            cols = [acc.link_id_column, "z"]
            cache.lookup(acc, "DESI_DR1", npix, cols, [101])
            cache.lookup(acc, "DESI_DR1", npix, cols, [101])
        assert len(calls) == 1

    def test_incremental_ids_merge_in_cache(self, tmp_path: Path, monkeypatch) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(
            lake, "DESI_DR1", norder=norder, npix=npix,
            source_ids=[101, 102], ra=[ra, ra + 0.001], dec=[dec, dec + 0.001],
            extra={"z": [0.5, 1.2]},
        )
        calls: list[int] = []
        orig = CatalogAccessor.get_sources_by_id_in_healpix_pixels

        def counting(self, source_ids, npixels, columns=None, fmt="polars", **kwargs):
            calls.append(1)
            return orig(self, source_ids, npixels, columns=columns, fmt=fmt, **kwargs)

        monkeypatch.setattr(
            CatalogAccessor, "get_sources_by_id_in_healpix_pixels", counting,
        )
        cache = PartnerTileCache(PartnerTileCacheConfig(max_bytes=10 * 1024 * 1024, max_tiles=8))
        with CatalogAccessor(lake, "DESI_DR1") as acc:
            cols = [acc.link_id_column, "z"]
            cache.lookup(acc, "DESI_DR1", npix, cols, [101])
            cache.lookup(acc, "DESI_DR1", npix, cols, [102])
            cache.lookup(acc, "DESI_DR1", npix, cols, [101, 102])
        assert len(calls) == 2

    def test_eviction_allows_reread(self, tmp_path: Path, monkeypatch) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(
            lake, "DESI_DR1", norder=norder, npix=npix,
            source_ids=[101], ra=[ra], dec=[dec], extra={"z": [0.5]},
        )
        calls: list[int] = []
        orig = CatalogAccessor.get_sources_by_id_in_healpix_pixels

        def counting(self, source_ids, npixels, columns=None, fmt="polars", **kwargs):
            calls.append(1)
            return orig(self, source_ids, npixels, columns=columns, fmt=fmt, **kwargs)

        monkeypatch.setattr(
            CatalogAccessor, "get_sources_by_id_in_healpix_pixels", counting,
        )
        cache = PartnerTileCache(
            PartnerTileCacheConfig(max_bytes=1, max_tiles=1, enabled=True),
        )
        with CatalogAccessor(lake, "DESI_DR1") as acc:
            cols = [acc.link_id_column, "z"]
            cache.lookup(acc, "DESI_DR1", npix, cols, [101])
            cache.lookup(acc, "DESI_DR1", npix + 1, cols, [101])
            cache.lookup(acc, "DESI_DR1", npix, cols, [101])
        assert len(calls) >= 2
