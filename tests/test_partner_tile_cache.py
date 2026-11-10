"""Tests for the dl-gather partner tile LRU cache (whole tiles, read once)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_lake.discovery.partner_tile_cache import PartnerTileCache, PartnerTileCacheConfig
from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, assign_healpix, healpix_dir
from data_lake.io.catalog import CatalogAccessor

NORDER = 5
RA, DEC = 120.0, 45.0
NPIX = int(assign_healpix(np.array([RA]), np.array([DEC]), NORDER)[0])


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


@pytest.fixture
def tile_reads(monkeypatch) -> list[list[int]]:
    """Record the tile list of every partner tile read from disk."""
    calls: list[list[int]] = []
    orig = CatalogAccessor.sources_in_healpix_pixels

    def counting(self, npixels, *args, **kwargs):
        calls.append([int(n) for n in npixels])
        return orig(self, npixels, *args, **kwargs)

    monkeypatch.setattr(CatalogAccessor, "sources_in_healpix_pixels", counting)
    return calls


@pytest.fixture
def lake(tmp_path: Path) -> Path:
    lake = tmp_path / "lake"
    _write_catalog_tile(
        lake, "DESI_DR1", norder=NORDER, npix=NPIX,
        source_ids=[101, 102], ra=[RA, RA + 0.001], dec=[DEC, DEC + 0.001],
        extra={"z": [0.5, 1.2]},
    )
    _write_catalog_tile(
        lake, "DESI_DR1", norder=NORDER, npix=NPIX + 1,
        source_ids=[103], ra=[RA + 0.01], dec=[DEC + 0.01], extra={"z": [2.0]},
    )
    return lake


def _cache(**kwargs) -> PartnerTileCache:
    cfg = {"max_bytes": 10 * 1024 * 1024, "max_tiles": 8, **kwargs}
    return PartnerTileCache(PartnerTileCacheConfig(**cfg))


class TestPartnerTileCache:
    def test_tile_read_once_for_different_ids(self, lake: Path, tile_reads) -> None:
        cache = _cache()
        with CatalogAccessor(lake, "DESI_DR1") as acc:
            cols = [acc.link_id_column, "z"]
            a = cache.lookup(acc, "DESI_DR1", NPIX, cols, [101])
            b = cache.lookup(acc, "DESI_DR1", NPIX, cols, [102])
            both = cache.lookup(acc, "DESI_DR1", NPIX, cols, [101, 102])
        assert tile_reads == [[NPIX]]
        assert a[LAKE_JOIN_ID_COLUMN].to_list() == [101]
        assert b["z"].to_list() == [1.2]
        assert both.height == 2
        assert (cache.hits, cache.misses) == (2, 1)

    def test_eviction_allows_reread(self, lake: Path, tile_reads) -> None:
        cache = _cache(max_bytes=1, max_tiles=1)
        with CatalogAccessor(lake, "DESI_DR1") as acc:
            cols = [acc.link_id_column, "z"]
            cache.lookup(acc, "DESI_DR1", NPIX, cols, [101])
            cache.lookup(acc, "DESI_DR1", NPIX + 1, cols, [103])
            cache.lookup(acc, "DESI_DR1", NPIX, cols, [101])
        assert len(tile_reads) == 3

    def test_lookup_many_reads_each_tile_once(self, lake: Path, tile_reads) -> None:
        cache = _cache()
        with CatalogAccessor(lake, "DESI_DR1") as acc:
            cols = [acc.link_id_column, "z"]
            out = cache.lookup_many(acc, "DESI_DR1", [NPIX, NPIX + 1], cols, [101, 103])
            again = cache.lookup_many(acc, "DESI_DR1", [NPIX + 1, NPIX], cols, [102])
        assert sorted(tile_reads) == [[NPIX], [NPIX + 1]]
        assert sorted(out[LAKE_JOIN_ID_COLUMN].to_list()) == [101, 103]
        assert again[LAKE_JOIN_ID_COLUMN].to_list() == [102]

    def test_absent_tile_remembered_without_slot(self, lake: Path, tile_reads) -> None:
        cache = _cache(max_tiles=1)
        with CatalogAccessor(lake, "DESI_DR1") as acc:
            cols = [acc.link_id_column, "z"]
            cache.lookup(acc, "DESI_DR1", NPIX, cols, [101])
            for _ in range(3):
                assert cache.lookup(acc, "DESI_DR1", 7, cols, [101]).is_empty()
            cache.lookup(acc, "DESI_DR1", NPIX, cols, [102])
        assert tile_reads == [[NPIX], [7]]

    def test_disabled_fetches_by_id(self, lake: Path, tile_reads) -> None:
        cache = _cache(enabled=False)
        with CatalogAccessor(lake, "DESI_DR1") as acc:
            cols = [acc.link_id_column, "z"]
            out = cache.lookup(acc, "DESI_DR1", NPIX, cols, [102])
        assert tile_reads == []
        assert out["z"].to_list() == [1.2]
