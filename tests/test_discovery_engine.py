"""Tests for the discovery engine (resolve_region)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.discovery import tile_index as ti
from data_lake.discovery.engine import resolve_region, round_count
from data_lake.discovery.region import Region
from data_lake.ingest.fits_to_parquet import assign_healpix, healpix_dir


def _write_catalog_tile(lake: Path, survey: str, norder: int, npix: int, n_rows: int) -> None:
    tile_dir = lake / "catalogs" / survey / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"x": pa.array(list(range(n_rows)), type=pa.int64())}),
        tile_dir / f"Npix={npix}.parquet",
    )


def _write_info(lake: Path, survey: str, norder: int, total_rows: int) -> None:
    (lake / "catalogs" / survey / "catalog_info.json").write_text(
        json.dumps({"hats_order": norder, "total_rows": total_rows})
    )


class TestRoundCount:
    def test_formatting(self) -> None:
        assert round_count(12) == "~12"
        assert round_count(12000) == "~12.0k"
        assert round_count(750_000_000) == "~750M"
        assert round_count(2_000_000_000) == "~2.0G"


class TestResolveRegion:
    def test_cone_overlap_single_tile(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        ra, dec, norder = 120.0, 45.0, 5
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(lake, "ALLWISE", norder, npix, n_rows=100)
        _write_info(lake, "ALLWISE", norder, total_rows=100)
        ti.write_tile_index(lake, "ALLWISE", "catalog")

        region = Region.cone(ra, dec, radius_arcsec=30.0)
        rows = resolve_region(lake, region, surveys="all", modalities=["catalog"])
        assert len(rows) == 1
        row = rows[0]
        assert row.survey == "ALLWISE"
        assert row.n_tiles_overlap == 1
        assert row.est_rows == 100
        assert row.exact_rows is None  # estimate by default

    def test_exact_count_footer_sum(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        ra, dec, norder = 10.0, -10.0, 5
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(lake, "S", norder, npix, n_rows=42)
        _write_info(lake, "S", norder, total_rows=999)  # deliberately wrong total
        ti.write_tile_index(lake, "S", "catalog")

        region = Region.cone(ra, dec, radius_arcsec=30.0)
        rows = resolve_region(lake, region, modalities=["catalog"], count=True)
        assert len(rows) == 1
        assert rows[0].exact_rows == 42  # footer truth, not the estimate

    def test_no_overlap_returns_empty(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        ra, dec, norder = 120.0, 45.0, 5
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(lake, "S", norder, npix, n_rows=10)
        _write_info(lake, "S", norder, total_rows=10)
        ti.write_tile_index(lake, "S", "catalog")

        # Opposite side of the sky.
        region = Region.cone(300.0, -45.0, radius_arcsec=30.0)
        rows = resolve_region(lake, region, modalities=["catalog"])
        assert rows == []

    def test_explicit_surveys(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        ra, dec, norder = 50.0, 5.0, 5
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        for s in ("A", "B"):
            _write_catalog_tile(lake, s, norder, npix, n_rows=5)
            _write_info(lake, s, norder, total_rows=5)
            ti.write_tile_index(lake, s, "catalog")

        region = Region.cone(ra, dec, radius_arcsec=30.0)
        rows = resolve_region(lake, region, surveys=["A"], modalities=["catalog"])
        assert {r.survey for r in rows} == {"A"}
