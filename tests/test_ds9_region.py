"""Tests for dl-export-ds9-region."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from data_lake.discovery import tile_index as ti
from data_lake.discovery.areas import make_area, save_area
from data_lake.discovery.ds9_region import (
    export_ds9_region,
    format_ds9_circle,
    format_ds9_polygon,
    healpix_tile_polygon_deg,
)
from data_lake.discovery.ds9_region_cli import cli as ds9_cli
from data_lake.discovery.region import Region
from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, assign_healpix, healpix_dir


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
def lake(tmp_path: Path) -> Path:
    lake = tmp_path / "lake"
    norder = 5
    ra, dec = 120.0, 45.0
    npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
    _write_catalog_tile(
        lake, "ALLWISE",
        norder=norder, npix=npix,
        source_ids=[1, 2, 3],
        ra=[ra, ra + 0.01, ra + 0.02],
        dec=[dec, dec + 0.01, dec + 0.02],
        extra={"mag": [10.0, 12.0, 14.0]},
    )
    ti.write_tile_index(lake, "ALLWISE", "catalog")
    area = make_area("MyCone", Region.cone(ra, dec, 3600.0))
    save_area(lake, area)
    return lake


class TestFormatters:
    def test_circle_and_polygon(self) -> None:
        assert format_ds9_circle(10.0, -5.0, 0.5) == 'circle(10.00000000,-5.00000000,0.5")'
        line = format_ds9_polygon([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
        assert line.startswith("polygon(")
        assert "1.00000000,2.00000000" in line

    def test_healpix_polygon_closed_enough(self) -> None:
        import healpy as hp

        nside = hp.order2nside(5)
        verts = healpix_tile_polygon_deg(nside, 100)
        assert len(verts) == 4
        assert all(abs(dec) <= 90 for _, dec in verts)


class TestExport:
    def test_footprint(self, lake: Path, tmp_path: Path) -> None:
        out = tmp_path / "foot.reg"
        result = export_ds9_region(
            lake, "ALLWISE", out, mode="footprint", overwrite=True,
        )
        text = out.read_text()
        assert result.n_footprint == 1
        assert result.n_sources == 0
        assert "fk5" in text
        assert "polygon(" in text
        assert "circle(" not in text

    def test_sources_where_limit(self, lake: Path, tmp_path: Path) -> None:
        out = tmp_path / "src.reg"
        result = export_ds9_region(
            lake, "ALLWISE", out,
            mode="sources",
            where="mag < 13",
            limit=1,
            overwrite=True,
        )
        text = out.read_text()
        assert result.n_sources == 1
        assert result.n_footprint == 0
        assert text.count("circle(") == 1
        assert '0.5"' in text

    def test_from_area_clip(self, lake: Path, tmp_path: Path) -> None:
        out = tmp_path / "area.reg"
        # Tiny cone far from the catalog → empty clip
        empty = Region.cone(0.0, 0.0, 1.0)
        result = export_ds9_region(
            lake, "ALLWISE", out,
            mode="both",
            region=empty,
            overwrite=True,
        )
        assert result.n_footprint == 0
        assert result.n_sources == 0

        # Area covering the sources
        covering = Region.cone(120.0, 45.0, 3600.0)
        out2 = tmp_path / "area2.reg"
        result2 = export_ds9_region(
            lake, "ALLWISE", out2,
            mode="both",
            region=covering,
            overwrite=True,
        )
        assert result2.n_footprint == 1
        assert result2.n_sources == 3

    def test_footprint_rejects_where(self, lake: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="--where"):
            export_ds9_region(
                lake, "ALLWISE", tmp_path / "bad.reg",
                mode="footprint", where="mag > 0", overwrite=True,
            )


class TestCli:
    def test_cli_both_from_area(self, lake: Path, tmp_path: Path) -> None:
        out = tmp_path / "cli.reg"
        runner = CliRunner()
        r = runner.invoke(
            ds9_cli,
            [
                str(lake), "--survey", "ALLWISE", "--mode", "both",
                "--from-area", "MyCone", "-o", str(out), "--overwrite",
            ],
        )
        assert r.exit_code == 0, r.output
        assert out.is_file()
        assert "footprint polygon" in r.output
        text = out.read_text()
        assert "polygon(" in text
        assert "circle(" in text

    def test_cli_guards(self, lake: Path, tmp_path: Path) -> None:
        runner = CliRunner()
        r = runner.invoke(
            ds9_cli,
            [
                str(lake), "--survey", "ALLWISE", "--mode", "footprint",
                "--where", "mag > 0", "-o", str(tmp_path / "x.reg"),
            ],
        )
        assert r.exit_code != 0
        assert "--where" in r.output

        r2 = runner.invoke(
            ds9_cli,
            [
                str(lake), "--survey", "ALLWISE",
                "--from-area", "MyCone", "--cone", "120", "45",
                "--cone-radius-arcsec", "60",
                "-o", str(tmp_path / "y.reg"),
            ],
        )
        assert r2.exit_code != 0
        assert "at most one" in r2.output.lower() or "Use at most one" in r2.output
