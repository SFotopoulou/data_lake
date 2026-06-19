"""Tests for MOC export (requires optional mocpy)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from data_lake.discovery import tile_index as ti
from data_lake.discovery.moc_export import (
    export_region_moc,
    export_survey_moc,
    region_to_moc,
)
from data_lake.discovery.moc_export_cli import cli as export_moc_cli
from data_lake.discovery.region import Region
from data_lake.discovery.region_cli import cli as region_cli
from data_lake.ingest.fits_to_parquet import assign_healpix, healpix_dir


@pytest.fixture
def lake(tmp_path: Path) -> Path:
    lake = tmp_path / "lake"
    ra, dec, norder = 120.0, 45.0, 5
    npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
    tile_dir = lake / "catalogs" / "ALLWISE" / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    (tile_dir / f"Npix={npix}.parquet").write_bytes(b"")
    (lake / "catalogs" / "ALLWISE" / "catalog_info.json").write_text(
        json.dumps({"hats_order": norder, "total_rows": 1})
    )
    ti.write_tile_index(lake, "ALLWISE", "catalog")
    return lake


class TestMocExport:
    def test_missing_mocpy_raises(self) -> None:
        region = Region.cone(120.0, 45.0, 60.0)
        try:
            import mocpy  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="mocpy"):
                region_to_moc(region, 5)
        else:
            pytest.skip("mocpy installed")

    def test_cone_fits_roundtrip(self, tmp_path: Path) -> None:
        pytest.importorskip("mocpy")
        region = Region.cone(120.0, 45.0, 600.0)
        out = tmp_path / "cone.moc.fits"
        export_region_moc(region, out, moc_order=5, overwrite=True)
        assert out.is_file()
        npix_in = region.to_npix(5)
        npix_back = Region.from_moc(path=out).to_npix(5)
        assert npix_back == npix_in

    def test_survey_footprint(self, lake: Path, tmp_path: Path) -> None:
        pytest.importorskip("mocpy")
        out = tmp_path / "allwise.moc.fits"
        result = export_survey_moc(
            lake, "ALLWISE", out, moc_order=5, overwrite=True,
        )
        assert result.n_cells >= 1
        assert out.is_file()

    def test_region_cli_export_moc(self, lake: Path, tmp_path: Path) -> None:
        pytest.importorskip("mocpy")
        out = tmp_path / "field.moc.fits"
        runner = CliRunner()
        r = runner.invoke(
            region_cli,
            [
                str(lake), "--cone", "120.0", "45.0", "--radius-arcsec", "60",
                "--export-moc", str(out), "--moc-order", "5", "--overwrite",
            ],
        )
        assert r.exit_code == 0, r.output
        assert out.is_file()
        assert "Exported MOC" in r.output

    def test_export_moc_cli(self, lake: Path, tmp_path: Path) -> None:
        pytest.importorskip("mocpy")
        out = tmp_path / "wise.moc.fits"
        runner = CliRunner()
        r = runner.invoke(
            export_moc_cli,
            [
                str(lake), "--survey", "ALLWISE", "--moc-order", "5",
                "-o", str(out), "--overwrite",
            ],
        )
        assert r.exit_code == 0, r.output
        assert out.is_file()
