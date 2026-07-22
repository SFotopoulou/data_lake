"""End-to-end tests for the dl-region CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from data_lake.discovery import tile_index as ti
from data_lake.discovery.region_cli import cli
from data_lake.ingest.fits_to_parquet import assign_healpix, healpix_dir


@pytest.fixture
def lake(tmp_path: Path) -> Path:
    lake = tmp_path / "lake"
    ra, dec, norder = 120.0, 45.0, 5
    npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
    tile_dir = lake / "catalogs" / "ALLWISE" / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"x": pa.array(list(range(100)), type=pa.int64())}),
        tile_dir / f"Npix={npix}.parquet",
    )
    (lake / "catalogs" / "ALLWISE" / "catalog_info.json").write_text(
        json.dumps({"hats_order": norder, "total_rows": 100})
    )
    ti.write_tile_index(lake, "ALLWISE", "catalog")
    return lake


class TestRegionCli:
    def test_cone_discovery(self, lake: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli, [str(lake), "--cone", "120.0", "45.0", "--radius-arcsec", "60"]
        )
        assert result.exit_code == 0, result.output
        assert "ALLWISE" in result.output
        assert "catalog" in result.output

    def test_exact_count(self, lake: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli, [str(lake), "--cone", "120.0", "45.0", "--radius-arcsec", "60", "--count"]
        )
        assert result.exit_code == 0, result.output
        assert "100" in result.output

    def test_save_as_and_from_area(self, lake: Path) -> None:
        runner = CliRunner()
        r1 = runner.invoke(
            cli,
            [str(lake), "--cone", "120.0", "45.0", "--radius-arcsec", "60",
             "--save-as", "Field1"],
        )
        assert r1.exit_code == 0, r1.output
        assert (lake / "areas" / "Field1.json").is_file()

        r2 = runner.invoke(cli, [str(lake), "--from-area", "Field1"])
        assert r2.exit_code == 0, r2.output
        assert "ALLWISE" in r2.output

    def test_requires_one_selector(self, lake: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, [str(lake)])
        assert result.exit_code != 0
        assert "exactly one region selector" in result.output

    def test_npix_requires_norder(self, lake: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, [str(lake), "--npix", "1,2,3"])
        assert result.exit_code != 0
        assert "norder" in result.output.lower()

    def test_from_area_crossmatch_modality(self, lake: Path) -> None:
        """CLI smoke: --from-area + --modalities crossmatch discovers XM trees."""
        ra, dec, norder = 120.0, 45.0, 5
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        tree = "A_x_B__r1.0"
        xm_dir = lake / "crossmatch" / tree / healpix_dir(norder, npix)
        xm_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({"x": pa.array([1, 2, 3], type=pa.int64())}),
            xm_dir / f"Npix={npix}.parquet",
        )
        (lake / "crossmatch" / tree / "crossmatch_info.json").write_text(
            json.dumps({"hats_order": norder, "total_rows": 3})
        )
        ti.write_tile_index(lake, tree, "crossmatch")

        runner = CliRunner()
        r_save = runner.invoke(
            cli,
            [str(lake), "--cone", "120.0", "45.0", "--radius-arcsec", "60",
             "--save-as", "XmField"],
        )
        assert r_save.exit_code == 0, r_save.output

        result = runner.invoke(
            cli,
            [str(lake), "--from-area", "XmField",
             "--modalities", "crossmatch", "--count"],
        )
        assert result.exit_code == 0, result.output
        assert tree in result.output
        assert "crossmatch" in result.output
        assert "3" in result.output
