"""Tests for dl-debug-specobj-lookup / probe_catalog_specobj_lookup."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from data_lake.ingest.fits_to_parquet import normalize_object_id
from data_lake.ingest.sdss_specobj_lookup import probe_catalog_specobj_lookup
from data_lake.ingest.sdss_specobj_lookup_debug import (
    cli,
    debug_specobj_lookup,
    format_debug_report,
)

_BOSS_SPPLATE = Path(__file__).resolve().parents[1] / "data" / "spPlate-3523-55144.fits"


@pytest.fixture
def boss_spplate_path() -> Path:
    return _BOSS_SPPLATE


class TestProbeCatalog:
    def test_probe_finds_specobjid_rows(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "sdss_spec" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "plate": pa.array([3523, 3523], type=pa.int32()),
                "mjd": pa.array([55144.0, 55144.0], type=pa.float64()),
                "fiber": pa.array([501, 502], type=pa.int16()),
                "specobjid": pa.array([9001, 9002], type=pa.int64()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        probe = probe_catalog_specobj_lookup("sdss_spec", 3523, 55144, catalog_root=lake)
        assert probe.ok
        assert probe.n_plate_mjd_rows == 2
        assert probe.fiber_map == {
            501: normalize_object_id(9001),
            502: normalize_object_id(9002),
        }

    def test_probe_reports_missing_columns(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "photo" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({"objid": pa.array([1], type=pa.int64())}),
            tile_dir / "Npix=1.parquet",
        )
        probe = probe_catalog_specobj_lookup("photo", 1, 2, catalog_root=lake)
        assert probe.error
        assert probe.fiber_map == {}


class TestDebugCli:
    def test_cli_plate_only(self, boss_spplate_path: Path) -> None:
        if not boss_spplate_path.is_file():
            pytest.skip("BOSS spPlate fixture not present")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(boss_spplate_path),
                "--survey",
                "boss_dr12",
                "--no-catalog",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "from_plate" in result.output
        assert "fiber mappings:" in result.output

    def test_cli_catalog_and_plate(self, boss_spplate_path: Path, tmp_path: Path) -> None:
        if not boss_spplate_path.is_file():
            pytest.skip("BOSS spPlate fixture not present")
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "boss_dr12" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "plate": pa.array([3523], type=pa.int32()),
                "mjd": pa.array([55144], type=pa.int64()),
                "fiber": pa.array([501], type=pa.int16()),
                "specobjid": pa.array([12345], type=pa.int64()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [str(boss_spplate_path), "--survey", "boss_dr12", str(lake)],
        )
        assert result.exit_code == 0, result.output
        assert "catalog" in result.output
        assert "rows with plate+mjd: 1" in result.output

    def test_format_report(self, boss_spplate_path: Path) -> None:
        if not boss_spplate_path.is_file():
            pytest.skip("BOSS spPlate fixture not present")
        report = debug_specobj_lookup(
            boss_spplate_path,
            "boss_dr12",
            try_from_catalog=False,
            specobj_id_layout="dr8plus",
        )
        text = format_debug_report(report, sample=2)
        assert "plate:" in text
        assert "from_plate" in text
