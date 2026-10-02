"""Tests for data_lake.io.preview.preview_sources."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_lake.ingest.fits_to_parquet import healpix_dir
from data_lake.io.preview import preview_sources
from synthetic_lake_helpers import SURVEY, ingest_synthetic_spectrum_lake


def _write_catalog_tile(lake: Path, survey: str, rows: list[dict], *, norder: int = 5) -> None:
    cat_root = lake / "catalogs" / survey
    cat_root.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    npix = int(rows[0]["_healpix_norder5"])
    out_dir = cat_root / healpix_dir(norder, npix)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / f"Npix={npix}.parquet")
    (cat_root / "catalog_info.json").write_text(
        json.dumps({"hats_order": norder, "link_id_column": "_source_id"})
    )


@pytest.fixture
def spectrum_lake(tmp_path: Path) -> Path:
    ingest_synthetic_spectrum_lake(tmp_path)
    return tmp_path


class TestPreviewSources:
    def test_catalog_peek(self, tmp_path: Path) -> None:
        rows = [
            {
                "_source_id": i,
                "_healpix_norder5": 10,
                "TARGETID": 1000 + i,
                "ra": 150.0 + 0.01 * i,
                "dec": 2.0,
            }
            for i in range(20)
        ]
        _write_catalog_tile(tmp_path, "PEEK_CAT", rows)
        df = preview_sources(tmp_path, "PEEK_CAT", modality="catalog", n=5)
        assert len(df) == 5
        assert "_source_id" in df.columns
        assert "ra" in df.columns

    def test_catalog_explicit_columns(self, tmp_path: Path) -> None:
        rows = [
            {
                "_source_id": 1,
                "_healpix_norder5": 3,
                "TARGETID": 9,
                "ra": 1.0,
                "dec": 2.0,
            }
        ]
        _write_catalog_tile(tmp_path, "PEEK_COLS", rows)
        df = preview_sources(
            tmp_path,
            "PEEK_COLS",
            modality="catalog",
            n=1,
            columns=["_source_id", "TARGETID"],
        )
        assert list(df.columns) == ["_source_id", "TARGETID"]

    def test_spectra_peek(self, spectrum_lake: Path) -> None:
        df = preview_sources(spectrum_lake, SURVEY, modality="spectra", n=3)
        assert len(df) == 3
        assert list(df.columns) == ["_source_id", "npix", "modality"]
        assert (df["modality"] == "spectra").all()
        assert df["_source_id"].dtype == pl.Int64

    def test_spectrum_alias(self, spectrum_lake: Path) -> None:
        df = preview_sources(spectrum_lake, SURVEY, modality="spectrum", n=2)
        assert len(df) == 2
        assert (df["modality"] == "spectra").all()

    def test_invalid_modality(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="modality"):
            preview_sources(tmp_path, "X", modality="bogus")  # type: ignore[arg-type]

    def test_n_must_be_positive(self, spectrum_lake: Path) -> None:
        with pytest.raises(ValueError, match="n must be"):
            preview_sources(spectrum_lake, SURVEY, n=0)

    def test_columns_only_for_catalog(self, spectrum_lake: Path) -> None:
        with pytest.raises(ValueError, match="columns="):
            preview_sources(
                spectrum_lake, SURVEY, modality="spectra", columns=["_source_id"]
            )


class TestPreviewSourcesCli:
    def test_cli_catalog_prints_rows(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.io.preview import cli

        rows = [
            {
                "_source_id": i,
                "_healpix_norder5": 7,
                "TARGETID": 100 + i,
                "ra": 10.0,
                "dec": 1.0,
            }
            for i in range(8)
        ]
        _write_catalog_tile(tmp_path, "CLI_PEEK", rows)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["CLI_PEEK", str(tmp_path), "--modality", "catalog", "-n", "3"],
        )
        assert result.exit_code == 0, result.output
        assert "_source_id" in result.output

    def test_cli_write_csv(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.io.preview import cli

        rows = [
            {
                "_source_id": 1,
                "_healpix_norder5": 2,
                "ra": 1.0,
                "dec": 2.0,
            }
        ]
        _write_catalog_tile(tmp_path, "CLI_OUT", rows)
        out = tmp_path / "peek.csv"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["CLI_OUT", str(tmp_path), "-n", "1", "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.is_file()
        assert "_source_id" in out.read_text()
