"""Smoke tests for data_lake.bench CLI."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from data_lake.bench.__main__ import cli
from synthetic_lake_helpers import ingest_synthetic_spectrum_lake


def test_bench_lookup_spectra(tmp_path: Path) -> None:
    ingest_synthetic_spectrum_lake(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["lookup-spectra", str(tmp_path), "--survey", "synthetic", "--n-ids", "3"],
    )
    assert result.exit_code == 0
    assert "lookup-spectra" in result.output
