"""Tests for dl-plot-source CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("matplotlib")


# ---------------------------------------------------------------------------
# Helpers to build a synthetic product catalog and spectrum
# ---------------------------------------------------------------------------


def _write_minimal_product_catalog(lake_root: Path, survey: str, source_id: int):
    """Write a single-row product catalog with phot_ab_w1 and phot_ab_g columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    catalog_dir = lake_root / "catalogs" / survey / "Norder=5" / "Dir=0"
    catalog_dir.mkdir(parents=True)

    from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, assign_healpix

    npix = assign_healpix(30.0, 5.0, norder=5)
    table = pa.table({
        LAKE_JOIN_ID_COLUMN: pa.array([source_id], type=pa.int64()),
        "_healpix_norder5": pa.array([npix], type=pa.int64()),
        "phot_ab_w1": pa.array([15.2], type=pa.float32()),
        "phot_ab_w1_err": pa.array([0.05], type=pa.float32()),
        "phot_ab_g": pa.array([18.5], type=pa.float32()),
    })
    pq.write_table(table, str(catalog_dir / f"Npix={npix}.parquet"))

    info = {
        "survey": survey,
        "hats_order": 5,
        "ra_column": "ra",
        "dec_column": "dec",
        "link_id_mode": "source_id",
        "kind": "product",
    }
    (lake_root / "catalogs" / survey / "catalog_info.json").write_text(
        json.dumps(info), encoding="utf-8"
    )
    return npix


def _write_minimal_spectrum(lake_root: Path, survey: str, source_id: int, spectrum_npix: int):
    """Write a one-source Zarr spectrum tile using the project's own ingest helpers."""
    from data_lake.ingest.fits_to_spectra_zarr import (
        SpectrumRecord,
        _meta_to_bytes,
        _META_DTYPE,
        _open_or_create_spectrum_tile,
        _write_spectrum_info,
    )
    from data_lake.ingest.fits_to_parquet import assign_healpix, healpix_dir, LAKE_JOIN_ID_COLUMN
    from data_lake.ingest.zarr_ids import zarr_join_array

    N_PIX = 32
    wave = np.linspace(3600.0, 9800.0, N_PIX)
    wcs_attrs = {
        "ctype": "WAVE", "crval": float(wave[0]),
        "cdelt": float(wave[1] - wave[0]), "crpix": 1.0,
        "unit": "Angstrom", "air_or_vacuum": "vacuum", "n_pix": N_PIX,
    }

    rec = SpectrumRecord(
        source_id=source_id,
        ra=30.0, dec=5.0,
        flux=np.full(N_PIX, 10.0, dtype=np.float32),
        ivar=np.full(N_PIX, 4.0, dtype=np.float32),
        mask=np.zeros(N_PIX, dtype=np.uint8),
        wavelength=wave,
        meta={"z": 0.1, "z_err": 0.001, "snr": 10.0, "exptime": 1000.0, "R": 3000.0, "instr": "TEST"},
    )

    spec_root = lake_root / "spectra" / survey
    tile_path = spec_root / healpix_dir(5, spectrum_npix) / f"Npix={spectrum_npix}.zarr"
    tile_path.parent.mkdir(parents=True, exist_ok=True)

    root = _open_or_create_spectrum_tile(tile_path, N_PIX, "shared", np.dtype(np.uint8), wcs_attrs)
    root["flux"].append(rec.flux[np.newaxis, :])
    root["ivar"].append(rec.ivar[np.newaxis, :])
    root["mask"].append(rec.mask[np.newaxis, :])
    zarr_join_array(root).append(np.array([rec.source_id], dtype=np.int64))
    meta_buf = np.frombuffer(_meta_to_bytes(rec.meta), dtype="|V" + str(_META_DTYPE.itemsize))
    root["meta"].append(meta_buf)
    root["wavelength"][:] = wave.astype(np.float64)

    _write_spectrum_info(spec_root, survey, 5, N_PIX, "shared", "uint8", wcs_attrs)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_lake(tmp_path):
    """A lake root with one product catalog row + one spectrum."""
    source_id = 42
    product = "TEST_PRODUCT"
    spectra_survey = "TEST_SPECTRA"

    npix = _write_minimal_product_catalog(tmp_path, product, source_id)
    _write_minimal_spectrum(tmp_path, spectra_survey, source_id, npix)
    return tmp_path, source_id, product, spectra_survey


def test_dl_plot_source_writes_png(synthetic_lake, tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    from click.testing import CliRunner
    from data_lake.plot.plot_cli import cli

    lake_root, source_id, product, spectra_survey = synthetic_lake
    out = tmp_path / "out.png"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--lake-root", str(lake_root),
        "--id", str(source_id),
        "--product", product,
        "--spectra-survey", spectra_survey,
        "-o", str(out),
        "--no-curves",  # skip curve loading to keep test fast
    ])

    if result.exit_code != 0:
        print(result.output)
        if result.exception:
            import traceback
            traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.stat().st_size > 0


def test_dl_plot_source_prints_output_path(synthetic_lake, tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    from click.testing import CliRunner
    from data_lake.plot.plot_cli import cli

    lake_root, source_id, product, spectra_survey = synthetic_lake
    out = tmp_path / "source.png"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--lake-root", str(lake_root),
        "--id", str(source_id),
        "--product", product,
        "--spectra-survey", spectra_survey,
        "-o", str(out),
        "--no-curves",
    ])
    assert result.exit_code == 0
    assert str(out) in result.output


def test_dl_plot_source_missing_product_gives_clear_error(tmp_path):
    from click.testing import CliRunner
    from data_lake.plot.plot_cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--lake-root", str(tmp_path),
        "--id", "1",
        "--product", "NONEXISTENT",
        "--spectra-survey", "SDSS_DR17",
    ])
    assert result.exit_code != 0
    assert "NONEXISTENT" in result.output or "not found" in result.output.lower()


def test_dl_plot_source_requires_lake_root():
    from click.testing import CliRunner
    from data_lake.plot.plot_cli import cli

    runner = CliRunner(env={"DATA_LAKE_CONFIG": ""})
    result = runner.invoke(cli, [
        "--id", "1",
        "--product", "PROD",
        "--spectra-survey", "SDSS",
    ])
    assert result.exit_code != 0


def test_dl_plot_spectrum_writes_png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    from click.testing import CliRunner

    from data_lake.plot.spectrum_cli import cli

    source_id = 99
    survey = "SPEC_ONLY"
    npix = 12
    _write_minimal_spectrum(tmp_path, survey, source_id, npix)
    out = tmp_path / "spec.png"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [survey, str(tmp_path), "--id", str(source_id), "-o", str(out), "-q"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.stat().st_size > 0
    assert str(out) in result.output


def test_dl_plot_spectrum_missing_survey(tmp_path):
    from click.testing import CliRunner

    from data_lake.plot.spectrum_cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["NOSPEC", str(tmp_path), "--id", "1"])
    assert result.exit_code != 0
    assert "NOSPEC" in result.output or "not found" in result.output.lower()
