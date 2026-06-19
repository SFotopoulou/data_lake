"""Tests for extract-time spectrum flux calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data_lake.export.spectra_calibration import (
    FluxCalibration,
    apply_flux_scale,
    calibration_sidecar_path,
    load_spectrum_flux_calibration,
    resolve_flux_calibration,
    write_calibration_sidecar,
)


class TestApplyFluxScale:
    def test_identity(self):
        flux = np.array([1.0, 2.0], dtype=np.float32)
        ivar = np.array([4.0, 9.0], dtype=np.float32)
        f, i = apply_flux_scale(flux, ivar, 1.0)
        np.testing.assert_array_equal(f, flux)
        np.testing.assert_array_equal(i, ivar)

    def test_scale_and_ivar(self):
        flux = np.array([2.0, 4.0], dtype=np.float32)
        ivar = np.array([1.0, 4.0], dtype=np.float32)
        f, i = apply_flux_scale(flux, ivar, 0.5)
        np.testing.assert_allclose(f, [1.0, 2.0])
        np.testing.assert_allclose(i, [4.0, 16.0])

    def test_zero_ivar_unchanged(self):
        flux = np.array([1.0], dtype=np.float32)
        ivar = np.array([0.0], dtype=np.float32)
        _, i = apply_flux_scale(flux, ivar, 2.0)
        assert i[0] == 0.0


class TestResolveFluxCalibration:
    def test_explicit_wins(self):
        cal = resolve_flux_calibration(None, "SDSS_DR17", flux_scale=2.0, apply_survey_calibration=True)
        assert cal is not None
        assert cal.flux_scale == 2.0

    def test_registry_sdss(self):
        cal = resolve_flux_calibration(None, "SDSS_DR17", apply_survey_calibration=True)
        assert cal is not None
        assert cal.flux_scale == 1e-17
        assert cal.output_flux_unit == "erg/s/cm2/Angstrom"

    def test_registry_desi(self):
        cal = resolve_flux_calibration(None, "DESI_DR1", apply_survey_calibration=True)
        assert cal is not None
        assert cal.flux_scale == 1e-17

    def test_missing_registry_raises(self):
        with pytest.raises(LookupError, match="No spectra.flux_calibration"):
            resolve_flux_calibration(None, "NONEXISTENT_SURVEY_XYZ", apply_survey_calibration=True)

    def test_invalid_scale(self):
        with pytest.raises(ValueError, match="positive"):
            resolve_flux_calibration(None, "SDSS_DR17", flux_scale=0.0)


class TestSidecar:
    def test_sidecar_paths(self, tmp_path: Path):
        assert calibration_sidecar_path(tmp_path / "out.zarr") == tmp_path / "out.calibration.json"
        assert calibration_sidecar_path(tmp_path / "out.parquet") == tmp_path / "out.calibration.json"
        (tmp_path / "dir").mkdir()
        assert calibration_sidecar_path(tmp_path / "dir") == tmp_path / "dir" / "calibration.json"
        assert calibration_sidecar_path(tmp_path / "fits_out") == tmp_path / "fits_out" / "calibration.json"

    def test_write_sidecar(self, tmp_path: Path):
        cal = FluxCalibration(survey="SDSS_DR17", flux_scale=1e-17, output_flux_unit="erg/s/cm2/Angstrom")
        path = write_calibration_sidecar(tmp_path / "subset.zarr", cal)
        data = json.loads(path.read_text())
        assert data["flux_scale"] == 1e-17
        assert data["survey"] == "SDSS_DR17"


class TestLoadBundled:
    def test_load_sdss(self):
        cal = load_spectrum_flux_calibration(None, "SDSS_DR17")
        assert cal is not None
        assert cal.flux_scale == 1e-17
