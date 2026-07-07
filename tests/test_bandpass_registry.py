"""Tests for BandpassRegistry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data_lake.homogenize.bandpass import BandpassRegistry


# ---------------------------------------------------------------------------
# Basic metadata
# ---------------------------------------------------------------------------


def test_available_bands_not_empty():
    bp = BandpassRegistry()
    bands = bp.available_bands()
    assert len(bands) > 0
    assert "phot_ab_w1" in bands
    assert "phot_ab_g" in bands


def test_lambda_eff_all_bands():
    bp = BandpassRegistry()
    for band in bp.available_bands():
        lam = bp.lambda_eff_um(band)
        assert lam is not None, f"Missing lambda_eff_um for {band}"
        assert 0.3 < lam < 30.0, f"lambda_eff_um={lam} out of plausible range for {band}"


def test_fwhm_all_bands():
    bp = BandpassRegistry()
    for band in bp.available_bands():
        fwhm = bp.fwhm_um(band)
        assert fwhm is not None, f"Missing fwhm_um for {band}"
        assert fwhm > 0


def test_band_meta_returns_dict():
    bp = BandpassRegistry()
    meta = bp.band_meta("phot_ab_w1")
    assert isinstance(meta, dict)
    assert "lambda_eff_um" in meta
    assert "survey_ref" in meta


def test_unknown_band_returns_none():
    bp = BandpassRegistry()
    assert bp.lambda_eff_um("phot_ab_not_a_real_band") is None
    assert bp.band_meta("phot_ab_not_a_real_band") is None
    assert bp.fwhm_um("phot_ab_not_a_real_band") is None


def test_all_ir_bands_present():
    bp = BandpassRegistry()
    expected = {
        "phot_ab_w1", "phot_ab_w2", "phot_ab_w3", "phot_ab_w4",
        "phot_ab_j", "phot_ab_h", "phot_ab_k", "phot_ab_ks",
        "phot_ab_y", "phot_ab_z",
    }
    missing = expected - set(bp.available_bands())
    assert not missing, f"Missing IR bands: {missing}"


# ---------------------------------------------------------------------------
# Curve loading
# ---------------------------------------------------------------------------


def test_load_curve_wise_w1():
    bp = BandpassRegistry()
    result = bp.load_curve("phot_ab_w1")
    assert result is not None, "WISE W1 curve file should be bundled"
    wave, throughput = result
    assert wave.dtype == np.float64
    assert throughput.dtype == np.float64
    assert len(wave) > 0
    assert len(wave) == len(throughput)
    assert throughput.min() >= 0.0
    assert throughput.max() <= 1.0 + 1e-9  # allow small float rounding


def test_load_curve_gaia_g():
    bp = BandpassRegistry()
    result = bp.load_curve("phot_ab_g")
    assert result is not None, "Gaia G curve file should be bundled"
    wave, throughput = result
    # Gaia G should peak somewhere between 5000–8000 Å
    peak_idx = np.argmax(throughput)
    assert 4000 < wave[peak_idx] < 9000


def test_load_curve_missing_band_returns_none():
    bp = BandpassRegistry()
    # phot_ab_w3 has no curve registered
    result = bp.load_curve("phot_ab_w3")
    assert result is None


def test_load_curve_unknown_band_returns_none():
    bp = BandpassRegistry()
    result = bp.load_curve("phot_ab_not_a_real_band")
    assert result is None


# ---------------------------------------------------------------------------
# Lake override
# ---------------------------------------------------------------------------


def test_lake_override_wins(tmp_path: Path):
    """A curve in the lake bandpasses dir is used instead of the bundled one."""
    # Create a minimal override ECSV
    override_dir = tmp_path / "shared" / "registry" / "bandpasses"
    override_dir.mkdir(parents=True)

    ecsv_content = """\
# %ECSV 1.0
# ---
# datatype:
# - {name: wavelength, unit: Angstrom, datatype: float64}
# - {name: throughput, datatype: float64}
# meta:
#   band: phot_ab_w1
#   system: AB
#   source: "test override"
# schema: astropy-2.0
wavelength throughput
30000 0.0
35000 1.0
40000 0.0
"""
    override_path = override_dir / "WISE_W1.ecsv"
    override_path.write_text(ecsv_content)

    bp = BandpassRegistry(lake_root=tmp_path)
    result = bp.load_curve("phot_ab_w1")
    assert result is not None
    wave, _ = result
    # Our override only has 3 rows — the bundled file has more
    assert len(wave) == 3


def test_no_lake_root_uses_bundled():
    bp = BandpassRegistry(lake_root=None)
    # WISE W1 is bundled
    result = bp.load_curve("phot_ab_w1")
    assert result is not None


# ---------------------------------------------------------------------------
# Shim back-compat
# ---------------------------------------------------------------------------


def test_registry_shim_still_works():
    from data_lake.homogenize.registry import load_bandpass_metadata

    data = load_bandpass_metadata()
    assert "bands" in data
    assert "phot_ab_w1" in data["bands"]
