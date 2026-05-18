"""
Tests for the DESI-specific ingest path in fits_to_spectra_zarr.py
and the resolution-related helpers in io/spectra.py.

These tests do NOT require desispec or real FITS files to be present.
They mock the desispec import and build synthetic data to exercise
every guard and helper that was added in the desispec switch.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers to build a synthetic Spectrum object
# ---------------------------------------------------------------------------

N_DIAG = 11
_HALF = N_DIAG // 2


def test_desi_read_spectra_skip_hdus():
    from data_lake.ingest.fits_to_spectra_zarr import _desi_read_spectra_skip_hdus

    base = _desi_read_spectra_skip_hdus(with_resolution=False)
    assert "EXP_FIBERMAP" in base
    assert "RESOLUTION" in base
    assert "RESOLUTION" not in _desi_read_spectra_skip_hdus(with_resolution=True)


def _make_spectrum(n_pix: int = 100, with_resolution: bool = True):
    """
    Return a minimal Spectrum with or without a resolution matrix.

    The synthetic resolution is built so that the dense (N_pix, N_pix) matrix
    has *rows* summing to 1.0 on the interior (i.e. rows for indices in
    [_HALF, N_pix - _HALF)).  Edge rows naturally sum to less than 1 because
    off-diagonal entries fall outside the matrix bounds.
    """
    from data_lake.io.spectra import Spectrum

    offsets = np.arange(-_HALF, _HALF + 1, dtype=np.int32)

    # Build a Gaussian LSF in banded form.  For dia_matrix with these offsets,
    # the dense matrix element R[i, j] = data[k, j] where j = i + offsets[k].
    # So (R @ ones)[i] = sum_k data[k, i + offsets[k]] for valid (i + offsets[k])
    # in [0, N_pix).  For interior i, the row sum equals sum_k data[k, *].
    sigma = 1.5
    profile = np.exp(-0.5 * (offsets.astype(np.float64) / sigma) ** 2)
    profile /= profile.sum()
    diags = np.broadcast_to(profile[:, None], (N_DIAG, n_pix)).astype(np.float32).copy()

    return Spectrum(
        source_id=1,
        flux=np.ones(n_pix, dtype=np.float32),
        ivar=np.ones(n_pix, dtype=np.float32),
        mask=np.zeros(n_pix, dtype=np.uint8),
        wavelength=np.linspace(3600, 9800, n_pix),
        meta={"z": 0.5, "z_err": 0.001, "snr": 10.0, "exptime": 3600.0, "R": 3000.0, "instr": "DESI"},
        wcs_attrs={"ctype": "WAVE", "crval": 3600.0, "cdelt": 6.2, "crpix": 1.0, "unit": "Angstrom"},
        resolution=diags if with_resolution else None,
        resolution_offsets=offsets if with_resolution else None,
    )


# ---------------------------------------------------------------------------
# Test 1: missing desispec raises ImportError with the install hint
# ---------------------------------------------------------------------------

class TestMissingDesispec:
    def test_importerror_contains_install_hint(self, monkeypatch):
        """_import_desispec() must raise ImportError with a pip install hint."""
        from data_lake.ingest.fits_to_spectra_zarr import _import_desispec

        # Force all desispec entries (existing or future) to be None so any
        # ``import desispec[.io|.coaddition]`` inside the helper raises.
        # We use monkeypatch.setitem so pytest restores sys.modules afterwards.
        for name in ("desispec", "desispec.io", "desispec.coaddition"):
            monkeypatch.setitem(sys.modules, name, None)

        with pytest.raises(ImportError, match="pip install"):
            _import_desispec()


# ---------------------------------------------------------------------------
# Test 2: with_resolution=True + wavelength_mode="per_source" raises ValueError
# ---------------------------------------------------------------------------

class TestWavelengthModeGuard:
    def test_per_source_wavelength_mode_rejected(self, tmp_path):
        """ingest_spectra_from_fits must reject per_source + with_resolution."""
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits

        # We just need to reach the guard – no actual file I/O needed.
        dummy_fits = tmp_path / "dummy.fits"
        dummy_fits.touch()

        with pytest.raises(ValueError, match="wavelength_mode='shared'"):
            ingest_spectra_from_fits(
                source_path=dummy_fits,
                output_root=tmp_path,
                survey_name="test",
                wavelength_mode="per_source",
                with_resolution=True,
            )

    def test_non_desi_format_rejected(self, tmp_path):
        """with_resolution=True on a non-DESI file must raise ValueError."""
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits

        dummy_fits = tmp_path / "dummy.fits"
        dummy_fits.touch()

        # Patch format detection to return sdss_boss so we bypass file I/O
        with patch(
            "data_lake.ingest.fits_to_spectra_zarr._detect_format_from_path",
            return_value="sdss_boss",
        ):
            with pytest.raises(ValueError, match="only supported for DESI"):
                ingest_spectra_from_fits(
                    source_path=dummy_fits,
                    output_root=tmp_path,
                    survey_name="test",
                    with_resolution=True,
                )


# ---------------------------------------------------------------------------
# Test 3: resolution_operator() returns a properly-normalised sparse matrix
# ---------------------------------------------------------------------------

class TestResolutionOperator:
    def test_returns_dia_matrix(self):
        """resolution_operator() returns a scipy dia_matrix of shape (N, N)."""
        pytest.importorskip("scipy", reason="scipy not installed")
        from scipy.sparse import dia_matrix

        spec = _make_spectrum(n_pix=100, with_resolution=True)
        R = spec.resolution_operator()

        assert isinstance(R, dia_matrix)
        assert R.shape == (100, 100)

    def test_row_sums_close_to_one(self):
        """Interior row sums of R should be ~1 (flux conservation).

        Edge rows (within ``_HALF`` pixels of either boundary) are expected to
        have lower sums because off-diagonal entries fall outside the matrix.
        """
        pytest.importorskip("scipy", reason="scipy not installed")
        n_pix = 200
        spec = _make_spectrum(n_pix=n_pix, with_resolution=True)
        R = spec.resolution_operator()

        row_sums = np.asarray(R.sum(axis=1)).ravel()
        interior = row_sums[_HALF:n_pix - _HALF]
        np.testing.assert_allclose(
            interior, 1.0, atol=1e-3,
            err_msg="Interior row sums of R are not close to 1 (flux not conserved)",
        )

    def test_raises_when_resolution_absent(self):
        """resolution_operator() must raise if resolution was not stored."""
        spec = _make_spectrum(n_pix=50, with_resolution=False)
        with pytest.raises(ValueError, match="--with-resolution"):
            spec.resolution_operator()

    def test_matrix_times_constant_gives_constant(self):
        """R @ ones ≈ ones in the interior (LSF of a flat spectrum is flat).

        Same edge-effect caveat as ``test_row_sums_close_to_one``.
        """
        pytest.importorskip("scipy", reason="scipy not installed")
        n_pix = 300
        spec = _make_spectrum(n_pix=n_pix, with_resolution=True)
        R = spec.resolution_operator()

        ones = np.ones(n_pix)
        result = R @ ones
        np.testing.assert_allclose(result[_HALF:n_pix - _HALF], 1.0, atol=1e-3)
