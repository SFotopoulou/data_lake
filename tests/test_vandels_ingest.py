"""Tests for VANDELS 1-D spectrum ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _write_vandels_like(path: Path, *, n_pix: int = 128) -> None:
    flux = np.linspace(1e-18, 2e-18, n_pix, dtype=np.float64)
    noise = np.full(n_pix, 2e-19, dtype=np.float64)
    noise[0] = 0.0

    primary = fits.PrimaryHDU(flux)
    primary.header["PND OBJRA"] = 34.5
    primary.header["PND OBJDEC"] = -5.2
    primary.header["PND Z"] = 3.6
    primary.header["PND OBJID"] = "UDS313141"
    primary.header["CRVAL1"] = 4800.0
    primary.header["CRPIX1"] = 1.0
    primary.header["CDELT1"] = 2.5

    exr2d = fits.ImageHDU(np.zeros((10, n_pix), dtype=np.float32), name="EXR2D")
    sky = fits.ImageHDU(np.zeros(n_pix, dtype=np.float64), name="SKY")
    noise_hdu = fits.ImageHDU(noise, name="NOISE")
    exr1d = fits.ImageHDU(flux.copy(), name="EXR1D")

    fits.HDUList([primary, exr2d, sky, noise_hdu, exr1d]).writeto(path, overwrite=True)


class TestVandelsIngest:
    def test_auto_detects_vandels(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        p = tmp_path / "sc_UDS313141_test.fits"
        _write_vandels_like(p)
        assert _detect_format_from_path(p) == "vandels"

    def test_reads_primary_and_noise(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_vandels_spectrum

        p = tmp_path / "sc_UDS313141_P3M1Q4_008_1.fits"
        _write_vandels_like(p)

        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_vandels_spectrum(hdul, p)

        r = records[0]
        assert r.source_id == normalize_object_id(p.name)
        assert r.flux.shape == (128,)
        assert r.ivar[1] > 0
        assert r.ivar[0] == 0
        assert r.mask[0] == 1
        assert r.meta["z"] == pytest.approx(3.6)
        assert r.ra == pytest.approx(34.5)
        assert r.dec == pytest.approx(-5.2)

    @pytest.mark.skipif(
        not Path("data/sc_UDS313141_P3M1Q4_008_1.fits").is_file(),
        reason="VANDELS example FITS not in data/",
    )
    def test_example_file(self) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import (
            _detect_format_from_path,
            _read_vandels_spectrum,
        )

        p = Path("data/sc_UDS313141_P3M1Q4_008_1.fits")
        assert _detect_format_from_path(p) == "vandels"
        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_vandels_spectrum(hdul, p)
        r = records[0]
        assert r.source_id == normalize_object_id(p.name)
        assert r.flux.shape == (2154,)
        assert (r.ivar > 0).all()
        assert r.meta["z"] == pytest.approx(3.6054)
