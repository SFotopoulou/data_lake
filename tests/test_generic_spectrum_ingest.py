"""Tests for generic 1-D spectrum ingest (flux + optional VARIANCE/IVAR HDU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _write_generic_with_variance(path: Path, *, n_pix: int = 64) -> None:
    flux = np.linspace(1.0, 2.0, n_pix, dtype=np.float32)
    variance = np.full(n_pix, 4.0, dtype=np.float32)  # ivar = 0.25
    variance[0] = np.nan

    primary = fits.PrimaryHDU(flux)
    primary.header["EXTNAME"] = "spectrum"
    primary.header["CRVAL1"] = 4000.0
    primary.header["CRPIX1"] = 1.0
    primary.header["CDELT1"] = 2.0
    primary.header["RA_OBJ"] = 10.0
    primary.header["DEC_OBJ"] = -5.0
    primary.header["Z"] = 0.1

    var_hdu = fits.ImageHDU(variance, name="VARIANCE")
    var_hdu.header["CRVAL1"] = 4000.0
    var_hdu.header["CRPIX1"] = 1.0
    var_hdu.header["CDELT1"] = 2.0

    fits.HDUList([primary, var_hdu]).writeto(path, overwrite=True)


def _write_generic_flux_only(path: Path, *, n_pix: int = 32) -> None:
    flux = np.ones(n_pix, dtype=np.float32)
    primary = fits.PrimaryHDU(flux)
    primary.header["CRVAL1"] = 5000.0
    primary.header["CRPIX1"] = 1.0
    primary.header["CDELT1"] = 1.0
    fits.HDUList([primary]).writeto(path, overwrite=True)


class TestReadGeneric1d:
    def test_reads_variance_extension_as_ivar(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _read_generic_1d

        p = tmp_path / "spec_var.fits"
        _write_generic_with_variance(p)
        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_generic_1d(
                hdul, ra_col="RA_OBJ", dec_col="DEC_OBJ",
            )

        assert len(records) == 1
        assert records[0].flux.shape == (64,)
        assert records[0].ivar[1] == pytest.approx(0.25)
        assert records[0].ivar[0] == 0.0  # NaN variance → ivar 0

    def test_flux_only_defaults_ivar_to_one(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _read_generic_1d

        p = tmp_path / "spec_only.fits"
        _write_generic_flux_only(p)
        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_generic_1d(hdul)

        assert (records[0].ivar == 1.0).all()

    def test_ivar_extension_used_directly(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _read_generic_1d

        n_pix = 16
        flux = np.ones(n_pix, dtype=np.float32)
        ivar = np.full(n_pix, 0.5, dtype=np.float32)
        primary = fits.PrimaryHDU(flux)
        primary.header["CRVAL1"] = 4000.0
        primary.header["CRPIX1"] = 1.0
        primary.header["CDELT1"] = 1.0
        ivar_hdu = fits.ImageHDU(ivar, name="IVAR")
        p = tmp_path / "spec_ivar.fits"
        fits.HDUList([primary, ivar_hdu]).writeto(p, overwrite=True)

        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_generic_1d(hdul)

        assert (records[0].ivar == 0.5).all()

    @pytest.mark.skipif(
        not Path("data/wig225415.fits").is_file(),
        reason="wig225415.fits not in data/",
    )
    def test_wig_example_file(self) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _read_generic_1d

        p = Path("data/wig225415.fits")
        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_generic_1d(
                hdul, ra_col="RA_OBJ", dec_col="DEC_OBJ",
            )

        assert records[0].flux.shape == (4904,)
        assert not (records[0].ivar == 1.0).all()
        assert (records[0].ivar > 0).any()
