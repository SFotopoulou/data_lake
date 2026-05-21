"""Tests for SDSS/BOSS spec-*.fits spectrum ingest (_read_sdss_boss)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _write_sdss_spec_fits(
    path: Path,
    *,
    n_pix: int = 32,
    include_and_mask: bool = False,
    mask_name: str = "and_mask",
) -> None:
    """Minimal BOSS-style spec file: primary header + COADD BINTABLE."""
    loglam = np.linspace(3.5, 3.6, n_pix)
    flux = np.ones(n_pix, dtype=np.float32) * 100.0
    ivar = np.ones(n_pix, dtype=np.float32) * 0.01
    cols = [
        fits.Column(name="loglam", format="D", array=loglam),
        fits.Column(name="flux", format="E", array=flux),
        fits.Column(name="ivar", format="E", array=ivar),
    ]
    if include_and_mask:
        cols.append(
            fits.Column(
                name=mask_name,
                format="I",
                array=np.zeros(n_pix, dtype=np.int16),
            )
        )
    coadd = fits.BinTableHDU.from_columns(cols, name="COADD")
    phdu = fits.PrimaryHDU()
    phdu.header["PLUG_RA"] = 120.0
    phdu.header["PLUG_DEC"] = 45.0
    phdu.header["OBJID"] = 1234567890123456789
    phdu.header["Z"] = 0.1
    fits.HDUList([phdu, coadd]).writeto(path, overwrite=True)


class TestReadSdssBoss:
    def test_coadd_without_mask_column(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _read_sdss_boss

        path = tmp_path / "spec-test.fits"
        _write_sdss_spec_fits(path, include_and_mask=False)
        with fits.open(path) as hdul:
            records, wcs = _read_sdss_boss(hdul, source_id_col="OBJID")
        assert len(records) == 1
        assert len(records[0].flux) == 32
        assert records[0].mask.shape == (32,)
        assert np.all(records[0].mask == 0)

    def test_coadd_with_uppercase_and_mask(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _read_sdss_boss

        path = tmp_path / "spec-mask.fits"
        _write_sdss_spec_fits(path, include_and_mask=True, mask_name="AND_MASK")
        with fits.open(path) as hdul:
            records, _ = _read_sdss_boss(hdul, source_id_col="OBJID")
        assert len(records[0].mask) == 32

    def test_detect_format(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        path = tmp_path / "spec-test.fits"
        _write_sdss_spec_fits(path)
        assert _detect_format_from_path(path) == "sdss_boss"
