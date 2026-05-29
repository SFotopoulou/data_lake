"""Tests for VVDS 1-D spectrum ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _write_vvds_like(path: Path, *, n_pix: int = 64, as_row: bool = True) -> None:
    flux = np.linspace(1e-18, 2e-18, n_pix, dtype=np.float32)
    data = flux[np.newaxis, :] if as_row else flux
    primary = fits.PrimaryHDU(data)
    primary.header["RA"] = 53.07825
    primary.header["DEC"] = -27.77536
    primary.header["CRVAL1"] = 5500.0
    primary.header["CRPIX1"] = -1.0
    primary.header["CDELT1"] = 7.14
    primary.header["ESO INS ID"] = "VIMOS"
    fits.HDUList([primary]).writeto(path, overwrite=True)


class TestVvdsIngest:
    def test_auto_detects_vvds(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        p = tmp_path / "sc_000030078_CDFS005_vmM1_red_30_1_atm_clean.fits"
        _write_vvds_like(p)
        assert _detect_format_from_path(p) == "vvds"

    def test_does_not_detect_vuds_as_vvds(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        flux = np.ones(32, dtype=np.float32)
        primary = fits.PrimaryHDU(flux)
        primary.header["LAM CESAM VO IDENT"] = 5101243705.0
        primary.header["CRVAL1"] = 3510.0
        primary.header["CRPIX1"] = 1.0
        primary.header["CDELT1"] = 5.0
        p = tmp_path / "sc_5101243705_test.fits"
        fits.HDUList([primary]).writeto(p, overwrite=True)
        assert _detect_format_from_path(p) == "vuds"

    def test_reads_row_primary_and_filename_id(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_vvds_spectrum

        p = tmp_path / "sc_000030078_CDFS005_vmM1_red_30_1_atm_clean.fits"
        _write_vvds_like(p, n_pix=128, as_row=True)

        with fits.open(p, memmap=True) as hdul:
            records, wcs = _read_vvds_spectrum(hdul, p)

        r = records[0]
        assert r.source_id == normalize_object_id(30078)
        assert r.flux.shape == (128,)
        assert (r.ivar == 1.0).all()
        assert r.mask.sum() == 0
        assert r.ra == pytest.approx(53.07825)
        assert r.dec == pytest.approx(-27.77536)
        assert wcs["n_pix"] == 128

    def test_reads_1d_primary(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _read_vvds_spectrum

        p = tmp_path / "sc_000030078_test.fits"
        _write_vvds_like(p, n_pix=32, as_row=False)

        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_vvds_spectrum(hdul, p)
        assert records[0].flux.shape == (32,)

    @pytest.mark.skipif(
        not Path("data/sc_000030078_CDFS005_vmM1_red_30_1_atm_clean.fits").is_file(),
        reason="VVDS example FITS not in data/",
    )
    def test_example_file(self) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import (
            _detect_format_from_path,
            _read_vvds_spectrum,
        )

        p = Path("data/sc_000030078_CDFS005_vmM1_red_30_1_atm_clean.fits")
        assert _detect_format_from_path(p) == "vvds"
        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_vvds_spectrum(hdul, p)
        r = records[0]
        assert r.source_id == normalize_object_id(30078)
        assert r.flux.shape == (557,)
