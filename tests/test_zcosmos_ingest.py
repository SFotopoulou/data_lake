"""Tests for zCOSMOS 1-D spectra ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _write_zcosmos_like(path: Path, *, n_pix: int = 128) -> None:
    """Minimal zCOSMOS-style spectral container FITS."""
    wave = np.linspace(5500.0, 5700.0, n_pix, dtype=np.float32)
    flux = np.linspace(1e-18, 2e-18, n_pix, dtype=np.float32)
    err = np.full(n_pix, 2e-19, dtype=np.float32)
    err[0] = 0.0  # masked pixel in ivar/mask

    primary = fits.PrimaryHDU()
    primary.header["RA"] = 150.10
    primary.header["DEC"] = 2.13
    primary.header["OBJECT"] = "960004"

    cols = fits.ColDefs(
        [
            fits.Column(name="WAVE", format=f"{n_pix}E", array=[wave]),
            fits.Column(name="FLUX_REDUCED", format=f"{n_pix}E", array=[flux]),
            fits.Column(name="ERR", format=f"{n_pix}E", array=[err]),
        ]
    )
    container = fits.BinTableHDU.from_columns(cols, name="SPECTRAL CONTAINER")
    container.header["RA"] = 150.10
    container.header["DEC"] = 2.13

    fits.HDUList([primary, container]).writeto(path, overwrite=True)


class TestZcosmosIngest:
    def test_auto_detects_zcosmos(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        p = tmp_path / "zCOSMOS_BRIGHT_DR3_test.fits"
        _write_zcosmos_like(p)
        assert _detect_format_from_path(p) == "zcosmos"

    def test_reads_spectral_container(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_zcosmos_spectrum

        p = tmp_path / "zCOSMOS_BRIGHT_DR3_000960004_ZCMRa65_M1_Q4_6_1.fits"
        _write_zcosmos_like(p)

        with fits.open(p, memmap=True) as hdul:
            records, wcs = _read_zcosmos_spectrum(hdul, p)
        assert len(records) == 1
        r = records[0]
        assert r.source_id == normalize_object_id(p.name)
        assert r.flux.shape == (128,)
        assert r.wavelength.shape == (128,)
        assert r.ivar[1] > 0
        assert r.ivar[0] == 0
        assert r.mask[0] == 1
        assert r.mask[1] == 0
        assert wcs["wcs_source"] == "explicit"

    def test_placeholder_err_uses_unit_ivar(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _read_zcosmos_spectrum

        p = tmp_path / "zCOSMOS_BRIGHT_DR3_placeholder.fits"
        _write_zcosmos_like(p, n_pix=64)
        with fits.open(p, mode="update", memmap=False) as hdul:
            hdul[1].data["ERR"][0] = np.zeros(64, dtype=np.float32)
            hdul.flush()

        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_zcosmos_spectrum(hdul, p)
        r = records[0]
        assert (r.ivar == 1.0).all()
        assert r.mask.sum() == 0

    @pytest.mark.skipif(
        not Path("data/zCOSMOS_BRIGHT_DR3_000960004_ZCMRa65_M1_Q4_6_1.fits").is_file(),
        reason="zCOSMOS example FITS not in data/",
    )
    def test_example_file(self) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import (
            _detect_format_from_path,
            _read_zcosmos_spectrum,
        )

        p = Path("data/zCOSMOS_BRIGHT_DR3_000960004_ZCMRa65_M1_Q4_6_1.fits")
        assert _detect_format_from_path(p) == "zcosmos"
        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_zcosmos_spectrum(hdul, p)
        r = records[0]
        assert r.source_id == normalize_object_id(p.name)
        assert r.flux.shape == (1642,)
        assert r.mask.shape == (1642,)
        assert r.mask.sum() == 0
        assert (r.ivar == 1.0).all()
        assert np.isfinite(r.wavelength).all()
