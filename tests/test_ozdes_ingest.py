"""Tests for OzDES stacked spectrum ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _write_ozdes_like(
    path: Path,
    *,
    source: str = "04D1qt",
    n_pix: int = 128,
    bad_fraction: float = 0.05,
) -> None:
    rng = np.random.default_rng(0)
    flux = rng.normal(10.0, 2.0, n_pix).astype(np.float32)
    variance = np.full(n_pix, 4.0, dtype=np.float32)
    bad = np.zeros(n_pix, dtype=np.float32)
    bad[: int(n_pix * bad_fraction)] = 1.0

    primary = fits.PrimaryHDU(flux)
    primary.header["SOURCE"] = source
    primary.header["RA"] = 36.6
    primary.header["DEC"] = -4.98
    primary.header["CRVAL1"] = 6000.0
    primary.header["CRPIX1"] = 1.0
    primary.header["CDELT1"] = 1.0
    primary.header["Z"] = -9.999

    var_hdu = fits.ImageHDU(variance, name="VARIANCE")
    bad_hdu = fits.ImageHDU(bad, name="BADPIX")
    fits.HDUList([primary, var_hdu, bad_hdu]).writeto(path, overwrite=True)


class TestOzdesIngest:
    def test_reads_stacked_flux_variance_badpix(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_ozdes_spectrum

        p = tmp_path / "OzDES-DR2_00099.fits"
        _write_ozdes_like(p, source="04D1qt", bad_fraction=0.1)
        catalog_id = normalize_object_id("04D1qt")

        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_ozdes_spectrum(hdul, p)

        assert len(records) == 1
        r = records[0]
        assert r.source_id == catalog_id
        assert r.flux.shape == (128,)
        assert r.ivar[0] == pytest.approx(0.25)
        assert r.mask.sum() == pytest.approx(12, abs=1)
        assert r.meta["z"] == 0.0  # sentinel cleared

    def test_auto_detects_ozdes(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        p = tmp_path / "OzDES-test.fits"
        _write_ozdes_like(p)
        assert _detect_format_from_path(p) == "ozdes"

    @pytest.mark.skipif(
        not Path("data/OzDES-DR2_00001.fits").is_file(),
        reason="OzDES example FITS not in data/",
    )
    def test_example_file(self) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import (
            _detect_format_from_path,
            _read_ozdes_spectrum,
        )

        p = Path("data/OzDES-DR2_00001.fits")
        assert _detect_format_from_path(p) == "ozdes"
        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_ozdes_spectrum(hdul, p, source_id_col="SOURCE")
        r = records[0]
        assert r.source_id == normalize_object_id("04D1qt")
        assert r.flux.shape == (5000,)
        assert r.mask.sum() == 10
        assert (r.ivar > 0).sum() == 4990
