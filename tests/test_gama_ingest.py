"""Tests for GAMA stacked 1-D spectrum ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _write_gama_like(
    path: Path,
    *,
    specid: str = "G23_Y7_015_265",
    n_pix: int = 256,
) -> None:
    rng = np.random.default_rng(0)
    flux = rng.normal(10.0, 2.0, n_pix).astype(np.float32)
    sigma = np.full(n_pix, 2.0, dtype=np.float32)
    flux_nocal = flux * 0.9
    sigma_nocal = sigma * 1.1
    sky = rng.normal(0.0, 0.5, n_pix).astype(np.float32)
    data = np.stack([flux, sigma, flux_nocal, sigma_nocal, sky], axis=0)

    primary = fits.PrimaryHDU(data)
    h = primary.header
    h["ORIGIN"] = "GAMA"
    h["ROW1"] = "Spectrum"
    h["ROW2"] = "Error"
    h["ROW3"] = "Spectrum_nocalib"
    h["ROW4"] = "Error_nocalib"
    h["ROW5"] = "Sky"
    h["SPECID"] = specid
    h["RA"] = 346.619
    h["DEC"] = -34.349
    h["Z"] = 0.18
    h["SN"] = 5.5
    h["T_EXP"] = 3600.0
    h["INSTRUME"] = "AAOMEGA-2dF"
    h["CTYPE1"] = "Wavelength"
    h["CUNIT1"] = "Angstrom"
    h["CRPIX1"] = (n_pix + 1) / 2.0
    h["CRVAL1"] = 6000.0
    h["CD1_1"] = 2.0
    fits.HDUList([primary]).writeto(path, overwrite=True)


class TestGamaIngest:
    def test_reads_stacked_rows(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_gama_spectrum

        specid = "G23_Y7_015_265"
        p = tmp_path / f"{specid}.fits"
        _write_gama_like(p, specid=specid, n_pix=128)
        expected_id = normalize_object_id(specid)

        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_gama_spectrum(hdul, p)

        assert len(records) == 1
        r = records[0]
        assert r.source_id == expected_id
        assert r.flux.shape == (128,)
        assert r.ivar[0] == pytest.approx(0.25)
        assert r.meta["z"] == pytest.approx(0.18)
        assert r.meta["snr"] == pytest.approx(5.5)

    def test_auto_detects_gama(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        p = tmp_path / "G23_Y7_015_999.fits"
        _write_gama_like(p)
        assert _detect_format_from_path(p) == "gama"

    def test_ingest_to_zarr(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits

        p = tmp_path / "G23_test.fits"
        lake = tmp_path / "lake"
        _write_gama_like(p)
        index = ingest_spectra_from_fits(
            p, lake, "GAMA_TEST", fmt="gama", norder=5, link_id_col="SPECID",
        )
        assert len(index) == 1

    @pytest.mark.skipif(
        not Path("data/G23_Y7_015_265.fit").is_file(),
        reason="GAMA example FITS not in data/",
    )
    def test_example_file(self) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import (
            _detect_format_from_path,
            _read_gama_spectrum,
        )

        p = Path("data/G23_Y7_015_265.fit")
        assert _detect_format_from_path(p) == "gama"
        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_gama_spectrum(hdul, p, link_id_col="SPECID")
        r = records[0]
        assert r.source_id == normalize_object_id("G23_Y7_015_265")
        assert r.flux.shape == (4953,)
        assert r.ra == pytest.approx(346.61896, rel=1e-5)
        assert (r.ivar > 0).sum() > 4000
        assert len(records) == 1
