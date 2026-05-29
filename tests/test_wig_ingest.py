"""Tests for WiggleZ (wig) spectrum ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _write_wig_like(path: Path, *, basename_key: str = "wig225415.fits") -> None:
    """Minimal WiggleZ-style FITS (matches layout of wig225415.fits)."""
    n_pix = 128
    flux = np.linspace(1.0, 2.0, n_pix, dtype=np.float32)
    variance = np.full(n_pix, 4.0, dtype=np.float32)

    primary = fits.PrimaryHDU(flux)
    primary.header["EXTNAME"] = "spectrum"
    primary.header["CRVAL1"] = 4000.0
    primary.header["CRPIX1"] = 1.0
    primary.header["CDELT1"] = 2.0
    primary.header["RA_OBJ"] = 10.0
    primary.header["DEC_OBJ"] = -5.0
    primary.header["Z"] = 0.5

    var_hdu = fits.ImageHDU(variance, name="VARIANCE")
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, var_hdu]).writeto(path, overwrite=True)
    assert path.name == basename_key or path.name.endswith(".fits")


class TestWigFormat:
    def test_source_id_matches_catalog_filename_column(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import (
            _read_wig_spectrum,
            _wig_source_id_from_path,
        )

        p = tmp_path / "wig225415.fits"
        _write_wig_like(p)
        catalog_id = normalize_object_id("wig225415.fits")
        assert _wig_source_id_from_path(p) == catalog_id

        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_wig_spectrum(hdul, p)

        assert len(records) == 1
        assert records[0].source_id == catalog_id
        assert records[0].ivar[0] == pytest.approx(0.25)

    def test_stem_without_fits_suffix_does_not_match(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _wig_source_id_from_path

        p = tmp_path / "wig225415.fits"
        _write_wig_like(p)
        assert _wig_source_id_from_path(p) != normalize_object_id("wig225415")

    def test_auto_detects_wig(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        p = tmp_path / "wig999001.fits"
        _write_wig_like(p)
        assert _detect_format_from_path(p) == "wig"

    def test_non_wig_prefix_stays_generic(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        n_pix = 64
        flux = np.ones(n_pix, dtype=np.float32)
        variance = np.ones(n_pix, dtype=np.float32)
        primary = fits.PrimaryHDU(flux)
        primary.header["EXTNAME"] = "spectrum"
        primary.header["CRVAL1"] = 4000.0
        primary.header["CRPIX1"] = 1.0
        primary.header["CDELT1"] = 1.0
        var_hdu = fits.ImageHDU(variance, name="VARIANCE")
        p = tmp_path / "other0001.fits"
        fits.HDUList([primary, var_hdu]).writeto(p, overwrite=True)
        assert _detect_format_from_path(p) == "generic"

    @pytest.mark.skipif(
        not Path("data/wig225415.fits").is_file(),
        reason="wig225415.fits not in data/",
    )
    def test_example_file(self) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import (
            _detect_format_from_path,
            _read_wig_spectrum,
        )

        p = Path("data/wig225415.fits")
        assert _detect_format_from_path(p) == "wig"
        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_wig_spectrum(hdul, p)
        assert records[0].source_id == normalize_object_id("wig225415.fits")
        assert records[0].flux.shape == (4904,)
