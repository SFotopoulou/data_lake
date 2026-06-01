"""Tests for VUDS 1-D spectrum ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _write_vuds_like(path: Path, *, n_pix: int = 64, obj_id: float = 5101243705.0) -> None:
    flux = np.linspace(1e-18, 2e-18, n_pix, dtype=np.float32)
    primary = fits.PrimaryHDU(flux)
    primary.header["LAM CESAM VO IDENT"] = obj_id
    primary.header["LAM CESAM VO ALPHA"] = 150.1519928
    primary.header["LAM CESAM VO DELTA"] = 2.30920005
    primary.header["LAM CESAM VO Z"] = 1.0962
    primary.header["CRVAL1"] = 3510.71
    primary.header["CRPIX1"] = 1.0
    primary.header["CDELT1"] = 5.355
    primary.header["ESO INS ID"] = "VIMOS"
    fits.HDUList([primary]).writeto(path, overwrite=True)


class TestVudsIngest:
    def test_auto_detects_vuds(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        p = tmp_path / "sc_5101243705_test.fits"
        _write_vuds_like(p)
        assert _detect_format_from_path(p) == "vuds"

    def test_reads_primary_and_id_header(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_vuds_spectrum

        p = tmp_path / "sc_5101243705_F51P006_join_A_10_1_atm_clean.fits"
        _write_vuds_like(p)

        with fits.open(p, memmap=True) as hdul:
            records, wcs = _read_vuds_spectrum(hdul, p)

        r = records[0]
        assert r.source_id == normalize_object_id(p.name)
        assert r.flux.shape == (64,)
        assert (r.ivar == 1.0).all()
        assert r.mask.sum() == 0
        assert r.ra == pytest.approx(150.1519928)
        assert r.dec == pytest.approx(2.30920005)
        assert r.meta["z"] == pytest.approx(1.0962)
        assert wcs["n_pix"] == 64

    @pytest.mark.skipif(
        not Path("data/sc_5101243705_F51P006_join_A_10_1_atm_clean.fits").is_file(),
        reason="VUDS example FITS not in data/",
    )
    def test_example_file(self) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import (
            _detect_format_from_path,
            _read_vuds_spectrum,
        )

        p = Path("data/sc_5101243705_F51P006_join_A_10_1_atm_clean.fits")
        assert _detect_format_from_path(p) == "vuds"
        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_vuds_spectrum(hdul, p)
        r = records[0]
        assert r.source_id == normalize_object_id(p.name)
        assert r.flux.shape == (1117,)
        assert r.meta["z"] == pytest.approx(1.0962)
