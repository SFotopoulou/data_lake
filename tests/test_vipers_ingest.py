"""Tests for VIPERS 1-D spectrum ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table


def _write_vipers_like(path: Path, *, n_pix: int = 64, obj_id: int = 406064719) -> None:
    wave = np.linspace(5500.0, 5700.0, n_pix, dtype=np.float32)
    flux = np.linspace(1e-18, 2e-18, n_pix, dtype=np.float32)
    noise = np.full(n_pix, 2e-19, dtype=np.float32)
    mask = np.zeros(n_pix, dtype=np.int32)
    mask[0] = 2
    mask[1] = 3

    primary = fits.PrimaryHDU()
    table = Table(
        {
            "WAVES": wave,
            "FLUXES": flux,
            "NOISE": noise,
            "SKY": np.zeros(n_pix, dtype=np.float32),
            "FLUXES_UNEDIT": flux.copy(),
            "MASK": mask,
        }
    )
    hdu = fits.BinTableHDU(table, name=f"VIPERS {obj_id}")
    hdu.header["ID"] = obj_id
    hdu.header["RA"] = 334.98
    hdu.header["DEC"] = 1.09
    hdu.header["REDSHIFT"] = 0.8596
    fits.HDUList([primary, hdu]).writeto(path, overwrite=True)


class TestVipersIngest:
    def test_auto_detects_vipers(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        p = tmp_path / "VIPERS_406064719.fits"
        _write_vipers_like(p)
        assert _detect_format_from_path(p) == "vipers"

    def test_reads_table_and_preserves_mask(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_vipers_spectrum

        p = tmp_path / "VIPERS_406064719.fits"
        _write_vipers_like(p, obj_id=406064719)

        with fits.open(p, memmap=True) as hdul:
            records, wcs = _read_vipers_spectrum(hdul, p)

        r = records[0]
        assert r.source_id == normalize_object_id(406064719)
        assert r.flux.shape == (64,)
        assert r.wavelength.shape == (64,)
        assert r.ivar[2] > 0
        assert r.mask[0] == 2
        assert r.mask[1] == 3
        assert r.meta["z"] == pytest.approx(0.8596)
        assert wcs["wcs_source"] == "explicit"

    @pytest.mark.skipif(
        not Path("data/VIPERS_406064719.fits").is_file(),
        reason="VIPERS example FITS not in data/",
    )
    def test_example_file(self) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import (
            _detect_format_from_path,
            _read_vipers_spectrum,
        )

        p = Path("data/VIPERS_406064719.fits")
        assert _detect_format_from_path(p) == "vipers"
        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_vipers_spectrum(hdul, p)
        r = records[0]
        assert r.source_id == normalize_object_id(406064719)
        assert r.flux.shape == (557,)
        assert set(np.unique(r.mask).tolist()) == {0, 2, 3}
        assert (r.ivar > 0).all()
