"""Tests for 6dFGS VR-spectrum ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _write_minimal_6df(path: Path, *, n_pix: int = 1024) -> None:
    """Create a minimal 6dFGS-like file with V, R, VR spectral extensions."""
    primary = fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32))
    primary.header["RA"] = 150.0
    primary.header["DEC"] = -30.0

    # Placeholders for image extensions (common in 6dF files)
    hdus = [primary]
    for i in range(1, 6):
        hdus.append(fits.ImageHDU(np.zeros((8, 8), dtype=np.float32), name=f"IMG{i}"))

    pix = np.arange(n_pix, dtype=np.float32)
    wave = 5000.0 + 2.0 * pix

    v_flux = np.ones(n_pix, dtype=np.float32) * 11.0
    r_flux = np.ones(n_pix, dtype=np.float32) * 22.0
    vr_flux = np.ones(n_pix, dtype=np.float32) * 33.0
    var = np.ones(n_pix, dtype=np.float32) * 4.0
    sky = np.zeros(n_pix, dtype=np.float32)

    target = path.stem
    hdu_v = fits.ImageHDU(np.stack([v_flux, var, sky]), name="V")
    hdu_r = fits.ImageHDU(np.stack([r_flux, var, sky]), name="R")
    hdu_vr = fits.ImageHDU(np.stack([vr_flux, var, sky, wave]), name="VR")
    hdu_v.header["TARGET"] = target
    hdu_r.header["TARGET"] = target
    hdu_vr.header["TARGET"] = target
    hdu_vr.header["CRVAL1"] = 4000.0  # intentionally different from explicit wave row
    hdu_vr.header["CRPIX1"] = 1.0
    hdu_vr.header["CDELT1"] = 1.0
    hdu_vr.header["Z"] = 0.12

    hdus.extend([hdu_v, hdu_r, hdu_vr])
    fits.HDUList(hdus).writeto(path, overwrite=True)


def test_detect_format_6df(tmp_path: Path) -> None:
    from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

    p = tmp_path / "g0001.fits"
    _write_minimal_6df(p)
    assert _detect_format_from_path(p) == "6df"


def test_read_6df_uses_vr_extension_only(tmp_path: Path) -> None:
    from data_lake.ingest.fits_to_spectra_zarr import _read_6df_spectrum

    p = tmp_path / "g0001.fits"
    _write_minimal_6df(p)
    with fits.open(p, memmap=True) as hdul:
        records, _ = _read_6df_spectrum(hdul, p)

    assert len(records) == 1
    rec = records[0]
    # Must come from VR flux row (33), not V (11) or R (22)
    assert np.allclose(rec.flux, 33.0)
    # 1 / variance(4.0)
    assert np.allclose(rec.ivar, 0.25)


def test_read_6df_prefers_explicit_wave_row(tmp_path: Path) -> None:
    from data_lake.ingest.fits_to_spectra_zarr import _read_6df_spectrum

    p = tmp_path / "g0001.fits"
    _write_minimal_6df(p, n_pix=16)
    with fits.open(p, memmap=True) as hdul:
        records, _ = _read_6df_spectrum(hdul, p)

    wave = records[0].wavelength
    assert wave is not None
    # Explicit row uses 5000 + 2*pix, while WCS header says 4000 + 1*pix.
    assert np.isclose(wave[0], 5000.0)
    assert np.isclose(wave[1], 5002.0)


def test_read_6df_uses_target_header_not_filename(tmp_path: Path) -> None:
    from data_lake.ingest.fits_to_parquet import normalize_object_id
    from data_lake.ingest.fits_to_spectra_zarr import _read_6df_spectrum

    p = tmp_path / "wrong_name.fits"
    _write_minimal_6df(p)
    with fits.open(p, mode="update") as hdul:
        for hdu in hdul:
            if (hdu.name or "").strip().upper() == "VR":
                hdu.header["TARGET"] = "g2259418-254505"
        hdul.flush()

    with fits.open(p, memmap=True) as hdul:
        records, _ = _read_6df_spectrum(hdul, p, source_id_col="targetname")

    assert records[0].source_id == normalize_object_id("g2259418-254505")


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "g2259418-254505.fits").is_file(),
    reason="requires data/g2259418-254505.fits",
)
def test_real_6df_detects_and_uses_target_header() -> None:
    from data_lake.ingest.fits_to_parquet import normalize_object_id
    from data_lake.ingest.fits_to_spectra_zarr import (
        _detect_format_from_path,
        _read_6df_spectrum,
    )

    p = Path(__file__).resolve().parents[1] / "data" / "g2259418-254505.fits"
    assert _detect_format_from_path(p) == "6df"
    with fits.open(p, memmap=True) as hdul:
        records, _ = _read_6df_spectrum(hdul, p, source_id_col="targetname")
    assert records[0].source_id == normalize_object_id("g2259418-254505")


def test_ingest_6df_end_to_end(tmp_path: Path) -> None:
    from data_lake.ingest.fits_to_parquet import normalize_object_id
    from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
    from data_lake.io.spectra import SpectrumAccessor

    p = tmp_path / "g00123.fits"
    _write_minimal_6df(p)

    lake = tmp_path / "lake"
    index_map = ingest_spectra_from_fits(
        p,
        lake,
        "SIXDF_DR3",
        fmt="6df",
        norder=5,
        on_duplicate_source_id="skip",
    )
    sid = normalize_object_id("g00123")
    assert sid in index_map

    acc = SpectrumAccessor(lake, "SIXDF_DR3")
    sp = acc.get_spectrum(sid)
    assert sp is not None
    assert sp.flux.shape[0] == 1024
    assert np.allclose(sp.flux, 33.0)

