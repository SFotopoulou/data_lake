"""Tests for 6dFGS VR-spectrum ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _write_minimal_6df(
    path: Path,
    *,
    n_pix: int = 1024,
    vr_observations: list[dict[str, object]] | None = None,
) -> None:
    """Create a minimal 6dFGS-like file with V, R, and one or more VR extensions."""
    primary = fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32))
    primary.header["RA"] = 150.0
    primary.header["DEC"] = -30.0

    hdus = [primary]
    for i in range(1, 6):
        hdus.append(fits.ImageHDU(np.zeros((8, 8), dtype=np.float32), name=f"IMG{i}"))

    pix = np.arange(n_pix, dtype=np.float32)
    wave = 5000.0 + 2.0 * pix
    var = np.ones(n_pix, dtype=np.float32) * 4.0
    sky = np.zeros(n_pix, dtype=np.float32)

    target = path.stem
    if vr_observations is None:
        vr_observations = [{"obsid_v": "V-00001", "obsid_r": "R-00001", "flux_scale": 33.0, "z": 0.12}]

    for obs_idx, obs in enumerate(vr_observations, start=1):
        flux_scale = float(obs.get("flux_scale", 33.0))
        obsid_v = str(obs.get("obsid_v", f"V-{obs_idx:05d}"))
        obsid_r = str(obs.get("obsid_r", f"R-{obs_idx:05d}"))
        z_val = float(obs.get("z", 0.12))
        obs_target = str(obs.get("target", target))
        kbestr = int(obs.get("kbestr", obs_idx))

        v_flux = np.ones(n_pix, dtype=np.float32) * (flux_scale - 22.0)
        r_flux = np.ones(n_pix, dtype=np.float32) * (flux_scale - 11.0)
        vr_flux = np.ones(n_pix, dtype=np.float32) * flux_scale

        hdu_v = fits.ImageHDU(np.stack([v_flux, var, sky]), name="V")
        hdu_r = fits.ImageHDU(np.stack([r_flux, var, sky]), name="R")
        hdu_vr = fits.ImageHDU(np.stack([vr_flux, var, sky, wave]), name="VR")
        for hdu in (hdu_v, hdu_r, hdu_vr):
            hdu.header["TARGET"] = obs_target
            hdu.header["OBSRA"] = 150.0
            hdu.header["OBSDEC"] = -30.0
            hdu.header["KBESTR"] = kbestr
        hdu_vr.header["OBSID_V"] = obsid_v
        hdu_vr.header["OBSID_R"] = obsid_r
        hdu_vr.header["CRVAL1"] = 4000.0
        hdu_vr.header["CRPIX1"] = 1.0
        hdu_vr.header["CDELT1"] = 1.0
        hdu_vr.header["Z"] = z_val
        hdus.extend([hdu_v, hdu_r, hdu_vr])

    fits.HDUList(hdus).writeto(path, overwrite=True)


def _expected_6df_source_id(
    target: str,
    obsid_v: str,
    obsid_r: str,
) -> int:
    from data_lake.ingest.fits_to_parquet import composite_link_label, normalize_object_id

    return normalize_object_id(composite_link_label(target, obsid_v, obsid_r))


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
    assert np.allclose(rec.flux, 33.0)
    assert np.allclose(rec.ivar, 0.25)


def test_read_6df_prefers_explicit_wave_row(tmp_path: Path) -> None:
    from data_lake.ingest.fits_to_spectra_zarr import _read_6df_spectrum

    p = tmp_path / "g0001.fits"
    _write_minimal_6df(p, n_pix=16)
    with fits.open(p, memmap=True) as hdul:
        records, _ = _read_6df_spectrum(hdul, p)

    wave = records[0].wavelength
    assert wave.shape == (16,)
    assert wave[0] == pytest.approx(5000.0)
    assert wave[-1] == pytest.approx(5000.0 + 2.0 * 15.0)


def test_read_6df_uses_target_obsid_v_obsid_r(tmp_path: Path) -> None:
    from data_lake.ingest.fits_to_spectra_zarr import _read_6df_spectrum

    p = tmp_path / "wrong_name.fits"
    _write_minimal_6df(
        p,
        vr_observations=[
            {
                "target": "g2259418-254505",
                "obsid_v": "V-00030",
                "obsid_r": "R-00030",
            },
        ],
    )

    with fits.open(p, memmap=True) as hdul:
        records, _ = _read_6df_spectrum(hdul, p)

    assert records[0].source_id == _expected_6df_source_id(
        "g2259418-254505", "V-00030", "R-00030",
    )


def test_read_6df_multi_vr_same_target(tmp_path: Path) -> None:
    from data_lake.ingest.fits_to_spectra_zarr import _read_6df_spectrum

    p = tmp_path / "g2302140-251235.fits"
    _write_minimal_6df(
        p,
        n_pix=32,
        vr_observations=[
            {
                "target": "g2302140-251235",
                "obsid_v": "V-00023",
                "obsid_r": "R-00023",
                "flux_scale": 33.0,
                "z": 0.032423,
            },
            {
                "target": "g2302140-251235",
                "obsid_v": "V-00024",
                "obsid_r": "R-00024",
                "flux_scale": 44.0,
                "z": 0.03237,
            },
        ],
    )

    with fits.open(p, memmap=True) as hdul:
        records, _ = _read_6df_spectrum(hdul, p)

    assert len(records) == 2
    sids = {r.source_id for r in records}
    expected = {
        _expected_6df_source_id("g2302140-251235", "V-00023", "R-00023"),
        _expected_6df_source_id("g2302140-251235", "V-00024", "R-00024"),
    }
    assert sids == expected
    by_flux = {float(r.flux[0]): r for r in records}
    assert by_flux[33.0].meta["z"] == pytest.approx(0.032423)
    assert by_flux[44.0].meta["z"] == pytest.approx(0.03237)


def test_read_6df_multi_vr_same_target_disambiguated_by_obsid(tmp_path: Path) -> None:
    """Two observations of the same target are uniquely keyed by OBSID_V/OBSID_R."""
    from data_lake.ingest.fits_to_spectra_zarr import _read_6df_spectrum

    target = "g1437140-385507"
    p = tmp_path / f"{target}.fits"
    _write_minimal_6df(
        p,
        n_pix=32,
        vr_observations=[
            {
                "target": target,
                "obsid_v": "V-00001",
                "obsid_r": "R-00001",
                "flux_scale": 33.0,
                "z": 0.0515,
            },
            {
                "target": target,
                "obsid_v": "V-00002",
                "obsid_r": "R-00002",
                "flux_scale": 44.0,
                "z": 1.83636,
            },
        ],
    )

    with fits.open(p, memmap=True) as hdul:
        records, _ = _read_6df_spectrum(hdul, p)

    assert len(records) == 2
    sids = {r.source_id for r in records}
    expected = {
        _expected_6df_source_id(target, "V-00001", "R-00001"),
        _expected_6df_source_id(target, "V-00002", "R-00002"),
    }
    assert sids == expected


def test_read_6df_missing_obsid_v_raises(tmp_path: Path) -> None:
    """Missing OBSID_V must raise ValueError (strict key enforcement)."""
    from data_lake.ingest.fits_to_spectra_zarr import _read_6df_spectrum

    p = tmp_path / "g0001.fits"
    _write_minimal_6df(
        p,
        vr_observations=[{"target": "g0001", "obsid_v": "V-00001", "obsid_r": "R-00001"}],
    )
    # Remove OBSID_V from VR HDU
    with fits.open(p, mode="update") as hdul:
        for hdu in hdul:
            if "OBSID_V" in hdu.header:
                del hdu.header["OBSID_V"]

    with fits.open(p, memmap=True) as hdul:
        with pytest.raises(ValueError, match="OBSID_V"):
            _read_6df_spectrum(hdul, p)


def test_read_6df_missing_obsid_r_raises(tmp_path: Path) -> None:
    """Missing OBSID_R must raise ValueError (strict key enforcement)."""
    from data_lake.ingest.fits_to_spectra_zarr import _read_6df_spectrum

    p = tmp_path / "g0001.fits"
    _write_minimal_6df(
        p,
        vr_observations=[{"target": "g0001", "obsid_v": "V-00001", "obsid_r": "R-00001"}],
    )
    with fits.open(p, mode="update") as hdul:
        for hdu in hdul:
            if "OBSID_R" in hdu.header:
                del hdu.header["OBSID_R"]

    with fits.open(p, memmap=True) as hdul:
        with pytest.raises(ValueError, match="OBSID_R"):
            _read_6df_spectrum(hdul, p)


def test_read_6df_missing_target_raises(tmp_path: Path) -> None:
    """Missing TARGET must raise ValueError (strict key enforcement)."""
    from data_lake.ingest.fits_to_spectra_zarr import _read_6df_spectrum

    p = tmp_path / "g0001.fits"
    _write_minimal_6df(
        p,
        vr_observations=[{"target": "g0001", "obsid_v": "V-00001", "obsid_r": "R-00001"}],
    )
    with fits.open(p, mode="update") as hdul:
        for hdu in hdul:
            if "TARGET" in hdu.header:
                del hdu.header["TARGET"]

    with fits.open(p, memmap=True) as hdul:
        with pytest.raises(ValueError, match="TARGET"):
            _read_6df_spectrum(hdul, p)


def test_6df_composite_catalog_source_ids(tmp_path: Path) -> None:
    import pyarrow as pa
    from data_lake.ingest.fits_to_parquet import (
        LAKE_JOIN_ID_COLUMN,
        ensure_catalog_source_ids,
    )

    table = pa.table({
        "targetname": pa.array(["g2302140-251235", "g2302140-251235"]),
        "obsid_v": pa.array(["V-00023", "V-00024"]),
        "obsid_r": pa.array(["R-00023", "R-00024"]),
        "ra": pa.array([345.55, 345.55]),
        "dec": pa.array([-25.21, -25.21]),
    })
    out, mode = ensure_catalog_source_ids(table, "targetname,obsid_v,obsid_r")
    assert mode == "composite:targetname,obsid_v,obsid_r"
    sids = out[LAKE_JOIN_ID_COLUMN].to_pylist()
    assert sids[0] == _expected_6df_source_id("g2302140-251235", "V-00023", "R-00023")
    assert sids[1] == _expected_6df_source_id("g2302140-251235", "V-00024", "R-00024")


def test_ingest_6df_end_to_end(tmp_path: Path) -> None:
    from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
    from data_lake.io.spectra import SpectrumAccessor

    p = tmp_path / "g00123.fits"
    _write_minimal_6df(p, vr_observations=[{"obsid_v": "V-00001", "obsid_r": "R-00001"}])

    lake = tmp_path / "lake"
    index_map = ingest_spectra_from_fits(
        p,
        lake,
        "SIXDF_DR3",
        fmt="6df",
        norder=5,
        on_duplicate_source_id="skip",
    )
    sid = _expected_6df_source_id("g00123", "V-00001", "R-00001")
    assert sid in index_map

    acc = SpectrumAccessor(lake, "SIXDF_DR3")
    sp = acc.get_spectrum(sid)
    assert sp is not None
    assert sp.flux.shape[0] == 1024
    assert np.allclose(sp.flux, 33.0)
