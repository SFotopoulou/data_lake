"""Tests for cutout Zarr ingest (fits_to_zarr)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from data_lake.ingest.fits_to_zarr import ingest_cutouts_from_fits
from data_lake.ingest.validate_cutout_ingest import run_validation


def _write_cutout_fits(path: Path, *, sid: int, ra: float, dec: float) -> None:
    data = np.random.RandomState(sid).rand(8, 8).astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    hdr = hdu.header
    hdr["RA"] = ra
    hdr["DEC"] = dec
    hdr["OBJ_ID"] = sid
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = ra
    hdr["CRVAL2"] = dec
    hdr["CRPIX1"] = 4.0
    hdr["CRPIX2"] = 4.0
    hdr["CD1_1"] = -1.0 / 3600.0
    hdr["CD1_2"] = 0.0
    hdr["CD2_1"] = 0.0
    hdr["CD2_2"] = 1.0 / 3600.0
    hdu.writeto(path, overwrite=True)


def test_ingest_cutouts_writes_zarr_and_passes_validation(tmp_path: Path) -> None:
    f1 = tmp_path / "a.fits"
    f2 = tmp_path / "b.fits"
    _write_cutout_fits(f1, sid=1001, ra=10.0, dec=20.0)
    _write_cutout_fits(f2, sid=1002, ra=10.01, dec=20.01)

    m1 = ingest_cutouts_from_fits(
        f1, tmp_path, "test_survey", ra_col="RA", dec_col="DEC",
        norder=3,
    )
    m2 = ingest_cutouts_from_fits(
        f2, tmp_path, "test_survey", ra_col="RA", dec_col="DEC",
        norder=3,
    )
    assert m1[1001][1] == 0   # local_index
    assert m2[1002][1] == 1   # local_index (second file appended to same tile)

    info = json.loads((tmp_path / "cutouts" / "test_survey" / "cutout_info.json").read_text())
    assert info["dtype"] == "float32"
    assert info["on_duplicate_source_id"] == "skip"

    rep = run_validation(tmp_path, "test_survey")
    assert rep.ok(strict=False), (rep.errors, rep.warnings)


def test_on_duplicate_error(tmp_path: Path) -> None:
    f = tmp_path / "dup.fits"
    _write_cutout_fits(f, sid=2001, ra=15.0, dec=25.0)
    ingest_cutouts_from_fits(
        f, tmp_path, "dup_survey", ra_col="RA", dec_col="DEC",
        norder=3, on_duplicate_source_id="append",
    )
    with pytest.raises(ValueError, match="already exists"):
        ingest_cutouts_from_fits(
            f, tmp_path, "dup_survey", ra_col="RA", dec_col="DEC",
            norder=3, on_duplicate_source_id="error",
        )


def test_link_id_col_targetid(tmp_path: Path) -> None:
    """--link-id-col reads TARGETID (DESI-style) for catalog linkage."""
    f = tmp_path / "desi_like.fits"
    data = np.ones((16, 16), dtype=np.float32)
    hdu = fits.PrimaryHDU(data)
    hdr = hdu.header
    tid = 9876543210123456
    hdr["TARGETID"] = tid
    hdr["TARGET_RA"] = 150.0
    hdr["TARGET_DEC"] = 2.5
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = 150.0
    hdr["CRVAL2"] = 2.5
    hdr["CRPIX1"] = 8.0
    hdr["CRPIX2"] = 8.0
    hdr["CD1_1"] = -1.0 / 3600.0
    hdr["CD1_2"] = 0.0
    hdr["CD2_1"] = 0.0
    hdr["CD2_2"] = 1.0 / 3600.0
    hdu.writeto(f, overwrite=True)

    m = ingest_cutouts_from_fits(
        f, tmp_path, "tid_survey",
        ra_col="TARGET_RA", dec_col="TARGET_DEC",
        link_id_col="TARGETID",
        norder=3,
    )
    assert m[tid][1] == 0   # local_index


def test_link_id_col_missing_raises(tmp_path: Path) -> None:
    f = tmp_path / "noid.fits"
    _write_cutout_fits(f, sid=1, ra=10.0, dec=20.0)
    with pytest.raises(KeyError, match="TARGETID"):
        ingest_cutouts_from_fits(
            f, tmp_path, "x",
            link_id_col="TARGETID",
            norder=3,
        )


def test_on_duplicate_skip(tmp_path: Path) -> None:
    f = tmp_path / "skip.fits"
    _write_cutout_fits(f, sid=3001, ra=50.0, dec=12.0)
    ingest_cutouts_from_fits(
        f, tmp_path, "skip_survey", ra_col="RA", dec_col="DEC",
        norder=3, on_duplicate_source_id="append",
    )
    m = ingest_cutouts_from_fits(
        f, tmp_path, "skip_survey", ra_col="RA", dec_col="DEC",
        norder=3, on_duplicate_source_id="skip",
    )
    assert m == {}
