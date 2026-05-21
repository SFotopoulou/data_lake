"""Tests for pre-ingest HEALPix norder recommendation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from data_lake.ingest.fits_to_parquet import read_catalog_sky_columns
from data_lake.ingest.recommend_norder import (
    format_recommendation_report,
    recommend_catalog_norder,
)


def _write_fits(tbl: Table, path: Path) -> None:
    tbl.write(path, overwrite=True)


def test_recommend_dense_catalog(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n = 20_000
    ra = rng.uniform(120.0, 120.5, n)
    dec = rng.uniform(44.0, 44.5, n)
    tbl = Table({"RA": ra, "DEC": dec, "ID": np.arange(n, dtype=np.int64)})
    fits_path = tmp_path / "dense.fits"
    _write_fits(tbl, fits_path)

    rec = recommend_catalog_norder(
        [fits_path],
        ra_col="RA",
        dec_col="DEC",
        sample_rows=n,
        norder_min=3,
        norder_max=7,
        seed=1,
    )
    assert 3 <= rec.recommended <= 7
    cand = next(c for c in rec.candidates if c.norder == rec.recommended)
    assert 1_000 <= cand.est_rows_per_tile <= 200_000
    assert rec.total_rows == n
    assert rec.sample_rows == n
    assert "Recommended --norder:" in format_recommendation_report(rec)


def test_sparse_allsky_lower_norder_than_dense_patch(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n_sparse = 5_000
    ra_s = rng.uniform(0.0, 360.0, n_sparse)
    dec_s = np.degrees(np.arcsin(rng.uniform(-1, 1, n_sparse)))
    _write_fits(Table({"ra": ra_s, "dec": dec_s}), tmp_path / "sparse.fits")

    n_dense = 20_000
    ra_d = rng.uniform(10.0, 10.3, n_dense)
    dec_d = rng.uniform(20.0, 20.3, n_dense)
    _write_fits(Table({"RA": ra_d, "DEC": dec_d}), tmp_path / "dense.fits")

    sparse = recommend_catalog_norder(
        [tmp_path / "sparse.fits"],
        sample_rows=n_sparse,
        norder_min=3,
        norder_max=7,
        seed=2,
    )
    dense = recommend_catalog_norder(
        [tmp_path / "dense.fits"],
        ra_col="RA",
        dec_col="DEC",
        sample_rows=n_dense,
        norder_min=3,
        norder_max=7,
        seed=1,
    )
    assert sparse.recommended <= dense.recommended


def test_no_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        recommend_catalog_norder([tmp_path / "missing.fits"])


def test_bintable_extension_and_merged_sky_columns(tmp_path: Path) -> None:
    """EUCLID-style: sky columns on extension 1 BINTABLE, not PRIMARY."""
    from astropy.io import fits

    n = 2_000
    rng = np.random.default_rng(0)
    ra = rng.uniform(50.0, 51.0, n)
    dec = rng.uniform(-10.0, -9.0, n)
    cols = fits.ColDefs([
        fits.Column(name="alpha_j2000_merged", format="D", array=ra),
        fits.Column(name="delta_j2000_merged", format="D", array=dec),
    ])
    tbl_hdu = fits.BinTableHDU.from_columns(cols)
    hdul = fits.HDUList([fits.PrimaryHDU(), tbl_hdu])
    path = tmp_path / "euclid_like.fits"
    hdul.writeto(path, overwrite=True)

    rec = recommend_catalog_norder(
        [path],
        ra_col="alpha_j2000_merged",
        dec_col="delta_j2000_merged",
        sample_rows=n,
        norder_min=3,
        norder_max=7,
    )
    assert rec.sample_rows == n
    assert rec.ra_col == "alpha_j2000_merged"


def test_read_catalog_sky_columns_matches_ingest_reader(tmp_path: Path) -> None:
    """Sky reader uses the same FITS Table path as catalog ingest."""
    from data_lake.ingest.fits_to_parquet import _read_fits_catalog_table

    n = 500
    rng = np.random.default_rng(1)
    ra = rng.uniform(50.0, 51.0, n)
    dec = rng.uniform(-10.0, -9.0, n)
    path = tmp_path / "sky.fits"
    Table({
        "alpha_j2000_merged": ra,
        "delta_j2000_merged": dec,
    }).write(path, overwrite=True)

    tbl = _read_fits_catalog_table(path)
    ra_ingest = np.asarray(tbl["alpha_j2000_merged"], dtype=np.float64)
    ra_read, dec_read = read_catalog_sky_columns(
        path, "alpha_j2000_merged", "delta_j2000_merged"
    )
    assert np.array_equal(ra_read, ra_ingest)
    assert dec_read.shape == (n,)
