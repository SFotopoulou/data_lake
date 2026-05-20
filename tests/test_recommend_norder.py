"""Tests for pre-ingest HEALPix norder recommendation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

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
