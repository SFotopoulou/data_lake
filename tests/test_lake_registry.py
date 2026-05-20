"""Tests for lake registry and master association metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.table import Table

from data_lake.ingest.fits_to_parquet import ingest_catalog
from data_lake.lake_registry import (
    REGISTRY_FILENAME,
    build_lake_registry_table,
    guess_master_meta,
    load_lake_registry,
    load_master_meta,
    master_meta_path,
    refresh_lake_registry,
    registry_path,
    write_master_meta,
)


def _ingest_mini_catalog(
    tmp_path: Path,
    survey: str,
    *,
    ra: float = 10.0,
    dec: float = 20.0,
    id_offset: int = 0,
) -> None:
    n = 6
    tbl = Table({
        "TARGETID": np.arange(id_offset, id_offset + n, dtype=np.int64),
        "TARGET_RA": np.full(n, ra),
        "TARGET_DEC": np.full(n, dec),
        "Z": np.linspace(0.1, 0.3, n),
        "MAG_G": np.linspace(18.0, 19.0, n),
    })
    fits = tmp_path / f"{survey}.fits"
    tbl.write(fits, overwrite=True)
    lake = tmp_path / "lake"
    ingest_catalog(
        source_path=fits,
        output_root=lake,
        survey_name=survey,
        ra_col="TARGET_RA",
        dec_col="TARGET_DEC",
        source_id_col="TARGETID",
        norder=5,
        overwrite=True,
    )


def test_refresh_lake_registry(tmp_path: Path) -> None:
    _ingest_mini_catalog(tmp_path, "SURV_A", id_offset=0)
    _ingest_mini_catalog(tmp_path, "SURV_B", id_offset=100, ra=50.0, dec=60.0)
    lake = tmp_path / "lake"

    out = refresh_lake_registry(lake)
    assert out == registry_path(lake)
    table = load_lake_registry(lake)
    names = set(table.column("survey").to_pylist())
    assert "SURV_A" in names
    assert "SURV_B" in names


def test_master_meta_guess_and_write(tmp_path: Path) -> None:
    _ingest_mini_catalog(tmp_path, "DESI_DR1", id_offset=0)
    _ingest_mini_catalog(tmp_path, "EUCLID_DR1", id_offset=1000, ra=50.0)
    lake = tmp_path / "lake"

    master = pa.table({
        "desi_targetid": pa.array([1, 2], type=pa.int64()),
        "euclid_source_id": pa.array([1000, 1001], type=pa.int64()),
        "sep_arcsec": pa.array([0.1, 0.2], type=pa.float32()),
    })
    master_path = lake / "associations" / "master.parquet"
    master_path.parent.mkdir(parents=True)
    pq.write_table(master, str(master_path))

    meta = guess_master_meta(master_path, lake)
    assert len(meta["partners"]) >= 1
    surveys = {p["survey"] for p in meta["partners"]}
    assert "DESI_DR1" in surveys or "EUCLID_DR1" in surveys

    write_master_meta(master_path, meta)
    assert master_meta_path(master_path).is_file()
    loaded = load_master_meta(master_path, lake, allow_guess=False)
    assert loaded["partners"][0]["master_column"]


def test_build_registry_table_empty_lake(tmp_path: Path) -> None:
    lake = tmp_path / "empty_lake"
    (lake / "catalogs").mkdir(parents=True)
    table = build_lake_registry_table(lake)
    assert table.num_rows == 0
