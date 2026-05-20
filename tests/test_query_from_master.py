"""Tests for want → master → catalog SQL generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table

from data_lake.ingest.fits_to_parquet import ingest_catalog
from data_lake.lake_registry import write_master_meta
from data_lake.query_from_master import (
    build_select_from_master,
    parse_column_picks,
    survey_sql_alias,
)


def _ingest(tmp_path: Path, survey: str, id_offset: int) -> None:
    n = 5
    tbl = Table({
        "TARGETID": np.arange(id_offset, id_offset + n, dtype=np.int64),
        "TARGET_RA": np.full(n, 10.0 + id_offset),
        "TARGET_DEC": np.full(n, 20.0),
        "Z": np.linspace(0.1, 0.3, n),
        "MAG_G": np.linspace(18.0, 19.0, n),
    })
    fits = tmp_path / f"{survey}.fits"
    tbl.write(fits, overwrite=True)
    ingest_catalog(
        source_path=fits,
        output_root=tmp_path / "lake",
        survey_name=survey,
        ra_col="TARGET_RA",
        dec_col="TARGET_DEC",
        source_id_col="TARGETID",
        norder=5,
        overwrite=True,
    )


def test_parse_column_picks() -> None:
    assert parse_column_picks(["DESI:Z,MAG_G", "EUCLID.SOURCE_ID"]) == {
        "DESI": ["Z", "MAG_G"],
        "EUCLID": ["SOURCE_ID"],
    }


def test_build_select_from_master_sql(tmp_path: Path) -> None:
    _ingest(tmp_path, "DESI_DR1", 0)
    _ingest(tmp_path, "EUCLID_DR1", 1000)
    lake = tmp_path / "lake"

    master_path = lake / "associations" / "master.parquet"
    master_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({
            "desi_targetid": pa.array([1, 2], type=pa.int64()),
            "euclid_source_id": pa.array([1000, 1001], type=pa.int64()),
            "sep_arcsec": pa.array([0.1, 0.2], type=pa.float32()),
        }),
        str(master_path),
    )

    meta = {
        "meta_version": "1",
        "primary_survey": "DESI_DR1",
        "partners": [
            {
                "survey": "DESI_DR1",
                "master_column": "desi_targetid",
                "catalog_id_column": "TARGETID",
            },
            {
                "survey": "EUCLID_DR1",
                "master_column": "euclid_source_id",
                "catalog_id_column": "TARGETID",
            },
        ],
    }
    write_master_meta(master_path, meta)

    plan = build_select_from_master(
        lake,
        master_path,
        "DESI_DR1",
        {
            "DESI_DR1": ["Z", "MAG_G"],
            "EUCLID_DR1": ["TARGETID"],
        },
        include_views=True,
    )

    assert "INNER JOIN master" in plan.sql
    assert "desi_targetid" in plan.sql
    assert survey_sql_alias("DESI_DR1") in plan.sql
    assert "CREATE OR REPLACE VIEW DESI_DR1" in plan.all_sql()
    assert "CREATE OR REPLACE VIEW master" in plan.all_sql()
    assert plan.aliases["DESI_DR1"] == "desi_dr1"

    # Execute in DuckDB with registered want table
    import duckdb

    con = duckdb.connect()
    for ddl in plan.view_ddls:
        con.execute(ddl)
    con.execute("CREATE TEMP TABLE want (id BIGINT)")
    con.execute("INSERT INTO want VALUES (1), (2)")
    rows = con.execute(plan.sql).fetchall()
    assert len(rows) == 2
    con.close()
