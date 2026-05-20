"""Tests for schema_manifest.json and dl-describe-survey helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from astropy.table import Table

from data_lake.ingest.fits_to_parquet import ingest_catalog
from data_lake.schema_registry import (
    MANIFEST_FILENAME,
    ROLE_ID,
    ROLE_PHOTOMETRY,
    ROLE_REDSHIFT,
    ROLE_SKY,
    build_catalog_schema_manifest,
    format_manifest_table,
    infer_column_role,
    load_catalog_schema_manifest,
    write_catalog_schema_manifest,
)


def _write_fits(tbl: Table, path: Path) -> None:
    tbl.write(path, overwrite=True)


def test_infer_column_roles() -> None:
    assert infer_column_role(
        "TARGETID",
        source_id_column="TARGETID",
        ra_column="TARGET_RA",
        dec_column="TARGET_DEC",
        redshift_column="Z",
    ) == ROLE_ID
    assert infer_column_role(
        "TARGET_RA",
        source_id_column="TARGETID",
        ra_column="TARGET_RA",
        dec_column="TARGET_DEC",
        redshift_column="Z",
    ) == ROLE_SKY
    assert infer_column_role(
        "Z",
        source_id_column="TARGETID",
        ra_column="TARGET_RA",
        dec_column="TARGET_DEC",
        redshift_column="Z",
    ) == ROLE_REDSHIFT
    assert infer_column_role(
        "MAG_G",
        source_id_column="TARGETID",
        ra_column="TARGET_RA",
        dec_column="TARGET_DEC",
        redshift_column="Z",
    ) == ROLE_PHOTOMETRY


def test_ingest_writes_schema_manifest(tmp_path: Path) -> None:
    import numpy as np

    n = 12
    tbl = Table({
        "TARGETID": np.arange(n, dtype=np.int64),
        "TARGET_RA": np.full(n, 120.0),
        "TARGET_DEC": np.full(n, 45.0),
        "Z": np.linspace(0.1, 0.5, n),
        "MAG_G": np.linspace(18.0, 20.0, n),
    })
    fits_path = tmp_path / "src.fits"
    _write_fits(tbl, fits_path)
    lake = tmp_path / "lake"

    ingest_catalog(
        source_path=fits_path,
        output_root=lake,
        survey_name="manifest_test",
        ra_col="TARGET_RA",
        dec_col="TARGET_DEC",
        norder=5,
        source_id_col="TARGETID",
        overwrite=True,
    )

    manifest_path = lake / "catalogs" / "manifest_test" / MANIFEST_FILENAME
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["survey"] == "manifest_test"
    assert manifest["source_id_column"] == "TARGETID"
    assert manifest["redshift_column"] == "Z"
    assert manifest["n_columns"] >= 7  # includes healpix + indices
    names = {c["name"] for c in manifest["columns"]}
    assert "MAG_G" in names
    assert "column_groups" in manifest


def test_build_and_format_manifest(tmp_path: Path) -> None:
    import numpy as np

    n = 8
    tbl = Table({
        "TARGETID": np.arange(n, dtype=np.int64),
        "TARGET_RA": np.full(n, 10.0),
        "TARGET_DEC": np.full(n, 20.0),
        "Z": np.zeros(n),
        "MAG_R": np.ones(n) * 19.0,
    })
    fits_path = tmp_path / "t.fits"
    _write_fits(tbl, fits_path)
    lake = tmp_path / "lake"
    ingest_catalog(
        source_path=fits_path,
        output_root=lake,
        survey_name="fmt_test",
        ra_col="TARGET_RA",
        dec_col="TARGET_DEC",
        source_id_col="TARGETID",
        overwrite=True,
    )
    root = lake / "catalogs" / "fmt_test"
    manifest = build_catalog_schema_manifest(
        root,
        "fmt_test",
        hats_order=5,
        ra_column="TARGET_RA",
        dec_column="TARGET_DEC",
        source_id_mode="column:TARGETID",
    )
    text = format_manifest_table(manifest, role=ROLE_PHOTOMETRY)
    assert "MAG_R" in text
    assert "photometry" in text

    write_catalog_schema_manifest(
        root,
        "fmt_test",
        hats_order=5,
        ra_column="TARGET_RA",
        dec_column="TARGET_DEC",
        source_id_mode="column:TARGETID",
    )
    loaded = load_catalog_schema_manifest(root)
    assert loaded["n_columns"] == manifest["n_columns"]


def test_load_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_catalog_schema_manifest(tmp_path / "nope")
