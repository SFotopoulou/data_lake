"""Tests for schema_manifest.json and dl-describe-survey helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from astropy.table import Table

from data_lake.ingest.fits_to_parquet import ingest_catalog
from data_lake.ingest.fits_to_spectra_zarr import _META_DTYPE
from data_lake.schema_registry import (
    MANIFEST_FILENAME,
    MODALITY_SPECTRA,
    ROLE_FLUX,
    ROLE_ID,
    ROLE_METADATA,
    ROLE_PHOTOMETRY,
    ROLE_REDSHIFT,
    ROLE_SKY,
    SPECTRUM_SKY_META_FIELDS,
    build_catalog_schema_manifest,
    build_spectra_schema_manifest,
    format_manifest_table,
    infer_column_role,
    load_catalog_schema_manifest,
    load_column_overlay,
    merge_column_overlays,
    write_catalog_schema_manifest,
    write_spectra_schema_manifest,
)


def _write_fits(tbl: Table, path: Path) -> None:
    tbl.write(path, overwrite=True)


def test_infer_column_roles() -> None:
    assert infer_column_role(
        "TARGETID",
        link_id_column="TARGETID",
        ra_column="TARGET_RA",
        dec_column="TARGET_DEC",
        redshift_column="Z",
    ) == ROLE_ID
    assert infer_column_role(
        "TARGET_RA",
        link_id_column="TARGETID",
        ra_column="TARGET_RA",
        dec_column="TARGET_DEC",
        redshift_column="Z",
    ) == ROLE_SKY
    assert infer_column_role(
        "Z",
        link_id_column="TARGETID",
        ra_column="TARGET_RA",
        dec_column="TARGET_DEC",
        redshift_column="Z",
    ) == ROLE_REDSHIFT
    assert infer_column_role(
        "MAG_G",
        link_id_column="TARGETID",
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
        link_id_col="TARGETID",
        tile_mode="overwrite",
    )

    manifest_path = lake / "catalogs" / "manifest_test" / MANIFEST_FILENAME
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["survey"] == "manifest_test"
    assert manifest["link_id_column"] == "_source_id"
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
        link_id_col="TARGETID",
        tile_mode="overwrite",
    )
    root = lake / "catalogs" / "fmt_test"
    manifest = build_catalog_schema_manifest(
        root,
        "fmt_test",
        hats_order=5,
        ra_column="TARGET_RA",
        dec_column="TARGET_DEC",
        link_id_mode="column:TARGETID",
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
        link_id_mode="column:TARGETID",
    )
    loaded = load_catalog_schema_manifest(root)
    assert loaded["n_columns"] == manifest["n_columns"]


def test_load_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_catalog_schema_manifest(tmp_path / "nope")


def test_spectra_schema_manifest(tmp_path: Path) -> None:
    spectra_root = tmp_path / "lake" / "spectra" / "DESI_TEST"
    spectra_root.mkdir(parents=True)
    info = {
        "survey_name": "DESI_TEST",
        "hats_order": 5,
        "n_pix": 100,
        "wavelength_mode": "shared",
        "meta_fields": ["z", "z_err"],
        "has_resolution": False,
    }
    (spectra_root / "spectrum_info.json").write_text(json.dumps(info))

    manifest = build_spectra_schema_manifest(spectra_root, "DESI_TEST")
    assert manifest["modality"] == MODALITY_SPECTRA
    assert manifest["n_pix"] == 100
    names = [c["name"] for c in manifest["columns"]]
    assert "flux" in names
    assert "meta.z" in names

    write_spectra_schema_manifest(spectra_root, "DESI_TEST")
    assert (spectra_root / MANIFEST_FILENAME).is_file()

    text = format_manifest_table(manifest, role=ROLE_FLUX)
    assert "flux" in text


def test_spectra_schema_manifest_sky_meta_fields(tmp_path: Path) -> None:
    spectra_root = tmp_path / "lake" / "spectra" / "VVDS_TEST"
    spectra_root.mkdir(parents=True)
    info = {
        "survey_name": "VVDS_TEST",
        "hats_order": 5,
        "n_pix": 557,
        "wavelength_mode": "shared",
        "meta_fields": list(_META_DTYPE.names),
        "has_resolution": False,
    }
    (spectra_root / "spectrum_info.json").write_text(json.dumps(info))

    manifest = build_spectra_schema_manifest(spectra_root, "VVDS_TEST")
    assert manifest["has_spectrum_sky_meta"] is True
    assert set(manifest["spectrum_sky_meta_fields"]) == set(SPECTRUM_SKY_META_FIELDS)
    assert "meta.sky" in manifest["column_groups"]

    by_name = {c["name"]: c for c in manifest["columns"]}
    assert by_name["meta.ra"]["role"] == ROLE_SKY
    assert by_name["meta.ra_key"]["role"] == ROLE_METADATA
    assert "FITS" in by_name["meta.ra_key"]["description"]
    assert by_name["meta.ra_key"]["dtype"] == "ascii[32]"

    text = format_manifest_table(manifest)
    assert "spectrum_sky_meta" in text
    assert "ra_key" in text

    sky_text = format_manifest_table(manifest, role=ROLE_SKY)
    assert "meta.ra" in sky_text
    assert "meta.dec" in sky_text

    meta_text = format_manifest_table(manifest, role=ROLE_METADATA)
    assert "meta.ra_key" in meta_text
    assert "meta.source_file" in meta_text


class TestCatalogResolver:
    """Tests for resolve_catalog_root / catalog_write_root."""

    def test_resolve_prefers_catalogs_when_both_exist(self, tmp_path: Path) -> None:
        from data_lake.schema_registry import resolve_catalog_root

        (tmp_path / "catalogs" / "SURVEY").mkdir(parents=True)
        (tmp_path / "products" / "SURVEY").mkdir(parents=True)
        result = resolve_catalog_root(tmp_path, "SURVEY")
        assert result == tmp_path / "catalogs" / "SURVEY"

    def test_resolve_falls_back_to_products(self, tmp_path: Path) -> None:
        from data_lake.schema_registry import resolve_catalog_root

        (tmp_path / "products" / "MY_PROD").mkdir(parents=True)
        result = resolve_catalog_root(tmp_path, "MY_PROD")
        assert result == tmp_path / "products" / "MY_PROD"

    def test_resolve_returns_products_path_when_nothing_exists(self, tmp_path: Path) -> None:
        from data_lake.schema_registry import resolve_catalog_root

        result = resolve_catalog_root(tmp_path, "NONEXISTENT")
        assert result == tmp_path / "products" / "NONEXISTENT"

    def test_catalog_write_root_returns_products(self, tmp_path: Path) -> None:
        from data_lake.schema_registry import catalog_write_root

        result = catalog_write_root(tmp_path, "MY_PROD")
        assert result == tmp_path / "products" / "MY_PROD"

    def test_catalog_write_root_raises_when_ingested_exists(self, tmp_path: Path) -> None:
        from data_lake.schema_registry import catalog_write_root

        (tmp_path / "catalogs" / "CONFLICT").mkdir(parents=True)
        with pytest.raises(FileExistsError, match="ingested survey already exists"):
            catalog_write_root(tmp_path, "CONFLICT")


def test_column_overlay_merge(tmp_path: Path) -> None:
    overlay_dir = tmp_path / "lake" / "shared" / "registry" / "overlays"
    overlay_dir.mkdir(parents=True)
    (overlay_dir / "WISE.catalog.json").write_text(
        json.dumps({"columns": {"MAG_W1": {"unit": "mag", "description": "W1 Vega"}}})
    )
    manifest = {
        "survey": "WISE",
        "modality": "catalog",
        "columns": [{"name": "MAG_W1", "dtype": "float64", "role": "photometry"}],
    }
    loaded = load_column_overlay(tmp_path / "lake", "WISE", "catalog")
    merged = merge_column_overlays(manifest, loaded)
    col = merged["columns"][0]
    assert col["unit"] == "mag"
    assert col["description"] == "W1 Vega"
    assert merged["has_column_overlay"]
