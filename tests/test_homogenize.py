"""Tests for survey homogenization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_lake.discovery.areas import make_area, save_area
from data_lake.discovery.region import Region
from data_lake.discovery import tile_index as ti
from data_lake.discovery.selection import selection_from_region
from data_lake.homogenize.engine import homogenize_catalog
from data_lake.homogenize.registry import load_transform, validate_transform_schema
from data_lake.homogenize.transforms import (
    apply_rules_to_frame,
    build_homogenized_view_sql,
    resolve_applicable_rules,
)
from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, assign_healpix, healpix_dir
from data_lake.schema_registry import PRODUCT_SUBTYPE_HOMOGENIZED


def _write_wise_tile(lake: Path, norder: int, npix: int, w1: float) -> None:
    tile_dir = lake / "catalogs" / "ALLWISE" / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    hp = f"_healpix_norder{norder}"
    pq.write_table(
        pa.table({
            LAKE_JOIN_ID_COLUMN: pa.array([1, 2], type=pa.int64()),
            "ra": pa.array([120.0, 120.001], type=pa.float64()),
            "dec": pa.array([45.0, 45.001], type=pa.float64()),
            hp: pa.array([npix, npix], type=pa.int64()),
            "w1mpro": pa.array([w1, w1 + 0.1], type=pa.float32()),
            "w1sigmpro": pa.array([0.05, 0.06], type=pa.float32()),
        }),
        tile_dir / f"Npix={npix}.parquet",
    )
    (lake / "catalogs" / "ALLWISE" / "catalog_info.json").write_text(json.dumps({
        "hats_order": norder,
        "ra_column": "ra",
        "dec_column": "dec",
        "link_id_column": LAKE_JOIN_ID_COLUMN,
        "link_id_mode": "column:_source_id",
        "total_rows": 2,
        "total_columns": 6,
    }))


class TestTransformRegistry:
    def test_load_phot_ab_v1(self) -> None:
        t = load_transform(None, "phot_ab_v1")
        assert t["transform_id"] == "phot_ab_v1"
        assert validate_transform_schema(t) == []


class TestTransforms:
    def test_wise_mag_offset(self) -> None:
        import polars as pl

        from data_lake.homogenize.survey_registry import resolve_catalog_rules
        from data_lake.homogenize.transforms import TransformRule

        rules = resolve_catalog_rules(None, "ALLWISE", "phot_ab_v1")
        assert len(rules) >= 1
        rule = next(r for r in rules if r.target_column == "phot_ab_w1")
        df = pl.DataFrame({"w1mpro": [10.0], "w1sigmpro": [0.05]})
        out, lin = apply_rules_to_frame(df, [rule])
        assert out["phot_ab_w1"][0] == pytest.approx(12.699)
        assert out["phot_ab_w1_err"][0] == pytest.approx(0.05)

    def test_unwise_flux_to_ab(self) -> None:
        import polars as pl

        from data_lake.homogenize.survey_registry import resolve_catalog_rules

        rules = resolve_catalog_rules(None, "UNWISE_W1", "phot_ab_v1")
        rule = next(r for r in rules if r.target_column == "phot_ab_w1")
        df = pl.DataFrame({"flux": [1.0, 0.0, -1.0], "dflux": [0.1, 0.1, 0.1]})
        out, _ = apply_rules_to_frame(df, [rule])
        assert out["phot_ab_w1"][0] == pytest.approx(8.906)
        assert out["phot_ab_w1_err"][0] == pytest.approx(0.1085736, rel=1e-4)
        assert out["phot_ab_w1"][1] is None
        assert out["phot_ab_w1"][2] is None

    def test_identity(self) -> None:
        import polars as pl

        rule = resolve_applicable_rules(
            {
                "transform_id": "phot_ab_v1",
                "rules": [
                    {
                        "survey": "TEST",
                        "source_column": "mag_r",
                        "target_column": "phot_ab_r",
                        "transform": {"type": "identity"},
                    }
                ],
            },
            "TEST",
            {"mag_r"},
        ).applied[0]
        df = pl.DataFrame({"mag_r": [17.5, -9999.0, float("nan"), 9999.0]})
        out, _ = apply_rules_to_frame(df, [rule])
        assert out["phot_ab_r"][0] == pytest.approx(17.5)
        assert out["phot_ab_r"][1] is None
        assert out["phot_ab_r"][2] is None
        assert out["phot_ab_r"][3] is None

    def test_null_if_sentinel_custom_values(self) -> None:
        import polars as pl

        rule = resolve_applicable_rules(
            {
                "transform_id": "phot_ab_v1",
                "rules": [
                    {
                        "survey": "TEST",
                        "source_column": "mag_r",
                        "target_column": "phot_ab_r",
                        "uncertainty_column": "mag_r_err",
                        "target_uncertainty_column": "phot_ab_r_err",
                        "transform": {"type": "null_if_sentinel", "values": [99.0, -99.0]},
                    }
                ],
            },
            "TEST",
            {"mag_r", "mag_r_err"},
        ).applied[0]
        df = pl.DataFrame({
            "mag_r": [17.5, 99.0, -99.0, -9999.0, float("nan")],
            "mag_r_err": [0.02, 0.02, 0.02, 0.02, 0.02],
        })
        out, _ = apply_rules_to_frame(df, [rule])
        assert out["phot_ab_r"][0] == pytest.approx(17.5)
        assert out["phot_ab_r"][1] is None   # custom sentinel 99.0
        assert out["phot_ab_r"][2] is None   # custom sentinel -99.0
        assert out["phot_ab_r"][3] is None   # built-in -9999
        assert out["phot_ab_r"][4] is None   # NaN
        assert out["phot_ab_r_err"][0] == pytest.approx(0.02)
        assert out["phot_ab_r_err"][1] is None  # uncertainty also nulled

    def test_explicit_uncertainty_transform_scale(self) -> None:
        """uncertainty_transform overrides the default auto-propagation."""
        import polars as pl

        rule = resolve_applicable_rules(
            {
                "transform_id": "phot_ab_v1",
                "rules": [
                    {
                        "survey": "TEST",
                        "source_column": "mag_r",
                        "target_column": "phot_ab_r",
                        "uncertainty_column": "mag_r_err",
                        "target_uncertainty_column": "phot_ab_r_err",
                        # value transform is identity (would default-copy err),
                        # but uncertainty is explicitly scaled
                        "transform": {"type": "identity"},
                        "uncertainty_transform": {"type": "scale", "factor": 0.001},
                    }
                ],
            },
            "TEST",
            {"mag_r", "mag_r_err"},
        ).applied[0]
        df = pl.DataFrame({"mag_r": [17.5], "mag_r_err": [20.0]})
        out, lin = apply_rules_to_frame(df, [rule])
        assert out["phot_ab_r"][0] == pytest.approx(17.5)
        assert out["phot_ab_r_err"][0] == pytest.approx(0.02)
        assert lin[0]["uncertainty_transform"]["type"] == "scale"

    def test_explicit_uncertainty_transform_flux_to_ab(self) -> None:
        """Explicit flux_to_ab uncertainty transform uses flux-error propagation."""
        import polars as pl

        rule = resolve_applicable_rules(
            {
                "transform_id": "phot_ab_v1",
                "rules": [
                    {
                        "survey": "TEST",
                        "source_column": "flux",
                        "target_column": "phot_ab_w1",
                        "uncertainty_column": "dflux",
                        "target_uncertainty_column": "phot_ab_w1_err",
                        "transform": {"type": "flux_to_ab", "zp": 8.906},
                        "uncertainty_transform": {"type": "flux_to_ab"},
                    }
                ],
            },
            "TEST",
            {"flux", "dflux"},
        ).applied[0]
        df = pl.DataFrame({"flux": [1.0], "dflux": [0.1]})
        out, _ = apply_rules_to_frame(df, [rule])
        assert out["phot_ab_w1"][0] == pytest.approx(8.906)
        assert out["phot_ab_w1_err"][0] == pytest.approx(0.1085736, rel=1e-4)

    def test_validate_uncertainty_transform_requires_columns(self) -> None:
        from data_lake.homogenize.survey_registry import validate_survey_homogenize

        msgs = validate_survey_homogenize({
            "survey": "TEST",
            "catalog": {
                "phot_ab_v1": {
                    "rules": [
                        {
                            "source_column": "mag_r",
                            "target_column": "phot_ab_r",
                            "transform": {"type": "identity"},
                            "uncertainty_transform": {"type": "scale", "factor": 2.0},
                        }
                    ]
                }
            },
        })
        assert any("uncertainty_transform requires" in m for m in msgs)

    def test_view_sql(self) -> None:
        t = load_transform(None, "phot_ab_v1")
        sql = build_homogenized_view_sql("ALLWISE", t, lake_root=None)
        assert "phot_ab_w1" in sql
        assert "2.699" in sql


class TestHomogenizeCatalog:
    def test_region_homogenize(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_wise_tile(lake, norder, npix, w1=10.0)
        ti.write_tile_index(lake, "ALLWISE", "catalog")

        region = Region.cone(ra, dec, 60.0)
        sel = selection_from_region(lake, "ALLWISE", region)
        result = homogenize_catalog(
            lake, "ALLWISE", "phot_ab_v1", sel,
            materialize_as="ALLWISE_ab_test",
        )
        assert result.n_rows == 2
        info = json.loads(
            (lake / "catalogs" / "ALLWISE_ab_test" / "catalog_info.json").read_text()
        )
        assert info["product_subtype"] == PRODUCT_SUBTYPE_HOMOGENIZED
        assert info["provenance"]["transform_id"] == "phot_ab_v1"

        tile = pq.read_table(
            lake / "catalogs" / "ALLWISE_ab_test" / healpix_dir(norder, npix) / f"Npix={npix}.parquet"
        )
        assert "phot_ab_w1" in tile.column_names
        assert tile.column("phot_ab_w1")[0].as_py() == pytest.approx(12.699)

    def test_check_only(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        npix = 0
        _write_wise_tile(lake, norder, npix, w1=11.0)
        ti.write_tile_index(lake, "ALLWISE", "catalog")
        sel = selection_from_region(
            lake, "ALLWISE", Region.from_npix([npix], norder),
        )
        result = homogenize_catalog(
            lake, "ALLWISE", "phot_ab_v1", sel,
            materialize_as="ALLWISE_ab_check",
            check_only=True,
        )
        assert result.check_only
        assert result.resolution["n_applied"] >= 1
        assert not (lake / "catalogs" / "ALLWISE_ab_check").exists()


def _write_minimal_lake_config(lake: Path) -> Path:
    cfg_path = lake.parent / "lake_config.toml"
    cfg_path.write_text(
        f"""
schema_version = "1"
[lake]
name = "test"
root = "{lake.as_posix()}"
[paths]
catalogs = "catalogs"
spectra = "spectra"
cutouts = "cutouts"
shared = "shared"
"""
    )
    return cfg_path


class TestHomogenizeCli:
    def test_data_lake_config_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from click.testing import CliRunner

        from data_lake.homogenize.homogenize_cli import cli

        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_wise_tile(lake, norder, npix, w1=10.0)
        ti.write_tile_index(lake, "ALLWISE", "catalog")

        area = make_area(
            "WiseCone",
            Region.cone(ra, dec, 60.0),
            homogenize={
                "survey": "ALLWISE",
                "transform": "phot_ab_v1",
                "materialize_as": "ALLWISE_from_config",
            },
        )
        save_area(lake, area)

        cfg_path = _write_minimal_lake_config(lake)
        monkeypatch.setenv("DATA_LAKE_CONFIG", str(cfg_path))

        result = CliRunner().invoke(cli, ["--from-area", "WiseCone"])
        assert result.exit_code == 0, result.output
        assert (lake / "catalogs" / "ALLWISE_from_config").is_dir()

    def test_from_area_homogenize_block(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.homogenize.homogenize_cli import cli

        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_wise_tile(lake, norder, npix, w1=10.0)
        ti.write_tile_index(lake, "ALLWISE", "catalog")

        area = make_area(
            "WiseCone",
            Region.cone(ra, dec, 60.0),
            homogenize={
                "survey": "ALLWISE",
                "transform": "phot_ab_v1",
                "materialize_as": "ALLWISE_from_area",
            },
        )
        save_area(lake, area)

        result = CliRunner().invoke(cli, [str(lake), "--from-area", "WiseCone"])
        assert result.exit_code == 0, result.output
        assert (lake / "catalogs" / "ALLWISE_from_area").is_dir()


def _write_gather_like_product(lake: Path, norder: int, npix: int, w1: float) -> None:
    tile_dir = lake / "catalogs" / "EUCLID_wise_native" / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    hp = f"_healpix_norder{norder}"
    pq.write_table(
        pa.table({
            LAKE_JOIN_ID_COLUMN: pa.array([1], type=pa.int64()),
            "ra": pa.array([120.0], type=pa.float64()),
            "dec": pa.array([45.0], type=pa.float64()),
            hp: pa.array([npix], type=pa.int64()),
            "ALLWISE_w1mpro": pa.array([w1], type=pa.float32()),
            "ALLWISE_w1sigmpro": pa.array([0.05], type=pa.float32()),
        }),
        tile_dir / f"Npix={npix}.parquet",
    )
    (lake / "catalogs" / "EUCLID_wise_native" / "catalog_info.json").write_text(
        json.dumps({
            "hats_order": norder,
            "ra_column": "ra",
            "dec_column": "dec",
            "link_id_column": LAKE_JOIN_ID_COLUMN,
            "kind": "product",
            "provenance": {
                "base_catalog": "EUCLID",
                "partners": [{"survey": "ALLWISE", "columns": ["w1mpro", "w1sigmpro"]}],
            },
        })
    )
    ti.write_tile_index(lake, "EUCLID_wise_native", "catalog")


class TestHomogenizeProduct:
    def test_from_product_wide_table(self, tmp_path: Path) -> None:
        from data_lake.discovery.selection import selection_from_all_tiles
        from data_lake.homogenize.engine import homogenize_product

        lake = tmp_path / "lake"
        norder = 5
        npix = 0
        _write_gather_like_product(lake, norder, npix, w1=10.0)
        sel = selection_from_all_tiles(lake, "EUCLID_wise_native")
        result = homogenize_product(
            lake, "EUCLID_wise_native", "phot_ab_v1", sel,
            materialize_as="EUCLID_wise_ab",
        )
        assert result.n_rows == 1
        from data_lake.io.catalog import CatalogAccessor

        with CatalogAccessor(lake, "EUCLID_wise_ab") as acc:
            df = acc.query("SELECT phot_ab_w1 FROM catalog", fmt="polars")
        assert df["phot_ab_w1"][0] == pytest.approx(12.699)
        info = json.loads(
            (lake / "catalogs" / "EUCLID_wise_ab" / "catalog_info.json").read_text()
        )
        assert info["provenance"]["source_product"] == "EUCLID_wise_native"

    def test_from_product_sparse_columns_finalize_metadata(self, tmp_path: Path) -> None:
        """Tiles missing partner photometry still share one schema after homogenize."""
        from data_lake.discovery.selection import selection_from_all_tiles
        from data_lake.homogenize.engine import homogenize_product

        lake = tmp_path / "lake"
        norder = 5
        product = "EUCLID_wise_native"
        hp = f"_healpix_norder{norder}"
        root = lake / "catalogs" / product

        # Tile with partner photometry
        rich_dir = root / healpix_dir(norder, 0)
        rich_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({
                LAKE_JOIN_ID_COLUMN: pa.array([1], type=pa.int64()),
                "ra": pa.array([120.0], type=pa.float64()),
                "dec": pa.array([45.0], type=pa.float64()),
                hp: pa.array([0], type=pa.int64()),
                "ALLWISE_w1mpro": pa.array([10.0], type=pa.float64()),
                "ALLWISE_w1sigmpro": pa.array([0.05], type=pa.float64()),
            }),
            rich_dir / "Npix=0.parquet",
        )
        # Tile without partner columns (sparse gather)
        sparse_dir = root / healpix_dir(norder, 10_500)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({
                LAKE_JOIN_ID_COLUMN: pa.array([2], type=pa.int64()),
                "ra": pa.array([121.0], type=pa.float64()),
                "dec": pa.array([46.0], type=pa.float64()),
                hp: pa.array([10_500], type=pa.int64()),
            }),
            sparse_dir / "Npix=10500.parquet",
        )
        (root / "catalog_info.json").write_text(json.dumps({
            "hats_order": norder,
            "ra_column": "ra",
            "dec_column": "dec",
            "link_id_column": LAKE_JOIN_ID_COLUMN,
            "kind": "product",
            "provenance": {
                "base_catalog": "EUCLID",
                "partners": [{"survey": "ALLWISE", "columns": ["w1mpro", "w1sigmpro"]}],
            },
        }))
        ti.write_tile_index(lake, product, "catalog")

        sel = selection_from_all_tiles(lake, product)
        result = homogenize_product(
            lake, product, "phot_ab_v1", sel,
            materialize_as="EUCLID_wise_ab_sparse",
        )
        assert result.n_rows == 2
        out_root = lake / "catalogs" / "EUCLID_wise_ab_sparse"
        assert (out_root / "_metadata").is_file()
        tiles = sorted(out_root.rglob("Npix=*.parquet"))
        assert len(tiles) == 2
        schemas = [pq.read_schema(str(p)) for p in tiles]
        assert schemas[0].equals(schemas[1])
        assert "phot_ab_w1" in schemas[0].names
        assert "phot_ab_w1_err" in schemas[0].names
        assert schemas[0].field("phot_ab_w1").type == pa.float64()

    def test_from_product_null_vs_float_partner_column(self, tmp_path: Path) -> None:
        """null-typed partner columns unify to float64 across tiles."""
        from data_lake.discovery.selection import selection_from_all_tiles
        from data_lake.homogenize.engine import homogenize_product

        lake = tmp_path / "lake"
        norder = 5
        product = "EUCLID_wise_native"
        hp = f"_healpix_norder{norder}"
        root = lake / "catalogs" / product

        typed_dir = root / healpix_dir(norder, 0)
        typed_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({
                LAKE_JOIN_ID_COLUMN: pa.array([1], type=pa.int64()),
                "ra": pa.array([120.0], type=pa.float64()),
                "dec": pa.array([45.0], type=pa.float64()),
                hp: pa.array([0], type=pa.int64()),
                "ALLWISE_w1mpro": pa.array([10.0], type=pa.float64()),
                "ALLWISE_w1sigmpro": pa.array([0.05], type=pa.float64()),
            }),
            typed_dir / "Npix=0.parquet",
        )
        null_dir = root / healpix_dir(norder, 10_500)
        null_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({
                LAKE_JOIN_ID_COLUMN: pa.array([2], type=pa.int64()),
                "ra": pa.array([121.0], type=pa.float64()),
                "dec": pa.array([46.0], type=pa.float64()),
                hp: pa.array([10_500], type=pa.int64()),
                "ALLWISE_w1mpro": pa.array([None], type=pa.null()),
                "ALLWISE_w1sigmpro": pa.array([None], type=pa.null()),
            }),
            null_dir / "Npix=10500.parquet",
        )
        (root / "catalog_info.json").write_text(json.dumps({
            "hats_order": norder,
            "ra_column": "ra",
            "dec_column": "dec",
            "link_id_column": LAKE_JOIN_ID_COLUMN,
            "kind": "product",
            "provenance": {
                "base_catalog": "EUCLID",
                "partners": [{"survey": "ALLWISE", "columns": ["w1mpro", "w1sigmpro"]}],
            },
        }))
        ti.write_tile_index(lake, product, "catalog")

        sel = selection_from_all_tiles(lake, product)
        result = homogenize_product(
            lake, product, "phot_ab_v1", sel,
            materialize_as="EUCLID_wise_ab_null",
        )
        assert result.n_rows == 2
        out_root = lake / "catalogs" / "EUCLID_wise_ab_null"
        assert (out_root / "_metadata").is_file()
        for path in out_root.rglob("Npix=*.parquet"):
            sch = pq.read_schema(str(path))
            assert sch.field("ALLWISE_w1mpro").type == pa.float64()
            assert sch.field("phot_ab_w1").type == pa.float64()


class TestValidateHomogenization:
    def test_golden_phot_ab(self) -> None:
        from data_lake.homogenize.validate_homogenization import validate_golden_transform

        rep = validate_golden_transform("phot_ab_v1")
        assert rep.errors == []

    def test_registry_lint(self) -> None:
        from data_lake.homogenize.validate_homogenization import validate_transform_registry

        rep = validate_transform_registry(None)
        assert rep.errors == []

    def test_survey_registry_allwise(self) -> None:
        from data_lake.homogenize.survey_registry import load_survey_homogenize, resolve_catalog_rules

        doc = load_survey_homogenize(None, "ALLWISE")
        assert doc is not None
        rules = resolve_catalog_rules(None, "ALLWISE", "phot_ab_v1")
        assert any(r.target_column == "phot_ab_w1" for r in rules)

    def test_transform_pack_has_no_survey_rules(self) -> None:
        t = load_transform(None, "phot_ab_v1")
        assert not t.get("rules")

    def test_missing_survey_recipe_raises(self) -> None:
        from data_lake.homogenize.survey_registry import SurveyHomogenizeNotFound, resolve_catalog_rules

        with pytest.raises(SurveyHomogenizeNotFound):
            resolve_catalog_rules(None, "NONEXISTENT_SURVEY_XX", "phot_ab_v1")


class TestZarrHomogenize:
    def test_spectra_flux_scale(self, tmp_path: Path) -> None:
        from data_lake.discovery.selection import selection_from_all_tiles
        from data_lake.homogenize.zarr_engine import homogenize_zarr
        from synthetic_lake_helpers import NORDER, SURVEY, ingest_synthetic_spectrum_lake

        lake = tmp_path / "lake"
        ingest_synthetic_spectrum_lake(lake)
        ti.write_tile_index(lake, SURVEY, "spectra")
        sel = selection_from_all_tiles(lake, SURVEY, modality="spectra")
        result = homogenize_zarr(
            lake, "spectra", SURVEY, "spec_observed_v1", sel,
            materialize_as="synthetic_spec_hom",
        )
        assert result.n_sources > 0
        import zarr

        tiles = list((lake / "spectra" / "synthetic_spec_hom").rglob("Npix=*.zarr"))
        assert tiles
        root = zarr.open_group(
            store=zarr.storage.LocalStore(str(tiles[0])), mode="r", zarr_format=3,
        )
        assert root["flux"].shape[0] > 0
