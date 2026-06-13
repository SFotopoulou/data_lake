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

        from data_lake.homogenize.transforms import TransformRule

        t = load_transform(None, "phot_ab_v1")
        res = resolve_applicable_rules(
            t, "ALLWISE", {"w1mpro", "w1sigmpro", "ra", "dec"},
        )
        assert len(res.applied) >= 1
        rule = next(r for r in res.applied if r.target_column == "phot_ab_w1")
        df = pl.DataFrame({"w1mpro": [10.0], "w1sigmpro": [0.05]})
        out, lin = apply_rules_to_frame(df, [rule])
        assert out["phot_ab_w1"][0] == pytest.approx(12.699)
        assert out["phot_ab_w1_err"][0] == pytest.approx(0.05)

    def test_view_sql(self) -> None:
        t = load_transform(None, "phot_ab_v1")
        sql = build_homogenized_view_sql("ALLWISE", t)
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


class TestHomogenizeCli:
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
