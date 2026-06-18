"""Smoke tests for lake explorer MCP tool handlers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_lake.discovery.areas import make_area, save_area
from data_lake.discovery.gather import PartnerSpec, gather_product
from data_lake.discovery.region import Region
from data_lake.discovery.selection import selection_from_region
from data_lake.discovery import tile_index as ti
from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, assign_healpix, healpix_dir
from data_lake.io.crossmatch import build_crossmatch
from data_lake.lake_registry import refresh_lake_registry
from data_lake.mcp_lake_server import (
    tool_describe_product,
    tool_discover_region,
    tool_estimate_operation_cost,
    tool_get_area,
    tool_lake_health,
    tool_list_areas,
    tool_list_crossmatches,
    tool_recommend_ingest,
    tool_validate_survey,
)


def _write_catalog_tile(
    lake: Path,
    survey: str,
    *,
    norder: int,
    npix: int,
    n_rows: int,
    lifecycle: str | None = None,
    finalized: bool | None = None,
) -> None:
    tile_dir = lake / "catalogs" / survey / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    hp = f"_healpix_norder{norder}"
    pq.write_table(
        pa.table({
            LAKE_JOIN_ID_COLUMN: pa.array(list(range(n_rows)), type=pa.int64()),
            "ra": pa.array([120.0] * n_rows, type=pa.float64()),
            "dec": pa.array([45.0] * n_rows, type=pa.float64()),
            hp: pa.array([npix] * n_rows, type=pa.int64()),
        }),
        tile_dir / f"Npix={npix}.parquet",
    )
    info: dict = {
        "hats_order": norder,
        "ra_column": "ra",
        "dec_column": "dec",
        "link_id_column": LAKE_JOIN_ID_COLUMN,
        "total_rows": n_rows,
        "total_columns": 4,
    }
    if lifecycle is not None:
        info["lifecycle"] = lifecycle
    if finalized is not None:
        info["finalized"] = finalized
    (lake / "catalogs" / survey / "catalog_info.json").write_text(json.dumps(info))


@pytest.fixture
def mini_lake(tmp_path: Path) -> Path:
    lake = tmp_path / "lake"
    norder = 5
    ra, dec = 120.0, 45.0
    npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
    _write_catalog_tile(lake, "ALLWISE", norder=norder, npix=npix, n_rows=50)
    ti.write_tile_index(lake, "ALLWISE", "catalog")
    area = make_area("Wide_47", Region.cone(ra, dec, 60.0))
    save_area(lake, area)
    refresh_lake_registry(lake)
    return lake


@pytest.fixture
def product_lake(tmp_path: Path) -> Path:
    lake = tmp_path / "lake"
    norder = 5
    ra, dec = 120.0, 45.0
    npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])

    for survey, z in [("EUCLID", None), ("DESI_DR1", [0.5, 1.2])]:
        tile_dir = lake / "catalogs" / survey / healpix_dir(norder, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        cols = {
            LAKE_JOIN_ID_COLUMN: pa.array([1, 2], type=pa.int64()),
            "ra": pa.array([ra, ra + 0.0005], type=pa.float64()),
            "dec": pa.array([dec, dec + 0.0005], type=pa.float64()),
            f"_healpix_norder{norder}": pa.array([npix, npix], type=pa.int64()),
        }
        if z is not None:
            cols["z"] = pa.array(z, type=pa.float64())
        pq.write_table(pa.table(cols), tile_dir / f"Npix={npix}.parquet")
        (lake / "catalogs" / survey / "catalog_info.json").write_text(json.dumps({
            "hats_order": norder,
            "ra_column": "ra",
            "dec_column": "dec",
            "link_id_column": LAKE_JOIN_ID_COLUMN,
            "total_rows": 2,
            "total_columns": 5 if z else 4,
        }))

    build_crossmatch(lake, "EUCLID", "DESI_DR1", radius_arcsec=2.0)
    region = Region.cone(ra, dec, 120.0)
    sel = selection_from_region(lake, "EUCLID", region)
    gather_product(
        lake, "EUCLID",
        [PartnerSpec("DESI_DR1", 2.0, ["z"])],
        sel,
        base_columns=["ra", "dec"],
        materialize_as="EUCLID_desi",
    )
    refresh_lake_registry(lake)
    return lake


def test_discover_region_cone(mini_lake: Path) -> None:
    payload = tool_discover_region(
        str(mini_lake),
        cone_ra=120.0,
        cone_dec=45.0,
        radius_arcsec=60.0,
    )
    assert payload["entries"]
    assert payload["entries"][0]["survey"] == "ALLWISE"


def test_list_and_get_area(mini_lake: Path) -> None:
    areas = tool_list_areas(str(mini_lake))
    assert "Wide_47" in areas["area_ids"]
    detail = tool_get_area(str(mini_lake), "Wide_47")
    assert detail["area_id"] == "Wide_47"
    assert detail["region"]["type"] == "cone"


def test_list_crossmatches(product_lake: Path) -> None:
    payload = tool_list_crossmatches(str(product_lake))
    assert payload["crossmatches"]
    xm = payload["crossmatches"][0]
    assert xm["survey_a"] == "EUCLID"
    assert xm["survey_b"] == "DESI_DR1"


def test_describe_product(product_lake: Path) -> None:
    payload = tool_describe_product(str(product_lake), "EUCLID_desi")
    assert payload["kind"] == "product"
    assert payload["provenance"]["base_catalog"] == "EUCLID"
    assert payload["crossmatch_trees"]


def test_validate_survey(mini_lake: Path) -> None:
    payload = tool_validate_survey(str(mini_lake), "ALLWISE", "catalog", max_tiles=1)
    assert payload["survey"] == "ALLWISE"
    assert payload["ok"] is True


def test_lake_health_flags_unfinalized(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    norder = 5
    npix = 0
    _write_catalog_tile(
        lake, "EUCLID", norder=norder, npix=npix, n_rows=10,
        lifecycle="live", finalized=False,
    )
    refresh_lake_registry(lake)
    health = tool_lake_health(str(lake))
    assert health["registry_present"] is True
    assert any(e["survey"] == "EUCLID" for e in health["unfinalized_live"])


def test_estimate_operation_cost(mini_lake: Path) -> None:
    payload = tool_estimate_operation_cost(
        str(mini_lake), "crossmatch", cone_ra=120.0, cone_dec=45.0, radius_arcsec=60.0,
    )
    assert payload["n_overlap_tiles"] >= 1
    assert "estimated_duration" in payload


def test_recommend_ingest() -> None:
    payload = tool_recommend_ingest("catalog", "GAIA", n_files=1000, file_list="gaia.txt")
    assert "dl-ingest-catalog-batch" in payload["command"]
    assert payload["slurm_script"]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("mcp") is None,
    reason="mcp optional extra not installed",
)
def test_create_lake_mcp_app() -> None:
    from data_lake.mcp_lake_server import create_lake_mcp_app

    app = create_lake_mcp_app()
    assert app is not None
