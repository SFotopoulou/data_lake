"""Smoke tests for MCP tool handlers (no stdio harness)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from data_lake.ingest.fits_to_parquet import ingest_catalog
from data_lake.lake_registry import refresh_lake_registry
from data_lake.mcp_inventory import tool_describe_lake, tool_describe_survey
from data_lake.mcp_server import (
    tool_get_doc_section,
    tool_list_cli_commands,
    tool_search_docs,
)


def _ingest_mini_catalog(tmp_path: Path, survey: str = "TEST_SURV") -> Path:
    n = 4
    tbl = Table({
        "TARGETID": np.arange(n, dtype=np.int64),
        "TARGET_RA": np.full(n, 10.0),
        "TARGET_DEC": np.full(n, 20.0),
        "Z": np.linspace(0.1, 0.3, n),
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
        link_id_col="TARGETID",
        norder=5,
        tile_mode="overwrite",
    )
    refresh_lake_registry(lake)
    return lake


def test_tool_search_docs() -> None:
    hits = tool_search_docs("dl-describe-lake", limit=3)
    assert hits
    assert any(
        "describe" in h["path"].lower()
        or "describe" in h.get("heading", "").lower()
        or "describe" in h.get("preview", "").lower()
        for h in hits
    )


def test_tool_get_doc_section() -> None:
    out = tool_get_doc_section("docs://troubleshooting")
    assert out["text"] is not None


def test_tool_list_cli_commands() -> None:
    cmds = tool_list_cli_commands()
    names = {c["name"] for c in cmds}
    assert "dl-ingest-catalog" in names
    assert "dl-mcp-docs" in names


def test_tool_describe_lake(tmp_path: Path) -> None:
    lake = _ingest_mini_catalog(tmp_path)
    payload = tool_describe_lake(str(lake), count_total=True)
    assert payload["entries"]
    assert "TEST_SURV" in {e["survey"] for e in payload["entries"]}
    assert "summary" in payload
    assert "filters_applied" in payload


def test_tool_describe_lake_kind_filter(tmp_path: Path) -> None:
    lake = _ingest_mini_catalog(tmp_path)
    payload = tool_describe_lake(str(lake), kind="ingested")
    assert payload["filters_applied"]["kind"] == "ingested"
    # No products in a fresh mini lake
    kinds = {e.get("kind") for e in payload["entries"]}
    assert "product" not in kinds


def test_tool_describe_survey(tmp_path: Path) -> None:
    lake = _ingest_mini_catalog(tmp_path)
    manifest = tool_describe_survey("TEST_SURV", lake_root=str(lake), modality="catalog")
    assert manifest.get("survey") == "TEST_SURV" or "columns" in manifest or "fields" in manifest


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("mcp") is None,
    reason="mcp optional extra not installed",
)
def test_create_mcp_app() -> None:
    from data_lake.mcp_server import create_mcp_app

    app = create_mcp_app()
    assert app is not None
