"""Shared lake inventory tools for both MCP servers (docs + lake explorer).

Extracted so that ``dl-mcp-lake`` can answer "what surveys/products exist?"
without requiring ``dl-mcp-docs`` to also be running.

All functions are read-only (no writes, no ingest, no token handling).
"""

from __future__ import annotations

from typing import Any

from data_lake.mcp_common import resolve_lake_root
from data_lake.schema_registry import (
    CATALOG_KIND_PRODUCT,
    MODALITY_CATALOG,
)


def tool_describe_lake(
    lake_root: str | None = None,
    *,
    kind: str | None = None,
    refresh: bool = False,
    count_total: bool = True,
    modality: str | None = None,
) -> dict[str, Any]:
    """Return lake registry summary (same shape as dl-describe-lake --json).

    Parameters
    ----------
    kind:
        Filter by registry kind: ``ingested``, ``product``, or ``crossmatch``.
        ``None`` (default) returns all entries.
    refresh:
        Rebuild the registry from disk before returning.
    count_total:
        Include a ``summary`` key with total row counts.
    modality:
        Filter by modality (``catalog``, ``spectra``, ``cutout``).
    """
    from data_lake.lake_registry import (
        coalesce_registry_kinds,
        filter_lake_registry_table,
        filter_registry_by_kind,
        load_lake_registry,
        refresh_lake_registry,
        registry_path,
        summarize_registry_row_counts,
    )

    root = resolve_lake_root(lake_root)
    if refresh or not registry_path(root).is_file():
        refresh_lake_registry(root)

    table = coalesce_registry_kinds(load_lake_registry(root))
    table = filter_registry_by_kind(table, kind)
    table = filter_lake_registry_table(table, modality)

    payload: dict[str, Any] = {
        "entries": table.to_pylist(),
        "filters_applied": {"kind": kind, "modality": modality},
    }
    if count_total:
        payload["summary"] = summarize_registry_row_counts(table)
    return payload


def tool_describe_survey(
    survey: str,
    *,
    lake_root: str | None = None,
    modality: str = MODALITY_CATALOG,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Return schema manifest for a survey layer (dl-describe-survey --json)."""
    from data_lake.schema_registry import get_survey_manifest

    root = resolve_lake_root(lake_root)
    return get_survey_manifest(root, survey, modality, rebuild=rebuild, apply_overlay=True)


def tool_list_products(lake_root: str | None = None) -> dict[str, Any]:
    """List all product catalogs in the lake registry.

    Returns names, subtype, modality and total_rows for each product so agents
    can discover product names before calling ``describe_product``.
    """
    from data_lake.lake_registry import (
        coalesce_registry_kinds,
        filter_registry_by_kind,
        load_lake_registry,
        registry_path,
        refresh_lake_registry,
    )

    root = resolve_lake_root(lake_root)
    if not registry_path(root).is_file():
        refresh_lake_registry(root)

    table = coalesce_registry_kinds(load_lake_registry(root))
    products = filter_registry_by_kind(table, CATALOG_KIND_PRODUCT)
    rows = products.to_pylist()

    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({
            "name": row.get("survey", ""),
            "modality": row.get("modality", MODALITY_CATALOG),
            "product_subtype": row.get("product_subtype"),
            "total_rows": row.get("total_rows"),
            "finalized": row.get("finalized"),
        })

    return {
        "n_products": len(out),
        "products": out,
        "hint": (
            "Use describe_product(name=...) for provenance and crossmatch lineage."
            if out
            else "No products found. Run dl-gather to build association products."
        ),
    }
