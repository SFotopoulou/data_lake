"""
Read-only MCP server exposing project documentation and lake discovery tools.

Run via ``dl-mcp-docs`` (stdio transport for Cursor and other MCP clients).
Requires the ``mcp`` optional extra: ``uv sync --extra mcp``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_lake.doc_index import (
    get_section,
    list_cli_commands,
    search_docs,
)


from data_lake.mcp_common import json_dumps, resolve_lake_root


def _resolve_lake_root(lake_root: str | None) -> Path:
    return resolve_lake_root(lake_root)


def tool_search_docs(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Search project documentation by keyword."""
    return search_docs(query, limit=limit)


def tool_get_doc_section(path_or_id: str) -> dict[str, str | None]:
    """Return a documentation section by path, docs:// URI, or heading."""
    text = get_section(path_or_id)
    return {"path_or_id": path_or_id, "text": text}


def tool_list_cli_commands() -> list[dict[str, str]]:
    """List all dl-* CLI entry points from pyproject.toml."""
    return list_cli_commands()


def tool_describe_lake(
    lake_root: str | None = None,
    *,
    refresh: bool = False,
    count_total: bool = True,
    modality: str | None = None,
) -> dict[str, Any]:
    """Return lake registry summary (same shape as dl-describe-lake --json)."""
    from data_lake.lake_registry import (
        filter_lake_registry_table,
        load_lake_registry,
        refresh_lake_registry,
        registry_path,
        summarize_registry_row_counts,
    )

    root = _resolve_lake_root(lake_root)
    if refresh or not registry_path(root).is_file():
        refresh_lake_registry(root)
    table = filter_lake_registry_table(load_lake_registry(root), modality)
    payload: dict[str, Any] = {"entries": table.to_pylist()}
    if count_total:
        payload["summary"] = summarize_registry_row_counts(table)
    return payload


def tool_describe_survey(
    survey: str,
    *,
    lake_root: str | None = None,
    modality: str = "catalog",
    rebuild: bool = False,
) -> dict[str, Any]:
    """Return schema manifest for a survey layer (dl-describe-survey --json)."""
    from data_lake.schema_registry import get_survey_manifest

    root = _resolve_lake_root(lake_root)
    manifest = get_survey_manifest(
        root, survey, modality, rebuild=rebuild, apply_overlay=True,
    )
    return manifest


def create_mcp_app() -> Any:
    """Build the FastMCP application (requires ``mcp`` package)."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("data-lake-docs")

    @mcp.tool()
    def search_docs(query: str, limit: int = 5) -> str:
        """Search astronomy data lake documentation by keyword."""
        return json.dumps(tool_search_docs(query, limit=limit), indent=2)

    @mcp.tool()
    def get_doc_section(path_or_id: str) -> str:
        """Fetch a documentation section by path (e.g. ingest/spectra.md) or docs:// URI."""
        return json.dumps(tool_get_doc_section(path_or_id), indent=2)

    @mcp.tool()
    def list_dl_commands() -> str:
        """List all dl-* CLI commands registered in pyproject.toml."""
        return json.dumps(tool_list_cli_commands(), indent=2)

    @mcp.tool()
    def describe_lake(
        lake_root: str | None = None,
        refresh: bool = False,
        count_total: bool = True,
        modality: str | None = None,
    ) -> str:
        """Summarize surveys and modalities on disk (dl-describe-lake --json)."""
        return json.dumps(
            tool_describe_lake(
                lake_root, refresh=refresh, count_total=count_total, modality=modality,
            ),
            indent=2,
            default=str,
        )

    @mcp.tool()
    def describe_survey(
        survey: str,
        lake_root: str | None = None,
        modality: str = "catalog",
        rebuild: bool = False,
    ) -> str:
        """Return column manifest for a survey (dl-describe-survey --json)."""
        return json.dumps(
            tool_describe_survey(
                survey, lake_root=lake_root, modality=modality, rebuild=rebuild,
            ),
            indent=2,
            default=str,
        )

    @mcp.resource("docs://{path}")
    def read_doc(path: str) -> str:
        """Read a documentation file under docs/ (path without .md suffix)."""
        text = get_section(path if path.endswith(".md") else path)
        if text is None:
            rel = path if path.endswith(".md") else f"{path}.md"
            text = get_section(rel)
        if text is None:
            return f"Documentation not found: docs://{path}"
        return text

    return mcp


def main() -> None:
    """Entry point for dl-mcp-docs."""
    mcp = create_mcp_app()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
