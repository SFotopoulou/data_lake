"""
Read-only MCP server exposing project documentation and lake discovery tools.

Run via ``dl-mcp-docs`` (stdio transport for Cursor and other MCP clients).
Requires the ``mcp`` optional extra: ``uv sync --extra mcp``.
"""

from __future__ import annotations

import json
from typing import Any

from data_lake.doc_index import (
    get_section,
    list_cli_commands,
    search_docs,
)


from data_lake.mcp_inventory import tool_describe_lake, tool_describe_survey


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


# tool_describe_lake and tool_describe_survey are imported from mcp_inventory
# so that dl-mcp-lake can share the same implementations.


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
        kind: str | None = None,
        refresh: bool = False,
        count_total: bool = True,
        modality: str | None = None,
    ) -> str:
        """Summarize surveys and modalities on disk (dl-describe-lake --json).

        kind: filter by 'ingested', 'product', or 'crossmatch' (default: all).
        """
        return json.dumps(
            tool_describe_lake(
                lake_root,
                kind=kind,
                refresh=refresh,
                count_total=count_total,
                modality=modality,
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
