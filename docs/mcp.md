# MCP server for agents

The `dl-mcp-docs` command exposes read-only documentation and lake discovery tools over the [Model Context Protocol](https://modelcontextprotocol.io/) (stdio transport). Use it from Cursor or other MCP clients so agents can search guides and inspect your deployment without shell access.

## Install

```bash
uv sync --extra mcp
```

## Configure Cursor

Copy the example config and set your lake path:

```bash
cp .cursor/mcp.json.example .cursor/mcp.json
```

Edit `.cursor/mcp.json`:

- Set `DATA_LAKE_CONFIG` to your `lake_config.toml` (same as CLI).
- Run from the repo root so `docs/` resolves correctly.

Example:

```json
{
  "mcpServers": {
    "data-lake-docs": {
      "command": "uv",
      "args": ["run", "--extra", "mcp", "dl-mcp-docs"],
      "env": {
        "DATA_LAKE_CONFIG": "/data/lake/lake_config.toml"
      }
    }
  }
}
```

Restart Cursor (or reload MCP servers) after changing the config.

## Tools

| Tool | Purpose |
|------|---------|
| `search_docs` | Keyword search across `docs/` (heading-weighted) |
| `get_doc_section` | Fetch a file or section by path (`ingest/spectra.md`) or `docs://` URI |
| `list_dl_commands` | All `dl-*` entry points from `pyproject.toml` |
| `describe_lake` | Registry summary (`dl-describe-lake --json`); optional `refresh`, `modality` |
| `describe_survey` | Column manifest (`dl-describe-survey --json`); `survey`, `modality`, `rebuild` |

Pass `lake_root` on `describe_*` tools when `DATA_LAKE_CONFIG` is not set.

## Resources

Documentation files are available as MCP resources:

- `docs://cli-reference`
- `docs://ingest/spectra`
- `docs://troubleshooting`
- … (mirrors the `docs/` tree)

See also [MCP lake explorer](mcp-lake.md) for region discovery, provenance, and ingest recommendations.

## Out of scope

The MCP server is **read-only**: no ingest, no token handling, no shell execution. Use the `dl-*` CLIs for writes.
