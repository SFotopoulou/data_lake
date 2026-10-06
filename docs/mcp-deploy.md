# MCP server deployment guide

This guide covers installing, configuring, and operating the two read-only MCP
servers (`dl-mcp-docs` and `dl-mcp-lake`) for team deployments of the data lake.

## Overview

| Server | CLI | Exposes |
|--------|-----|---------|
| `data-lake-docs` | `dl-mcp-docs` | Documentation search, `docs://` resources, lake/survey inventory |
| `data-lake-explorer` | `dl-mcp-lake` | Lake inventory (self-sufficient), region discovery, provenance, QA, ingest advice |

The explorer is **self-sufficient** for lake operations: it includes `describe_lake`,
`describe_survey`, `list_products`, and `list_homogenize_recipes`. Add the docs
server only when you also need documentation search or `docs://` resources.

## Install

```bash
uv sync --extra mcp
```

Both servers are included in the same extra. No additional packages are needed.

## Lake configuration

Both servers discover the lake via `DATA_LAKE_CONFIG` (path to `lake_config.toml`).

Priority:
1. `DATA_LAKE_CONFIG` environment variable
2. `lake_config.toml` in the current working directory
3. `lake_root` argument on individual tool calls

Set `DATA_LAKE_CONFIG` in the MCP server env block so agents never need to pass
`lake_root` explicitly:

```json
{
  "env": {
    "DATA_LAKE_CONFIG": "/shared/como/lake_config.toml"
  }
}
```

## Cursor configuration

Copy the example and fill in your lake path:

```bash
cp .cursor/mcp.json.example .cursor/mcp.json
```

Edit `DATA_LAKE_CONFIG` in both server blocks, then restart Cursor (or reload MCP
servers from the Cursor settings panel).

**Explorer only** (minimal — covers all lake operations):

```json
{
  "mcpServers": {
    "data-lake-explorer": {
      "command": "uv",
      "args": ["run", "--extra", "mcp", "dl-mcp-lake"],
      "env": { "DATA_LAKE_CONFIG": "/shared/como/lake_config.toml" }
    }
  }
}
```

**Both servers** (adds documentation search + `docs://` URI resources):

```json
{
  "mcpServers": {
    "data-lake-docs": {
      "command": "uv",
      "args": ["run", "--extra", "mcp", "dl-mcp-docs"],
      "env": { "DATA_LAKE_CONFIG": "/shared/como/lake_config.toml" }
    },
    "data-lake-explorer": {
      "command": "uv",
      "args": ["run", "--extra", "mcp", "dl-mcp-lake"],
      "env": { "DATA_LAKE_CONFIG": "/shared/como/lake_config.toml" }
    }
  }
}
```

## Read-only boundary

Both MCP servers are **strictly read-only**. They never:

- Ingest, modify, or delete any data on disk
- Handle or validate ingest tokens
- Submit jobs (Slurm or otherwise)
- Execute SQL (queries are built as strings, not run)
- Spawn subprocesses beyond the FastMCP stdio loop

All write operations go through the `dl-*` CLIs directly.

If your team deployment has a shared lake on a read-only mount, the MCP servers
work correctly: they only read registry files, Parquet tile metadata, and JSON
sidecar files.

## Shared team lake

For a team lake (e.g. `/shared/como`):

1. Set `DATA_LAKE_CONFIG=/shared/como/lake_config.toml` in the MCP server env.
2. Each user runs their own stdio process — the servers are single-process and
   stateless; there is no daemon or shared socket.
3. The lake registry (`shared/registry/lake_registry.parquet`) is read at startup
   and cached per-process. A `refresh=True` call or restarting the server picks up
   new surveys.
4. There is no authentication on the MCP layer. Access control is filesystem-level:
   restrict `DATA_LAKE_CONFIG` and the lake directory as needed.

## Troubleshooting

**"No lake root: pass lake_root or set DATA_LAKE_CONFIG"**
→ `DATA_LAKE_CONFIG` is not set or the path does not exist.

**"Run dl-refresh-lake-registry to build the registry"**
→ The lake has data on disk but no registry file yet.
Run `dl-refresh-lake-registry` (or `dl-ingest-catalog` which refreshes automatically).

**"Registry is N h old"** (from `lake_health`)
→ New surveys were ingested since the registry was last refreshed.
Run `dl-refresh-lake-registry`.

**MCP server does not appear in Cursor**
→ Ensure `uv sync --extra mcp` ran successfully in the project root.
Check `.cursor/mcp.json` has the correct `command` path and `DATA_LAKE_CONFIG`.
Restart Cursor after any config change.

## See also

- [MCP lake explorer](mcp-lake.md) — full tool reference
- [MCP docs server](mcp.md) — documentation search reference
- [lake-config.md](lake-config.md) — `lake_config.toml` reference
- [cli-reference.md](cli-reference.md) — all `dl-*` commands
