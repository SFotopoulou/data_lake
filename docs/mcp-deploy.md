# MCP server deployment guide

This guide covers installing, configuring and running the two Model Context Protocol (MCP) servers, `dl-mcp-docs` and `dl-mcp-lake`, for yourself or for a team sharing one lake. Read it when you set up agents against a shared lake or when a server does not behave as expected.

## Overview

| Server name | Command | Exposes |
|-------------|---------|---------|
| `data-lake-explorer` | `dl-mcp-lake` | Lake inventory, region discovery, provenance, QA and ingest advice |
| `data-lake-docs` | `dl-mcp-docs` | Documentation search, `docs://` resources, and lake/survey inventory |

Use the explorer alone unless you also want documentation search. It already includes the inventory tools (`describe_lake`, `describe_survey`, `list_products`, `list_homogenize_recipes`). Tool details are in [MCP lake explorer](mcp-lake.md) and [MCP docs server](mcp.md).

## Install

```bash
uv sync --extra mcp
```

Both servers come with the same `mcp` extra; nothing else is needed.

## Lake configuration

Each tool call finds the lake in this order:

1. The `lake_root` argument, if the agent passes one.
2. The `DATA_LAKE_CONFIG` environment variable (path to `lake_config.toml`).
3. A `lake_config.toml` in the server's working directory or any parent directory.

Set `DATA_LAKE_CONFIG` in the server's `env` block so agents never need to pass `lake_root`:

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

Edit `DATA_LAKE_CONFIG` in both server blocks, then restart Cursor (or reload MCP servers from the Cursor settings panel).

Explorer only (covers all lake operations):

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

Both servers (adds documentation search and `docs://` resources):

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

Neither server ingests, modifies or deletes survey data. They never:

- Handle or validate ingest tokens
- Submit jobs (Slurm or otherwise)
- Execute SQL (`build_query` returns the SQL as text)
- Run shell commands

Use the `dl-*` CLIs for all writes.

A few calls do write lake metadata files when they are missing or when the agent asks for a rebuild:

| Call | Writes |
|------|--------|
| `describe_lake(refresh=True)`, or `describe_lake` / `list_products` when no registry exists | The lake registry `shared/registry/surveys.parquet` and the tile indices under `shared/registry/tile_index/` |
| `describe_survey(rebuild=True)`, or `describe_survey` / `describe_product` when the survey has no `schema_manifest.json` | That survey's `schema_manifest.json` |

On a read-only lake mount these calls fail (`describe_product` instead returns an empty column summary), but everything else works, because the servers otherwise only read the registry, Parquet tile metadata and JSON sidecar files. Run `dl-refresh-lake-registry` and `dl-finalize-catalog` from an account with write access so the files already exist.

## Shared team lake

For a team lake (for example `/shared/como`):

1. Set `DATA_LAKE_CONFIG=/shared/como/lake_config.toml` in each server's `env` block.
2. Each user's MCP client starts its own stdio process. There is no daemon or shared socket.
3. The lake registry is read from disk on each tool call, so new surveys appear as soon as the registry is refreshed with `dl-refresh-lake-registry` (or `describe_lake(refresh=True)`). The documentation search index is built once per process; restart the docs server to pick up edited pages.
4. The MCP layer has no authentication. Access control is at the filesystem level: restrict who can read `lake_config.toml` and the lake directory.

## Troubleshooting

**"No lake root: pass lake_root or set DATA_LAKE_CONFIG / lake_config.toml."**
No lake was found by any of the methods in [lake configuration](#lake-configuration). Set `DATA_LAKE_CONFIG` in the server's `env` block and check the path exists.

**"No lake registry at … Run: dl-refresh-lake-registry"**, or `lake_health` reports "Run dl-refresh-lake-registry to build the registry."
The lake has data on disk but no registry file yet. Run `dl-refresh-lake-registry`.

**"Registry is N h old — consider dl-refresh-lake-registry."** (from `lake_health`)
The registry is more than 24 hours old, so surveys ingested since then may be missing. Run `dl-refresh-lake-registry`.

**`lake_health` reports unfinalized live surveys**
Run `dl-finalize-catalog` for those surveys.

**`lake_health` reports surveys missing a tile index**
Run `dl-refresh-lake-registry`, which rebuilds all tile indices.

**MCP server does not appear in Cursor**
Check that `uv sync --extra mcp` succeeded in the project root, that `.cursor/mcp.json` has the right `command` and `DATA_LAKE_CONFIG`, and that the repository is open as the Cursor workspace. Restart Cursor after any config change.

## See also

- [MCP lake explorer](mcp-lake.md): full tool reference
- [MCP docs server](mcp.md): documentation search reference
- [lake-config.md](lake-config.md): `lake_config.toml` reference
- [cli-reference.md](cli-reference.md): all `dl-*` commands
