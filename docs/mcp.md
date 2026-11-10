# MCP server for agents

The `dl-mcp-docs` command runs a Model Context Protocol (MCP) server that lets an AI agent in Cursor, or any other MCP client, search this project's documentation and look up what is in your lake without shell access. It talks to the client over stdio (standard input/output). Use it when you want agents to answer "how do I…" questions from `docs/`. For lake exploration alone, the [MCP lake explorer](mcp-lake.md) (`dl-mcp-lake`) is enough.

## Install

```bash
uv sync --extra mcp
```

## Configure Cursor

Copy the example config, which defines both MCP servers:

```bash
cp .cursor/mcp.json.example .cursor/mcp.json
```

In `.cursor/mcp.json`, set `DATA_LAKE_CONFIG` to the path of your `lake_config.toml` (the same file the `dl-*` commands use). To run only the docs server:

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

Open the repository as your Cursor workspace so `uv run` finds the project. The server reads the `docs/` folder that sits next to the `data_lake` package, so it needs a source checkout of the repository. Restart Cursor (or reload MCP servers) after changing the config.

## Tools

| Tool | Purpose |
|------|---------|
| `search_docs` | Keyword search across `docs/`; matches in headings and file paths rank higher. Arguments: `query`, `limit` (default 5). |
| `get_doc_section` | Fetch a whole file by path (`ingest/spectra.md`) or `docs://` URI, or one section by its heading or section ID. |
| `list_dl_commands` | List every `dl-*` entry point in `pyproject.toml`. |
| `describe_lake` | Lake registry summary, same output as `dl-describe-lake --json`. Optional `kind` (`ingested`, `product`, `crossmatch`), `modality`, `refresh`, `count_total` (default true). |
| `describe_survey` | Column manifest for one survey, same output as `dl-describe-survey --json`. Arguments: `survey`, `modality` (default `catalog`), `rebuild`. |

`describe_lake` and `describe_survey` find the lake through `DATA_LAKE_CONFIG`. Pass `lake_root` to them when that variable is not set; see [lake configuration](mcp-deploy.md#lake-configuration) for the full lookup order.

## Resources

Every documentation file is also available as an MCP resource. The URI is the path under `docs/` without the `.md` suffix, for example:

- `docs://cli-reference`
- `docs://ingest/spectra`
- `docs://troubleshooting`

## Out of scope

The server does not ingest data, handle ingest tokens, or run shell commands. Use the `dl-*` CLIs for anything that changes the lake. Two calls can write lake metadata: `describe_lake` rebuilds the lake registry when `refresh=True` or when no registry exists yet, and `describe_survey` rebuilds the survey's `schema_manifest.json` when `rebuild=True` or when it is missing. See [read-only boundary](mcp-deploy.md#read-only-boundary).
