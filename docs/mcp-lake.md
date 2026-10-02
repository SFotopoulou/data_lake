# MCP lake explorer for agents

The `dl-mcp-lake` command exposes read-only **lake intelligence** tools over the
[Model Context Protocol](https://modelcontextprotocol.io/) (stdio transport). Use
it alongside `dl-mcp-docs` so agents can discover data in regions, inspect
provenance, build SQL, check lake health, estimate job cost, and recommend ingest
commands — without shell access or writes.

## Install

```bash
uv sync --extra mcp
```

## Configure Cursor

Add a second MCP server entry (see `.cursor/mcp.json.example`):

```json
{
  "mcpServers": {
    "data-lake-docs": {
      "command": "uv",
      "args": ["run", "--extra", "mcp", "dl-mcp-docs"],
      "env": {
        "DATA_LAKE_CONFIG": "/data/lake/lake_config.toml"
      }
    },
    "data-lake-explorer": {
      "command": "uv",
      "args": ["run", "--extra", "mcp", "dl-mcp-lake"],
      "env": {
        "DATA_LAKE_CONFIG": "/data/lake/lake_config.toml"
      }
    }
  }
}
```

Restart Cursor after changing MCP config.

## Tools

| Tool | Purpose |
|------|---------|
| `discover_region` | Survey × modality overlap for a region (`from_area`, cone, bbox, npix, moc); rounded counts or `--count` exact |
| `list_areas` / `get_area` | List or load `areas/<id>.json` (region, crossmatch_plan, gather); plans are also editable via [`dl-area`](discovery/areas-cli.md) |
| `list_crossmatches` | Crossmatch trees (sky + column) with metadata |
| `describe_crossmatch` | Detail for one crossmatch tree (`dl-describe-crossmatch --json`) |
| `describe_product` | Product catalog provenance and contributing crossmatch trees |
| `build_query` | DuckDB SQL from a master association file (never executes) |
| `validate_survey` | Sample ingest validation (`catalog` / `spectra` / `cutout`) |
| `lake_health` | Unfinalized live catalogs, missing tile indices |
| `recommend_norder` | Suggest `--norder` from sample catalog files |
| `estimate_operation_cost` | Rough crossmatch/gather/ingest duration for a region |
| `recommend_ingest` | Recommend `dl-*` command + Slurm script for a survey ingest |

Pass `lake_root` when `DATA_LAKE_CONFIG` is not set.

## Example agent workflows

**“What data exists in the Euclid North area?”**

```
discover_region(from_area="Euclid_North", modalities=["catalog","spectra"])
```

**“How was EUCLID_north_joined built?”**

```
describe_product(name="EUCLID_north_joined")
```

**“How should I ingest 300k 2dF spectra on Slurm?”**

```
recommend_ingest(modality="spectra", survey="2DFGRS_DR3", fmt="2df", n_files=300000)
```

## Out of scope

Read-only: no ingest, no token handling, no job submission, no SQL execution.
Use the `dl-*` CLIs for writes.

See also: [MCP docs server](mcp.md) for documentation search and registry describe.
