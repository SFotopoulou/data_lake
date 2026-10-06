# MCP lake explorer for agents

The `dl-mcp-lake` command exposes read-only **lake intelligence** tools over the
[Model Context Protocol](https://modelcontextprotocol.io/) (stdio transport).

The explorer is self-sufficient for lake operations: it includes inventory tools
(`describe_lake`, `describe_survey`, `list_products`) so agents can discover what
exists without also needing `dl-mcp-docs`. Run both servers when you also want
documentation search and the `docs://` resource URI.

## Install

```bash
uv sync --extra mcp
```

## Configure Cursor

Copy the example config and set your lake path:

```bash
cp .cursor/mcp.json.example .cursor/mcp.json
# Edit DATA_LAKE_CONFIG to point at your lake_config.toml
```

To run the explorer only (minimal setup):

```json
{
  "mcpServers": {
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

To run both (adds docs search + `docs://` resources):

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

### Inventory (new — no need for dl-mcp-docs)

| Tool | Purpose |
|------|---------|
| `describe_lake` | Registry summary — all surveys, modalities, row counts. Filter by `kind` (`ingested`/`product`/`crossmatch`) and/or `modality`. |
| `describe_survey` | Column manifest for a survey (`dl-describe-survey --json`). |
| `list_products` | All product catalogs with name, subtype, and row count. |
| `list_homogenize_recipes` | Per-survey homogenization recipes (lake overrides + bundled defaults). |

### Discovery and provenance

| Tool | Purpose |
|------|---------|
| `discover_region` | Survey × modality overlap for a region (`from_area`, cone, bbox, npix, moc); rounded counts or `--count` exact |
| `list_areas` / `get_area` | List or load `areas/<id>.json` (region, crossmatch_plan, gather, **homogenize**); editable via [`dl-area`](discovery/areas-cli.md) |
| `list_crossmatches` | Crossmatch trees (sky + column) with metadata |
| `describe_crossmatch` | Detail for one crossmatch tree (`dl-describe-crossmatch --json`) |
| `describe_product` | Product provenance, crossmatch trees, columns summary, and homogenize provenance |
| `build_query` | DuckDB SQL from a master association file (never executes) |

### QA and advice

| Tool | Purpose |
|------|---------|
| `validate_survey` | Sample ingest validation (`catalog` / `spectra` / `cutout`) |
| `lake_health` | Registry entry counts by kind/modality, unfinalized surveys, missing tile indices, stale registry hint |
| `recommend_norder` | Suggest `--norder` from sample catalog files |
| `estimate_operation_cost` | Rough crossmatch/gather/ingest duration for a region (cone, bbox, moc, npix, or area) |
| `recommend_ingest` | Recommend `dl-*` command + Slurm script for a survey ingest |

Pass `lake_root` to any tool when `DATA_LAKE_CONFIG` is not set.

## Example agent workflows

**"What surveys exist in this lake?"**

```
describe_lake()                                    # all surveys
describe_lake(kind="ingested")                     # primary ingests only
describe_lake(kind="product")                      # derived products
list_products()                                    # quick product index
```

**"What data exists in the Euclid North area?"**

```
discover_region(from_area="Euclid_North", modalities=["catalog","spectra"])
```

**"How was EUCLID_north_joined built?"**

```
describe_product(name="EUCLID_north_joined")
```

**"Which surveys have homogenization recipes?"**

```
list_homogenize_recipes()
```

**"How should I ingest 300k 2dF spectra on Slurm?"**

```
recommend_ingest(modality="spectra", survey="2DFGRS_DR3", fmt="2df", n_files=300000)
```

## Out of scope

Read-only: no ingest, no token handling, no job submission, no SQL execution.
Use the `dl-*` CLIs for writes.

See also: [MCP docs server](mcp.md) for documentation search, `docs://` resources,
and CLI reference.
