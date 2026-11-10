# MCP lake explorer for agents

The `dl-mcp-lake` command runs a Model Context Protocol (MCP) server, over stdio, that lets an AI agent answer questions about your lake: which surveys and products exist, what data covers a sky region, how a product was built, whether an ingest looks healthy, and how to ingest a new survey. Use it unless you also want documentation search; for that, run the [MCP docs server](mcp.md) (`dl-mcp-docs`) alongside it.

## Install

```bash
uv sync --extra mcp
```

## Configure Cursor

Copy the example config and set `DATA_LAKE_CONFIG` to the path of your `lake_config.toml`:

```bash
cp .cursor/mcp.json.example .cursor/mcp.json
# Edit DATA_LAKE_CONFIG to point at your lake_config.toml
```

The example defines both servers. To run the explorer only:

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

To also get documentation search and `docs://` resources, run both:

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

Restart Cursor after changing the MCP config.

## Example agent workflows

These are the tool calls an agent makes; you normally just ask the question in chat.

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

**"How should I ingest 93k Legacy Survey brick catalogs?"**

```
recommend_ingest(modality="catalog", survey="Legacy_DR11_north", n_files=93000,
                 layout="spatial", file_area_deg2=0.0625, norder=7)
```

This returns map and reduce `dl-ingest-from-shard` commands, a Slurm array script for the map step, and the reduce job script in `extra_slurm_scripts["reduce"]`. Without `layout` or `file_area_deg2`, the advisor keeps the batch-append recommendation and adds a warning that the file layout is unknown.

## Tools

Every tool that reads the lake finds it through `DATA_LAKE_CONFIG`. Pass `lake_root` when that variable is not set; see [lake configuration](mcp-deploy.md#lake-configuration) for the full lookup order. `recommend_norder` and `recommend_ingest` work on input files and take no `lake_root`.

### Inventory

| Tool | Purpose |
|------|---------|
| `describe_lake` | Registry summary of all surveys, modalities and row counts, same output as `dl-describe-lake --json`. Optional `kind` (`ingested`, `product`, `crossmatch`), `modality`, `refresh`, `count_total` (default true). |
| `describe_survey` | Column manifest for a survey, same output as `dl-describe-survey --json`. Arguments: `survey`, `modality` (default `catalog`), `rebuild`. |
| `list_products` | All product catalogs with name, modality, subtype, row count and finalized flag. |
| `list_homogenize_recipes` | Surveys with a homogenization recipe, whether it is a lake override or a bundled default, and which modalities it covers. |

### Discovery and provenance

| Tool | Purpose |
|------|---------|
| `discover_region` | Which surveys and modalities overlap a sky region, with rounded row counts. Set `count=True` for exact counts. |
| `list_areas` / `get_area` | List saved area IDs, or load one `areas/<id>.json` (region, `discover`, `crossmatch_plan`, `gather` and `homogenize` blocks). Edit areas with [`dl-area`](discovery/areas-cli.md). |
| `list_crossmatches` | Crossmatch trees (sky and column matches) with metadata; optional `survey_a`, `survey_b` filters. |
| `describe_crossmatch` | Detail for one crossmatch tree, same output as `dl-describe-crossmatch --json`. |
| `describe_product` | Product provenance, the crossmatch trees it was built from, the first 20 columns, and homogenization provenance. |
| `build_query` | Build DuckDB SQL from a master association file. Arguments: `master`, `primary_survey`, `columns` (list of `SURVEY:col1,col2` picks). The SQL is returned, never executed. |

`discover_region` and `estimate_operation_cost` need exactly one region selector:

| Selector | Arguments |
|----------|-----------|
| Saved area | `from_area` |
| HEALPix pixels | `npix` plus `norder` (the order those pixels are given in) |
| Cone | `cone_ra`, `cone_dec`, `radius_arcsec` |
| Box | `bbox_ra_min`, `bbox_ra_max`, `bbox_dec_min`, `bbox_dec_max` |
| MOC (Multi-Order Coverage map) | `moc` (file path) |

`discover_region` also takes `surveys` (comma-separated names, default `all`) and `modalities` (default catalog, spectra and cutout).

### QA and advice

| Tool | Purpose |
|------|---------|
| `validate_survey` | Run ingest validation on a sample of tiles. Arguments: `survey`, `modality` (`catalog`, `spectra`, `cutout`), `max_tiles` (default 5). |
| `lake_health` | Registry entry counts by kind and modality, live surveys that are not finalized, surveys missing a tile index, and a hint when the registry is more than 24 hours old. |
| `recommend_norder` | Suggest `--norder` for a catalog ingest from sample files (`paths` or `file_list`; `ra_col`/`dec_col` default to `ra`/`dec`). |
| `estimate_operation_cost` | Order-of-magnitude runtime for `operation` = `crossmatch`, `gather` or `ingest` over a region, based on the number of overlapping catalog tiles. |
| `recommend_ingest` | Recommend a `dl-*` ingest command and Slurm script for a survey. |

For catalog file lists, `recommend_ingest` chooses between batch append and `dl-ingest-from-shard`. Tell it how the files are laid out with `layout`, or give `file_area_deg2` (sky area per file) together with `norder`:

| `layout` | Use for |
|----------|---------|
| `spatial` | Bricks or RA/Dec patches |
| `unsorted` | Files not grouped on the sky |
| `partitioned` | Files already aligned with lake tiles |

Live surveys (`lifecycle="live"`) always get batch append with `--defer-finalize`, because the shard reduce step always finalizes. Other arguments: `file_list`, `sample_file`, `fmt`, `n_files`, `total_size_gb` (above 8 GB switches on streaming), `streaming`.

## Out of scope

The explorer does not ingest data, handle ingest tokens, submit jobs or execute SQL. Use the `dl-*` CLIs for anything that changes the lake. A few calls can refresh lake metadata files (the registry, tile indices, schema manifests); see [read-only boundary](mcp-deploy.md#read-only-boundary).

See also: [MCP docs server](mcp.md) for documentation search and `docs://` resources, and [MCP deployment](mcp-deploy.md) for team setups and troubleshooting.
