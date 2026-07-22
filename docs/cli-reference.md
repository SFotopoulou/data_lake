# CLI quick reference

Every `dl-*` command in one table. Pass `--help` to any command for full flag docs.

All commands that read from or write to a lake honour `--config PATH` (or `$DATA_LAKE_CONFIG`) to supply the lake root and defaults without repeating flags.

**Deploy**

| Command | Purpose |
|---------|---------|
| `dl-init` | Scaffold a new deployment (`lake_config.toml`, dirs, `.gitignore`) |
| `dl-set-ingest-token` | Rotate the ingest token hash on an existing deployment |

**Catalog ingest**

| Command | Purpose |
|---------|---------|
| `dl-ingest-catalog` | Single FITS/CSV/Parquet/VOTable → HATS-partitioned Parquet; required: `--survey`, `--link-id-col` (source integer ID column), `--ra-col`, `--dec-col`; `--defer-finalize`/`--lifecycle live` for incremental live surveys |
| `dl-ingest-catalog-batch` | Parallel decode, single-thread writer (large file lists) |
| `dl-ingest-catalog-from-list` | Sequential (default) or parallel catalog file-list ingest |
| `dl-finalize-catalog` | Rebuild `catalog_info.json` + `_metadata` from tiles (no re-ingest) |
| `dl-repair-catalog-metadata` | Repair/rebuild link IDs, `--check-only`, `--rebuild-link-id` |
| `dl-recommend-catalog-norder` | Sample a catalog to suggest a good `--norder` |
| `dl-check-fits-table-format` | Header-only FITS layout probe (standard vs packed-vector; est. source count) |

**Spectra ingest**

| Command | Purpose |
|---------|---------|
| `dl-ingest-spectra` | Single spectrum FITS → Zarr (all supported formats) |
| `dl-ingest-spectra-batch-desi-coadds` | Multi-process DESI coadd batch |
| `dl-ingest-spectra-batch-spplate` | Multi-process spPlate batch ingest |
| `dl-ingest-spectra-from-list` | Spectrum file-list ingest (sequential default; `--n-workers > 1` for parallel decode) |
| `dl-rebuild-catalog-indices` | Backfill `_spectrum_index` / `_cutout_index` in Parquet tiles; `--kind spectrum\|cutout`, `--n-workers N` for parallel catalog patch; progress bars are on by default (`--no-progress` to disable) |

**Cutout ingest**

| Command | Purpose |
|---------|---------|
| `dl-ingest-cutouts` | Single cutout FITS → Zarr |
| `dl-ingest-cutouts-from-list` | Sequential cutout file-list ingest |
| `dl-generate-cutout-fits` | Generate per-object stamp FITS from band images + catalog |

**Validate and repair**

| Command | Purpose |
|---------|---------|
| `dl-validate-catalog-ingest` | Check Parquet tiles for required columns and schema (`--survey` or `--all`) |
| `dl-validate-spectra-ingest` | Check Zarr spectrum tiles (`--survey` or `--all`) |
| `dl-validate-cutout-ingest` | Check Zarr cutout tiles (`--survey` or `--all`) |
| `dl-validate-catalog-spectra-link` | Verify `_spectrum_index` ↔ Zarr `_source_id` agreement (`--survey` or `--all`); `--n-workers N` for parallel Zarr scan; progress on by default (`--no-progress` to disable) |
| `dl-widen-spectrum-tiles` | Pad narrower `Npix=*.zarr` tiles to survey `n_pix` in `spectrum_info.json` (`--survey` or `--all`) |

**Export and extract**

| Command | Purpose |
|---------|---------|
| `dl-extract-spectra-subset` | Export a curated ID list → flat Zarr / Parquet / HDF5 / FITS; progress on by default (`--no-progress` to disable) |
| `dl-extract-catalog` | Project catalog columns → Parquet / CSV / FITS / VOTable; `--all-columns` for full schema; progress on by default (`--no-progress` to disable) |
| `dl-extract-spplate-catalog` | Build a specObj-style catalog from spPlate files |
| `dl-pack-tile` | Package one HEALPix tile as a `.tar` for sharing |

**Discovery and inventory**

| Command | Purpose |
|---------|---------|
| `dl-describe-lake` | Print survey × modality summary from registry; `--count-total`, `--modality`, `--kind ingested\|product\|crossmatch`, `--areas`, `--refresh`, `--verbose`, `--pair-surveys` |
| `dl-describe-survey` | Column manifest for one survey layer; `--modality`, `--role`, `--rebuild` |
| `dl-describe-master` | Show master association columns mapped to catalog schemas |
| `dl-refresh-lake-registry` | Scan lake, write `shared/registry/surveys.parquet`, rebuild tile indices |
| `dl-region` | Discover survey × modality data in a region (`--from-area`/`--cone`/`--bbox`/`--npix`/`--moc`); `--modalities` may include `crossmatch`; rounded counts, `--count` exact for catalog/crossmatch, `--save-as` area, `--export-moc`, `--moc-format fits\|json\|ascii` ([docs](discovery/regions-and-areas.md)) |
| `dl-area` | Manage `areas/<id>.json`: `list`, `show`, `validate`, `set-crossmatch`, `set-gather`, `set-homogenize`, `import` ([docs](discovery/areas-cli.md)) |
| `dl-export-moc` | Write survey tile footprint (or region-clipped) as IVOA MOC FITS/JSON/ASCII; `--moc-order` ([docs](discovery/regions-and-areas.md#export-as-moc)) |
| `dl-gather` | Materialise a derived product catalog joining base × partners over a selection; `--from-area`, `--matches-only` (default: keep all base rows), partner cache flags, nearest/`--multiplicity all`, `--where-joined`, `--extract-modalities`; progress on by default (`--no-progress`) ([docs](discovery/gather.md)) |
| **Workflow** | End-to-end region → ML export: [discovery/workflow.md](discovery/workflow.md), [notebook](../notebooks/14_discovery_workflow.ipynb) |
| `dl-build-query-from-master` | Generate DuckDB SQL from a master association file |
| `dl-crossmatch` | Catalog↔catalog match at lake scale; sky mode (default, `--radius-arcsec`) or column equality (`--match-mode column --match-col-a COL --match-col-b COL`); `--from-area`/`--plan` for region-bounded sky plans; progress on by default (`--no-progress`) |
| `dl-debug-specobj-lookup` | Diagnose SDSS specObj fiber-to-ID mapping issues |

**Visualisation**

| Command | Purpose |
|---------|---------|
| `dl-plot-source` | Two-panel SED + 1D spectrum figure for one source (`--id`, `--product`, `--spectra-survey`, `-o`); requires `--extra viz` ([docs](plotting.md)) |

**Agents (MCP)**

| Command | Purpose |
|---------|---------|
| `dl-homogenize` | Apply transform registry to a survey+region → homogenized product catalog; progress on by default (`--no-progress`) ([docs](homogenization.md)) |
| `dl-validate-homogenization` | Lint transform registry, golden spot checks, homogenized product validation |
| `dl-mcp-docs` | Stdio MCP server: search docs, describe lake/survey (requires `--extra mcp`) |
| `dl-mcp-lake` | Stdio MCP server: region discovery, provenance, query builder, QA, ingest advisor ([docs](mcp-lake.md)) |

