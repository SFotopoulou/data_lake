# CLI quick reference

Every `dl-*` command in one table. Pass `--help` to any command for full flag docs.

**Deploy**

| Command | Purpose |
|---------|---------|
| `dl-init` | Scaffold a new deployment (`lake_config.toml`, dirs, `.gitignore`) |
| `dl-set-ingest-token` | Rotate the ingest token hash on an existing deployment |

**Catalog ingest**

| Command | Purpose |
|---------|---------|
| `dl-ingest-catalog` | Single FITS/CSV/Parquet/VOTable → HATS-partitioned Parquet |
| `dl-ingest-catalog-batch` | Parallel decode, single-thread writer (large file lists) |
| `dl-ingest-catalog-from-list` | Sequential (default) or parallel catalog file-list ingest |
| `dl-finalize-catalog` | Rebuild `catalog_info.json` + `_metadata` from tiles (no re-ingest) |
| `dl-repair-catalog-metadata` | Repair/rebuild link IDs, `--check-only`, `--rebuild-link-id` |
| `dl-recommend-catalog-norder` | Sample a catalog to suggest a good `--norder` |

**Spectra ingest**

| Command | Purpose |
|---------|---------|
| `dl-ingest-spectra` | Single spectrum FITS → Zarr (all supported formats) |
| `dl-ingest-spectra-batch-desi-coadds` | Multi-process DESI coadd batch |
| `dl-ingest-spectra-batch-spplate` | Multi-process spPlate batch ingest |
| `dl-ingest-spectra-from-list` | Spectrum file-list ingest (sequential default; `--n-workers > 1` for parallel decode) |
| `dl-rebuild-catalog-indices` | Backfill `_spectrum_index` / `_cutout_index` in Parquet tiles |

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
| `dl-validate-catalog-spectra-link` | Verify `_spectrum_index` ↔ Zarr `_source_id` agreement (`--survey` or `--all`) |
| `dl-widen-spectrum-tiles` | Pad narrower `Npix=*.zarr` tiles to survey `n_pix` in `spectrum_info.json` (`--survey` or `--all`) |

**Export and extract**

| Command | Purpose |
|---------|---------|
| `dl-extract-spectra-subset` | Export a curated ID list → flat Zarr / Parquet / HDF5 / FITS |
| `dl-extract-catalog` | Project catalog columns → Parquet / CSV / FITS |
| `dl-extract-spplate-catalog` | Build a specObj-style catalog from spPlate files |
| `dl-pack-tile` | Package one HEALPix tile as a `.tar` for sharing |

**Discovery and inventory**

| Command | Purpose |
|---------|---------|
| `dl-describe-lake` | Print survey × modality summary from registry; `--count-total`, `--modality`, `--refresh`, `--verbose`, `--pair-surveys` |
| `dl-describe-survey` | Column manifest for one survey layer; `--modality`, `--role`, `--rebuild` |
| `dl-describe-master` | Show master association columns mapped to catalog schemas |
| `dl-refresh-lake-registry` | Scan lake and write `shared/registry/surveys.parquet` |
| `dl-build-query-from-master` | Generate DuckDB SQL from a master association file |
| `dl-crossmatch` | Positional catalog↔catalog match at lake scale (`--match-backend astropy\|rapids`, `--gpu-id`) |
| `dl-debug-specobj-lookup` | Diagnose SDSS specObj fiber-to-ID mapping issues |

**Agents (MCP)**

| Command | Purpose |
|---------|---------|
| `dl-mcp-docs` | Stdio MCP server: search docs, describe lake/survey (requires `--extra mcp`) |

