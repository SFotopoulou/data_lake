# Astronomy Data Lake

A local-first data lake for multi-survey astronomy catalogs, galaxy image cutouts, and 1-D spectra. Data is stored in [HATS](https://hats.readthedocs.io/en/stable/)-partitioned Parquet (catalogs — HATS = HEALPix Adaptive Tiling Scheme) and sharded Zarr v3 (spectra, cutouts). FITS is the ingest/export format; internal storage is Parquet and Zarr.

Each survey lives in its own directory under `catalogs/`, `spectra/`, or `cutouts/` — these three directories are the three **modalities**. Every object carries a `_source_id` integer that links its catalog row to its spectrum and cutout tiles across all modalities.

> **New here?** Start with the zero-data smoke test below, then follow the [Day-1 quickstart](docs/quickstart.md) to initialise your own lake, ingest a catalog, and run your first query.

## Quick start (zero data required)

```bash
# 1. Install uv (https://github.com/astral-sh/uv) and clone
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/SFotopoulou/data_lake.git && cd data_lake

# 2. Create the environment and run the test suite
uv venv --python 3.11 .venv && uv sync --extra dev --extra fitsio
source .venv/bin/activate
pytest                                  # all tests; no external data needed

# 3. Run the self-contained cross-survey demo (synthetic lake, no downloads)
python examples/cross_survey_lsst_desi_euclid/demo.py
```

Once the tests pass, follow [docs/quickstart.md](docs/quickstart.md) to create a real lake and ingest your first survey:

```bash
dl-init /data/lake --ingest-token 'your-secret'
export DATA_LAKE_CONFIG=/data/lake/lake_config.toml
dl-ingest-catalog survey.fits --survey MY_SURVEY \
  --ra-col RA --dec-col DEC --link-id-col TARGETID
dl-describe-lake --count-total
```

> **Have DESI coadds?** See [ingest/spectra.md](docs/ingest/spectra.md) for `dl-ingest-spectra-batch-desi-coadds` and the `dry_run_desi_ingest.py` smoke script.

## Documentation

### Getting started

| Topic | Guide |
|-------|--------|
| Install, deployment, ingest token | [quickstart.md](docs/quickstart.md) |
| All `dl-*` commands | [cli-reference.md](docs/cli-reference.md) |
| Term definitions (HATS, `_source_id`, modality, …) | [glossary.md](docs/glossary.md) |
| `lake_config.toml` | [lake-config.md](docs/lake-config.md) |
| Example notebooks | [notebooks.md](docs/notebooks.md) |
| MCP access for agents | [mcp.md](docs/mcp.md), [mcp-lake.md](docs/mcp-lake.md) |

### Ingest

| Topic | Guide |
|-------|--------|
| Catalog ingest | [ingest/catalog.md](docs/ingest/catalog.md) |
| Cutout ingest | [ingest/cutouts.md](docs/ingest/cutouts.md) |
| Spectrum ingest | [ingest/spectra.md](docs/ingest/spectra.md) |
| Batch, file-list, checkpoints | [ingest/batch-and-checkpoints.md](docs/ingest/batch-and-checkpoints.md) |
| Logging and long runs (`-q`, heartbeat) | [ingest/logging.md](docs/ingest/logging.md) |

### Operations

| Topic | Guide |
|-------|--------|
| Validate and repair | [validate-and-repair.md](docs/validate-and-repair.md) |
| Export and sharing | [export-and-sharing.md](docs/export-and-sharing.md) |
| Troubleshooting | [troubleshooting.md](docs/troubleshooting.md) |

### Discovery and query

| Topic | Guide |
|-------|--------|
| Lake registry | [discovery/registry.md](docs/discovery/registry.md) |
| Schema registry (`dl-describe-survey`) | [discovery/schema-registry.md](docs/discovery/schema-registry.md) |
| Crossmatch and associations | [discovery/crossmatch.md](docs/discovery/crossmatch.md) |
| Regions, areas, `dl-region` | [discovery/regions-and-areas.md](docs/discovery/regions-and-areas.md) |
| Area plans (`dl-area`) | [discovery/areas-cli.md](docs/discovery/areas-cli.md) |
| MOC export (`dl-export-moc`) | [regions-and-areas.md § Export as MOC](docs/discovery/regions-and-areas.md#export-as-moc) |
| Gather products (`dl-gather`) | [discovery/gather.md](docs/discovery/gather.md) |
| **Reference workflow** (region → ML export) | [discovery/workflow.md](docs/discovery/workflow.md) · [notebook](notebooks/14_discovery_workflow.ipynb) |
| Homogenization (`dl-homogenize`) | [homogenization.md](docs/homogenization.md) |
| SED + spectrum plot (`dl-plot-source`) | [plotting.md](docs/plotting.md) |
| Performance tuning | [performance.md](docs/performance.md) |
| DuckDB ID-list queries | [discovery/duckdb-queries.md](docs/discovery/duckdb-queries.md) |

### Layout and design

| Topic | Guide |
|-------|--------|
| Repository layout | [layout/repository.md](docs/layout/repository.md) |
| Data on disk (HEALPix tiles) | [layout/data-on-disk.md](docs/layout/data-on-disk.md) |
| Design decisions | [design-decisions.md](docs/design-decisions.md) |

## Which command for this file?

```
1-D spectrum FITS  → dl-ingest-spectra / dl-ingest-spectra-from-list / dl-ingest-spectra-batch-desi-coadds
Catalog table      → dl-check-fits-table-format (FITS pre-check) → dl-ingest-catalog / dl-ingest-catalog-from-list / dl-ingest-catalog-batch
Image cutouts      → dl-ingest-cutouts / dl-ingest-cutouts-from-list
What's in the lake → dl-describe-lake --count-total
```

See [design-decisions.md](docs/design-decisions.md) for the full decision tree.

## Dependencies

Core: `pyarrow`, `zarr>=3`, `numcodecs`, `duckdb`, `astropy`, `healpy`, `numpy`, `polars`, `torch`, `tqdm`, `click`.

Optional: `desispec` (`uv sync --extra desi`), `fitsio` (`--extra fitsio`), dev/notebooks (`--extra dev`), MCP server (`--extra mcp`), visualisation (`--extra viz`, adds `matplotlib`).

Details: [docs/quickstart.md#dependencies](docs/quickstart.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Documentation edits go in `docs/`; keep this README as a navigation hub.
