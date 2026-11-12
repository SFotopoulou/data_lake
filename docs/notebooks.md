# Example notebooks

The `notebooks/` directory holds worked examples for every layer of the lake: ingest, queries, visualisation, ML and the discovery workflow. Use this page to pick a notebook and to check what data and packages it needs before you open it.

## Setup

Run the notebooks in the uv project environment from the [Quick-start](quickstart.md), not with a bare system Python or a `pip install` outside uv. The `dev` extra provides Jupyter, matplotlib and napari, and PyTorch is a core dependency.

```bash
uv sync --extra dev
uv run python -m ipykernel install --user --name=data-lake --display-name "Python (data-lake)"
```

Then select the **Python (data-lake)** kernel. Notebook headers use Python 3.12 and `uv sync --extra dev`. Add `uv sync --extra desi` only if you ingest DESI coadd files (none of the notebooks do that as written).

**Lake path:** most notebooks read `LAKE_ROOT` from `$DATA_LAKE_ROOT` (fallback `/data/lake`). Notebooks `04`, `14`, `15` and `17` resolve a deployment via `$DATA_LAKE_CONFIG` / `lake_config.toml` instead. Edit survey names in the first code cells to match your lake.

**Tiny dry-run lake:** for cutout / catalog query / ML patterns without a full deployment:

```bash
uv run dl-demo-lake /tmp/demo
export DATA_LAKE_ROOT=/tmp/demo/data
# then set SURVEY = "DEMO_PHOT" in notebooks 11–13
```

Only `01` and `03` run without existing data, because they build a temporary demo lake in the notebook itself. Set `USE_TEMP_LAKE = False` to point them at your own FITS inputs.

## Notebook list

| Notebook | What it teaches | Data it needs | Extras |
|----------|-----------------|---------------|--------|
| [`01_catalog_ingest.ipynb`](../notebooks/01_catalog_ingest.ipynb) | Ingest a FITS catalog into HEALPix Parquet, validate it and query with `CatalogAccessor` (lake join key `_source_id`), plus the CLI equivalents | None in demo mode (writes a synthetic FITS to a temp lake). Set `USE_TEMP_LAKE = False` to use your own FITS file | `dev` |
| [`02_spectrum_workflow.ipynb`](../notebooks/02_spectrum_workflow.ipynb) | Ingest spectra, link `_spectrum_index`, plot, transform, `locate` / bulk reads, subset export with catalog redshift, DuckDB-to-spectra, PyTorch datasets, FITS round-trip | A lake at `$DATA_LAKE_ROOT` and an SDSS spectrum FITS file (survey `sdss_dr17`). §6 also uses a DESI target list and `DESI_DR1` / `EUCLID_DR1` catalogs, and writes to `/scratch` | `dev` |
| [`03_cutout_ingest.ipynb`](../notebooks/03_cutout_ingest.ipynb) | Ingest FITS stamps into Zarr cutout stacks, validate them, read with `CutoutAccessor` (`locate` + `get_cutout_at`), patch `_cutout_index`. §6b covers catalog extract then cutouts by `_source_id` or positional match | None in demo mode (two synthetic stamps in a temp lake). Set `USE_TEMP_LAKE = False` to use your own FITS stamps | `dev` |
| [`04_ingestion_report.ipynb`](../notebooks/04_ingestion_report.ipynb) | Summarise what a deployment holds (catalogs, spectra, cutouts, crossmatches) and run optional linkage and crossmatch-orphan checks. Pairs with `dl-describe-lake --count-total` | An existing deployment (`lake_config.toml`). The optional checks use `EUCLID_DR1` × `DESI_DR1` | `dev` |
| [`11_duckdb_catalog_query.ipynb`](../notebooks/11_duckdb_catalog_query.ipynb) | SQL over Parquet catalogs: schema, cone search, Astropy output, multi-survey and crossmatch joins, tile access. Uses `cat.link_id_column` and `catalog_info` RA/Dec names. §8 covers WISE Vega to AB in SQL, §9 a master association table with ID lists, §10 column discovery | A lake at `$DATA_LAKE_ROOT` with a catalog (`des_dr2` by default; `DEMO_PHOT` works for §§1–4). §8 needs `ALLWISE`. §9 needs `associations/master_desi_euclid.parquet` plus `DESI_DR1` and `EUCLID_DR1` | `dev` |
| [`12_visualization.ipynb`](../notebooks/12_visualization.ipynb) | Cutout quick-looks with matplotlib, ZScale/asinh display, RGB composites, browsing a tile, an optional napari viewer, FITS export for DS9/Aladin with WCS round-trip | A cutout survey at `$DATA_LAKE_ROOT` (`des_dr2` by default; `DEMO_PHOT` after `dl-demo-lake`), plus a real `_source_id` and tile `npix` | `dev` |
| [`13_pytorch_training_loop.ipynb`](../notebooks/13_pytorch_training_loop.ipynb) | PyTorch `DataLoader` over Zarr cutouts: `CutoutDataset` resolves IDs once at construction (pass `catalog_accessor=` for a fast lookup), then workers read by tile position; prefer `TileSequentialSampler` over a plain shuffle. `TileCutoutDataset` streams whole tiles | A catalog and cutout survey at `$DATA_LAKE_ROOT` (`des_dr2` by default; `DEMO_PHOT` after `dl-demo-lake`) | `dev` |
| [`14_discovery_workflow.ipynb`](../notebooks/14_discovery_workflow.ipynb) | The end-to-end reference: inventory, area, tile overlap, crossmatch, gather, homogenize (`phot_ab_v1`) and ML export. §8 covers spectra and cutouts by `_source_id` or positional partner. See the [workflow guide](discovery/workflow.md) and [`dl-area`](discovery/areas-cli.md) | A deployment (`$DATA_LAKE_CONFIG`) with `EUCLID_DR1`, `DESI_DR1` and `ALLWISE`, the template `examples/areas/multi_survey_cone.example.json`, and a writable scratch path | `dev` |
| [`15_Photoz_estimation.ipynb`](../notebooks/15_Photoz_estimation.ipynb) | Build a photometric-redshift (photo-z) training catalog: `EUCLID_DR1` bounding box, crossmatch, gather, EUCLID `phot_ab_v1` recipe (lake override), AB homogenization, FITS export with spec-z columns, optional DESI spectra extract | A deployment (`$DATA_LAKE_CONFIG`) with `EUCLID_DR1`. Partners `ALLWISE`, `VIKING_DR4`, `GAIA_DR3_source`, `DESI_DR1` and `SDSS_DR17` are used when ingested. Needs a writable `/scratch/euclid_photoz` (editable) | `dev` |
| [`16_cutout_day1_onboarding.ipynb`](../notebooks/16_cutout_day1_onboarding.ipynb) | First-time cutout modality gates on the synthetic demo lake: inventory, `_cutout_index`, validate, `locate`/`get_cutout_at`, extract subset, optional `CutoutDataset` + `TileSequentialSampler`; writes a friction JSON | None (calls `ensure_demo_lake`). Not for production ingest — see [`docs/talks/2026-11-cutout-real-lake-friction-plan.md`](talks/2026-11-cutout-real-lake-friction-plan.md) | `dev` |
| [`17_region_extract_samp.ipynb`](../notebooks/17_region_extract_samp.ipynb) | Save a bounding-box area (EDFN example), `dl-extract-catalog --from-area` for `DESI_DR1` (`TARGETID`, `TARGET_RA`, `TARGET_DEC`, `Z`), optionally `dl-samp-send-catalog` into a running TOPCAT | A deployment with `DESI_DR1`; writable scratch; TOPCAT open for the SAMP step | `dev` |

## Notes

- The lake join key is always `_source_id` (see `CatalogAccessor.link_id_column`). Native ID columns (`source_id`, `TARGETID`, …) may still exist alongside it — use the join key when feeding datasets, extract CLIs, or cross-survey products.
- Notebook `15` disables its heavy steps by default. Set `RUN_CROSSMATCH`, `RUN_GATHER`, `RUN_HOMOGENIZE`, `RUN_EXPORT` and `RUN_EXTRACT_SPECTRA` to `True` one at a time as each earlier step finishes.
- Notebooks `14` and `15` call the `dl-*` commands through `subprocess`, so the uv environment's `bin/` must be on `PATH` for the kernel.
- DataLoader cells default to `num_workers=0` (load in the kernel). Raise the value in a script under `if __name__ == "__main__":`; in a notebook the cells already run as main.
- For many spectra or cutouts in Python (outside the datasets), call `locate(ids)` once and then `get_spectrum_at` / `get_cutout_at`, instead of `get_spectrum` / `get_cutout` in a loop. See [Performance → Retrieval](performance.md#retrieval).
