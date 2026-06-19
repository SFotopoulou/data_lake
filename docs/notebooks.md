# Example notebooks

All notebooks expect the **uv** project environment from [Quick-start](quickstart.md)
(`.venv` + `uv sync --extra desi --extra dev`, kernel **Python (data-lake)**).
Do not use a bare system Python or an ad-hoc `pip install` outside `uv`.

See `notebooks/` for worked examples:

1. **`01_catalog_ingest.ipynb`** — FITS → HEALPix Parquet ingest, validation, and `CatalogAccessor` queries (self-contained temp lake or your paths)
2. **`02_spectrum_workflow.ipynb`** — Ingest spectra, query, transform, subset export (catalog `Z`), ML loop, FITS export + round-trip
3. **`03_cutout_ingest.ipynb`** — FITS stamps → Zarr cutout stacks, validation, `CutoutAccessor`, optional `_cutout_index` catalog patch
4. **`04_ingestion_report.ipynb`** — Summarise what is on disk under a deployment (`lake_config.toml`); pairs with `dl-describe-lake --count-total`
5. **`11_duckdb_catalog_query.ipynb`** — SQL over Parquet catalogs; §9 master table + ID-list joins. Use `dl-describe-survey <name>` to choose columns before building joins.
6. **`12_visualization.ipynb`** — Matplotlib / Napari cutout visualization + DS9 FITS export
7. **`13_pytorch_training_loop.ipynb`** — PyTorch DataLoader over Zarr cutouts
8. **`14_discovery_workflow.ipynb`** — End-to-end reference: region → crossmatch → gather → homogenize → ML export ([workflow guide](discovery/workflow.md), [`dl-area`](discovery/areas-cli.md); template in `examples/areas/`)
