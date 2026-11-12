# Repository layout

Where to look in the source tree. Each `dl-*` command is a thin CLI over one of these modules; see [cli-reference.md](../cli-reference.md) for the command list.

```
data_lake/
  config.py, knobs.py       Lake config resolution (lake_config.toml) and tunable defaults
  lake_registry.py          Survey × modality registry (dl-describe-lake, dl-refresh-lake-registry, dl-describe-master)
  schema_registry.py        Column manifests (dl-describe-survey); schema read from _common_metadata
  query_from_master.py      SQL from a master association file (dl-build-query-from-master)
  mcp_server.py, mcp_lake_server.py   MCP servers for agents (dl-mcp-docs, dl-mcp-lake)

  ingest/                   Raw files → lake
    fits_to_parquet.py        Catalog → HATS Parquet tiles; finalize writes _common_metadata + _metadata
    catalog_ingest_cli.py     dl-ingest-catalog / dl-finalize-catalog
    catalog_parallel_ingest.py  dl-ingest-catalog-batch (parallel decode, single writer)
    catalog_shard_ingest.py   dl-ingest-from-shard (map → per-tile parts, reduce → tiles)
    catalog_mef_ingest.py     dl-ingest-catalog-mef (row-aligned FITS extensions joined)
    fits_to_zarr.py           Cutout FITS → sharded Zarr v3
    fits_to_spectra_zarr.py   1-D spectrum FITS → sharded Zarr v3
    spectra_ingest_cli.py     dl-ingest-spectra
    desi_parallel_ingest.py, spplate_parallel_ingest.py, spectra_parallel_ingest.py
                              Multi-process spectrum batch ingest
    update_catalog_indices.py Write _spectrum_index / _cutout_index back into catalog tiles
    validate_*.py, repair_catalog_metadata.py, widen_spectrum_tiles.py
                              Validation and repair commands

  io/                       Reading the lake
    catalog.py                CatalogAccessor: DuckDB over Parquet tiles (SQL, cone, ID lookups)
    cutouts.py, spectra.py    Zarr accessors; locate() resolves many IDs at once, get_*_at() reads by position
    id_lookup.py              _source_id → (tile, row) resolution shared by the accessors
    crossmatch.py             Tiled sky / column matching library
    crossmatch_cli.py         dl-crossmatch / dl-describe-crossmatch
    crossmatch_matchers.py    Matcher backends (astropy on CPU, RAPIDS cuML on GPU)
    crossmatch_b_cache.py     Per-worker LRU of partner (B) tiles
    duckdb_limits.py          Per-connection DuckDB memory/thread caps and footer cache

  discovery/                Regions, areas and derived products
    region.py, engine.py, region_cli.py   Region selection and dl-region
    areas.py, area_cli.py     Area JSON (dl-area)
    gather.py, gather_cli.py  dl-gather: base × partners → joined product catalog
    partner_tile_cache.py     Whole-tile LRU used by dl-gather for partner lookups
    moc_export.py             dl-export-moc
    ds9_region.py             dl-export-ds9-region

  homogenize/               dl-homogenize: transform packs, bandpasses, per-survey recipes
  export/                   dl-extract-catalog, spectra/cutout subsets, FITS writers
  ml/                       PyTorch datasets (cutouts, spectra) with tile-ordered sampling
  plot/                     dl-plot-source, dl-plot-spectrum
  share/                    dl-pack-tile
  admin/                    dl-init, dl-set-ingest-token
  cutouts/                  dl-generate-cutout-fits

notebooks/                  Guided walk-throughs; see docs/notebooks.md
examples/                   Self-contained demos (synthetic cross-survey lake, area JSON, survey scripts)
scripts/                    Dev/ops helpers (Click CLIs; not installed as ``dl-*`` entry points)
tests/                      pytest suite; needs no external data
docs/                       This documentation
```

