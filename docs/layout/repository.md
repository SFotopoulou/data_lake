# Repository layout

```
data_lake/
  ingest/
    fits_to_parquet.py        FITS/VOTable → HATS-partitioned Parquet (dl-ingest-catalog)
    catalog_parallel_ingest.py  Parallel decode, single-thread writer (dl-ingest-catalog-batch)
    fits_to_zarr.py           FITS cutouts → Zarr v3 sharded stacks
    fits_to_spectra_zarr.py   FITS 1-D spectra → Zarr v3 sharded stacks
    desi_parallel_ingest.py   Multi-process DESI coadd batch ingest
    spectra_parallel_ingest.py  Parallel file-list spectrum ingest (non-DESI)
    update_catalog_indices.py Patch _cutout_index / _spectrum_index in Parquet tiles; dl-rebuild-catalog-indices backfill CLI
  io/
    catalog.py           DuckDB-backed Parquet accessor
    cutouts.py           Zarr cutout accessor (O(1) source_id lookup)
    spectra.py           Zarr spectrum accessor (O(1) source_id lookup)
    crossmatch.py        Build & query precomputed cross-match catalogs
  ml/
    dataset.py           PyTorch Dataset / IterableDataset (cutouts)
    spectrum_dataset.py  PyTorch Dataset / IterableDataset (spectra) + transforms
  homogenize/
    bandpass.json        Per-band effective wavelength + FWHM registry (all phot_ab_* bands)
    bandpass.py          BandpassRegistry: lake-override resolution + ECSV curve loading
    bandpasses/          Bundled transmission-curve ECSV files (one per band; illustrative)
    surveys/             Per-survey homogenization recipes (<SURVEY>.json)
    transforms/          Transform-pack contracts (phot_ab_v1.json, etc.)
    registry.py          load_homogenize_transforms / load_bandpass_metadata helpers
  plot/
    sed.py               Assemble SEDPoints from a homogenized product catalog row
    source_figure.py     Two-panel SED + 1D spectrum matplotlib figure (lazy import)
    plot_cli.py          dl-plot-source CLI entry point
  discovery/
    gather.py / gather_cli.py     Product catalog materialisation (dl-gather)
    areas.py / area_cli.py        Area JSON management (dl-area)
    engine.py                     Region query engine (dl-region)
    moc_export.py / moc_export_cli.py  MOC export (dl-export-moc)
  admin/
    init_lake.py         dl-init: scaffold a new lake deployment
    set_ingest_token.py  dl-set-ingest-token
  bench/
    __main__.py          Benchmarking entry point
  cutouts/
    generate_fits.py     dl-generate-cutout-fits: generate per-object stamps from band images
  share/
    pack_tile.py         Per-tile .tar packaging + MANIFEST.json
  export/
    catalog_extract.py   dl-extract-catalog: project columns → Parquet / CSV / FITS / VOTable
    spectra_calibration.py  Flux calibration helpers for spectrum export
    spectra_subset.py    Curated source-id subset → single flat Zarr group (dl-extract-spectra-subset)
    spplate_catalog.py   dl-extract-spplate-catalog: specObj-style catalog from spPlate files
    to_fits.py           Zarr cutout → standards-compliant FITS export
    to_spectrum_fits.py  Zarr spectrum → 1-D FITS + BINTABLE export
notebooks/
  01_catalog_ingest.ipynb        02_spectrum_workflow.ipynb
  03_cutout_ingest.ipynb         04_ingestion_report.ipynb
  11_duckdb_catalog_query.ipynb  12_visualization.ipynb
  13_pytorch_training_loop.ipynb 14_discovery_workflow.ipynb
examples/
  cross_survey_lsst_desi_euclid/   # synthetic lake + DuckDB join (master + modalities)
  areas/                            # example area JSON files
  imaging/ spectroscopy/ photometry/  # survey-specific example scripts
data/                               # gitignored; populated locally with FITS fixtures for tests
                                    # (see tests/ for the actual committed test assets)
```

