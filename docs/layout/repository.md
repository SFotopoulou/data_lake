# Repository layout

```
data_lake/
  ingest/
    fits_to_parquet.py        FITS/VOTable → HATS-partitioned Parquet
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
  plot/
    sed.py               Assemble SEDPoints from a homogenized product catalog row
    source_figure.py     Two-panel SED + 1D spectrum matplotlib figure (lazy import)
    plot_cli.py          dl-plot-source CLI entry point
  share/
    pack_tile.py         Per-tile .tar packaging + MANIFEST.json
  export/
    to_fits.py           Zarr cutout → standards-compliant FITS export
    to_spectrum_fits.py  Zarr spectrum → 1-D FITS + BINTABLE export
    spectra_subset.py    Curated source-id subset → single flat Zarr group
notebooks/
  01_catalog_ingest.ipynb        02_spectrum_workflow.ipynb
  03_cutout_ingest.ipynb         04_ingestion_report.ipynb
  11_duckdb_catalog_query.ipynb  12_visualization.ipynb
  13_pytorch_training_loop.ipynb # see "Example notebooks" below
examples/
  cross_survey_lsst_desi_euclid/   # synthetic lake + DuckDB join (master + modalities)
data/                               # committed FITS fixtures for tests and smoke runs
  389442.fits                       # 2dFGRS spectrum (2 SPECTRUM extensions; SPFILE+FIBRE link)
  154714.fits / 161216.fits         # additional 2dFGRS spectra
  g2302140-251235.fits              # 6dFGS target file (VR extensions; stem+OBSID_V+OBSID_R)
  g1437140-385507.fits / g2259418-254505.fits  # additional 6dFGS files
  G23_Y7_015_265.fit                # GAMA spectrum (SPECID header key)
  OzDES-DR2_00001.fits              # OzDES spectrum (filename link)
  sc_*.fits                         # VIPERS / VUDS / VVDS / VANDELS spectra (filename link)
  VIPERS_406064719.fits             # VIPERS spectrum
  wig225415.fits                    # WiggleZ spectrum (filename link — full name incl. ext)
  spPlate-*.fits                    # SDSS/BOSS spPlate fixtures (2 plates, different n_pix)
  zCOSMOS_BRIGHT_DR3_*.fits         # zCOSMOS spectrum (filename link)
  sdss-specobjid.txt                # specObjID sidecar for spPlate tests
```

