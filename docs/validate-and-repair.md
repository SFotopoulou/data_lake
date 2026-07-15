# Validate and repair

#### Validate Parquet / Zarr survey directories

```bash
dl-validate-catalog-ingest --survey des_dr2
dl-validate-cutout-ingest --survey des_dr2
dl-validate-spectra-ingest --survey desi_edr
# Or validate every discovered survey in that modality:
dl-validate-spectra-ingest --all
dl-validate-catalog-spectra-link --all
```

These accept ``--file-list``, ``--checkpoint``, ``--inflight``, ``--max-tiles``, and ``--strict`` (same semantics as the spectrum validator).

#### Verify catalog ↔ spectra linkage

After spectrum ingest (with ``--update-catalog``, the default), each catalog row
should carry ``_source_id``, ``_healpix_norder{N}``, and ``_spectrum_index`` pointing
at the matching row in the paired ``Npix=*.zarr`` tile.  Cross-check with:

```bash
# 1. Zarr internal layout (flux/ivar/mask/_source_id shapes)
dl-validate-spectra-ingest --survey zCOSMOS_DR3

# 2. Catalog row index ↔ Zarr _source_id agreement (per HEALPix tile)
dl-validate-catalog-spectra-link --survey zCOSMOS_DR3

# Quick smoke: first tile only, sample 100 linked rows per tile
dl-validate-catalog-spectra-link --survey zCOSMOS_DR3 --max-tiles 1 --sample 100

# Every survey with both catalogs/ and spectra/ trees
dl-validate-catalog-spectra-link --all
```

If step 2 reports **unpatched catalog** (``_spectrum_index=-1`` but Zarr row exists)
or stale indices, rebuild from on-disk Zarr without re-ingesting FITS:

```bash
# --norder defaults to catalog_info.json hats_order (not always 5)
dl-rebuild-catalog-indices --survey zCOSMOS_DR3 --kind spectrum
dl-validate-catalog-spectra-link --survey zCOSMOS_DR3
```

Rebuild sets ``_spectrum_index`` from each Zarr tile's ``_source_id`` array and
clears the index to ``-1`` for catalog rows with no matching spectrum in that tile
(fixes ``out of range`` / ``wrong id`` after Zarr was replaced or shrunk).

At survey scale (millions of spectra), rebuild scans all Zarr tiles once to build
an in-memory ``{source_id: (zarr_npix, local_index)}`` map, then patches each
catalog Parquet tile in a single pass — the same O(n_zarr + n_catalog) pattern as
``dl-validate-catalog-spectra-link``.  Parallelise the catalog patch with
``--n-workers``:

```bash
dl-rebuild-catalog-indices --survey SDSS_DR17 --kind spectrum --n-workers 8
```

Use ``--strict`` to treat orphan Zarr rows and unpatched catalog warnings as errors.
Override partitioning only when needed: ``--norder 1`` (must match catalog ``hats_order``).

Use ``-q`` / ``--quiet`` on ``dl-validate-catalog-spectra-link`` when a survey has
many orphan spectra — summary counts are still printed without per-row ``WARNING`` lines.

#### Performance at survey scale

For surveys with ``_spectrum_npix`` (the post-v0.2 catalog format used by SDSS_DR17
and later), ``dl-validate-catalog-spectra-link`` builds a catalog link index in a
single Parquet pass before validating any Zarr tiles.  This eliminates the previous
O(n_zarr × n_catalog) read pattern and reduces runtime for 6.5 M spectra / 5.8 M
catalog rows from several hours to a few minutes.

The remaining cost is proportional to the number of Zarr tiles (one shard open per
tile).  Parallelise with ``--n-workers``:

```bash
# Full validation with 8 parallel Zarr-tile workers + tqdm progress bars
dl-validate-catalog-spectra-link --survey SDSS_DR17 --n-workers 8

# Smoke test while waiting for the fix — first 50 tiles, sample 200 rows each
dl-validate-catalog-spectra-link --survey SDSS_DR17 -q --max-tiles 50 --sample 200
```

``--n-workers`` defaults to 1 (serial).  A good starting value for large surveys is
``$(nproc) - 1``.  The catalog index is built once in the main process and inherited
by workers via fork, so memory overhead is proportional to the catalog size (roughly
50–150 MB for 5 M rows at three int64 columns).

#### Widen spectrum tiles

Pad narrower Zarr tiles to the survey ``n_pix`` in ``spectrum_info.json``:

```bash
dl-widen-spectrum-tiles --survey my_survey
dl-widen-spectrum-tiles --all --dry-run
```

#### Repair catalog metadata

```bash
dl-repair-catalog-metadata /data/lake --survey my_survey --check-only
dl-repair-catalog-metadata /data/lake --survey my_survey --rebuild-link-id TARGETID
```

