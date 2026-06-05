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

Use ``--strict`` to treat orphan Zarr rows and unpatched catalog warnings as errors.
Override partitioning only when needed: ``--norder 1`` (must match catalog ``hats_order``).

Use ``-q`` / ``--quiet`` on ``dl-validate-catalog-spectra-link`` when a survey has
many orphan spectra — summary counts are still printed without per-row ``WARNING`` lines.

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

