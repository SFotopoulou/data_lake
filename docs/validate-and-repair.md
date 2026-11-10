# Validate and repair

Use these commands after an ingest to check that the catalog Parquet tiles and
the spectrum and cutout Zarr stores are internally consistent and correctly
linked to each other, and to fix common problems without re-ingesting the
original FITS files.

## Validate Parquet / Zarr survey directories

Each modality has its own validator. Name surveys with `--survey` (repeatable)
or check every survey of that modality with `--all`:

```bash
dl-validate-catalog-ingest --survey des_dr2
dl-validate-cutout-ingest --survey des_dr2
dl-validate-spectra-ingest --survey desi_edr
# Or validate every discovered survey in that modality:
dl-validate-spectra-ingest --all
dl-validate-catalog-spectra-link --all
```

`dl-validate-catalog-ingest`, `dl-validate-cutout-ingest` and
`dl-validate-spectra-ingest` share these options:

| Flag | Meaning |
|------|---------|
| `--file-list` | Ingest file list; warn about entries missing from the checkpoint |
| `--checkpoint` | Override the checkpoint path (default `<survey>/.ingest_checkpoint.json`) |
| `--inflight` | Override the in-flight journal path (default `<survey>/.ingest_inflight.json`) |
| `--max-tiles N` | Check only the first N tiles (sorted path order) |
| `--strict` | Treat warnings (for example a stale in-flight journal) as errors |

## Verify catalog ↔ spectra linkage

After a spectrum ingest (with `--update-catalog`, the default), every catalog
row that has a spectrum carries `_source_id`, `_healpix_norder{N}` and
`_spectrum_index`, which points at the matching row in the paired
`Npix=*.zarr` spectrum tile. Check this in two steps:

```bash
# 1. Zarr internal layout (flux/ivar/mask/_source_id shapes)
dl-validate-spectra-ingest --survey zCOSMOS_DR3

# 2. Catalog row index ↔ Zarr _source_id agreement (per HEALPix tile)
dl-validate-catalog-spectra-link --survey zCOSMOS_DR3

# Quick smoke test: first tile only, sample 100 linked rows per tile
dl-validate-catalog-spectra-link --survey zCOSMOS_DR3 --max-tiles 1 --sample 100

# Every survey with both catalogs/ and spectra/ trees
dl-validate-catalog-spectra-link --all
```

### Fix stale or missing indices

If step 2 reports an **unpatched catalog** (`_spectrum_index = -1` although the
Zarr row exists) or stale indices, rebuild the index from the Zarr tiles on
disk; no FITS re-ingestion is needed:

```bash
# --norder defaults to catalog_info.json hats_order (not always 5)
dl-rebuild-catalog-indices --survey zCOSMOS_DR3 --kind spectrum
dl-validate-catalog-spectra-link --survey zCOSMOS_DR3
```

The rebuild sets `_spectrum_index` from each Zarr tile's `_source_id` array and
resets it to `-1` for catalog rows with no matching spectrum. This fixes
`out of range` and `wrong id` reports after Zarr tiles were replaced or
shrunk. Use `--kind cutout` to rebuild `_cutout_index` the same way.

At survey scale (millions of spectra) the rebuild reads every Zarr tile once to
build an in-memory map from source ID to tile and row, then patches each
catalog tile in a single pass. Parallelise the catalog patch with
`--n-workers`:

```bash
dl-rebuild-catalog-indices --survey SDSS_DR17 --kind spectrum --n-workers 8
```

Override the partitioning only when needed: `--norder 1` (must match the
catalog's `hats_order`).

### Linkage validator options

| Flag | Meaning |
|------|---------|
| `--max-tiles N` | Validate only the first N Zarr tiles (sorted path order) |
| `--sample N` | Check at most N random linked rows per tile (`--seed`, default 0) |
| `--strict` | Treat orphan Zarr rows and unpatched-catalog warnings as errors |
| `-q` / `--quiet` | Print summary counts only, without a `WARNING` line per row; useful when a survey has many orphan spectra |
| `--n-workers N` | Parallel worker processes for Zarr tiles (default 1) |
| `--norder`, `--link-id-col` | Override the HEALPix order or catalog join column |

### Performance at survey scale

When the catalog has the `_spectrum_npix` column (SDSS_DR17 and later
ingests), `dl-validate-catalog-spectra-link` first reads the catalog once to
build a link index, then checks each Zarr tile against that index instead of
re-reading the catalog per tile. For about 6.5 million spectra and 5.8 million
catalog rows this takes minutes rather than hours. Catalogs without
`_spectrum_npix` use a slower per-tile path.

The remaining cost is one shard open per Zarr tile, so parallelise with
`--n-workers`:

```bash
# Full validation with 8 parallel Zarr-tile workers + tqdm progress bars
dl-validate-catalog-spectra-link --survey SDSS_DR17 --n-workers 8

# Smoke test: first 50 tiles, sample 200 rows each, summary only
dl-validate-catalog-spectra-link --survey SDSS_DR17 -q --max-tiles 50 --sample 200
```

`--n-workers` defaults to 1 (serial); `$(nproc) - 1` is a good starting value
for large surveys.

**Memory with `--n-workers`.** The catalog link index is built once in the main
process and handed to each worker through the process-pool initializer
(`dl-rebuild-catalog-indices --n-workers` does the same with its source-ID
map). These commands do not choose a process start method, so Python's
platform default applies. With `fork` (the Linux default up to Python 3.13)
workers start from a copy-on-write image of the parent. With `spawn` (macOS)
or `forkserver` (the Linux default from Python 3.14), each worker receives its
own pickled copy of the index. The index is held as Python sets and
dictionaries keyed by source ID, so it is several times larger than the raw
ID columns. With `spawn` or `forkserver`, budget roughly that size once per
worker plus once for the main process.

## Widen spectrum tiles

Pad Zarr spectrum tiles that are narrower than the survey's `n_pix` in
`spectrum_info.json`:

```bash
dl-widen-spectrum-tiles --survey my_survey
dl-widen-spectrum-tiles --all --dry-run
```

`--dry-run` only reports which tiles would be widened. `--n-pix` sets an
explicit target width, and `--max-tiles N` limits the run to the first N tiles
per survey.

## Repair catalog metadata

`dl-repair-catalog-metadata` checks and fixes the join columns of catalog
tiles:

```bash
dl-repair-catalog-metadata /data/lake --survey my_survey --check-only
dl-repair-catalog-metadata /data/lake --survey my_survey --rebuild-link-id TARGETID
```

| Flag | Meaning |
|------|---------|
| `--check-only` | Report join-column status; exit 1 if any tile is missing `_source_id`. Add `--spectra` / `--cutouts` to include those Zarr tiles |
| `--rebuild-link-id COL` | Recompute `_source_id` from column `COL` on every tile. This resets `_spectrum_index` and `_cutout_index`, so run `dl-rebuild-catalog-indices` afterwards |
| `--allow-incomplete-link-id` | With `--rebuild-link-id`: set `_source_id` to null when parts of a composite key are missing |
| `--normalize-column-names` | Strip leading/trailing whitespace from column names (fixes padded FITS `TTYPE` names such as `' dec'`) |
| `--check-padded-columns` | Report padded column names without rewriting tiles |
| `--survey` (repeatable) / `--all` | Which catalogs to process |

## Rebuild catalog footers without re-ingesting

After a killed batch job, a `--defer-finalize` ingest, or to add `_common_metadata`
to an older catalog (so schema lookups stop parsing the large `_metadata` file):

```bash
dl-finalize-catalog --survey MY_SURVEY
dl-refresh-lake-registry
```

This rewrites `catalog_info.json`, `_common_metadata`, `_metadata` and
`schema_manifest.json` from the tiles already on disk. It does not re-read the
original FITS. See [Data on disk](layout/data-on-disk.md).
