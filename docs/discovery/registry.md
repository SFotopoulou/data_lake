# Lake registry and overlays

The lake registry is an index of every survey and modality on disk, so you can see what the
lake holds without walking the directory tree. Use this page to list surveys, get row totals,
map a master association table to catalog ID columns, and add analyst notes (units,
descriptions) to columns.

## Quick start

```bash
dl-refresh-lake-registry      # scan disk, write shared/registry/surveys.parquet
dl-describe-lake              # print the survey × modality summary
```

`dl-describe-lake` builds the registry automatically if `surveys.parquet` does not exist yet.
Both commands take the lake root as an optional positional argument, or read it from
`$DATA_LAKE_CONFIG`.

## `dl-describe-lake` options

```bash
dl-describe-lake --modality catalog       # one modality: catalog, spectra, cutout, crossmatch
dl-describe-lake --kind ingested          # primary ingest (catalogs, spectra, cutouts)
dl-describe-lake --kind product           # derived products (gather, homogenize)
dl-describe-lake --kind crossmatch        # crossmatch trees
dl-describe-lake --count-total            # footer with per-modality totals and grand total
dl-describe-lake --modality catalog --count-total
dl-describe-lake --pair-surveys           # footer: catalog vs spectra hats_order per survey
dl-describe-lake --areas                  # append the list of saved areas
dl-describe-lake --verbose                # wide table columns (not DEBUG logging)
dl-describe-lake --refresh                # rebuild the registry from disk first, then print
dl-describe-lake --json                   # {"entries": [...]}, one per survey × modality
dl-describe-lake --json --count-total     # adds a "summary" object
dl-describe-lake --json --pair-surveys    # adds a "pairing" object
dl-describe-lake -q --log-file describe.log   # quiet stderr; file gets INFO (no -v log flag)
```

`--verbose` on this command means **extra table columns** (path, tiles, ingest sidecars,
meta fields, …). It does **not** turn on DEBUG logging. Unlike most `dl-*` tools, there is
no `-v` log switch here; use `-q` / `--log-file` only (see [Logging](../ingest/logging.md)).

For crossmatch trees specifically:

```bash
dl-describe-crossmatch                             # list every sky and column tree
dl-describe-crossmatch A B --radius-arcsec 1.0     # details of one tree
```

### Row counts and freshness

`--count-total` sums the `total_rows` stored in the registry; it does not rescan tiles. The
registry records counts when it is built, so after new ingests run
`dl-describe-lake --refresh --count-total` for current numbers.

How `total_rows` is filled in when the registry is built:

- **Catalogs:** `total_rows` from `catalog_info.json` (or the schema manifest).
- **Spectra:** the sum of `_source_id` lengths across all `Npix=*.zarr` tiles, the same
  number the ingestion report notebook shows as `n_spectra_in_zarr`.
- **Crossmatch:** recounted from the on-disk Parquet footers, not from
  `crossmatch_info.json`, so the lake-wide total stays correct even if an older
  region-bounded run left a smaller sidecar.

For counts inside one area (including crossmatch tile overlap and exact `--count`), use
[`dl-region --from-area`](regions-and-areas.md) with `--modalities …,crossmatch`.

### Table columns

Besides survey, modality, HEALPix order, column count, and rows, the default table shows:

- **sky:** the `ra_column` / `dec_column` pair;
- **detail:** `link_id_mode` for catalogs, `n_pix` and `wavelength_mode` for spectra, and the
  band × height × width stack for cutouts;
- **id_col:** the join ID column;
- **mf:** `Y` if a `schema_manifest.json` exists, `.` otherwise.

All registry fields (including `link_id_mode`, `native_id_column`, `n_tiles`, ingest
checkpoint flags, `hats_order_match`, and `created_utc`) are in
`shared/registry/surveys.parquet` and in `--json` output.

| Question | Registry field | Also see |
|----------|----------------|----------|
| How are catalog IDs defined? | `link_id_mode`, `native_id_column` | `catalog_info.json`, `dl-describe-survey` |
| Sky columns for joins / crossmatch | `ra_column`, `dec_column` | `dl-extract-catalog`, STILTS |
| Catalog vs spectra HEALPix order | `hats_order_match`, `--pair-surveys` | `dl-validate-catalog-spectra-link` |
| Spectrum pixel width / wavelength layout | `n_pix`, `wavelength_mode` | `spectrum_info.json` |
| Per-spectrum Zarr metadata (redshift, sky provenance, …) | `meta_fields`, `spectrum_sky_meta_fields`, `has_spectrum_sky_meta` | `dl-describe-lake --verbose`; `dl-describe-survey NAME --modality spectra` |
| Is an ingest still running? | `has_ingest_checkpoint`, `has_ingest_inflight` | the survey's `.ingest_checkpoint.json` / `.ingest_inflight.json` |

## Master association tables

A master association table maps your science sample to the IDs of each survey. Keep the
master Parquet thin (IDs and match QA only) and describe its columns in a sidecar
`<master>.meta.json`:

```json
{
  "meta_version": "1",
  "master_parquet": "associations/master_desi_euclid.parquet",
  "primary_survey": "DESI_DR1",
  "partners": [
    {"survey": "DESI_DR1", "master_column": "desi_targetid", "catalog_id_column": "TARGETID"},
    {"survey": "EUCLID_DR1", "master_column": "euclid_source_id", "catalog_id_column": "SOURCE_ID"}
  ]
}
```

Inspect a master and save the mapping:

```bash
dl-describe-master associations/master.parquet
dl-describe-master associations/master.parquet --write-meta   # save guessed .meta.json
```

If `.meta.json` is missing, `dl-describe-master` guesses the mapping by matching master column
names against each survey's `schema_manifest.json`. Pass `--primary-survey NAME` to set the
home survey of the guess, and `--json` for machine-readable output. For catalogs without a
manifest, run `dl-describe-survey <name> --rebuild` first.

## Spectra and cutout manifests

Spectra and cutout layers get a `schema_manifest.json` when ingest finalizes, built from
`spectrum_info.json` / `cutout_info.json`. It lists the Zarr array names, dtypes, and roles
(`flux`, `ivar`, `mask`, `wavelength`, `image`, `metadata`, …). See
[Schema registry](schema-registry.md) for catalog manifests.

```bash
dl-describe-survey DESI_DR1 --modality catalog    # default
dl-describe-survey DESI_DR1 --modality spectra
dl-describe-survey LSST_DR1 --modality cutout
dl-describe-survey ALLWISE --rebuild
```

## Column overlays (optional)

Overlays are analyst-written JSON files under `<lake_root>/shared/registry/overlays/` that add
`unit`, `description`, and homogenization hints (for example the WISE Vega-to-AB offset) to
columns. `dl-describe-survey` merges them at display time, looking first for
`<survey>.<modality>.json` and then `<survey>.json`. The format is described in
[`shared/registry/overlays/README.md`](../../shared/registry/overlays/README.md).

## Discovery workflow

To go from "what is in the lake" to a SQL query:

1. Run `dl-refresh-lake-registry`, then `dl-describe-lake`.
2. Run `dl-describe-master associations/master.parquet` (add `--write-meta` once).
3. Run `dl-describe-survey <partner>` for each catalog, with `--modality spectra` if needed.
4. Join your ID list to the master and catalogs in DuckDB; see
   [DuckDB queries](duckdb-queries.md) and `notebooks/11_duckdb_catalog_query.ipynb` §9–§10.
