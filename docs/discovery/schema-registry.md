# Schema registry

Every survey layer in the lake has a `schema_manifest.json` that lists its columns, types, and
roles. Use `dl-describe-survey` to browse it when choosing columns for a query, a gather, or a
join to a [master association table](registry.md#master-association-tables).

## Quick start

```bash
dl-describe-survey DESI_DR1                       # catalog columns (default modality)
dl-describe-survey EUCLID_DR1 --role photometry   # only photometry columns
dl-describe-survey DESI_DR1 --modality spectra    # Zarr arrays and per-spectrum metadata
dl-describe-survey ALLWISE --rebuild              # regenerate the manifest from disk
dl-describe-survey DESI_DR1 --json                # full manifest for tooling
```

The lake root is an optional second positional argument (`dl-describe-survey DESI_DR1
/data/lake`), or comes from `$DATA_LAKE_CONFIG`. Derived products under `products/` are found
by name the same way as ingested catalogs.

## Options

| Flag | Meaning |
|------|---------|
| `--modality` | `catalog` (default, Parquet), `spectra` (Zarr), or `cutout` (Zarr). |
| `--role ROLE` | Show only columns with this role. |
| `--limit N` | Print at most `N` columns. |
| `--rebuild` | Regenerate `schema_manifest.json` from on-disk data before printing. |
| `--json` | Print the full manifest as JSON. |

## What a catalog manifest contains

`catalogs/<survey>/schema_manifest.json` records:

- every column name, Arrow dtype, and nullability;
- a heuristic **role** per column: `id`, `sky`, `redshift`, `healpix`, `index`,
  `classification`, `photometry`, or `science` (the fallback);
- the join ID column (`link_id_column`), sky columns, redshift column, and `link_id_mode`;
- **column groups**: columns sharing a prefix, for example `MAG` for `MAG_G`, `MAG_R`;
- `total_rows` and the HEALPix order.

Spectra and cutout manifests list Zarr array names, dtypes, and roles instead (`flux`, `ivar`,
`mask`, `wavelength`, `image`, `metadata`, …); see [Lake registry](registry.md#spectra-and-cutout-manifests).

Keep master association files ID-only; science columns come from these per-survey catalogs.

## When manifests are written

Catalog finalize (`dl-finalize-catalog`, which also runs at the end of every non-deferred
ingest) writes `schema_manifest.json` together with `_common_metadata`, `_metadata`, and
`catalog_info.json`. Spectra and cutout manifests are written when their ingest finalizes.

If a manifest is missing, `dl-describe-survey` builds it on first use. Use `--rebuild` to
refresh one after the data changed.

## Where the schema is read from

Catalog schemas are read from the cheapest file available under the catalog root
(`catalog_schema_source` in `data_lake/schema_registry.py`):

1. `_common_metadata`: schema only, a few KiB. Written by `dl-finalize-catalog`.
2. One `Npix=*.parquet` tile, for older catalogs without `_common_metadata`.
3. The aggregate `_metadata` footer as a last resort. It holds every tile's row groups and can
   be hundreds of MiB for wide catalogs with many tiles.

If an older catalog lacks `_common_metadata`, run `dl-finalize-catalog` on it once to add it.
Joined products written by `dl-gather` have tiles and `catalog_info.json` only, so their schema
is read from a tile.
