# Design decisions

## Schema versioning policy

Parquet handles column add/drop natively.  The following rules apply:

| Change type | Policy |
|---|---|
| Add new column to existing survey | Add in-place with nullable default; regenerate `_metadata` |
| Remove column | Write tombstone `null` column in new files; drop from `_metadata` |
| Rename column | Add new column + deprecate old (keep both for one release cycle) |
| Breaking structural change | Bump `schema_version` in `catalog_info.json`; write new directory `<survey>_v2/` |

The `schema_version` field in `catalog_info.json` is a plain integer starting at `"1"`.

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Catalog format | Parquet v2 + Zstd | No column limit; columnar projection; column stats for pushdown |
| Catalog partitioning | HEALPix HATS (`--norder`, default 5) | Default ~3.4 deg² tiles; see [Data layout on disk](layout/data-on-disk.md) |
| Cutout format | Zarr v3, sharded | Avoids file-per-cutout; sequential shard reads for ML |
| Cutout dtype | float32 | Full science precision; halve to float16 only for ML-only mirrors |
| Spectra format | Zarr v3, sharded (flux/ivar/mask) | Symmetric with cutout layer; shared wavelength saves ~30% space |
| Wavelength mode | shared (default) / per-source | Shared = one array per tile; per-source when grids differ across spectra |
| Spectrum mask | uint8 (default) / uint16 | 8 bits covers SDSS/DESI defaults; bump if >8 flag bits needed |
| Sharing unit | Per-tile .tar (catalog + cutouts + spectra) | Matches partition granularity; already compressed inside |
| ML dataloader | CutoutDataset / SpectrumDataset (map) or Tile* (iterable) | Map-style for random sampling; tile-iterable for full-epoch streaming |

## Which command for this file?

```
Do you have a 1-D spectrum FITS file?
  → dl-ingest-spectra FILE --survey NAME        # try auto-detect first
  → dl-ingest-spectra FILE --survey NAME --fmt <name>   # if detection fails
  → dl-ingest-spectra-from-list file_list.txt --survey NAME  # many files (add --n-workers N for parallel)
  → dl-ingest-spectra-batch-desi-coadds --survey NAME --file-list coadds.txt  # DESI coadds only

Do you have a catalog table? (FITS, CSV, Parquet, VOTable)
  → dl-ingest-catalog cat.fits --survey NAME --ra-col RA --dec-col DEC --link-id-col ID
  → dl-ingest-catalog-from-list list.txt --survey NAME ...   # file list
  → dl-ingest-catalog-batch list.txt --survey NAME ...       # parallel decode

Do you have image cutout stamps?
  → dl-ingest-cutouts stamps.fits --survey NAME --ra-col RA --dec-col DEC

After spectra or cutouts are ingested, patch catalog indices:
  → dl-rebuild-catalog-indices --survey NAME --kind spectrum
  → dl-validate-catalog-spectra-link --survey NAME

Check what is in the lake:
  → dl-describe-lake --count-total
  → dl-describe-survey NAME
```
