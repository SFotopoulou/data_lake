# Design decisions

This page explains the storage choices behind the lake, how catalog schemas are versioned, and which ingest command to use for a given input file. Read it when choosing an ingest path, or when you need to know why data is laid out the way it is.

Publication figures (TikZ): layered architecture
[`docs/figures/architecture_layers.tex`](figures/architecture_layers.tex) and
end-to-end pipeline
[`docs/figures/pipeline_flowchart.tex`](figures/pipeline_flowchart.tex).

## Which command for this file?

Start with the single-file command for your data type. Switch to the list or batch variant once you have many files.

```
Do you have a 1-D spectrum FITS file?
  → dl-ingest-spectra FILE --survey NAME        # try auto-detect first
  → dl-ingest-spectra FILE --survey NAME --fmt <name>   # if detection fails
  → dl-ingest-spectra-from-list file_list.txt --survey NAME  # many files (add --n-workers N for parallel)
  → dl-ingest-spectra-batch-desi-coadds --survey NAME --file-list coadds.txt  # DESI coadds only
  → dl-ingest-spectra-batch-spplate --survey NAME --file-list plates.txt     # SDSS spPlate files only

Do you have a catalog table? (FITS, CSV, Parquet, VOTable)
  → dl-ingest-catalog cat.fits --survey NAME --ra-col RA --dec-col DEC --link-id-col ID
  → dl-ingest-catalog-from-list list.txt --survey NAME ...   # file list
  → dl-ingest-catalog-batch list.txt --survey NAME ...       # parallel decode
  → dl-ingest-catalog-mef file.fits --survey NAME ...        # row-aligned multi-extension FITS tables

Do you have image cutout stamps?
  → dl-ingest-cutouts stamps.fits --survey NAME --ra-col RA --dec-col DEC --link-id-col ID

Patch catalog indices (only if spectra/cutouts were ingested with --no-update-catalog,
or the catalog was ingested after them):
  → dl-rebuild-catalog-indices --survey NAME --kind spectrum
  → dl-validate-catalog-spectra-link --survey NAME

Check what is in the lake:
  → dl-describe-lake --count-total
  → dl-describe-survey NAME
```

Catalog ingest always requires `--link-id-col`. For cutouts and spectra, `--link-id-col` must name the same object ID as the catalog so that the rows link through `_source_id`. `dl-ingest-spectra` and `dl-ingest-cutouts` write `_spectrum_index` / `_cutout_index` into the catalog by default (`--update-catalog`).

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Catalog format | Parquet v2 + Zstd | No column limit; columnar projection; column statistics for predicate pushdown |
| Catalog partitioning | HEALPix HATS (`--norder`, default 5) | About 3.4 deg² per tile at the default; see [Data layout on disk](layout/data-on-disk.md) |
| Catalog metadata | `_common_metadata` (schema only) + `_metadata` (aggregate footer), written by `dl-finalize-catalog` | Schema reads open one small file instead of a tile |
| Cutout format | Zarr v3, sharded | Avoids one file per cutout; sequential shard reads for ML |
| Cutout dtype | float32 (`--dtype`) | Full science precision; use float16 only for ML-only mirrors |
| Spectra format | Zarr v3, sharded (flux/ivar/mask) | Same layout as the cutout layer; a shared wavelength array saves about 30% space |
| Wavelength mode | shared (default) / per-source | Shared stores one array per tile; use per-source when grids differ between spectra |
| Spectrum mask | uint8 (default) / uint16 | 8 bits cover the SDSS/DESI defaults; use uint16 if you need more than 8 flag bits |
| Sharing unit | Per-tile `.tar` (`dl-pack-tile`: catalog + cutouts + spectra) | Matches the partition granularity; the contents are already compressed |
| ML data loading | `CutoutDataset` / `SpectrumDataset` (map-style) or `TileCutoutDataset` / `TileSpectrumDataset` (iterable) | Map-style for random sampling; tile-iterable for full-epoch streaming |

The map-style ML datasets resolve all source IDs to tile positions once, when they are constructed, and warn about missing IDs at that point. DataLoader workers then read by position and never query the catalog. Use `TileSequentialSampler` to group samples by tile and reduce random reads. Scripts that start worker pools (a DataLoader with `num_workers > 0`, or `dl-crossmatch` from Python) need an `if __name__ == "__main__":` guard.

## Schema versioning policy

Parquet handles adding and dropping columns natively. Schema changes follow these rules:

| Change type | Policy |
|---|---|
| Add new column to existing survey | Add in place with a nullable default; run `dl-finalize-catalog` (rewrites `_common_metadata` and `_metadata`) |
| Remove column | Write a tombstone `null` column in new files; drop it from `_metadata` via `dl-finalize-catalog` |
| Rename column | Add the new column and deprecate the old one (keep both for one release cycle) |
| Breaking structural change | Bump `schema_version` in `catalog_info.json`; write a new directory `<survey>_v2/` |

`schema_version` in `catalog_info.json` is stored as a string holding an integer, starting at `"1"`.

## Trust boundaries

Free-form DuckDB predicates (`--where`, `--where-joined`) and path-built view
DDL are trusted-operator inputs for HPC/lab deployments, not a multi-tenant SQL
API. Details: [security.md](security.md).
