# Performance tuning

Quick reference for ingest, retrieval, and crossmatch throughput in the data lake.

## Retrieval

- Attach a **catalog** when opening `SpectrumAccessor` or `CutoutAccessor` so
  `_spectrum_index` / `_cutout_index` enable bulk DuckDB lookup.
- Use **`get_batch()`** instead of loops over `get_spectrum()` / `get_cutout()`.
- For large extracts, prefer **`dl-extract-spectra-subset`** or
  `extract_subset_to_zarr`.
- Pre-build **crossmatch association** Parquet once; join in DuckDB/Polars rather
  than re-running positional matching in analysis loops.

## Ingest

**First step for FITS catalogs:** **`dl-check-fits-table-format --file-list paths.txt`**
(header-only; reports `standard-bintable` vs `packed-vector` and total estimated
sources). Use the source total to sanity-check ingest; avoid parallel whole-file
batch on **`packed-vector`** (colfits) exports — convert to row-normal FITS or
Parquet first. See [Check FITS layout before ingest](ingest/catalog.md#check-fits-layout-before-ingest).

| Scenario | Flags |
|----------|--------|
| Many small spectrum FITS | `--n-workers 8 --files-per-worker 16 --fits-memmap auto` |
| Gaia / CSV catalog shards | `dl-ingest-catalog-batch --n-workers 8 --files-per-worker 4` |
| One huge catalog FITS | `dl-ingest-catalog --streaming --fits-memmap on` |
| Huge FITS + many cores | add `--streaming-parallel N` with `--tile-mode append` |
| Wide FITS (many columns) | `--columns col1,col2,...` on streaming or parallel catalog |
| Files grouped by directory | `--partition-by-dir` on parallel file-list ingest |
| Many cutout FITS | `dl-ingest-cutouts-from-list --n-workers 4 --files-per-worker 8` |

After batch catalog jobs, run **`dl-finalize-catalog`** so DuckDB predicate
pushdown works. Ensure spectrum/cutout ingest runs with **`--update-catalog`**
(default) so index columns are populated.

See [Batch ingest and checkpoints](ingest/batch-and-checkpoints.md) for FITS
memmap policy and worker guidance on HDD vs NVMe.

## Crossmatch

- Default **`dl-crossmatch`** is tile-parallel on CPU (`astropy` backend).
- Amortize DuckDB startup: **`--tiles-per-worker 8`** with **`--n-workers`** tuned
  to CPU cores.
- GPU: **`--match-backend rapids --n-workers 1 --gpu-id 0`** (requires
  `uv sync --extra rapids`).
- Use a coarser **`--norder-b`** when science allows to shrink survey-B candidates.

## Benchmarks

```bash
python -m data_lake.bench lookup-spectra /data/lake --survey DESI_DR1 --n-ids 5000
python -m data_lake.bench catalog-streaming-columns huge.fits \
  --columns TARGETID,RA,DEC --ra-col RA --dec-col DEC --link-id-col TARGETID
```

Output is JSON lines suitable for CI regression tracking (compare `elapsed_s` /
`ids_per_s` with tolerance).

## Storage

Keep hot lake trees on **local SSD**. On NFS/Lustre, prefer fewer workers and
higher **`--files-per-worker`** batching; the tools log a warning when
`--n-workers > 8` on likely shared mounts.
