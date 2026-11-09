# Performance tuning

Quick reference for ingest, retrieval, and crossmatch throughput in the data lake.

## Operational knobs (`data_lake.knobs`)

Shared defaults live in [`data_lake/knobs.py`](../data_lake/knobs.py). CLIs and
hot paths import from there so magic numbers stay consistent. Lake-config
overrides for these values are a later pass; until then edit the module or
pass the matching CLI flags.

| Knob | Default | Used by |
|------|---------|---------|
| `READ_MAX_OPEN_TILES` | 8 | `SpectrumAccessor` / `CutoutAccessor` (extract & query) |
| `WRITE_MAX_OPEN_TILES` | 64 | Parallel ingest writers (`--max-open-tiles`) |
| `PARTNER_TILE_CACHE_MAX_TILES` / `_BYTES` | 48 / 512 MiB | Sky + column `dl-crossmatch` B-cache; `dl-gather` partner cache |
| `SPECTRA_CHUNKS_PER_SHARD` | 512 | Spectrum Zarr shards; lake `partitioning.chunks_per_shard` |
| `CUTOUT_CHUNKS_PER_SHARD` | 1024 | Cutout image-stack Zarr shards |
| `ZSTD_INGEST` / `ZSTD_GATHER` / `ZSTD_COMPACT` | 3 / 7 / 9 | Catalog/XM ingest; gather products; `compact()` |
| `DEFAULT_FITS_CHUNK_ROWS` | 50 000 | Catalog-FITS export chunking |
| `MAX_PARALLEL_WORKERS_CAP` | 64 | Homogenize pool ceiling |
| `DEFAULT_HATS_ORDER` | 5 | Fallback when catalog/spectrum info lacks order |

**Intentional asymmetries** (do not unify without a design pass): read vs write
open-tile LRUs; spectra vs cutout shard sizes; ZSTD speed vs density tiers;
CLI ID/sky column defaults (`ra`/`dec` on catalog ingest; spectra subset
`--target-id-col TARGETID`; cutout subset `_source_id`). See module docstring.

## Retrieval

- Attach a **catalog** when opening `SpectrumAccessor` or `CutoutAccessor` so
  `_spectrum_index` / `_cutout_index` enable bulk DuckDB lookup.
- Use **`get_batch()`** instead of loops over `get_spectrum()` / `get_cutout()`.
- For large extracts, prefer **`dl-extract-spectra-subset`** or
  `extract_subset_to_zarr`. One-pass extract/`get_batch` paths release Zarr
  tiles after each HEALPix tile when fan-out exceeds `READ_MAX_OPEN_TILES`,
  so raising that knob does not help multi-tile extracts.
- Pre-build **crossmatch association** Parquet once; join in DuckDB/Polars rather
  than re-running positional matching in analysis loops.

### Open-tile LRU (FD policy)

| Path | Cap | What is held | Tune when |
|------|-----|--------------|-----------|
| Read accessors | `READ_MAX_OPEN_TILES=8` | Open Zarr groups | Random single-ID lookups across a few tiles in one process |
| Parallel ingest writers | `WRITE_MAX_OPEN_TILES=64` (`--max-open-tiles`) | Open Zarr groups + optional `_source_id` sets | Writer thrash / `ulimit -n`; prefer sky-local file order + `--on-duplicate append` when safe |
| XM / gather partner caches | 48 tiles / 512 MiB | Polars frames (RAM, not Zarr FDs) | HDD/NFS crossmatch; raise `--b-cache-mib` / `--partner-cache-mb` |

Do **not** bump read LRU for large extracts — those already stream one tile at a
time. Watch **DataLoader `num_workers` × 8** open groups per process.

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
