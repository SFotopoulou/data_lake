# Performance tuning

This page collects the settings that most affect speed and memory when you
ingest, read, crossmatch or gather data, and says when to change them. Start
with the section for the step that is slow; the defaults suit most single-node
runs.

## Operational knobs (`data_lake.knobs`)

Shared defaults live in [`data_lake/knobs.py`](../data_lake/knobs.py), and
command-line tools and libraries read them from there. They cannot be set in
`lake_config.toml`; change them by passing the matching CLI flag, the listed
environment variable, or by editing the module.

| Knob | Default | Used by |
|------|---------|---------|
| `READ_MAX_OPEN_TILES` | 8 | `SpectrumAccessor` / `CutoutAccessor` (extract and query) |
| `WRITE_MAX_OPEN_TILES` | 64 | Parallel ingest writers (`--max-open-tiles`) |
| `PARTNER_TILE_CACHE_MAX_TILES` / `_BYTES` | 48 / 512 MiB | `dl-crossmatch` survey-B cache (sky and column modes); `dl-gather` partner cache |
| `DUCKDB_WORKER_MEMORY_FRACTION` | 0.25 | Share of each crossmatch worker's memory given to DuckDB |
| `CATALOG_PARQUET_METADATA_CACHE` | `True` | `CatalogAccessor` footer cache (env `DATA_LAKE_PARQUET_METADATA_CACHE`) |
| `SPECTRA_CHUNKS_PER_SHARD` | 512 | Spectrum Zarr shards; lake `partitioning.chunks_per_shard` |
| `CUTOUT_CHUNKS_PER_SHARD` | 1024 | Cutout image-stack Zarr shards |
| `ZSTD_INGEST` / `ZSTD_GATHER` / `ZSTD_COMPACT` | 3 / 7 / 9 | Catalog and crossmatch tiles; gather products; `compact()` |
| `DEFAULT_FITS_CHUNK_ROWS` | 50 000 | Catalog-FITS export chunking (`--fits-chunk-rows`) |
| `MAX_PARALLEL_WORKERS_CAP` | 64 | Homogenize worker-pool ceiling |
| `CROSSMATCH_DENSE_TILE_WARN` | 50 000 | `dl-crossmatch` warns when one survey-A tile has more sources than this |
| `DEFAULT_HATS_ORDER` | 5 | Fallback HEALPix order when catalog/spectrum info lacks one |

Environment variables read at run time:

| Variable | Effect |
|----------|--------|
| `DATA_LAKE_PARQUET_METADATA_CACHE` | `0`/`false`/`no`/`off` or `1`/`true`/`yes`/`on`; overrides `CATALOG_PARQUET_METADATA_CACHE` |
| `DATA_LAKE_DUCKDB_THREADS` | DuckDB threads per connection (overrides the per-worker split) |
| `DATA_LAKE_DUCKDB_MEMORY_LIMIT_MIB` | DuckDB memory limit per connection in MiB (overrides the per-worker split) |
| `DATA_LAKE_FITS_MEMMAP` | Default for `--fits-memmap` (`auto`, `on`, `off`) |

Some differences between knobs are deliberate; do not unify them without a
design review. Readers keep few tiles open and writers many. Spectra use
smaller shards than cutouts. ZSTD levels trade speed (ingest) against size
(products). Default ID columns differ by command: catalog ingest uses
`ra`/`dec`, `dl-extract-spectra-subset` uses `--target-id-col TARGETID`, and
`dl-extract-cutout-subset` uses `_source_id`. The module docstring explains
each choice.

### Catalog footer cache

A `CatalogAccessor` opens one in-memory DuckDB connection. With the footer
cache on (the default), DuckDB keeps the parsed Parquet footer of every tile it
has read (`parquet_metadata_cache`), so repeated queries on the same accessor
skip re-parsing them. The cost is roughly one footer per tile read, about the
size of the catalog's `_metadata` file, held for the life of the accessor.

Leave it on for interactive work and for repeated queries. Turn it off for
pipelines that read each tile only once, or when many accessors are open at the
same time:

```python
from data_lake.io.catalog import CatalogAccessor

cat = CatalogAccessor("/data/lake", "DESI_DR1", metadata_cache=False)
```

```bash
export DATA_LAKE_PARQUET_METADATA_CACHE=0   # default for every accessor in this process
```

An explicit `metadata_cache=` argument wins over the environment variable,
which wins over the knob. Crossmatch workers always open their catalogs with
the cache off.

Large ID lists passed to a `CatalogAccessor` lookup are sent to DuckDB as a
registered table in chunks of 1,000,000 IDs, not as a literal `IN (...)` list,
so lookups of millions of IDs stay bounded in memory.

## Retrieval

For more than a handful of sources, resolve all IDs in one call and then read
by position. Attach a `CatalogAccessor` so the lookup can use the catalog's
`_spectrum_index` / `_cutout_index` columns:

```python
cat = CatalogAccessor(lake, survey)
with SpectrumAccessor(lake, survey, catalog_accessor=cat) as acc:
    npix, local = acc.locate(ids)          # -1 where an ID is not in the lake
    for p, i in zip(npix.tolist(), local.tolist()):
        if p >= 0:
            sp = acc.get_spectrum_at(p, i)
```

`CutoutAccessor` works the same way with `get_image_at` / `get_cutout_at`.
`locate()` issues one catalog query per million IDs and falls back to a tile
scan for IDs the catalog cannot resolve. Avoid loops over `get_spectrum(id)` /
`get_cutout(id)`, which look up one ID at a time. See
[Export and sharing](export-and-sharing.md#reading-many-spectra-or-cutouts-in-python).

- For large extracts to disk, use `dl-extract-spectra-subset` /
  `dl-extract-cutout-subset` or `extract_subset_to_zarr`. These, and
  `get_batch()`, close each Zarr tile after reading it when the extract spans
  more tiles than `READ_MAX_OPEN_TILES`, so raising that knob does not speed up
  multi-tile extracts.
- Build crossmatch association tables once and join them in DuckDB or Polars,
  instead of re-running positional matching inside analysis loops.

### Machine-learning datasets

`data_lake.ml.dataset.CutoutDataset` and
`data_lake.ml.spectrum_dataset.SpectrumDataset` resolve every source ID once,
when the dataset is built, and store the tile position of each sample (two
int64 arrays, 16 bytes per sample). PyTorch `DataLoader` workers then read Zarr
directly and never query the catalog. Pass `catalog_accessor=` to make that
one-time lookup use the catalog; the accessor is not kept, so workers do not
share its DuckDB connection.

IDs that are not found are reported in one warning at construction; accessing
such a sample raises `KeyError`, so drop them first (`ds.tile_npix < 0` marks
them). Both datasets default to `norder=5`; pass `norder=` if the survey uses a
different `hats_order`.

`TileSequentialSampler` (for `CutoutDataset`) yields samples grouped by tile,
shuffling the tile order by default, so consecutive reads hit the same Zarr
shard. For a full pass over whole tiles, use the iterable `TileCutoutDataset` /
`TileSpectrumDataset` instead.

```python
from torch.utils.data import DataLoader
from data_lake.io.catalog import CatalogAccessor
from data_lake.ml.dataset import CutoutDataset, TileSequentialSampler


def main():
    cat = CatalogAccessor(lake, "des_dr2")
    ds = CutoutDataset(lake, "des_dr2", source_ids=ids, catalog_accessor=cat)
    loader = DataLoader(ds, batch_size=64, num_workers=4,
                        sampler=TileSequentialSampler(ds))
    for batch in loader:
        ...


if __name__ == "__main__":
    main()
```

With `num_workers > 0`, keep the `if __name__ == "__main__":` guard: on
platforms that start workers with `spawn` or `forkserver`, a script without it
fails at start-up. Each worker opens its own accessor, so expect up to
`READ_MAX_OPEN_TILES` (8) open Zarr tiles per worker.

### Open-tile LRU (file-descriptor policy)

Accessors and writers keep a least-recently-used (LRU) set of open Zarr tiles
to bound open file descriptors.

| Path | Cap | What is held | Tune when |
|------|-----|--------------|-----------|
| Read accessors | `READ_MAX_OPEN_TILES=8` | Open Zarr groups | Random single-ID lookups that revisit a few tiles in one process |
| Parallel ingest writers | `WRITE_MAX_OPEN_TILES=64` (`--max-open-tiles`) | Open Zarr groups + optional `_source_id` sets | Writer thrash or `ulimit -n` errors; prefer sky-local file order, and `--on-duplicate append` when safe |
| Crossmatch / gather partner caches | 48 tiles / 512 MiB | Polars frames in RAM (no Zarr file descriptors) | Crossmatch on HDD/NFS; raise `--b-cache-mib` / `--partner-cache-mb` |

Do not raise the read LRU for large extracts; they already stream one tile at
a time.

## Ingest

For FITS catalogs, first run `dl-check-fits-table-format --file-list paths.txt`.
It reads headers only, reports `standard-bintable` or `packed-vector` per file,
and estimates the total number of sources, which you can use to sanity-check
the ingest. Do not run parallel whole-file batch ingest on `packed-vector`
(colfits) exports; convert them to row-normal FITS or Parquet first. See
[Check FITS layout before ingest](ingest/catalog.md#check-fits-layout-before-ingest).

| Scenario | Flags |
|----------|--------|
| Many small spectrum FITS | `--n-workers 8 --files-per-worker 16 --fits-memmap auto` |
| Gaia / CSV catalog shards | `dl-ingest-catalog-batch --n-workers 8 --files-per-worker 4` |
| One huge catalog FITS | `dl-ingest-catalog --streaming --fits-memmap on` |
| Huge FITS + many cores | add `--streaming-parallel N` with `--tile-mode append` |
| Wide FITS (many columns) | `--columns col1,col2,...` on streaming or parallel catalog ingest |
| Files grouped by directory | `--partition-by-dir` on parallel file-list ingest |
| Many cutout FITS | `dl-ingest-cutouts-from-list --n-workers 4 --files-per-worker 8` |

Every catalog ingest finishes with `dl-finalize-catalog`, which writes the
`_metadata` and `_common_metadata` footer files that let DuckDB skip tiles
when filtering. Run `dl-finalize-catalog --survey NAME` yourself only after
`--defer-finalize` ingests or a batch job that was killed. Keep spectrum and
cutout ingest on `--update-catalog` (the default) so the catalog index columns
are filled in.

See [Batch ingest and checkpoints](ingest/batch-and-checkpoints.md) for the
FITS memory-mapping policy and worker guidance on HDD vs NVMe.

## Crossmatch

`dl-crossmatch` runs on CPU with the `astropy` backend by default and is
parallel over HEALPix tiles with `--n-workers` (default 1).

```bash
dl-crossmatch DESI_DR1 EUCLID_DR1 /data/lake --radius-arcsec 1.0 \
  --n-workers 16 --strips-per-worker 4 --memory-budget-gb 120
```

**Work strips.** Pending survey-A tiles are put in sky order and cut into
`--n-workers × --strips-per-worker` contiguous strips of roughly equal
estimated cost (survey-A rows plus survey-B rows in the covering tiles). Strips
are handed out heaviest first as workers become free, and each worker keeps its
survey-B tile cache across the strips it processes. `--strips-per-worker`
defaults to 4; raise it if some workers sit idle while others finish (the
summed phase time divided by the number of workers is much smaller than the
wall time). `--tiles-per-worker` only sets how often progress is logged; it
does not change how work is split.

**Memory and threads.** `--memory-budget-gb` is the total memory for the run.
It is split evenly across workers, and each worker gives
`DUCKDB_WORKER_MEMORY_FRACTION` (25 %, at least 256 MiB) of its share to
DuckDB; the rest is left for Polars frames, the survey-B cache and the matcher.
DuckDB threads are set to the available CPUs divided by `--n-workers`. Without
`--memory-budget-gb`, the budget is the smallest of physical RAM, the cgroup
memory limit and the Slurm allocation. `DATA_LAKE_DUCKDB_THREADS` and
`DATA_LAKE_DUCKDB_MEMORY_LIMIT_MIB` override the computed values. If workers
are killed (`BrokenProcessPool`), lower `--n-workers` or `--b-cache-mib`, or
set `--memory-budget-gb` to the memory the job really has.

**Survey-B tile cache.** Each worker keeps whole survey-B tiles in an LRU cache
(`--b-cache-tiles`, default 48; `--b-cache-mib`, default 512 MiB per worker) so
neighbouring survey-A tiles reuse them. `--no-b-cache` turns it off, which is
mainly useful for comparison.

**Worker start-up.** Workers are started with `forkserver` (or `spawn` where
`forkserver` is unavailable), never by forking the parent. If you call
crossmatch from your own Python script, put the call under
`if __name__ == "__main__":`.

**Where the time goes.** Before adding workers or switching to GPU, check the
logged phase timings or the `perf` block in `crossmatch_info.json`:
`t_a_read_s`, `t_b_load_s`, `t_match_s`, `t_write_s`, plus `b_cache_hits` and
`b_cache_misses`. A large `t_b_load_s` points to I/O (try a bigger survey-B
cache); a large `t_match_s` points to compute.

- GPU: `--match-backend rapids --n-workers 1 --gpu-id 0` (requires
  `uv sync --extra rapids`).
- Use a coarser `--norder-b` when the science allows, to shrink the set of
  survey-B candidates.

More detail: [Crossmatch performance tuning](discovery/crossmatch.md#performance-tuning).

## Gather

`dl-gather` reads partner catalogs through a whole-tile LRU cache. Each partner
tile is read from disk once per run (only the requested columns) and filtered
in memory for every base tile that needs it, so neighbouring base tiles do not
re-read the same partner tile. Tiles with no rows are remembered separately and
do not count against the tile limit.

| Flag | Default | Effect |
|------|---------|--------|
| `--partner-cache-mb` | 512 | Maximum cache size in MiB; `0` disables the cache |
| `--partner-cache-tiles` | 48 | Maximum number of cached partner tiles |
| `--no-partner-cache` | off | Disable the cache |

Raise the limits on slow (HDD or network) storage when partner tiles are
large. `--where-joined` filters the joined rows after partner columns are
attached; use `--where` for predicates on base columns only, which is cheaper.
See [Partner tile cache](discovery/gather.md#partner-tile-cache).

## Benchmarks

```bash
python -m data_lake.bench lookup-spectra /data/lake --survey DESI_DR1 --n-ids 5000
python -m data_lake.bench catalog-streaming-columns huge.fits \
  --columns TARGETID,RA,DEC --ra-col RA --dec-col DEC --link-id-col TARGETID
```

Each benchmark prints JSON lines suitable for continuous-integration regression
tracking; compare `elapsed_s` / `ids_per_s` (lookup) or `full_elapsed_s` /
`subset_elapsed_s` (streaming columns) with a tolerance.

## Storage

Keep frequently used lake trees on local SSD. On NFS or Lustre, use fewer
workers and a higher `--files-per-worker`. Parallel catalog and cutout ingest
and `dl-crossmatch` log a warning when `--n-workers` is above 8 on what looks
like shared storage.
