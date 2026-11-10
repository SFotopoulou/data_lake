# Batch ingest, checkpoints, and duplicate policy

This page covers ingesting many input files at once: file lists, parallel catalog batches,
and the two-phase map-reduce ingest for very large FITS (Flexible Image Transport System)
collections. It also explains how re-runs resume from checkpoints and how duplicate object
IDs are handled. Read it when a single `dl-ingest-*` call per file is no longer practical.

Pick the catalog command from how your files are laid out on the sky. Use
`dl-ingest-catalog-batch` when about one file feeds each lake tile, and
`dl-ingest-from-shard` when many small or unsorted files share a tile. The full decision
table is in [Choosing a catalog ingest command](catalog.md#choosing-a-catalog-ingest-command).
Cutouts and spectra use the `-from-list` commands, or the dedicated DESI and spPlate batch
commands.

Every `dl-ingest-*` command needs a deployment config (`--config` or `$DATA_LAKE_CONFIG`)
and the ingest token (`--ingest-token` or `$LAKE_INGEST_TOKEN`).

## Sequential file-list ingest (catalogs, cutouts, spectra)

The simplest batch is a text file with one input path per line, passed to a `-from-list`
command. Each command records finished files in a checkpoint, so re-running the same
command after a crash continues where it stopped.

```bash
# Catalog: several FITS files into one survey
find /data/cats -name '*.fits' > cat_files.txt
dl-ingest-catalog-from-list cat_files.txt --survey my_survey \
  --ra-col RA --dec-col DEC --link-id-col OBJID \
  --tile-mode append

# Cutouts: one stamp FITS per object
find /data/cutouts -name '*.fits' > cutout_files.txt
dl-ingest-cutouts-from-list cutout_files.txt --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID \
  --band-names r,i,z --on-duplicate skip

# DESI coadd spectra (parallel DESI batch command)
dl-ingest-spectra-batch-desi-coadds --survey desi_dr1 --file-list coadds.txt --n-workers 16 \
  --on-duplicate skip

# Generic / SDSS spectra
dl-ingest-spectra-from-list spec_files.txt --survey sdss_dr17 \
  --link-id-col SPECOBJID --on-duplicate skip

# spPlate file lists (same flags as dl-ingest-spectra)
dl-ingest-spectra-from-list spPlate_files.txt --survey boss_dr12 \
  --fmt sdss_spplate --specobj-lookup /path/to/lookup.parquet --on-duplicate skip
```

Replace `OBJID` with your catalog's object ID column; `--link-id-col` is required for
catalog ingest.

Pass `--tile-mode append` when several catalog files can land in the same HEALPix
(Hierarchical Equal Area isoLatitude Pixelization) tile. The sequential default is `skip`,
which leaves an existing tile untouched, so rows from later files that fall into an
already-written tile are dropped.

Spectrum and cutout ingest patch `_spectrum_index` / `_cutout_index` in the catalog when
`--update-catalog` is on (the default).

SDSS spectrum lists often differ slightly in pixel count. `--on-length-mismatch` defaults
to `error`, but for the `sdss_boss` and `sdss_spplate` formats ingest switches `error` to
`pad`, widening existing tiles when a longer spectrum arrives. Pass
`--on-length-mismatch truncate` to cut spectra to the tile length instead (same as
`dl-ingest-spectra`).

**Checkpoints and failures.** Checkpoints default to `.ingest_checkpoint.json` under
`catalogs/<survey>/`, `cutouts/<survey>/` or `spectra/<survey>/` in the lake root; override
with `--checkpoint`, or ignore them with `--no-skip-completed`. `--failures-log PATH` appends
one JSONL (JSON Lines) record per failed file.

## Parallel catalog batch (large file lists)

> Batch append is fast when about one file feeds each lake tile. If many files share a
> tile (bricks, RA/Dec patches, unsorted run folders), use
> [`dl-ingest-from-shard`](#two-phase-map-reduce-ingest-dl-ingest-from-shard) instead; see
> [Choosing a catalog ingest command](catalog.md#choosing-a-catalog-ingest-command).

Use `dl-ingest-catalog-batch` for surveys shipped as many catalog files (for example Gaia
`GaiaSource_*.csv.gz`). Worker processes decode files in parallel, and a single writer
thread merges their rows into the HEALPix tiles, so overlapping tiles are combined safely.

```bash
dl-ingest-catalog-batch gaia_files.txt --survey GAIA_DR3_source \
  --ra-col ra --dec-col dec --link-id-col source_id --norder 5 \
  --tile-mode append --on-duplicate-id skip --n-workers 8 --files-per-worker 4
```

**Before you start**, run `dl-check-fits-table-format --file-list <paths.txt>` on FITS
catalogs. Its summary reports the estimated number of sources and the total on-disk size;
keep it to compare with `dl-describe-survey` after ingest (allowing for rows dropped by
`--on-duplicate-id skip`). Files reported as `packed-vector` (for example STILTS colfits)
are not suited to parallel whole-file decode; convert them to row-normal FITS or Parquet
first. Details: [Check FITS layout before ingest](catalog.md#check-fits-layout-before-ingest).

Row-normal FITS shards in one tree can instead be merged first with
[`concatenate_fits.py`](catalog.md#concatenate-fits-shards-pre-ingest) and ingested as a
single file. At TB scale batch ingest is usually the better choice because it checkpoints
and appends tile by tile.

**Useful options:**

- `--files-per-worker N` sends N files to each worker task (try 4–16 for small CSV shards).
- `--partition-by-dir` orders the queue by parent directory for better disk locality on
  sharded surveys (Gaia run directories, DESI run folders).
- `--max-in-flight` limits how many decoded files are buffered; the default is
  `n_workers`.

`dl-ingest-catalog-from-list` with `--n-workers > 1` uses the same parallel engine and
accepts the same flags (including `--allow-incomplete-link-id` and `--tile-mode`). Only the
sequential path (`--n-workers 1`, the default) supports `--streaming`. When `--tile-mode` is
omitted, the parallel path defaults to `append` and the sequential path to `skip`.

**Finalize.** Every batch exit, including a run where every file was already in the
checkpoint, finalizes the catalog. Finalize writes `catalog_info.json`, `_common_metadata`
(the schema only, a few KiB), `_metadata` (the aggregate Parquet footer) and
`schema_manifest.json`. If a Slurm job is killed mid-run, tiles can exist without this
metadata; run `dl-finalize-catalog --survey <name>` before `dl-refresh-lake-registry`.

**Memory.** Peak RAM scales roughly with `n_workers × largest catalog file`, because each
worker decodes a whole file in memory. Workers spool their tile tables to temporary Parquet
files rather than sending them back to the parent. On wide surveys (for example ALLWISE)
lower `--n-workers` and keep fewer columns with `--columns`. FITS files larger than
`DATA_LAKE_PARALLEL_CATALOG_MAX_BYTES` (default 512 MiB) are rejected with a message to use
sequential `dl-ingest-catalog --streaming`.

If the operating system kills a worker for running out of memory, you may see
`BrokenProcessPool` in the log. The batch tool restarts the pool and re-queues the
in-flight files. If this keeps happening, use fewer workers and a smaller `--columns` set.

The command exits with status 1 if any file failed.

## Two-phase map-reduce ingest (`dl-ingest-from-shard`)

> **Use this instead of `dl-ingest-catalog-batch` when more than ~3 input files feed
> each lake tile**: files smaller than a tile (Legacy Survey bricks, RA/Dec patches,
> pointings), or files not sorted on the sky. This holds for thousands of files, not only
> very large surveys. Decision table and how to estimate files-per-tile:
> [Choosing a catalog ingest command](catalog.md#choosing-a-catalog-ingest-command).

### Why the append path does not scale

`dl-ingest-catalog-batch` and `dl-ingest-catalog-from-list` have a single writer that, for
every source file, reads the existing tile, removes duplicates, concatenates the new rows
and rewrites the tile. At `--norder 4` one HEALPix pixel (about 3.7°) is shared by hundreds
of bricks, so later files rewrite ever-larger tiles. That costs about 20 s per file, or
about 2,000 CPU-hours for 400,000 files. Do not concatenate 400,000 FITS files into one
large file first either: the merge step alone needs TB-scale memory.

### How it works

```
FITS → decode_catalog_file_to_batches → immutable Npix shard parts   (phase map)
Npix shard parts → one concat per Npix → lake tile                   (phase reduce)
```

In the **map** phase each Slurm array task handles one RA folder and writes
`<shard-root>/<survey>/Norder=N/Dir=D/Npix=P/part-<stem>.parquet`. Parts are never
modified once written, so tasks run fully in parallel. A JSONL checkpoint lets a killed
task resume.

In the **reduce** phase each worker owns a separate set of tiles. It reads all parts for a
tile once, removes duplicates according to `--on-duplicate-id`, and writes the final lake
tile. No tile locking is needed, so reduce scales linearly with workers. After all tiles
succeed, reduce finalizes the catalog (see [Finalize](#parallel-catalog-batch-large-file-lists)).

### Choosing norder

Run `dl-recommend-catalog-norder` on a few sample files with `--norder-max 8`. For
tractor-format bricks (Legacy Survey), `--norder 6` or `7` is usually best: smaller tiles
mean smaller concatenations in reduce and more parallel reduce tasks. If an existing
`norder=4` tree must be kept, ingest under a new survey name.

### Slurm recipe

Reduce writes into the lake given as the `OUTPUT_ROOT` argument, defaulting to `lake.root`
from the config.

```bash
export DATA_LAKE_CONFIG=/path/to/lake_config.toml
export LAKE_INGEST_TOKEN=...   # or --ingest-token

# Prepare: list all RA folder paths (one per array task)
ls -d /data/legacy/north/[0-9]* > ra_folders.txt

# Map array job (one task per RA folder)
#SBATCH --array=0-N%64
#SBATCH --cpus-per-task=8 --mem=32G --time=4:00:00

FOLDER=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" ra_folders.txt)
dl-ingest-from-shard --phase map \
  --input-dir "$FOLDER" \
  --survey Legacy_DR11_north \
  --shard-root $SCRATCH/shards \
  --ra-col ra --dec-col dec --link-id-col ls_id_dr11 \
  --norder 7 --n-workers 8 \
  --no-progress -q \
  --log-file "logs/map_${SLURM_ARRAY_TASK_ID}.log" \
  --failures-log "logs/map_${SLURM_ARRAY_TASK_ID}_failures.jsonl"

# Reduce job (single job, depends on array)
#SBATCH --cpus-per-task=32 --mem=200G --time=1-00:00:00 --dependency=afterok:<array_jobid>

dl-ingest-from-shard /data/lake --phase reduce \
  --survey Legacy_DR11_north \
  --shard-root $SCRATCH/shards \
  --staging-dir "$TMPDIR/dl_stage" \
  --n-workers 32 --on-duplicate-id skip \
  --no-progress -q --log-file logs/reduce.log --failures-log logs/reduce_failures.jsonl
```

If your files are not grouped in folders, sort the full list and split it into chunks
(`sort files.txt | split -l 2000 -d -a 4 - chunks/chunk_`), then pass
`--file-list chunks/chunk_NNNN` to each array task. The ingest advisor
(`recommend_ingest(..., layout="spatial")`) generates both scripts in this form.

### Retrying failures

Collect failed inputs from every map task and re-run map on just those. Finished files are
skipped through the checkpoint, so re-running a whole task is also safe.

```bash
cat logs/map_*_failures.jsonl | jq -r .path | sort -u > retry.txt
dl-ingest-from-shard --phase map --file-list retry.txt --survey Legacy_DR11_north \
  --shard-root $SCRATCH/shards --link-id-col ls_id_dr11 --norder 7
```

Run reduce only after map has no failures. If any tile fails, reduce skips finalize; fix
the cause and re-run reduce. Both phases exit with status 1 if any file or tile failed. A
settings mismatch exits with a one-line error instead of a traceback.

**Expected throughput:** map takes about 0.2–1 s per file (one FITS decode plus one
Parquet write per tile hit), and reduce time grows with the number of tiles, divided across
workers. 400,000 files take hours of array wall-clock time instead of 2,000 CPU-hours of
serial append.

### Adding to an existing survey

Map the new files into a fresh `--shard-root` and reduce with the default
`--tile-mode append`. Each touched tile is read once, merged with the new rows, and
rewritten once. Map records `norder` and `link_id_mode` in
`<shard-root>/<survey>/shard_info.json`. Reduce aborts before writing anything if those
disagree with `--norder` or with the existing survey's `catalog_info.json`, and map aborts
if a later task uses another `--link-id-col` or `--norder` on the same shard tree. Reduce
takes `--norder` from `shard_info.json` when you omit it.

### Small local SSD

Keep `--shard-root` and the lake (`OUTPUT_ROOT`) on large shared storage and pass
`--staging-dir $TMPDIR/dl_stage`. Writes go to the local SSD (solid-state drive) first, and
finished files are moved atomically (a cross-device copy to `<final>.tmp`, then a rename).
Readers never see a partial tile, and the SSD only needs room for the tiles currently in
flight.

### Flags reference

| Flag | Phase | Default | Notes |
|------|-------|---------|-------|
| `OUTPUT_ROOT` (argument) | reduce | config `lake.root` | Lake root; tiles go to `catalogs/<survey>/`. `--lake-root` is a deprecated alias |
| `--config` | both | `$DATA_LAKE_CONFIG` | Deployment config (required) |
| `--ingest-token` | both | `$LAKE_INGEST_TOKEN` | Ingest operator token (required) |
| `--phase` | both | required | `map` or `reduce` |
| `--input-dir` | map | — | Non-recursive directory of `*.fits` (one RA folder) |
| `--file-list` | map | — | Text file, one FITS path per line (relative paths resolve against the list) |
| `--shard-root` | both | required | Root for immutable shard parts |
| `--survey` | both | required | Survey name (also used as lake sub-directory) |
| `--ra-col` / `--dec-col` | both | `ra` / `dec` | Sky coordinate columns |
| `--link-id-col` | map | required | ID column or composite spec (e.g. `objid,brickname`) |
| `--norder` | both | shard tree, then config `hats_order`, then 5 | HEALPix order; must match between map and reduce |
| `--columns` | map | all | Comma-separated subset of columns to keep |
| `--allow-incomplete-link-id` | map | off | Null `_source_id` for rows with missing composite parts |
| `--fits-memmap` | map | `auto` | FITS read policy (`$DATA_LAKE_FITS_MEMMAP`) |
| `--hdu` | map | first table HDU | Table extension: index, `EXTNAME[,EXTVER]`, or `all` (`$DATA_LAKE_FITS_HDU`); recorded in `shard_info.json`, carried to the lake by reduce |
| `--on-duplicate-id` | reduce | `skip` | `skip` keeps the first/existing row, `last` keeps the newest, `error` aborts the tile |
| `--tile-mode` | reduce | `append` | Existing lake tile: `append` merges (one rewrite per tile per reduce run), `skip` leaves it, `overwrite` replaces it |
| `--compact` | reduce | off | Smaller tiles (ZSTD-9, no stats/dictionary) |
| `--n-workers` | both | `4` | Worker processes |
| `--max-in-flight` | both | `2 × n_workers` | Tasks queued to workers at once |
| `--staging-dir` | both | `$DATA_LAKE_STAGING_DIR` | Fast local scratch; each file is written here then moved to the final area (holds ~`n_workers` files; falls back to direct write when space is short) |
| `--checkpoint` | map | `<shard-root>/<survey>/map_checkpoint.jsonl` | JSONL of finished input files |
| `--no-skip-completed` | map | off | Ignore the checkpoint (existing part files are still never rewritten) |
| `--failures-log` | both | — | JSONL with `error` and `traceback`; map writes one line per FITS `path`, reduce one per `npix` with its `parts` |
| `--progress/--no-progress` | both | on | tqdm progress bar |
| `--log-file` | both | — | Append INFO+ logs to a file (with `-q`, the file still gets INFO) |
| `-q` / `-v` | both | — | Terminal shows warnings only / DEBUG; mutually exclusive |

## Duplicate / resume flags by command

Cutout and spectrum ingest default to `--on-duplicate skip`, so re-running a file list or
batch does not create duplicate rows. Use `append` only when you intentionally want
duplicate Zarr rows. Catalog append uses `--on-duplicate-id skip` (the default) together
with `--tile-mode append`.

| Command | Flag | Values | Notes |
|---------|------|--------|--------|
| `dl-ingest-catalog` | `--tile-mode` | `skip`, `overwrite`, `append` | Default `skip` (existing tile left untouched); `overwrite` replaces it; `append` merges rows |
| `dl-ingest-catalog` | `--on-duplicate-id` | `skip`, `error`, `last` | Default `skip`; only applies with `--tile-mode append` |
| `dl-ingest-catalog` | `--allow-incomplete-link-id` | flag | Null `_source_id` for rows with missing composite/label parts; row kept, `_spectrum_index` = -1 |
| `dl-ingest-catalog` | `--streaming` | flag | FITS only; bounded RAM. Not available on `dl-ingest-catalog-batch` or parallel `-from-list` |
| `dl-ingest-catalog` | `--fits-memmap` | `auto`, `on`, `off` | FITS read policy (default `auto`: memory-map files ≥ 8 MiB) |
| `dl-ingest-catalog`, `-batch`, `-from-list`, `dl-ingest-from-shard` | `--hdu` | index, `EXTNAME[,EXTVER]`, `all` | FITS table extension(s) per file; default first table HDU; `$DATA_LAKE_FITS_HDU` ([multi-extension FITS](catalog.md#multi-extension-fits---hdu)) |
| `dl-ingest-catalog-from-list` | `--on-duplicate-id` | same | same |
| `dl-ingest-catalog-from-list` | `--allow-incomplete-link-id` | flag | Same as single-file command; forwarded to the parallel path when `--n-workers > 1` |
| `dl-ingest-catalog-from-list` | `--tile-mode` | `skip`, `overwrite`, `append` | Sequential default `skip`; parallel default `append` (when omitted with `--n-workers > 1`) |
| `dl-ingest-catalog-batch` | `--on-duplicate-id` | same | Default `skip` |
| `dl-ingest-catalog-batch` | `--allow-incomplete-link-id` | flag | Same as single-file command |
| `dl-ingest-catalog-batch` | `--tile-mode` | `skip`, `overwrite`, `append` | Default `append` (recommended for multi-file ingest) |
| `dl-finalize-catalog` | — | — | Rebuild `catalog_info.json`, `_common_metadata`, `_metadata`, `schema_manifest.json` from tiles |
| `dl-repair-catalog-metadata` | — | — | Rebuild the same metadata from tiles; `--check-only` reports `_source_id` status without rewriting |
| `dl-repair-catalog-metadata` | `--rebuild-link-id` | column name | Recompute catalog `_source_id` from a column (catalog only); then run `dl-rebuild-catalog-indices` |
| `dl-repair-catalog-metadata` | `--allow-incomplete-link-id` | flag | Allow null `_source_id` when rebuilding link IDs |
| `dl-ingest-cutouts` | `--on-duplicate` | `skip`, `error`, `append` | Default `skip`; checked per `_source_id` in each `Npix=*.zarr` |
| `dl-ingest-cutouts-from-list` | `--on-duplicate` | same | same |
| `dl-ingest-spectra` | `--on-duplicate` | same | same |
| `dl-ingest-spectra-from-list` | `--on-duplicate` | same | Also `--on-length-mismatch`, `--wavelength-mode` |
| `dl-ingest-spectra-from-list` | `--n-workers` | int | Default `1`; `>1` decodes in parallel with a single Zarr writer (not for `desi_coadd`) |
| `dl-ingest-spectra-from-list` | `--max-in-flight` | int | Buffered decodes when `--n-workers > 1` (default `n_workers`) |
| `dl-ingest-spectra-from-list` | `--fits-memmap` | `auto`, `on`, `off` | FITS read policy for spectrum decode |
| `dl-ingest-spectra-from-list` | `--files-per-worker` | int | FITS files per worker task (default 1; try 8–32 for small spectra) |
| `dl-ingest-spectra-batch-desi-coadds` | `--on-duplicate` | same | Default `skip` |
| `dl-ingest-spectra-batch-desi-coadds` | `--link-id-col` | string | Object ID spec; default `TARGETID,SURVEY,PROGRAM` |
| `dl-ingest-spectra-batch-desi-coadds` | `--files-per-worker` | int | Coadd FITS files decoded per worker task |
| `dl-ingest-spectra-batch-spplate` | `--on-duplicate` | same | Default `skip`; `--on-length-mismatch` defaults to `pad` |

## FITS I/O tuning (memmap and workers)

Local FITS ingest uses Astropy memory mapping (not Astropy's remote download cache). The
operating system's page cache keeps recently read file pages in RAM across opens.

The FITS ingest commands (`dl-ingest-catalog`, `dl-ingest-catalog-batch`, the three `-from-list` commands,
`dl-ingest-from-shard`, `dl-ingest-catalog-mef`, `dl-ingest-spectra` and
`dl-ingest-cutouts`) accept `--fits-memmap`, which can also be set with
`$DATA_LAKE_FITS_MEMMAP`. With the default `auto`, files of at least 8 MiB are
memory-mapped (large catalogs, spPlate files, cutout cubes) and smaller files are read into
RAM once, which avoids the mapping overhead for typical `spec-*.fits`. Use
`--fits-memmap on` or `off` to force one behaviour.

| Environment variable | Default | Effect |
|----------------------|---------|--------|
| `DATA_LAKE_FITS_MEMMAP` | `auto` | Same as `--fits-memmap` |
| `DATA_LAKE_FITS_SMALL_BYTES` | `8388608` (8 MiB) | Size threshold for `auto` |
| `DATA_LAKE_PARALLEL_CATALOG_MAX_BYTES` | 512 MiB | Larger FITS files are rejected by parallel catalog ingest |

**Large single catalog FITS:** use sequential `dl-ingest-catalog --streaming`. Streaming
reads tile rows in file-order runs to reduce random disk I/O on spinning disks.

**Many small spectrum FITS:** the parallel path sorts the file list for disk locality.
`--files-per-worker 16` reduces inter-process overhead and keeps the page cache warm. For
`--n-workers`, use 2–4 on spinning disks, about the number of CPU cores on local NVMe, and
fewer workers with larger batches on NFS (Network File System).

```bash
# Many SDSS spec-*.fits on SSD
dl-ingest-spectra-from-list specs.txt --survey sdss_dr17 \
  --link-id-col SPECOBJID --n-workers 8 --files-per-worker 16 --fits-memmap auto

# One huge DESI target catalog FITS
dl-ingest-catalog huge_targets.fits --survey desi_targets \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID \
  --streaming --fits-memmap on
```

## Retrieval prerequisites (index columns and finalize)

Fast spectrum and cutout lookup and bulk extraction depend on catalog metadata written at
ingest time. Check these before relying on `get_batch`, `extract_subset_to_zarr`, or
crossmatch at scale:

1. **Index columns.** Catalog tiles should carry `_spectrum_index` and/or `_cutout_index`,
   plus `_spectrum_npix` / `_cutout_npix` when the spectrum or cutout HEALPix order differs
   from the catalog. Spectrum and cutout ingest patch these when `--update-catalog` is on
   (the default).
2. **Finalize after interrupted batch jobs.** Run `dl-finalize-catalog --survey <name>` so
   `_common_metadata`, `_metadata`, `catalog_info.json` and `schema_manifest.json` exist.
   Schema lookups read `_common_metadata`; older catalogs without it fall back to reading
   one tile, and a single `dl-finalize-catalog` run adds it. `dl-validate-catalog-ingest`
   warns when `_metadata` is missing.
3. **HEALPix order.** Use `dl-recommend-catalog-norder` on a sample file when unsure.
   `--norder` must match `hats_order` in `catalog_info.json` for retrieval and crossmatch
   tile paths.
4. **Storage locality.** Keep the lake root on local SSD when possible; on NFS or Lustre use
   fewer parallel workers (see [FITS I/O tuning](#fits-io-tuning-memmap-and-workers)).

If `_spectrum_index` is missing, retrieval falls back to scanning Zarr tiles for the
requested IDs, which is much slower. Repair it by re-running spectrum ingest with
`--update-catalog`, or with `dl-rebuild-catalog-indices`.
