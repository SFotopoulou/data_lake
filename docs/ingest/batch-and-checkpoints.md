# Batch ingest, checkpoints, and duplicate policy

#### Duplicate / resume flags by command

| Command | Flag | Values | Notes |
|---------|------|--------|--------|
| `dl-ingest-catalog` | `--on-duplicate-id` | `skip`, `error`, `last` | Only when `--tile-mode append` (Parquet rows) |
| `dl-ingest-catalog` | `--allow-incomplete-link-id` | flag | Null `_source_id` for rows with missing composite/label parts; row kept, `_spectrum_index` = -1 |
| `dl-ingest-catalog` | `--tile-mode` | `append`, `replace` | Default `append`; `replace` overwrites existing tile |
| `dl-ingest-catalog` | `--streaming` | flag | FITS-only; bounded RAM; not available on batch path |
| `dl-ingest-catalog` | `--fits-memmap` | `auto`, `on`, `off` | FITS read policy (default `auto`: mmap files ≥ 8 MiB) |
| `dl-ingest-catalog`, `-batch`, `-from-list`, `dl-ingest-from-shard` | `--hdu` | index, `EXTNAME[,EXTVER]`, `all` | FITS table extension(s) per file; default first table HDU; `$DATA_LAKE_FITS_HDU` ([multi-extension FITS](catalog.md#multi-extension-fits---hdu)) |
| `dl-ingest-catalog-from-list` | `--on-duplicate-id` | same | same |
| `dl-ingest-catalog-from-list` | `--allow-incomplete-link-id` | flag | Same semantics as single-file command; forwarded to parallel path when `--n-workers > 1` |
| `dl-ingest-catalog-from-list` | `--tile-mode` | `skip`, `overwrite`, `append` | Sequential default `skip`; parallel default `append` (when omitted with `--n-workers > 1`) |
| `dl-ingest-catalog-batch` | `--on-duplicate-id` | same | Parallel decode; default `--tile-mode append`; writes manifest at finalize |
| `dl-ingest-catalog-batch` | `--allow-incomplete-link-id` | flag | Same semantics as single-file command |
| `dl-ingest-catalog-batch` | `--tile-mode` | `skip`, `overwrite`, `append` | Default `append` (recommended for multi-file ingest) |
| `dl-finalize-catalog` | — | — | Rebuild ``catalog_info.json``, ``_metadata``, ``schema_manifest.json`` from tiles |
| `dl-repair-catalog-metadata` | `--rebuild-link-id` | column name | Recompute catalog ``_source_id`` from column (catalog only); then run ``dl-rebuild-catalog-indices`` |
| `dl-repair-catalog-metadata` | `--allow-incomplete-link-id` | flag | Allow null `_source_id` when rebuilding link IDs |
| `dl-repair-catalog-metadata` | — | — | Repair ``catalog_info.json``; ``--check-only``; ``--rebuild-link-id`` |
| `dl-ingest-cutouts` | `--on-duplicate` | `skip`, `error`, `append` | Default **`skip`**; per `source_id` in each `Npix=*.zarr` |
| `dl-ingest-cutouts-from-list` | `--on-duplicate` | same | same |
| `dl-ingest-spectra` | `--on-duplicate` | same | same |
| `dl-ingest-spectra-from-list` | `--on-duplicate` | same | Also ``--on-length-mismatch``, ``--wavelength-mode`` |
| `dl-ingest-spectra-from-list` | `--n-workers` | `1` (default) | `>1` parallel decode + single Zarr writer (not for ``desi_coadd``) |
| `dl-ingest-spectra-from-list` | `--max-in-flight` | — | Buffered decodes when ``--n-workers > 1`` (default: ``n_workers``) |
| `dl-ingest-spectra-from-list` | `--fits-memmap` | `auto`, `on`, `off` | FITS read policy for spectrum decode |
| `dl-ingest-spectra-from-list` | `--files-per-worker` | int | FITS files per worker task (default 1; try 8–32 for small specs) |
| `dl-ingest-spectra-batch-desi-coadds` | `--on-duplicate` | same | DESI parallel batch; default **`skip`** |
| `dl-ingest-spectra-batch-desi-coadds` | `--link-id-col` | string | Object ID spec; default **`TARGETID,SURVEY,PROGRAM`** |
| `dl-ingest-spectra-batch-desi-coadds` | `--files-per-worker` | int | Coadd FITS files decoded per worker task |

Cutout/spectrum ingest defaults to **`--on-duplicate skip`** so file-list and batch re-runs
are idempotent. Use **`append`** only when you intentionally want duplicate Zarr rows.
Catalog append uses **`--on-duplicate-id skip`** (default) with **`--tile-mode append`**.

#### Two-phase map-reduce ingest (`dl-ingest-from-shard`)

> **Use this instead of `dl-ingest-catalog-batch` when more than ~3 input files feed
> each lake tile**: files smaller than a tile (Legacy Survey bricks, RA/Dec patches,
> pointings), or files not sorted on the sky. This holds for thousands of files, not only
> very large surveys. Decision table and how to estimate files-per-tile:
> [Choosing a catalog ingest command](catalog.md#choosing-a-catalog-ingest-command).

**Why the append path does not scale**

`dl-ingest-catalog-batch` and `dl-ingest-catalog-from-list` use a single-writer
append model: for every source file, the main thread reads the existing Npix tile,
deduplicates, concatenates, and rewrites it.  At `--norder 4`, one ~3.7° HEALPix
pixel is shared by hundreds of bricks → later files rewrite ever-larger tiles →
~20 s/file → 2 000 CPU-hours for 400 k files.  **Do not concatenate 400 k FITS into
one large file first** — the merge step alone has a TB-scale OOM spike.

**How the shard approach works**

```
FITS → decode_catalog_file_to_batches → immutable Npix shard parts   (phase map)
Npix shard parts → one concat per Npix → lake tile                   (phase reduce)
```

- **Phase map** is embarrassingly parallel: each Slurm array task owns one RA folder
  (no pixel overlap *within* one folder), writes `<shard-root>/<survey>/Norder=N/Dir=D/Npix=P/part-<stem>.parquet` and never modifies an existing part.
  A JSONL checkpoint makes the phase resume-safe if a job is killed.
- **Phase reduce** reads all parts for each Npix once, deduplicates (`--on-duplicate-id`), and writes the final lake tile.
  Workers own disjoint Npix sets → no tile locking, linear parallelism.

**Recommended norder**

Run `dl-recommend-catalog-norder` on a few sample files with `--norder-max 8`.
For tractor-format bricks (Legacy Survey), `--norder 6` or `7` is usually best:
smaller tiles → smaller per-Npix concats in reduce, more parallel reduce tasks.
Start a **new survey name** if an existing `norder=4` tree must be kept.

**Slurm recipe**

Like every `dl-ingest-*` command, both phases need a deployment config
(`--config` or `$DATA_LAKE_CONFIG`) and the ingest token (`$LAKE_INGEST_TOKEN`).
Reduce writes into the lake given as the `OUTPUT_ROOT` argument, defaulting to
`lake.root` from the config.

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

Files not grouped in folders? Sort the full list and split it into chunks instead
(`sort files.txt | split -l 2000 -d -a 4 - chunks/chunk_`), then pass
`--file-list chunks/chunk_NNNN` per array task. The ingest advisor
(`recommend_ingest(..., layout="spatial")`) generates both scripts in this form.

**Retrying failures:** collect failed inputs from every map task and re-run map on just
those (finished files are skipped through the checkpoint, so re-running a whole task is
also safe):

```bash
cat logs/map_*_failures.jsonl | jq -r .path | sort -u > retry.txt
dl-ingest-from-shard --phase map --file-list retry.txt --survey Legacy_DR11_north \
  --shard-root $SCRATCH/shards --link-id-col ls_id_dr11 --norder 7
```

Run reduce only after map has no failures; reduce skips finalize when a tile fails, so
fix the cause and re-run reduce.

Reduce takes `--norder` from the shard tree's `shard_info.json` when omitted. Both
phases exit with status 1 if any file or tile failed. A settings mismatch exits with a
one-line error instead of a traceback.

**Expected throughput:** map ≈ 0.2–1 s/file (one FITS decode + one Parquet write per
Npix hit); reduce ≈ O(n_pixels) with linear parallelism.  400 k files in hours of
array wall-clock, not 2 000 CPU-hours of serial append.

**Flags reference**

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

**Adding to an existing survey:** map the new files into a fresh `--shard-root` and
reduce with the default `--tile-mode append`. Each touched tile is read once, merged with
the new rows, and rewritten once. Map records `norder` and `link_id_mode` in
`<shard-root>/<survey>/shard_info.json`. Reduce aborts before writing anything if those
disagree with `--norder` or with the existing survey's `catalog_info.json`, and map aborts
if a later task uses another `--link-id-col` or `--norder` on the same shard tree.

**Small local SSD:** keep `--shard-root` and the lake (`OUTPUT_ROOT`) on large shared storage and
pass `--staging-dir $TMPDIR/dl_stage`. Writes go to the SSD first; finished files are
moved atomically (cross-device copy to `<final>.tmp` then rename), so readers never see a
partial tile and the SSD only needs room for the tiles currently in flight.

---

#### Parallel catalog batch (large file lists)

> Batch append is fast when about one file feeds each lake tile. If many files share a
> tile (bricks, RA/Dec patches, unsorted run folders), use
> [`dl-ingest-from-shard`](#two-phase-map-reduce-ingest-dl-ingest-from-shard) instead; see
> [Choosing a catalog ingest command](catalog.md#choosing-a-catalog-ingest-command).

**Pre-flight:** run **`dl-check-fits-table-format --file-list <paths.txt>`** on FITS
catalogs before batch ingest. The summary reports total estimated **sources** and
on-disk **size** — keep that output to compare against `dl-describe-survey` after
ingest (accounting for `--on-duplicate-id skip`). Files classified as
**`packed-vector`** (e.g. STILTS colfits) are not suited to parallel whole-file
decode; convert to row-normal FITS or Parquet first. Details:
[Catalog ingest — check FITS layout](catalog.md#check-fits-layout-before-ingest).

For surveys shipped as **many catalog files** (e.g. Gaia `GaiaSource_*.csv.gz`), use
parallel decode with a **single-thread Parquet writer** so overlapping HEALPix tiles
are merged safely.

For **row-normal FITS shards** in one tree, you can alternatively merge them first
with [`concatenate_fits.py`](catalog.md#concatenate-fits-shards-pre-ingest) and ingest
the single output file — batch ingest is usually better at TB scale (checkpoints,
append tiles). See [Catalog ingest — concatenate FITS shards](catalog.md#concatenate-fits-shards-pre-ingest).

```bash

```bash
dl-ingest-catalog-batch gaia_files.txt --survey GAIA_DR3_source \
  --ra-col ra --dec-col dec --link-id-col source_id --norder 5 \
  --tile-mode append --on-duplicate-id skip --n-workers 8 --files-per-worker 4
```

`--files-per-worker` batches multiple catalog files per worker task (same as
spectrum ingest). `--partition-by-dir` orders the queue by parent directory for
better disk locality on sharded surveys (Gaia run directories, DESI run folders).

`dl-ingest-catalog-from-list` with `--n-workers > 1` uses the same parallel engine and accepts the same flags as `dl-ingest-catalog-batch` (including `--allow-incomplete-link-id` and `--tile-mode`). The sequential path (`--n-workers 1`, the default) additionally supports `--streaming`. When `--tile-mode` is omitted, the parallel path defaults to `append` and the sequential path defaults to `skip`.

Each batch exit (including when every file is already in the checkpoint) runs
**finalize**: ``catalog_info.json``, Parquet ``_metadata``, and
``schema_manifest.json``. If a Slurm job is killed mid-run, tiles may exist without
a manifest — run ``dl-finalize-catalog --survey <name>`` before
``dl-refresh-lake-registry``.

Peak RAM scales roughly as **`O(n_workers × largest catalog file)`** — each worker
still decodes a full file in memory, but tile tables are **spooled to temp Parquet
in the worker** (not pickled back to the parent). Lower `--n-workers` and use
`--columns` on wide surveys (ALLWISE). Default **`--max-in-flight`** is **`n_workers`**
(not `n_workers + 2` like spectra batch).

If the OS kills a worker (**OOM**), you may see `BrokenProcessPool`; the batch tool
**restarts the pool** and re-queues in-flight files. Persistent OOM → fewer workers
and a smaller `--columns` set.

#### Sequential file-list ingest (catalogs, cutouts, spectra)

```bash
find /data/cats -name '*.fits' > cat_files.txt
dl-ingest-catalog-from-list cat_files.txt --survey my_survey --ra-col RA --dec-col DEC

find /data/cutouts -name '*.fits' > cutout_files.txt
dl-ingest-cutouts-from-list cutout_files.txt --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID \
  --band-names r,i,z --on-duplicate skip

ls coadds.txt  # one DESI coadd path per line
dl-ingest-spectra-batch-desi-coadds --survey desi_dr1 --file-list coadds.txt --n-workers 16 \
  --on-duplicate skip

# Generic / SDSS spectra (sequential, not parallel DESI batch):
dl-ingest-spectra-from-list spec_files.txt --survey sdss_dr17 \
  --link-id-col SPECOBJID --on-duplicate skip

# spPlate file lists (same flags as dl-ingest-spectra):
dl-ingest-spectra-from-list spPlate_files.txt --survey boss_dr12 \
  --fmt sdss_spplate --specobj-lookup /path/to/lookup.parquet --on-duplicate skip

SDSS spec lists: pixel lengths differ slightly; ingest auto-pads (or widens the
existing tile) when format is ``sdss_boss`` or ``sdss_spplate``.  Override with
``--on-length-mismatch pad`` (default; widens tiles for longer incoming spectra)
or ``truncate`` (same as ``dl-ingest-spectra``).

# Patch _cutout_index / _spectrum_index when --update-catalog (default).
```

Checkpoints default to ``catalogs/<survey>/.ingest_checkpoint.json`` or
``cutouts/<survey>/.ingest_checkpoint.json`` under the lake root.  Optional
``--failures-log`` writes JSONL per-file errors.

#### FITS I/O tuning (memmap and workers)

Local FITS ingest uses Astropy **memory mapping** (not Astropy's remote download
cache).  The OS **page cache** keeps recently read file pages in RAM across opens.

**`--fits-memmap auto`** (default) is available on every `dl-*` command that reads FITS
files (ingest, extract, generate, debug).  It also applies via ``$DATA_LAKE_FITS_MEMMAP``.
Commands that only read Parquet/Zarr/lake metadata (``dl-describe-*``, ``dl-validate-*``,
``dl-crossmatch``, ``dl-pack-tile``, ``dl-mcp-docs``, ``dl-init``, …) are unaffected.

- Files **≥ 8 MiB** → `memmap=True` (large catalogs, spPlate, cutout cubes)
- Files **< 8 MiB** → load into RAM once (typical `spec-*.fits`; avoids mmap setup overhead)

Override with `--fits-memmap on` or `--fits-memmap off`.  Environment:
`DATA_LAKE_FITS_MEMMAP`, `DATA_LAKE_FITS_SMALL_BYTES` (default 8388608),
`DATA_LAKE_PARALLEL_CATALOG_MAX_BYTES` (default 512 MiB — parallel catalog ingest
rejects larger FITS with a message to use `--streaming`).

**Large single catalog FITS:** use sequential `dl-ingest-catalog --streaming`.
Streaming reads tile rows in **file-order runs** to reduce random disk I/O on HDD.

**Many small spectrum FITS:**

- Sort file lists (done automatically) for disk locality on spinning rust
- **`--files-per-worker 16`** amortizes process IPC and keeps the page cache warm
- **`--n-workers`:** HDD ≈ 2–4; local NVMe ≈ CPU cores; NFS → lower workers + batching

```bash
# Many SDSS spec-*.fits on SSD
dl-ingest-spectra-from-list specs.txt --survey sdss_dr17 \
  --link-id-col SPECOBJID --n-workers 8 --files-per-worker 16 --fits-memmap auto

# One huge DESI target catalog FITS
dl-ingest-catalog huge_targets.fits --survey desi_targets \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID \
  --streaming --fits-memmap on
```

#### Retrieval prerequisites (index columns and finalize)

Fast spectrum/cutout lookup and bulk extract depend on catalog metadata written at
ingest time. Before relying on ``get_batch``, ``extract_subset_to_zarr``, or
crossmatch at scale:

1. **Index columns** — catalog tiles should carry ``_spectrum_index`` and/or
   ``_cutout_index`` (and ``_spectrum_npix`` / ``_cutout_npix`` when spectrum or
   cutout HEALPix order differs from the catalog). Spectrum and cutout ingest
   patch these when ``--update-catalog`` is enabled (default on file-list ingest).
2. **Finalize after batch jobs** — run ``dl-finalize-catalog --survey <name>`` so
   Parquet ``_metadata`` exists. DuckDB predicate pushdown in
   ``CatalogAccessor`` depends on it.
3. **HEALPix order** — use ``dl-recommend-catalog-norder`` on a sample file when
   unsure; ``--norder`` must match ``hats_order`` in ``catalog_info.json`` for
   retrieval and crossmatch tile paths.
4. **Storage locality** — keep the lake root on local SSD when possible; on NFS or
   Lustre use fewer parallel workers (see FITS I/O tuning above).

If ``_spectrum_index`` is missing, retrieval falls back to scanning every Zarr tile
(O(tiles × tile_size)) — repair with spectrum ingest + ``--update-catalog`` or
``dl-rebuild-catalog-indices``.

