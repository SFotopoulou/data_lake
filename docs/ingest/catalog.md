# Catalog ingest

This page explains how to turn a survey catalog (FITS, CSV, Parquet or VOTable) into
HEALPix-partitioned Parquet tiles in the lake. Read it when you ingest a new catalog,
add files to an existing one, or need to pick a HEALPix order or a link-ID column.

## Ingest a survey catalog

One input file goes through `dl-ingest-catalog`. You must name the RA and Dec columns
(defaults `ra` and `dec`) and the survey's object-ID column with `--link-id-col`
(required):

```bash
dl-ingest-catalog survey_catalog.fits --survey des_dr2 \
  --ra-col RA --dec-col DEC --link-id-col COADD_OBJECT_ID
```

The lake root comes from your deployment config (`$DATA_LAKE_CONFIG` or `--config`).
Without a config, pass it as the second argument; tiles are written under
`<lake root>/catalogs/<survey>/`:

```bash
dl-ingest-catalog survey_catalog.fits /data/lake --survey des_dr2 \
  --ra-col RA --dec-col DEC --link-id-col COADD_OBJECT_ID
```

Flags given on the command line (`--norder` and so on) override config defaults.

Supported inputs, chosen by file suffix:

| Suffix | Read as |
|--------|---------|
| `.fits`, `.fit`, `.fz`, `.fits.gz` | FITS binary table |
| `.parquet`, `.pq` | Parquet |
| `.xml`, `.vot`, `.votable` | VOTable |
| `.ecsv` | ECSV |
| `.csv`, `.csv.gz`, `.tsv`, `.tsv.gz` | Delimited text. For `.csv` files the delimiter (comma, tab, semicolon or pipe) is detected from the first lines, so tab-separated `.csv.gz` exports such as Gaia's work |

Every row's RA and Dec must be a valid sky position (finite, Dec within [-90, 90], and
not an SDSS-style sentinel of −9000 or below). Any invalid row aborts the ingest before
HEALPix (Hierarchical Equal Area isoLatitude Pixelization) pixels are assigned.

### What an ingest writes

Each non-empty sky pixel becomes one tile, `Norder=<N>/Dir=<D>/Npix=<P>.parquet`. At
the end of every ingest (unless you pass `--defer-finalize`), the finalize step scans
**all** tiles on disk and refreshes:

| File | Contents |
|------|----------|
| `_common_metadata` | Schema only (a few KiB). Readers use it for column names and types |
| `_metadata` | Aggregate Parquet footer covering every tile |
| `catalog_info.json` | Survey description, including `total_rows`, `hats_order`, `link_id_mode` |
| `schema_manifest.json` | Column manifest |

You can run the same step on its own with `dl-finalize-catalog --survey <name>`. Do this
when a batch job was killed before it finalized, after a checkpoint-only re-run left
the files missing, or after the last `--defer-finalize` append on a live survey.
Running it once on an older catalog also adds `_common_metadata`; until then, schema
reads fall back to opening one tile.

### Large FITS files (`--streaming`)

For large FITS catalogs (the CLI help suggests above about 50 million rows), add
`--streaming`. It memory-maps the file, sorts only the RA, Dec and ID columns, and writes
one tile at a time:

```bash
dl-ingest-catalog zall-pix-iron.fits --survey desi_dr1 \
    --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID \
    --streaming
```

Peak memory stays at about one tile's worth of rows (tens of MB at order 5) instead of
roughly three times the table size. The Parquet output is the same as without
`--streaming`. The cost is about 1.5–2× longer wall-clock time from scattered reads, so
use it only when memory is the constraint. Streaming is FITS-only.

`--streaming-parallel N` splits the streaming ingest across N worker processes; it
implies `--streaming` and requires `--tile-mode append`.

### Reducing disk use (`--compact`, `--columns`)

Parquet tiles are often larger than a compressed FITS file, because the lake stores a
typed, queryable columnar layout with per-column statistics. To shrink it on re-ingest:

```bash
# Smaller tiles: ZSTD-9, no statistics or dictionary encoding, narrow strings per tile
dl-ingest-catalog zall-pix-iron.fits --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID \
  --streaming --tile-mode overwrite --compact

# Largest win: keep only the columns you query or join on
dl-ingest-catalog ... --columns TARGETID,TARGET_RA,TARGET_DEC,Z,MAG_G,MAG_R,MAG_Z,SPECTYPE
```

Combining `--compact` with a sensible `--columns` list typically gives tiles 2–4× smaller
than a default ingest. Sky, ID, HEALPix and index columns are always kept.
`--compression-level` sets the ZSTD level (default 3) when you do not use `--compact`.

### Adding more files to one survey (`--tile-mode`)

`--tile-mode` decides what happens when a tile already exists:

| `--tile-mode` | Behaviour |
|---------------|-----------|
| `skip` | Leave the existing tile alone. Default for `dl-ingest-catalog` and for `dl-ingest-catalog-from-list` with one worker |
| `append` | Read the tile, add the new rows, rewrite it. Default for `dl-ingest-catalog-batch`, `dl-ingest-catalog-mef`, and `dl-ingest-catalog-from-list` with `--n-workers` > 1 |
| `overwrite` | Replace the tile with rows from the current file only |

To build one survey from several files, use `append`:

```bash
dl-ingest-catalog-from-list desi_files.txt --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID \
  --streaming --tile-mode append
```

When appending, `--on-duplicate-id skip|error|last` (default `skip`) controls rows whose
`_source_id` is already in the tile. With `append` and `skip`, re-ingesting the same file
is safe: rows already on disk are dropped and unchanged tiles are not rewritten.

Files in one survey may disagree on numeric types (for example AllWISE or GALEX columns
stored as float32 in one FITS and float64 in another). Ingest promotes floats to float64
and integers to int64, including the elements of fixed-size vector columns, before
writing, so appends do not fail on type mismatches. When finalize finds older tiles
still on narrower types, it rewrites them before building `_metadata`.

`--defer-finalize` skips the finalize scan on each `dl-ingest-catalog` run and records
the survey as `lifecycle=live`, `finalized=false`. Use it for live surveys that append
tile by tile, then run `dl-finalize-catalog` once at the end.

## Choosing a catalog ingest command

With many input files, pick the command from **how the files are laid out on the sky**,
not from the total survey size.

| Input layout on disk | Command |
|----------------------|---------|
| One file (any size) | `dl-ingest-catalog` (add `--streaming` for large FITS) |
| Many files, about one file per lake tile: each file covers at least one tile, or files are already split by HEALPix | `dl-ingest-catalog-batch` or `dl-ingest-catalog-from-list` |
| Many files smaller than a tile: bricks, RA/Dec patches, pointings, plates | [`dl-ingest-from-shard`](batch-and-checkpoints.md#two-phase-map-reduce-ingest-dl-ingest-from-shard) |
| Many files not sorted on the sky: run folders, time-ordered dumps, each file spanning the whole footprint | [`dl-ingest-from-shard`](batch-and-checkpoints.md#two-phase-map-reduce-ingest-dl-ingest-from-shard) |

The reason is write cost. Batch ingest appends, so every file rewrites each tile it
touches. If *k* files feed one tile, that tile is rewritten *k* times as it grows and the
survey is written roughly (*k*+1)/2 times in total. `dl-ingest-from-shard` writes each row
exactly twice (once to immutable shard parts, once in a single reduce per tile),
whatever *k* is. Break-even is at *k* ≈ 3; above that, map-reduce wins by a margin that
grows linearly with *k*.

To estimate *k* for spatially organised files, divide the tile area by the file area
(tile areas are in [Choosing HEALPix order](#choosing-healpix-order---norder--hats_order)). For unsorted files,
*k* is roughly the number of files, since each file touches most tiles.

| Example | File area | norder 4 (13.4 deg²) | norder 5 (3.36 deg²) | norder 7 (0.21 deg²) |
|---------|-----------|----------------------|----------------------|----------------------|
| Legacy Survey brick (0.25° × 0.25°) | 0.0625 deg² | k ≈ 215 | k ≈ 54 | k ≈ 3.4 |
| 1° × 1° RA/Dec patch | 1 deg² | k ≈ 13 | k ≈ 3.4 | k ≈ 1 |
| 10° × 10° RA/Dec block | 100 deg² | k ≈ 1 | k ≈ 1 | k ≈ 1 |

Do not raise the order just to lower *k*. Choose the order for query performance (rows
per tile), then pick the command from *k*.

Map-reduce has its own costs: a scratch `--shard-root` about the size of the final
survey, two Slurm steps (a map array, then a dependent reduce), and a reduce that always
finalizes. Live surveys that rely on `--defer-finalize` should stay on batch ingest.

Merging thousands of bricks into one FITS with `concatenate_fits.py` first does not help:
it moves the same cost into a single-node, in-memory stack. Use it only for moderate
inputs (see [Concatenate FITS shards](#concatenate-fits-shards-pre-ingest)).

The ingest advisor applies the same rule: `recommend_ingest(..., layout="spatial")`, or
`file_area_deg2=...` together with `norder=...`, returns the shard map and reduce
commands plus both Slurm scripts (see [MCP lake server](../mcp-lake.md)).

## Check FITS layout before ingest

Before ingesting FITS, especially long file lists, run `dl-check-fits-table-format`. It
reads headers only and reports how the lake will read each file:

| Report field | Meaning |
|--------------|---------|
| `format` | `standard-bintable` (one row per source) or `packed-vector` (column-oriented STILTS "colfits" layout, `NAXIS2=1`) |
| `sources` | Estimated source count from `NAXIS2` or `TDIMn` |
| `file size` | On-disk bytes, useful for planning worker RAM |
| `ingest` | Fast memory-mapped/streaming path, or slow column-by-column reader |
| `warning` | FITS warnings for the file, including truncation (file shorter than its header says). The command exits 1 if any file warns |

With a file list, the summary line adds up total size and total estimated sources:

```bash
# One file
dl-check-fits-table-format /data/galex/photoobjall_001.fits

# Same paths as batch ingest
dl-check-fits-table-format --file-list galex_files.txt > galex.summary.txt

# Machine-readable (one JSON object per file)
dl-check-fits-table-format --file-list gaia_files.txt --json
```

Use the total as a baseline after ingest. `dl-describe-survey` counts unique rows in the
Parquet tiles, which can be lower than the input total when `--on-duplicate-id skip`
(the default) dropped duplicates. Ingest progress logs usually report rows read from
FITS, not rows written, so a small gap is normal.

If any file is `packed-vector`, expect long ingest times and high RAM per worker, which
makes parallel whole-file ingest a poor fit. Re-export it as row-normal FITS (STILTS
`col=false`) or Parquet first. See [Performance tuning](../performance.md).

## Choosing HEALPix order (`--norder` / `hats_order`)

Catalogs, cutouts and spectra are partitioned on HEALPix nested pixels. Each non-empty
pixel becomes one file (`Norder=<N>/Dir=<D>/Npix=<P>.parquet`, or `.zarr` for cutouts and
spectra). Disk footprint and inode count grow with the number of files, not only with
row count, so a sparse all-sky catalog at a high order can end up much larger than its
source FITS. The default order is 5.

Mean tile area on the full sphere (equal-area pixels, same convention as
`healpy.order2nside`):

| `--norder` | `nside` | Max tiles (full sky) | Mean tile area |
|------------|---------|----------------------|----------------|
| 0 | 1 | 12 | 3438 deg² |
| 1 | 2 | 48 | 859 deg² |
| 2 | 4 | 192 | 215 deg² |
| 3 | 8 | 768 | 53.7 deg² |
| 4 | 16 | 3 072 | 13.4 deg² |
| 5 | 32 | 12 288 | 3.36 deg² (default) |
| 6 | 64 | 49 152 | 0.84 deg² |
| 7 | 128 | 196 608 | 0.21 deg² |
| 8 | 256 | 786 432 | 0.052 deg² |
| 9 | 512 | 3 145 728 | 0.013 deg² |
| 10 | 1024 | 12 582 912 | 0.0033 deg² |

Only tiles containing at least one source are written, but for a sparse all-sky
catalog the file count still climbs quickly with order, approaching one file per object
at very high orders.

Aim for roughly 10⁴–10⁵ rows per non-empty tile. Too many tiny files add metadata and
filesystem overhead (a 2 MB FITS can become many MB on disk); too few huge tiles make
appends slow and need a lot of RAM per tile. In practice:

| Catalog type | Typical `--norder` | Why |
|--------------|-------------------|-----|
| Sparse all-sky (10⁶–10⁷ rows, full sphere) | 3–4 | Fewer, larger tiles and fewer inodes |
| Dense survey footprint (DESI, deep fields) | 5 (default) | About 3.4 deg² tiles; matches common HATS (Hierarchical Adaptive Tiling Scheme) and Rubin/LSST `lsdb` practice |
| Very local, high density | 6–7 | Only if tiles still hold many rows each |

Base the choice on source density, not on the survey's name. Override the default per
run:

```bash
dl-ingest-catalog sparse_allsky.fits --survey my_sparse --norder 4 \
  --ra-col RA --dec-col DEC --link-id-col ID --streaming
```

Catalog, spectra and cutouts can use different orders. The catalog stores
`_healpix_norder{N}` (its own partition key) alongside `_spectrum_npix` and
`_cutout_npix` (the pixel of the matching spectrum or cutout tile), so accessors find the
right Zarr tile regardless of order. Linking uses `_source_id`, not equal HEALPix pixels,
and cross-survey crossmatch does not need both catalogs at the same order.

Before a TB-scale ingest, check the order with `dl-recommend-catalog-norder`, which reads
FITS headers and a subsample of RA/Dec without writing anything:

```bash
# Files or a directory of FITS (--glob filters a directory)
dl-recommend-catalog-norder /path/to/catalogs/*.fits \
  --ra-col TARGET_RA --dec-col TARGET_DEC

# File list (same paths as batch ingest)
dl-recommend-catalog-norder --file-list allwise_files.txt \
  --ra-col ra --dec-col dec --max-files 32 --sample-rows 500000
```

It prints candidate orders with estimated rows per tile, tile count and
`pixel_sky_frac`, and marks the order closest to the target of 50 000 rows per occupied
pixel (`--target-rows-per-tile`). Increase `--sample-rows` for large, clustered
footprints. Alternatively, ingest a subset and check the tile count and
`du -sh catalogs/<survey>`.

## Link identifiers (`--link-id-col`)

The lake joins catalogs, cutouts and spectra on one internal int64 column, `_source_id`,
stored in the Parquet tiles and in Zarr `_source_id/` arrays. Your survey's own ID
columns (for example UltraVISTA `SOURCE_ID` or DESI `TARGETID`) are kept unchanged. Any
column starting with `_` is lake bookkeeping (`_healpix_norder*`, `_cutout_index`,
`_spectrum_index`, and so on).

`--link-id-col` is required and must name an existing column, or a comma-separated list
of columns. There is no auto-detection and no sequential numbering. Ingest records how
`_source_id` was built as `link_id_mode` in `catalog_info.json`, sets
`link_id_column: "_source_id"`, and stores the column you passed as `native_id_column`.

| Input type | Example | Parquet columns | `link_id_mode` |
|------------|---------|-----------------|----------------|
| Integer column | DESI `TARGETID` | `TARGETID` (int64) + `_source_id` (same values) | `column:TARGETID` |
| Unsigned / uint64 column | SDSS `objid` (> `2**63-1`) | Native cast + `_source_id` | `column:objid` |
| Decimal string column | `"39627658462934656"` in FITS ASCII | Parsed native + `_source_id` | `column:TARGETID` |
| Vector ID column | SDSS `OBJID` shape `(5,)` | Error: use scalar `objid` | — |
| Alphanumeric labels | `J000000.00-314627.5` in `NAME` | `NAME` kept; `_source_id` = stable hash | `label:NAME` |
| Composite labels | `targetname` + `obsid_v` + `obsid_r` (6dF) | Columns kept; `_source_id` = hash of `target\|obsid_v\|obsid_r` | `composite:targetname,obsid_v,obsid_r` |
| Composite with header (2dF) | `SPFILE` + `FIBRE` per extension | Columns kept; `_source_id` = hash of `spfile\|fibre` | `composite:SPFILE,FIBRE` |

Leading and trailing spaces are stripped from ID values before parsing or hashing (common
with fixed-width FITS strings); internal spaces are kept.

### Alphanumeric and composite IDs

For text labels, the readable name stays in your column (for example `NAME`) and
`_source_id` holds a deterministic 64-bit signed hash (BLAKE2b). You can compute the same
value in Python:

```python
from data_lake.ingest.fits_to_parquet import normalize_object_id
from data_lake.io.spectra import SpectrumAccessor

sid = normalize_object_id("J000000.00-314627.5")  # same int64 as catalog _source_id
acc = SpectrumAccessor("/data/lake", "MY_SURVEY")
acc.get_spectrum(sid)
```

In SQL, filter on the name directly: `SELECT * FROM catalog WHERE NAME = 'J000000.00-314627.5'`.
Spectrum and cutout FITS headers must use the same spelling (after stripping) with
`--link-id-col NAME`, so their hashes match the catalog.

When the key spans several catalog columns, as in 6dF, pass them comma-separated:
`--link-id-col targetname,obsid_v,obsid_r`. All columns are kept and `_source_id` is the
hash of `targetname|obsid_v|obsid_r`. On the spectrum side the same string is built from
the spectrum filename stem plus `OBSID_V` from the V header and `OBSID_R` from the R
header.

### Rows with missing ID parts (`--allow-incomplete-link-id`)

If some rows lack part of a composite or label key (for example a blank `SPFILE`), the
ingest aborts on the first one. Pass `--allow-incomplete-link-id` to keep those rows
instead: they keep all science columns, but `_source_id` is null and `_spectrum_index`
stays -1, so they are not linked to spectra.

```bash
# 2dF catalog where some rows have no SPFILE
dl-ingest-catalog 2dF_cat.fits --survey 2DFGRS_DR3 \
  --link-id-col SPFILE,FIBRE --allow-incomplete-link-id

# Also available on batch, from-list and repair commands
dl-ingest-catalog-batch --allow-incomplete-link-id ...
dl-repair-catalog-metadata /lake --survey 2DFGRS_DR3 \
  --rebuild-link-id SPFILE,FIBRE --allow-incomplete-link-id
```

### Changing the link column after ingest

To recompute `_source_id` from a different column without re-reading the FITS (catalog
Parquet only), then rebuild and check the spectrum links:

```bash
dl-repair-catalog-metadata /lake --survey zCOSMOS_DR3 --rebuild-link-id filename
dl-rebuild-catalog-indices --survey zCOSMOS_DR3 --kind spectrum
dl-validate-catalog-spectra-link --survey zCOSMOS_DR3
```

Science columns (for example `id`) are unchanged. `_spectrum_index` and `_cutout_index`
are reset, which is why `dl-rebuild-catalog-indices` must run afterwards.

## Injecting constant columns (`--set-column`)

In some surveys the native ID alone is not unique, because the same object appears in
several observations (visits). If the visit number is not stored in the file, inject it
as a constant column; this happens before `_source_id` is computed, so it can be part of
the key:

```bash
dl-ingest-catalog visit7.fits /data/lake \
  --survey MYSURVEY \
  --set-column VISIT=7 \
  --link-id-col OBJECT_ID,TILE_ID,VISIT \
  --tile-mode append --on-duplicate-id error
```

`_source_id` is then the hash of `<OBJECT_ID>|<TILE_ID>|7`, and `VISIT` is a normal
filterable column in every tile. Repeat the flag to add several columns:

```bash
dl-ingest-catalog obs.fits /data/lake --survey S \
  --set-column VISIT=3 \
  --set-column INSTRUMENT=HST \
  --link-id-col OBJ_ID,VISIT
```

The type is inferred as int64, then float64, then string. Force it with
`NAME:TYPE=VALUE`, where TYPE is `int`, `float`, `str` or `bool`. Only the first `=`
splits name from value, and the `:TYPE` tag is looked for only in the name part, so
values may contain `=` or `:`.

| Spec | Column type | Value |
|------|------------|-------|
| `VISIT=7` | int64 | `7` |
| `WEIGHT=1.5` | float64 | `1.5` |
| `EPOCH=J2000` | string | `"J2000"` |
| `VISIT:str=007` | string | `"007"` (keeps zero padding) |
| `FLAG:bool=true` | bool | `True` |

Constraints:

- The column must not already exist in the source file.
- Lake-internal names (`_source_id`, `_healpix_norder*`, `_cutout_*`, `_spectrum_*`) are
  rejected.
- Every ingest into a survey must inject the same set of columns, starting with the first
  write. If earlier tiles were written without it, `--tile-mode append` fails with a
  schema-mismatch error that names the cause.
- The values are recorded in `catalog_info.json` as `set_columns: {NAME: VALUE}` and kept
  by `dl-finalize-catalog`.

## Multi-extension FITS (`--hdu`)

By default every catalog command reads the first table HDU (header-data unit) of each
FITS file, skipping the primary and image extensions. When a file holds several table
extensions (per-CCD or per-field tables, an `OBJECTS` table next to a `SUMMARY` table,
`EXTVER`-numbered copies), choose one with `--hdu`. The flag works the same on
`dl-ingest-catalog`, `dl-ingest-catalog-batch`, `dl-ingest-catalog-from-list`,
`dl-ingest-from-shard` (map phase), `dl-check-fits-table-format` and
`dl-recommend-catalog-norder`, and can be set once with `$DATA_LAKE_FITS_HDU`. It is
ignored for non-FITS inputs.

| `--hdu` value | Reads |
|---------------|-------|
| *(omitted)* | First table HDU |
| `2` | HDU index 2 (must be a table) |
| `OBJECTS` | The table with `EXTNAME=OBJECTS` (case-insensitive; error if several match) |
| `OBJECTS,2` | `EXTNAME=OBJECTS`, `EXTVER=2` |
| `all` | Every non-empty table HDU, stacked. Column names must match in every extension; numeric widths are promoted. Otherwise the file fails with a message naming the extension |

A missing or non-table selection fails that file with an error listing its HDUs
(`index:EXTNAME(type)`). In batch and shard runs the error goes to `--failures-log`.

`dl-check-fits-table-format` reports, for example, `table HDU: 2 (of 3 table HDUs)` and
adds a hint when a file has more table extensions than the selection reads, so run it
first:

```bash
dl-check-fits-table-format --file-list files.txt              # what the default reads
dl-check-fits-table-format --file-list files.txt --hdu all    # sources across all tables
dl-ingest-catalog-batch files.txt --survey MY_SURVEY --link-id-col OBJID --hdu OBJECTS
```

A non-default selection is recorded as `fits_hdu` in `catalog_info.json` (and in
`shard_info.json` for map-reduce). A later ingest into the same survey with a different
`--hdu` logs a warning but still runs, since mixing extensions may be intended.

## Joined-extension catalogs (`dl-ingest-catalog-mef`)

`--hdu all` stacks rows and needs identical column names in every table. Some
multi-extension FITS (MEF) files have the other shape: the same objects in the same order
across extensions, a few shared key columns, coordinates on one extension and
measurements on the others. FastSpec (`METADATA` + `SPECPHOT` + `FASTSPEC`) is an example.

`dl-ingest-catalog-mef` joins such extensions side by side into one table, then writes
tiles the usual way. It works for any MEF whose table HDUs share columns and row count,
not only FastSpec:

```bash
dl-ingest-catalog-mef data/spectroscopy/fastspec-iron-cmx-other.fits \
  --survey FASTSPEC_IRON --ra-col RA --dec-col DEC \
  --link-id-col TARGETID,SURVEY,PROGRAM
```

How the join works:

- It uses every non-empty table HDU in file order. Repeat `--hdu NAME` to restrict the
  set (at least two).
- Every selected HDU must have the same row count.
- Shared columns are the names present on all selected HDUs (case-insensitive; at least
  one). `--key-cols` narrows that set, and any other name still found on more than one HDU
  is an error.
- Shared values must match row by row (FITS strings are compared after stripping).
- Shared columns are taken from the first HDU and the remaining columns are appended.
  Vector columns (for example a length-5 `COEFF`) are kept as they are.

There is no streaming path. It takes one file, or `--file-list` processed sequentially
with `--tile-mode append` by default so the list builds one survey. The extensions used
are recorded as, for example, `fits_hdu: "join:METADATA+SPECPHOT+FASTSPEC"`.

To flatten the same files outside the lake (one Parquet table per MEF, for TOPCAT or a
later `dl-ingest-catalog`):

```bash
python scripts/flatten_fits_mef.py data/spectroscopy/fastspec-iron-cmx-other.fits
python scripts/flatten_fits_mef.py --file-list files.txt --output-dir ./flat
```

`dl-check-fits-table-format` suggests `dl-ingest-catalog-mef` when table HDUs share
columns and `NAXIS2` but their schemas differ. Use `--hdu all` only when the extensions
really are one table split into pieces.

## Concatenate FITS shards (pre-ingest)

Some surveys ship as many small row-normal binary-table files in one directory (Legacy
Survey bricks, Gaia run folders). Usually you should ingest them directly; see
[Choosing a catalog ingest command](#choosing-a-catalog-ingest-command). Merging them
into one FITS with [`scripts/concatenate_fits.py`](../../scripts/concatenate_fits.py)
is only worth it for moderate sizes:

| Approach | Best for |
|----------|----------|
| [`dl-ingest-from-shard`](batch-and-checkpoints.md#two-phase-map-reduce-ingest-dl-ingest-from-shard) | Many files per tile (*k* > 3): bricks, RA/Dec patches, unsorted run folders; any size, multi-node |
| [`dl-ingest-catalog-batch`](batch-and-checkpoints.md#parallel-catalog-batch-large-file-lists) | About one file per tile (*k* ≲ 3), resume from checkpoints, live surveys appending over time |
| `concatenate_fits.py`, then one `dl-ingest-catalog` | Inputs that fit in RAM on one node, or when you want one merged file for archiving or other tools |

The script stacks tables with Astropy `vstack`. Every shard must be a catalog FITS with
the same column names in the same order and the same table HDU index; otherwise it stops
with an error, and you should fix the export or pass `--hdu`. Packed-vector files
(`NAXIS2=1`) are rejected; convert them to row-normal FITS or Parquet first.

A typical run checks the shards, merges them and ingests the result:

```bash
source .venv/bin/activate   # project environment, from the repo root

# 1. Check layout (headers only)
dl-check-fits-table-format /data/shards/*.fits

# 2. Merge
python scripts/concatenate_fits.py /data/shards \
  -o /data/merged_catalog.fits \
  --overwrite

# 3. Ingest (streaming recommended for a large merged file)
dl-ingest-catalog /data/merged_catalog.fits --survey MY_SURVEY \
  --ra-col RA --dec-col DEC --link-id-col SOURCE_ID --streaming
```

More examples:

```bash
# List matching files without writing output
python scripts/concatenate_fits.py /data/legacy/bricks \
  -o /data/legacy/legacy_dr10_merged.fits \
  --dry-run

# Gaia-style run folders: search subdirectories
python scripts/concatenate_fits.py /data/gaia/dr3/csv_fits_runs \
  -o /data/gaia/gaia_dr3_merged.fits \
  --pattern 'GaiaSource_*.fits' \
  --recursive \
  --overwrite

# Explicit table HDU when extensions differ between files
python scripts/concatenate_fits.py /data/my_survey/shards \
  -o /data/my_survey/merged.fits \
  --hdu 1 \
  --overwrite

# Same FITS memory-map policy as dl-ingest-catalog (large merged writes)
python scripts/concatenate_fits.py /data/shards \
  -o /data/merged.fits \
  --fits-memmap auto \
  --overwrite
```

For many shard folders under one parent, `scripts/concatenate_fits_folders.sh` writes one
merged FITS per folder, named `{SURVEY}_{FOLDER}_merged.fits`:

```bash
scripts/concatenate_fits_folders.sh LEGACY_DR10 /data/legacy/bricks /data/legacy/merged \
  --overwrite

scripts/concatenate_fits_folders.sh GAIA_DR3 /data/gaia/runs /data/gaia/merged \
  --pattern 'GaiaSource_*.fits' --recursive --overwrite

# Preview planned outputs and estimated sizes (sum of input shards)
scripts/concatenate_fits_folders.sh LEGACY_DR10 /data/legacy/bricks /data/legacy/merged \
  --dry-run --summary
```

`concatenate_fits.py` flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `input_dir` | — | Directory containing FITS table shards |
| `-o`, `--output` | required | Output FITS path |
| `--pattern` | `*.fits` | Glob for input files, relative to `input_dir` |
| `--recursive` | off | Search subdirectories |
| `--hdu` | first binary table | Table HDU index or name in every file |
| `--join` | `exact` | Column alignment when stacking (`exact` or `outer`) |
| `--fits-memmap` | `auto` | FITS read policy (same as `dl-ingest-catalog`) |
| `--overwrite` | off | Replace an existing output file |
| `--dry-run` | off | List input paths and exit |
| `--no-progress` | off | Disable the progress bar |

## Troubleshooting

**DuckDB reports `Referenced column "dec" not found; Candidate bindings: " dec"`.** The
FITS column names (`TTYPE` keywords) had leading or trailing spaces. Ingest strips them
so Parquet names match `--ra-col` and `--dec-col`, but surveys ingested before this
normalization may still have padded names on disk. Check and fix in place:

```bash
# Report padded column names
dl-repair-catalog-metadata /data/lake --survey ALLWISE --check-padded-columns

# Rewrite tiles in place and refresh metadata
dl-repair-catalog-metadata /data/lake --survey ALLWISE --normalize-column-names
```

**`_metadata`, `_common_metadata` or `catalog_info.json` is missing or stale** after a
killed batch job or a checkpoint-only re-run: run `dl-finalize-catalog --survey <name>`
(see [What an ingest writes](#what-an-ingest-writes)).

**Ingest aborts on invalid sky positions.** Some rows have non-finite RA/Dec, Dec outside
[-90, 90], or sentinel values of −9000 or below. Filter them out of the input first.

**`--tile-mode append` fails with a schema mismatch after adding `--set-column`.** The
earlier tiles were written without the injected column; see
[Injecting constant columns](#injecting-constant-columns---set-column).
