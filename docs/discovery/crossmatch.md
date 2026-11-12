# Crossmatch and associations

`dl-crossmatch` links two ingested catalogs in the lake, either by sky position
(nearest neighbour within a radius) or by equality on a shared ID column, and
writes the result as a crossmatch tree under `<lake>/crossmatch/`. Use this page
when you need to know which sources in survey A have a counterpart in survey B,
or before running [`dl-gather`](gather.md) to build a multi-survey product.

## Quick start

```bash
# Nearest survey-B neighbour within 1 arcsec for every survey-A source
dl-crossmatch SURVEY_A SURVEY_B /data/lake --radius-arcsec 1.0

# Same, using 8 worker processes
dl-crossmatch SURVEY_A SURVEY_B /data/lake \
  --radius-arcsec 1.0 --n-workers 8
```

The positional arguments are survey A, survey B and the lake root. Survey A
defines the partitioning: the match is A-centric, so each A source gets at most
one B neighbour (the nearest within `--radius-arcsec`, default `1.0`). The
output goes to `crossmatch/SURVEY_A_x_SURVEY_B__r1.0/`; query it with
`CrossmatchAccessor` or DuckDB over that tree.

Runs are resumable: tiles that already exist in the output tree are skipped.
Pass `--overwrite` to rebuild them.

If you call `build_crossmatch(..., n_workers > 1)` from your own script, wrap the
call in an `if __name__ == "__main__":` guard, because workers are started in
fresh processes (`forkserver`).

## Output tree

Crossmatch trees live under `<lake>/crossmatch/<A>_x_<B>__r<radius>/`, next to
`catalogs/`, `spectra/` and `cutouts/`. The match radius is part of the tree
name, so runs at different radii are separate trees and never collide.

Each tree has a `crossmatch_info.json` sidecar recording the radius, the
survey-A and survey-B `hats_order`, the match backend and, for GPU runs, the
`gpu_id`. When you re-run into an existing tree, a mismatch in survey order or
backend is rejected unless you pass `--overwrite`.

Each per-tile Parquet file has these columns:

| Column | Meaning |
|--------|---------|
| `source_id_a` | Survey-A source ID |
| `source_id_b` | Survey-B source ID |
| `sep_arcsec` | Angular separation (`0.0` in column mode) |
| `_healpix_norder<N_a>` | Survey-A HEALPix pixel (the partition key) |
| `healpix_npix_b` | Survey-B HEALPix pixel holding the partner row |

[`dl-gather`](gather.md) uses `healpix_npix_b` to open exactly the right partner
tile. Trees written before this column existed still work through a geometric
fallback; re-run `dl-crossmatch` with `--overwrite` to add it.

### Sky columns and HEALPix order

By default each survey's RA/Dec columns and HEALPix order come from its
`catalog_info.json`. Override them per survey when needed:

```bash
dl-crossmatch SURVEY_A SURVEY_B /data/lake \
  --ra-col RA --dec-col DEC --norder 5 \
  --ra-col-b RAJ2000 --dec-col-b DEJ2000 --norder-b 6
```

Only populated survey-A tiles are processed. `--all-tiles` scans every HEALPix
pixel instead, which is slow.

To also write every match row to one file, add `--export-parquet PATH` or
`--export-fits PATH`.

## Column equality match

When two catalogs already share an identifier column (for example `TARGETID` or
`SOURCE_ID`), build the association tree by equality join instead of sky
matching. This is useful for linking pre-matched or spectroscopic-redshift
products to photometric catalogs.

```bash
dl-crossmatch SURVEY_A SURVEY_B /data/lake \
  --match-mode column \
  --match-col-a TARGETID \
  --match-col-b TARGETID
```

The two columns may have different names. Both are cast to string before
comparison. All matching pairs are written (many-to-many), and `sep_arcsec` is
`0.0`. The output tree is named `{A}_x_{B}__col_{col_a}__{col_b}`, for example
`crossmatch/EUCLID_DR1_x_DESI_DR1__col_TARGETID__TARGETID/`. It has the same
columns as a sky tree, so `CrossmatchAccessor`, `--export-parquet` and
`--export-fits` work the same way.

In `dl-describe-lake` output, the `detail` column of a column tree shows
`col:COL_A:COL_B`. To use that tree as a partner in an area plan, pass
`SURVEY:col:COL_A:COL_B` to `dl-area set-crossmatch --partner` (see
[Area plans](areas-cli.md)); `dl-gather --from-area` then resolves the `__col_`
tree by the column pair.

Column mode assumes spatial locality. For each survey-A tile, it loads only the
B tiles whose footprint overlaps that tile or its neighbours, padded by half the
maximum pixel radius of the coarser of the two HEALPix orders, so sources near
tile edges and surveys at different orders still find their partners. Rows whose
sky positions are far apart are not matched, even if their key values are equal.
If matched IDs can sit at arbitrary sky positions, pre-partition both catalogs by
the shared key outside the lake before ingesting.

Column mode uses the same worker strips and B-tile cache as sky mode (see
[Performance tuning](#performance-tuning)). It cannot be combined with
`--from-area` or `--plan` in the same command; put column partners in the plan
instead.

## Region-bounded plans (`--from-area` / `--plan`)

Instead of matching whole surveys pair by pair, you can run a crossmatch plan: a
base catalog matched against several partners, each with its own radius or
match columns, optionally restricted to a sky region.

The usual setup is to save a cone (or npix list, or box) with
`dl-region --save-as`, then attach a `crossmatch_plan` with
[`dl-area set-crossmatch`](areas-cli.md) (or `dl-area import`). See
[End-to-end workflow](regions-and-areas.md#end-to-end-workflow-cone--crossmatch--gather).

### Draft a plan (`--suggest-plan`)

To get a starting JSON with an automatic **sky vs column** choice per partner
(from schema manifests), without running any match:

```bash
dl-crossmatch --suggest-plan \
  --base EUCLID_Q1_MER \
  --partner GAIA_DR3_source \
  --partner ALLWISE \
  -o euclid_partners.suggest.json
```

Heuristics (first hit wins): shared ID-like column names; foreign-key style
columns such as `GAIA_ID` → partner `source_id`; otherwise sky with
`--radius-arcsec` as the default (1.0). Reasons are printed on stderr and stored
under `_suggestions` in the JSON. Edit radii/columns by hand, then run with
`--plan` or copy partners into an area. `_suggestions` is ignored at execute
time. Suggest never joins on `_source_id` (lake-local hashes differ by survey).

```bash
# Run the area's crossmatch_plan, bounded to the area's region (gap-fill only)
dl-crossmatch /data/lake --from-area Euclid_North --n-workers 8

# Or run a standalone plan file (no region restriction; --overwrite allowed)
dl-crossmatch /data/lake --plan plans/euclid_partners.json
```

In plan mode the only positional argument is the lake root. A plan has the same
shape as the `crossmatch_plan` block of an area (see
[Regions and areas](regions-and-areas.md)):

```json
{"base_catalog": "EUCLID",
 "partners": [
   {"survey": "ALLWISE", "match_mode": "sky", "radius_arcsec": 2.0},
   {"survey": "DESI_DR1", "match_mode": "column",
    "match_col_a": "TARGETID", "match_col_b": "TARGETID"}
 ],
 "reuse_existing": true}
```

`--from-area` only fills gaps. Existing tiles, including those outside the area,
are never deleted or rewritten, so `reuse_existing` is effectively always on and
`--overwrite` is rejected together with `--from-area`. To rebuild a tree, use
pair mode (`dl-crossmatch A B LAKE --overwrite`). Re-running after more tiles
have been ingested only fills the new gaps.

After every run, `total_rows` and `n_tiles` in `crossmatch_info.json` are
recounted from the files on disk, so a region-bounded run never makes the
sidecar report fewer rows than the tree holds. The command prints both the rows
matched in this run and the on-disk tree total.

Column partners in a plan are written as `__col_<col_a>__<col_b>` trees. Use
[`dl-gather`](gather.md) to materialise the joined columns.

## Inspecting crossmatch trees

`dl-describe-lake` (and `dl-describe-lake --refresh`) counts crossmatch rows
lake-wide from Parquet footers. For a focused view of sky and column trees, use
`dl-describe-crossmatch`:

```bash
# List every tree under crossmatch/
dl-describe-crossmatch /data/lake
dl-describe-crossmatch /data/lake --json

# Detail for one sky tree (give the radius when several exist)
dl-describe-crossmatch EUCLID_DR1 2MASS_PSC /data/lake --radius-arcsec 1.0
dl-describe-crossmatch EUCLID_DR1_x_2MASS_PSC__r1.0 /data/lake
dl-describe-crossmatch --name EUCLID_DR1_x_DESI_DR1__col_TARGETID__TARGETID /data/lake

# Authoritative row/tile recount from Parquet footers
dl-describe-crossmatch EUCLID_DR1 2MASS_PSC /data/lake --radius-arcsec 1.0 --recount
```

For crossmatch coverage of one area (tile overlap and exact counts), use
`dl-region --from-area AREA --modalities catalog,spectra,crossmatch --count`.

## Performance tuning

For large surveys, set `--n-workers` to the number of CPU cores and leave the
other settings at their defaults. On HDD or NFS storage, where reading survey-B
tiles dominates, fewer workers are often faster.

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-workers` | `1` | Worker processes |
| `--strips-per-worker` | `4` | Work strips per worker (only with `--n-workers > 1`) |
| `--tiles-per-worker` | `1` | Heartbeat log interval inside a strip (`1` means about every 50 tiles) |
| `--memory-budget-gb` | detected | Total memory split across workers for DuckDB |
| `--b-cache` / `--no-b-cache` | on | Per-worker cache of survey-B tiles |
| `--b-cache-tiles` | `48` | Maximum survey-B tiles per worker cache |
| `--b-cache-mib` | `512` | Maximum in-memory survey-B cache size per worker (MiB) |
| `--match-backend` | `astropy` | `astropy` (CPU) or `rapids` (GPU) |
| `--gpu-id` | `0` | CUDA device for `--match-backend rapids` |
| `--progress` / `--no-progress` | on | Progress bars; with `--no-progress`, workers log heartbeats instead |

### Work strips

With more than one worker, pending survey-A tiles are sorted by a coarse nested
HEALPix parent (then by `npix`) so neighbouring tiles stay together. They are
then cut into `--n-workers × --strips-per-worker` contiguous strips of roughly
equal estimated cost. The cost of a tile is its A rows plus the B rows in the
tiles covering it, estimated from file sizes and a few Parquet footers. Strips
are handed out heaviest first as workers become free, so a dense sky region does
not leave the other workers idle at the end. Each worker keeps its catalog
accessors and B-tile cache across all the strips it runs; processes do not share
caches.

Keep the default of 4 strips per worker. More strips give better load balance;
raise it if some workers still finish long before others. Use
`--strips-per-worker 1` for one strip per worker.

`--tiles-per-worker` does not change how work is split; it only sets how often a
worker logs progress inside its strip.

### Survey-B tile cache

Neighbouring survey-A tiles often need the same survey-B tiles. In both sky and
column mode, each worker keeps a least-recently-used (LRU) cache of whole B
tiles, using the same defaults as the `dl-gather` partner cache
(`data_lake.knobs.PARTNER_TILE_CACHE_*`: 48 tiles, 512 MiB). In sky mode the
A tile's RA/Dec bounding box is applied to the cached B tile in memory.

`--b-cache-mib` limits the in-memory frames (ID, RA, Dec and HEALPix columns
only), not the Parquet file size. Run with `--verbose` to log each B tile's row
count, on-disk and in-memory size, evictions, and a per-worker summary. Many
evictions mean the B tiles one A tile needs (about 9 at the same order) do not
fit, so each A tile re-reads them; raise `--b-cache-tiles` or `--b-cache-mib`.

```bash
dl-crossmatch SURVEY_A SURVEY_B /data/lake \
  --radius-arcsec 1.0 --n-workers 8 --tiles-per-worker 50

# Compare without B-tile reuse (slower on HDD/NFS)
dl-crossmatch SURVEY_A SURVEY_B /data/lake --no-b-cache --overwrite
```

### DuckDB limits per worker

Each worker's DuckDB connections get `available CPUs / --n-workers` threads and
a memory limit of a quarter of `--memory-budget-gb / --n-workers`
(`data_lake.knobs.DUCKDB_WORKER_MEMORY_FRACTION`). The default budget is the
smallest of physical RAM, the cgroup limit and the Slurm allocation. Without
these limits every connection would take all cores and 80 % of the node's RAM,
oversubscribing the node and getting workers killed (`BrokenProcessPool`).
Override with `$DATA_LAKE_DUCKDB_THREADS` and
`$DATA_LAKE_DUCKDB_MEMORY_LIMIT_MIB`; the chosen values are logged at start.

At the end of a run, phase timings and B-cache hit/miss counts are logged at
INFO (`a_read=… b_load=… match=… write=…`), stored under `perf` in
`crossmatch_info.json` as `t_a_read_s`, `t_b_load_s`, `t_match_s`, `t_write_s`,
`b_cache_hits` and `b_cache_misses`, and appended to the final CLI summary line
(`click.echo`) so they still print under `-q`/`--quiet` (which raises the
terminal log level to WARNING). With `--log-file` and `-q`, the file still
receives the INFO phase line as well.

### GPU backend

`--match-backend rapids` uses cuML nearest-neighbour search on unit-sphere
coordinates. Output layout and match policy (A-centric nearest neighbour within
`--radius-arcsec`) are identical to the default astropy backend. It needs the
optional `rapids` extra (Linux with NVIDIA CUDA 12). Use `--n-workers 1` on a
single GPU, because several processes contend for one device.

```bash
uv sync --extra rapids
dl-crossmatch DESI_DR1 GAIA_DR3 /data/lake \
  --radius-arcsec 1.0 --match-backend rapids --gpu-id 0 --n-workers 1
```

## Export columns for matching

Use `dl-crossmatch` for matching at lake scale; it streams tile by tile and does
not load whole surveys into memory. Use `dl-extract-catalog` when an external
tool such as STILTS needs a portable extract. It projects columns from raw
catalog files (FITS, VOTable, Parquet, CSV and others) or from an ingested lake
catalog:

```bash
# Raw survey catalog → Parquet for STILTS
# -c accepts a comma-separated list or repeated flags
dl-extract-catalog survey_a.fits -o a_sky.parquet \
  -c TARGETID,RA,DEC --valid-sky-only

# Rename columns for STILTS (NAME:alias)
dl-extract-catalog survey_b.fits -o b_sky.csv --format csv \
  -c ID:id,ra:RA,dec:DEC

# Already-ingested lake catalog (streams; does not load full survey into RAM)
dl-extract-catalog --lake-root /data/lake --survey DESI_DR1 \
  -o desi_sky.parquet -c _source_id,ra,dec --engine tiles

# Predicate + cap without a separate DuckDB step (lake or raw files)
dl-extract-catalog --lake-root /data/lake --survey DESI_DR1 \
  -c TARGETID,Z --where "upper(trim(SPECTYPE)) = 'STAR'" --limit 1000 \
  -o desi_stars.parquet
dl-extract-catalog zall-pix-iron.fits -c TARGETID,Z \
  --where "upper(trim(SPECTYPE)) = 'STAR'" --limit 1000 -o desi_stars.parquet

# Same lake export to CSV or FITS (CSV streams tile-by-tile; FITS uses a temp Parquet pass)
dl-extract-catalog --lake-root /data/lake --survey zCOSMOS_DR3 \
  -o zCOSMOS_sky.csv -c _source_id,ra,dec_
dl-extract-catalog --lake-root /data/lake --survey zCOSMOS_DR3 \
  -o zCOSMOS_sky.fits -c _source_id,ra,dec_

# 100M+ rows: tiled export (parallel STILTS / bounded memory)
dl-extract-catalog --lake-root /data/lake --survey GAIA_DR3 \
  --output-dir /scratch/gaia_sky/ -c source_id,ra,dec

# Large FITS before ingest
dl-extract-catalog huge_cat.fits -o sky.parquet --streaming \
  -c TARGETID,RA,DEC --valid-sky-only

# Many files
dl-extract-catalog --file-list catalog_paths.txt -o all_a.parquet \
  -c serial,RA,DEC --add-input-path

# Full gathered product → single FITS (or Parquet/CSV/VOTable)
dl-extract-catalog --lake-root /data/lake --survey EUCLID_north_joined \
  --all-columns -o EUCLID_north_joined.fits --format fits
```

`-o` writes one merged file; `--output-dir` writes a HATS tile tree (Parquet
only). Prefer `--output-dir` for very large catalogs. For lake catalogs,
`--engine` chooses between DuckDB `COPY` (`auto`, the default, or `duckdb`) and
Arrow tile streaming (`tiles`).

### Catalog first, then spectra or cutouts

Extract by lake survey name (`--survey`) and keep `_source_id`. Then either pull
that survey's spectra or cutouts directly, or crossmatch to a partner and pull
the partner's rows. See
[Export and sharing](../export-and-sharing.md#catalog-by-survey-name-then-spectra--cutouts).

```bash
dl-extract-catalog --survey DESI_DR1 \
  -c _source_id,TARGET_RA,TARGET_DEC,SPECTYPE \
  --where "upper(trim(SPECTYPE)) = 'STAR'" --limit 1000 \
  -o /scratch/desi_stars.parquet

# Partner catalog columns via an existing tree (one command).
# --target-id-col values are merged into the Parquet for join-back.
dl-extract-catalog --survey Legacy_DR11_north \
  --via-crossmatch DESI_DR1_x_Legacy_DR11_north__r1.0 \
  --target-list /scratch/desi_stars.parquet --target-id-col _source_id \
  -c _source_id,ra,dec,flux_g,flux_r,flux_z \
  -o /scratch/legacy_for_desi_stars.parquet

# Same survey (spectra/cutout also write <output>.target_ids.parquet)
dl-extract-spectra-subset --survey DESI_DR1 \
  --target-list /scratch/desi_stars.parquet --target-id-col _source_id \
  --output /scratch/desi_stars.zarr
dl-extract-cutout-subset --survey DESI_DR1 \
  --target-list /scratch/desi_stars.parquet --target-id-col _source_id \
  --output /scratch/desi_stars_cutouts.zarr

# Partner survey (match tree: source_id_a = DESI, source_id_b = EUCLID)
dl-crossmatch DESI_DR1 EUCLID_DR1 /data/lake --radius-arcsec 1.0
# join extract._source_id = source_id_a, then:
dl-extract-cutout-subset --survey EUCLID_DR1 \
  --target-list /scratch/desi_x_euclid.parquet --target-id-col source_id_b \
  --output /scratch/euclid_matched_cutouts.zarr
```

## Associations with STILTS

[STILTS](https://www.starlink.ac.uk/stilts/) is useful when you need match
semantics beyond `dl-crossmatch` (A-centric nearest neighbour within a radius),
such as all neighbours within a radius, symmetric or mutual best matches, or
proper-motion corrections. At hundreds of millions of rows, prefer
`dl-crossmatch`; use STILTS on smaller extracts or on per-tile exports from
`dl-extract-catalog --output-dir`.

A typical workflow:

1. Write the inputs with `dl-extract-catalog` (above) or DuckDB
   `COPY (SELECT …)` as FITS, VOTable or Parquet, with the ID and sky columns
   you need. Keep native IDs consistent with how each catalog was (or will be)
   ingested.
2. Run STILTS, for example `tskymatch2` or `tmatch2`, with your chosen `find=`
   policy, error circles and output columns for both tables.
3. Convert the STILTS output to Parquet to create the association file (see
   below). Add `healpix_npix` and `norder` if they are missing, computed with
   the same `assign_healpix` and `hats_order` as the lake so tile paths stay
   consistent.
4. Join in DuckDB or Polars: filter the primary catalog, join it to the
   association file on `source_id`, and optionally join partner catalogs or open
   Zarr stores using `npix` and row indices. A minimal end-to-end pattern lives
   in `examples/cross_survey_lsst_desi_euclid/` (synthetic tiles and
   `query.sql`); step-by-step examples are in
   `notebooks/11_duckdb_catalog_query.ipynb` section 9.

If you later ingest new spectra or cutouts for matched IDs, call
`update_index_column` so `_spectrum_index` and `_cutout_index` on the native
survey catalog stay in sync. The association file can carry partner IDs and
separations, while the lake catalog keeps the row indices used by the fast
accessors.

## Association files and lake inventory

The lake does not produce one automatic "master file" listing every survey and
object. Each modality keeps its own discovery metadata: `catalog_info.json`,
`cutout_info.json` and `spectrum_info.json` sidecars; for finalized catalogs,
the Parquet `_common_metadata` (schema only) and `_metadata` (aggregate footer)
files written by `dl-finalize-catalog`; Zarr `_source_id` arrays; and optional
ingest checkpoints. Gathered products from `dl-gather` have tiles and
`catalog_info.json` only.

For multi-survey science, use `dl-crossmatch` trees where they fit. When you
need something they do not provide (for example STILTS output), maintain your
own association table, usually Parquet or CSV, recording how identifiers line
up and, when needed, how to open the right Zarr row.

### What goes in an association file

Shape it for how you query (DuckDB, Polars, ADQL). Typical columns:

| Column | Purpose |
|--------|--------|
| Primary `source_id` | int64 key for your "home" survey catalog row (or hash of a string label; see [Catalog vs spectrum CLI flags](../ingest/spectra.md#catalog-vs-spectrum-cli-flags) for how IDs are resolved) |
| Partner IDs | e.g. `desi_targetid`, `euclid_source_id` — whatever the other survey stores |
| `sep_arcsec` | Sky separation from the matcher (optional but good for QA) |
| `match_rank` / `find` flag | Disambiguates when the matcher returns multiple neighbours |
| `healpix_npix`, `norder` | Same nested HEALPix index and order as the lake tiles (must match ingest `hats_order` / `--norder`) |
| Zarr row indices | Optional, after ingest: `desi_spectrum_row`, `euclid_cutout_row` — tile-local indices aligned with `_spectrum_index` / `_cutout_index` |

Tile paths in the lake follow `healpix_dir(norder, npix)` from
`data_lake.ingest.fits_to_parquet` (for example `Norder=5/Dir=10000/Npix=12345`).
Either store `npix` and `norder` and build paths in SQL or Python, or join the
association file back to an ingested catalog on `source_id` and read
`_healpix_norder5` from its Parquet.

Store the file wherever suits you. Many teams use
`<lake_root>/shared/associations/<name>.parquet`, which is easy to back up and
not confused with a formal `catalogs/<survey>/` ingest; or use a dedicated tree
under `catalogs/` if you want the same validation tooling as other catalogs.

Catalog-to-catalog association is positional (`dl-crossmatch`, STILTS).
Catalog-to-spectrum linkage is a separate workflow: the catalog row must carry
the native spectrum key (or `_spectrum_index` after ingest), not a positional
match to Zarr tiles.

### Building an inventory of what is in the lake

From fastest to slowest:

1. Run `dl-refresh-lake-registry`, then `dl-describe-lake --count-total` for a
   survey × modality table with per-modality and grand-total row counts.
   Catalog and spectra counts come from sidecars recorded at ingest; crossmatch
   counts are summed from Parquet footers.
2. Run `dl-describe-survey <name>` for column names, dtypes and roles.
3. Open `notebooks/04_ingestion_report.ipynb`, which walks the whole deployment
   and plots per-survey statistics.
4. Query ad hoc with DuckDB or Polars over
   `read_parquet('.../catalogs/<survey>/**/*.parquet')`, or read each survey's
   `catalog_info.json`, when you need custom filters.

You rarely need one wide table of the whole lake; a small association Parquet
plus on-demand joins to the native catalog tiles is usually enough.

## Troubleshooting

- **Workers killed with `BrokenProcessPool`**: the node is oversubscribed. Lower
  `--n-workers`, or set `--memory-budget-gb` to what your job is actually
  allowed to use.
- **Many B-cache evictions in `--verbose` output**: raise `--b-cache-tiles` or
  `--b-cache-mib`, or use fewer workers so each has more memory.
- **`--overwrite cannot be combined with --from-area`**: region-bounded runs only
  fill gaps. Rebuild with pair mode, `dl-crossmatch A B LAKE --overwrite`.
- **Existing trees under `catalogs/crossmatch/` are not found**: trees must live
  under the top-level `crossmatch/` directory with the radius in the name. Move
  them (`mv catalogs/crossmatch/* crossmatch/`, renaming each to add
  `__r<radius>`) or recompute them.
- **Tree has no `healpix_npix_b` column**: `dl-gather` falls back to reading a
  ring of neighbouring partner tiles, which is slower. Re-run `dl-crossmatch`
  with `--overwrite` to add the column.
