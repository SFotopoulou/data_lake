# Crossmatch and associations

The library does **not** emit one automatic “master file” that lists every survey
and object in the lake. **Discovery** is by convention: each modality keeps its
own metadata (`catalog_info.json`, `cutout_info.json`, `spectrum_info.json`,
Parquet `_metadata`, Zarr `_source_id` arrays, and optional ingest checkpoints).
For **multi-survey science** you maintain a separate **association table**
(usually columnar Parquet or CSV) that records how identifiers line up and,
when needed, how to open the right Zarr row.

### What goes in a master / association file

Shape it for how you query (DuckDB, Polars, ADQL). Typical columns:

| Column | Purpose |
|--------|--------|
| Primary `source_id` | int64 key for your “home” survey catalog row (or hash of a string label; see [Catalog vs spectrum CLI flags](../ingest/spectra.md#catalog-vs-spectrum-cli-flags) for how IDs are resolved) |
| Partner IDs | e.g. `desi_targetid`, `euclid_source_id` — whatever the other survey stores |
| `sep_arcsec` | Sky separation from the matcher (optional but good for QA) |
| `match_rank` / `find` flag | If the matcher can return multiple neighbours, disambiguate |
| `healpix_npix`, `norder` | Same HEALPix **nested** index and order used for lake tiles (must match ingest `hats_order` / `--norder`) |
| Zarr row indices | After ingest, optional `desi_spectrum_row`, `euclid_cutout_row` — tile-local indices aligned with `_spectrum_index` / `_cutout_index` |

Tile paths under the lake follow `healpix_dir(norder, npix)` from
`data_lake.ingest.fits_to_parquet` (e.g. `Norder=5/Dir=10000/Npix=12345`).
Either store `npix` + `norder` and build paths in SQL/Python, or **join** the
master back to an ingested catalog on `source_id` and read `_healpix_norder5`
from Parquet.

Put the file wherever you prefer: many teams use
`<lake_root>/shared/associations/<name>.parquet` (easy to back up, not confused
with a formal `catalogs/<survey>/` ingest), or a dedicated tree under
`catalogs/` if you want the same validation tooling as other catalogs.

### Building an inventory of “what is in the lake”

Recommended steps (fastest to slowest):

1. **Registry summary** — `dl-refresh-lake-registry` then `dl-describe-lake --count-total` gives a survey × modality table with per-modality and grand-total row counts in seconds, without rescanning any tile.
2. **Survey column manifest** — `dl-describe-survey <name>` for column names, dtypes, and roles.
3. **Deployment tree notebook** — `notebooks/04_ingestion_report.ipynb` walks the full deployment and plots per-survey statistics.
4. **Ad-hoc SQL** — `duckdb` / `polars` over `read_parquet('.../catalogs/<survey>/**/*.parquet')` or each survey’s `catalog_info.json` when you need custom filters.

There is no requirement to materialise a single wide table of the whole lake;
often a **small association Parquet** plus **on-demand joins** to native
catalog tiles is enough.

### Catalog–catalog association (positional)

Catalog↔catalog matching is **sky-based** (STILTS, ``build_crossmatch``, etc.).
Catalog↔spectrum linkage is a **separate workflow**: the catalog row must carry
the native spectrum key (or ``_spectrum_index`` after ingest), not a positional
match to Zarr tiles.

### Export columns for matching

``dl-extract-catalog`` projects any columns from raw catalog files (FITS, VOTable, Parquet, CSV, …) or from an ingested lake catalog:

```bash
# Raw survey catalog → Parquet for STILTS
dl-extract-catalog survey_a.fits -o a_sky.parquet \
  -c TARGETID -c RA -c DEC --valid-sky-only

# Rename columns for STILTS (NAME:alias)
dl-extract-catalog survey_b.fits -o b_sky.csv --format csv \
  -c ID:id -c ra:RA -c dec:DEC

# Already-ingested lake catalog (streams; does not load full survey into RAM)
dl-extract-catalog --lake-root /data/lake --survey DESI_DR1 \
  -o desi_sky.parquet -c _source_id -c ra -c dec --engine tiles

# Same lake export to CSV or FITS (CSV streams tile-by-tile; FITS uses a temp Parquet pass)
dl-extract-catalog --lake-root /data/lake --survey zCOSMOS_DR3 \
  -o zCOSMOS_sky.csv -c _source_id -c ra -c dec_
dl-extract-catalog --lake-root /data/lake --survey zCOSMOS_DR3 \
  -o zCOSMOS_sky.fits -c _source_id -c ra -c dec_

# 100M+ rows: tiled export (parallel STILTS / bounded memory)
dl-extract-catalog --lake-root /data/lake --survey GAIA_DR3 \
  --output-dir /scratch/gaia_sky/ -c source_id -c ra -c dec

# Large FITS before ingest
dl-extract-catalog huge_cat.fits -o sky.parquet --streaming \
  -c TARGETID -c RA -c DEC --valid-sky-only

# Many files
dl-extract-catalog --file-list catalog_paths.txt -o all_a.parquet \
  -c serial -c RA -c DEC --add-input-path

# Full gathered product → single FITS (or Parquet/CSV/VOTable)
dl-extract-catalog --lake-root /data/lake --survey EUCLID_north_joined \
  --all-columns -o EUCLID_north_joined.fits --format fits

# -o writes one merged file; --output-dir writes a HATS tile tree (Parquet only)
```

**Very large surveys** — avoid materialising hundreds of millions of rows in
one process. Prefer **in-lake cross-match** (``dl-crossmatch``) which streams
tile-by-tile like ingest. Use ``dl-extract-catalog`` only when an external tool
(STILTS) needs a portable extract.

```bash
# Positional catalog↔catalog match at lake scale (survey A defines partition)
dl-crossmatch SURVEY_A SURVEY_B /data/lake \
  --radius-arcsec 1.0 --n-workers 8

# Per-survey sky columns / Norder (defaults: each catalog_info.json)
dl-crossmatch SURVEY_A SURVEY_B /data/lake \
  --ra-col RA --dec-col DEC --norder 5 \
  --ra-col-b RAJ2000 --dec-col-b DEJ2000 --norder-b 6

# Output: crossmatch/SURVEY_A_x_SURVEY_B__r1.0/   (top-level modality; radius in name)
# Query: CrossmatchAccessor or DuckDB over that tree

# GPU sky matching (optional [rapids] extra; same match semantics as default)
uv sync --extra rapids
dl-crossmatch DESI_DR1 GAIA_DR3 /data/lake \
  --radius-arcsec 1.0 --match-backend rapids --gpu-id 0 --n-workers 1
```

**GPU backend (`--match-backend rapids`)** — uses cuML nearest-neighbour on
unit-sphere coordinates. Output layout and match policy (survey-A-centric
nearest neighbour within `--radius-arcsec`) are identical to the default
astropy backend. Install with `uv sync --extra rapids` (Linux + NVIDIA CUDA 12).
Prefer `--n-workers 1` on a single GPU; multiple processes can contend for one
device. `crossmatch_info.json` records `match_backend` and `gpu_id` for provenance.

### Crossmatch is a top-level modality

Crossmatch trees live under `<lake>/crossmatch/<A>_x_<B>__r<radius>/` (a
first-class modality alongside `catalogs/`, `spectra/`, `cutouts/`), **not** under
`catalogs/`. The **match radius is part of the tree name** (`__r1.0`), so different
radii are distinct trees that never collide. Each tree carries a
`crossmatch_info.json` sidecar (radius, survey-A/B `hats_order`, backend). On reuse,
a mismatch in survey order or backend is rejected unless you pass `--overwrite`
(radius can't mismatch — it's path-encoded).

There is no backward compatibility for the old `catalogs/crossmatch/` location:
move existing trees up one level (`mv catalogs/crossmatch/* crossmatch/`, renaming
to add `__r<radius>`) or recompute them.

**Per-tile Parquet columns:** `source_id_a`, `source_id_b`, `sep_arcsec`,
`_healpix_norder<N_a>` (base partition key), and `healpix_npix_b` (partner survey
B pixel, written on new crossmatch runs). [`dl-gather`](gather.md) uses
`healpix_npix_b` for exact partner tile reads; older trees without it still work
via a geometric fallback. Re-run `dl-crossmatch` with `--overwrite` to refresh.

### Region-bounded plans (`--from-area` / `--plan`)

Instead of matching whole surveys, drive a **crossmatch plan** (a base catalog ×
N partners, each with its own radius) and restrict it to a sky region.

**Typical setup:** save a cone (or npix/bbox) with `dl-region --save-as`, then
attach `crossmatch_plan` with [`dl-area set-crossmatch`](areas-cli.md) (or
`dl-area import`). See [End-to-end workflow](regions-and-areas.md#end-to-end-workflow-cone--crossmatch--gather).

```bash
# Run the area's crossmatch_plan, bounded to the area region (reuses + gap-fills)
dl-crossmatch /data/lake --from-area Euclid_North --n-workers 8

# Or a standalone plan file (no region restriction)
dl-crossmatch /data/lake --plan plans/euclid_partners.json
```

A plan is the `crossmatch_plan` block of an area (see
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

Existing tiles are skipped (resume), so re-running after more live tiles arrive
only fills gaps. Column partners are written as ``__col_<col_a>__<col_b>`` trees;
``dl-gather --from-area`` resolves them by the column pair. See [`dl-gather`](gather.md)
to materialise the joined columns.

**Worker batching:** ``--tiles-per-worker`` (default 1) runs multiple survey-A
HEALPix tiles per worker process, amortizing DuckDB catalog registration. Try
4–16 on large surveys with ``--n-workers`` set to CPU core count.

Lake exports read **one HEALPix tile at a time** (or use DuckDB
``COPY`` via ``--engine duckdb`` for a single Parquet file). Prefer
``--output-dir`` when you need a portable extract for external tools.

### Column equality match

When two catalogs already share a common identifier column (e.g. `TARGETID`, `SOURCE_ID`) you can build an association tree by **equality join** instead of sky matching.  This is useful for linking pre-matched or spec-z products to photometric catalogs, or for joining on a common observation ID.

```bash
dl-crossmatch SURVEY_A SURVEY_B /data/lake \
  --match-mode column \
  --match-col-a TARGETID \
  --match-col-b TARGETID
```

`--match-col-a` and `--match-col-b` may have different names in each survey; both are cast to string before comparison.  All matching pairs are written (many-to-many); `sep_arcsec` is set to `0.0`.

**Output tree name:** `{A}_x_{B}__col_{col_a}__{col_b}` — the column names are directly embedded in the path, so `dl-describe-lake` shows exactly what you need to reconstruct the partner spec.

Example: `crossmatch/EUCLID_DR1_x_DESI_DR1__col_TARGETID__TARGETID/`

**Reading from `dl-describe-lake`:** The `detail` column shows `col:COL_A:COL_B`. To use a column tree as a `dl-gather` partner, pass `SURVEY:col:COL_A:COL_B`, copying the values directly from the describe output.

**Same schema as sky trees:** `source_id_a`, `source_id_b`, `sep_arcsec`, `_healpix_norder{N}`, `healpix_npix_b` — readable by `CrossmatchAccessor` and exportable with `--export-parquet` / `--export-fits`.

**Spatial-locality assumption:** For each survey-A tile, B tiles are loaded if their HEALPix footprint overlaps that tile **or** its neighbours.  Overlap uses the A-tile geometric extent plus a **half-pixel pad** of the coarser of the two surveys (`0.5 × max_pixrad`), so edge sources and `norder` mismatches still find partner tiles.  Rows whose sky positions lie in completely disjoint regions remain out of scope, even if their key values match.  Per-process LRU reuse of prepared B frames avoids reopening the same B Parquet for adjacent A tiles.

If you know that matched IDs can span arbitrary sky positions, pre-partition both catalogs by the shared key outside the lake before ingesting.

**Performance and progress:** Progress bars are on by default (`--no-progress` to disable).  `--n-workers` / `--tiles-per-worker` apply to column mode.  When progress is disabled, a log line is emitted every 50 tiles (and every 30 s in parallel mode) with tile count and row count so far.

```bash
dl-crossmatch A B /data/lake \
  --match-mode column --match-col-a ID --match-col-b ID \
  --n-workers 4 --tiles-per-worker 8
```

**Limitations:**
- Pair-mode ``--match-mode column`` cannot be combined with ``--from-area`` / ``--plan`` on the same invocation (use plan partners instead — see below).
- When an area plan includes column partners, ``dl-crossmatch --from-area`` and ``dl-gather --from-area`` resolve ``__col_`` trees by column pair.

### Associations with STILTS

[STILTS](https://www.starlink.ac.uk/stilts/) is useful when you need
**explicit match semantics** (all neighbours in a radius, symmetric / mutual
best matches, proper motions, etc.) beyond ``dl-crossmatch`` (nearest neighbour
within a radius, survey-A-centric partitioning). At hundreds of millions of
rows, prefer ``dl-crossmatch``; use STILTS on smaller extracts or per-tile
exports from ``dl-extract-catalog --output-dir``.

**Suggested workflow:**

1. **Materialise inputs** — Use ``dl-extract-catalog`` (above) or DuckDB
   ``COPY (SELECT …)`` to write FITS/VOTable/Parquet with the ID and sky columns
   you need for matching. Keep **native IDs** consistent with how each catalog
   was (or will be) ingested.
2. **Run STILTS** — e.g. `tskymatch2` / `tmatch2` with your chosen `find=`
   policy, error circles, and output columns for both tables.
3. **Write the master** — Convert STILTS output to **Parquet** (columnar,
   typed). Add `healpix_npix` / `norder` if missing (recompute with the same
   `assign_healpix` / `hats_order` as the lake so tile paths stay consistent).
4. **Use in analysis** — DuckDB / Polars joins: filter the primary catalog,
   join to the master on `source_id`, optionally join to partner catalogs or
   open Zarr using `npix` + row indices. A minimal end-to-end pattern lives in
   `examples/cross_survey_lsst_desi_euclid/` (synthetic tiles + `query.sql`).
   Step-by-step examples: **`notebooks/11_duckdb_catalog_query.ipynb` §9**.

**After STILTS:** if you ingest new spectra or cutouts for matched IDs, call
`update_index_column` so `_spectrum_index` / `_cutout_index` on the **native**
survey catalog stay in sync; the master table can carry partner IDs and
separations while the lake catalog keeps machine indices for fast accessors.

