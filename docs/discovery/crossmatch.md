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
| Primary `source_id` | int64 key for your “home” survey catalog row (or hash of a string label; see [Object identifiers](#object-identifiers---link-id-col)) |
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

**Export columns for matching** — ``dl-extract-catalog`` projects any columns
from raw catalog files (FITS, VOTable, Parquet, CSV, …) or from an ingested
lake catalog:

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
  --output-dir /scratch/gaia_sky/ -c source_id -c ra -c dec --progress

# Large FITS before ingest
dl-extract-catalog huge_cat.fits -o sky.parquet --streaming \
  -c TARGETID -c RA -c DEC --valid-sky-only

# Many files
dl-extract-catalog --file-list catalog_paths.txt -o all_a.parquet \
  -c serial -c RA -c DEC --add-input-path
```

**Very large surveys** — avoid materialising hundreds of millions of rows in
one process. Prefer **in-lake cross-match** (``dl-crossmatch``) which streams
tile-by-tile like ingest. Use ``dl-extract-catalog`` only when an external tool
(STILTS) needs a portable extract.

```bash
# Positional catalog↔catalog match at lake scale (survey A defines partition)
dl-crossmatch SURVEY_A SURVEY_B /data/lake \
  --radius-arcsec 1.0 --n-workers 8 --progress

# Per-survey sky columns / Norder (defaults: each catalog_info.json)
dl-crossmatch SURVEY_A SURVEY_B /data/lake \
  --ra-col RA --dec-col DEC --norder 5 \
  --ra-col-b RAJ2000 --dec-col-b DEJ2000 --norder-b 6

# Output: catalogs/crossmatch/SURVEY_A_x_SURVEY_B/
# Query: CrossmatchAccessor or DuckDB over that tree
```

Lake exports read **one HEALPix tile at a time** (or use DuckDB
``COPY`` via ``--engine duckdb`` for a single Parquet file). Prefer
``--output-dir`` when you need a portable extract for external tools.

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

