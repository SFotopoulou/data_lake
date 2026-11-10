# DuckDB queries

Catalog tiles are plain Parquet, so you can query them with DuckDB from Python or SQL. Use this
page when you have a list of object IDs and want columns from one or more survey catalogs.
Retrieval is keyed on native object IDs, not on HEALPix pixels, so it works even when partner
catalogs use different `--norder` values (crossmatching itself is by sky position).

## Quick start: one survey, many IDs

```python
from data_lake.io.catalog import CatalogAccessor

cat = CatalogAccessor(lake_root, "DESI_DR1")
print(cat.link_id_column)          # e.g. "TARGETID"

z = cat.redshifts_for_source_ids(target_ids)                  # {id: redshift}
vals = cat.values_for_source_ids(target_ids, ["MAG_G", "MAG_R"])  # {column: {id: value}}
```

IDs are values of the catalog's `link_id_column`. Duplicates are ignored.

## ID lookups with `CatalogAccessor`

`CatalogAccessor(lake_root, survey, norder=None, *, metadata_cache=None)` opens an in-memory
DuckDB connection with a `catalog` view over `Norder=<norder>/**/*.parquet`. `norder` defaults
to `hats_order` from `catalog_info.json`.

| Method | Returns | How IDs are passed |
|--------|---------|--------------------|
| `redshifts_for_source_ids(ids)` | `{id: z}` | Registered ID table, chunked |
| `values_for_source_ids(ids, columns)` | `{column: {id: value}}` | Registered ID table, chunked |
| `get_sources_by_id_in_healpix_pixels(ids, npixels, columns=None)` | Table (Polars by default) | Registered ID table, chunked; reads only the listed tiles |
| `get_sources_by_id(ids, columns=None)` | Table (Polars by default) | Registered ID table, chunked; scans the whole survey |

All four methods deduplicate the IDs, register them as a one-column DuckDB table in chunks of
1,000,000 (`ID_SCAN_CHUNK`, about 8 MB per chunk), and semi-join against it, so each chunk costs
one catalog scan. Override the chunk size with `batch_size=`. When you already know which tiles
the IDs live in, `get_sources_by_id_in_healpix_pixels` reads only those files:

```python
df = cat.get_sources_by_id(
    target_ids,
    columns=["TARGETID", "Z", "MAG_G", "MAG_R", "MAG_Z"],
)
```

If you know which tiles hold your objects (for example the master stores each object's
`npix`), `get_sources_by_id_in_healpix_pixels` reads only those tiles instead of the whole
survey.

### Parquet footer cache

The accessor's DuckDB connection keeps parsed Parquet footers between queries, so repeated
queries over the same tiles skip re-reading them. Memory grows with the number of tiles read.
The default comes from the environment variable `DATA_LAKE_PARQUET_METADATA_CACHE`
(`1`/`true`/`yes`/`on` or `0`/`false`/`no`/`off`) when set, otherwise from the knob
`CATALOG_PARQUET_METADATA_CACHE` (`True`). To turn it off for one accessor:

```python
cat = CatalogAccessor(lake_root, "DESI_DR1", metadata_cache=False)
```

`MultiCatalogAccessor` uses the same default.

## Several surveys: ID list, master, catalogs

To pull columns from several surveys, join your ID list to a master association table (see
[Lake registry](registry.md#master-association-tables)), then to each catalog on its ID column.

A master file is a flat Parquet, for example `<lake_root>/associations/master_desi_euclid.parquet`:

| Column | Purpose |
|--------|---------|
| `desi_targetid` (example) | Primary key for your science sample |
| `euclid_source_id`, … | Partner survey IDs from the matcher |
| `sep_arcsec` | Match separation (quality check) |
| `desi_norder`, `desi_npix` | Optional; that survey's tile, for single-tile reads |

### Generate the SQL from the master

After choosing columns with `dl-describe-survey`, let the builder write the views and the join
from `<master>.meta.json`:

```python
from data_lake.query_from_master import build_select_from_master

plan = build_select_from_master(
    lake_root,
    lake_root / "associations" / "master_desi_euclid.parquet",
    primary_survey="DESI_DR1",
    columns={
        "DESI_DR1": ["Z", "MAG_G", "MAG_R"],
        "EUCLID_DR1": ["SOURCE_ID"],
    },
)
print(plan.all_sql())  # CREATE VIEW statements, then the SELECT
```

The CLI runs the same logic and prints the SQL:

```bash
dl-build-query-from-master associations/master.parquet /data/lake \
  --primary-survey DESI_DR1 \
  --column DESI_DR1:Z,MAG_G,MAG_R \
  --column EUCLID_DR1:SOURCE_ID
```

Useful CLI flags: `--want-table` / `--want-id-column` (defaults `want` / `id`),
`--master-column` (extra master columns; default partner IDs and `sep_arcsec`), `--no-views`
(SELECT only), and `--json`.

To run the plan, register your IDs as a table named `want` with an `id` column, execute
`plan.view_ddls`, then `plan.sql`. With `MultiCatalogAccessor`, register on its connection,
for example `multi._con.register("want", df)`. The full example is in
`notebooks/11_duckdb_catalog_query.ipynb` §9.

### Write the SQL by hand

Run from the lake root so the relative paths resolve. Register each catalog with the same glob
`CatalogAccessor` uses:

```sql
CREATE VIEW desi AS
SELECT * FROM parquet_scan('catalogs/DESI_DR1/Norder=5/**/*.parquet', hive_partitioning=false);

CREATE VIEW euclid AS
SELECT * FROM parquet_scan('catalogs/EUCLID_DR1/Norder=6/**/*.parquet', hive_partitioning=false);

CREATE VIEW master AS
SELECT * FROM read_parquet('associations/master_desi_euclid.parquet');
```

Put the ID list in a table rather than a long literal `IN (...)`:

```sql
CREATE TEMP TABLE want (id BIGINT);
-- INSERT from read_csv('my_ids.csv') or register from Python (see notebook §9)
```

Then join. DuckDB reads only the listed columns from Parquet:

```sql
SELECT
    w.id,
    m.euclid_source_id,
    m.sep_arcsec,
    d.Z,
    d.MAG_G,
    d.MAG_R,
    e.SOURCE_ID
FROM want AS w
INNER JOIN master AS m ON m.desi_targetid = w.id
INNER JOIN desi AS d ON d.TARGETID = w.id
INNER JOIN euclid AS e ON e.SOURCE_ID = m.euclid_source_id;
```

Replace `TARGETID` / `SOURCE_ID` with each catalog's real ID column
(`CatalogAccessor(...).link_id_column`).

### Optional: read one tile per object

When the master stores `desi_npix` at DESI's order, you can read the exact tile file instead of
scanning the survey glob (tiles are grouped in `Dir=` folders of 10,000):

```sql
SELECT d.TARGETID, d.Z
FROM want w
JOIN master m ON m.desi_targetid = w.id
JOIN read_parquet(
  'catalogs/DESI_DR1/Norder=' || m.desi_norder::VARCHAR
  || '/Dir=' || ((m.desi_npix // 10000) * 10000)::VARCHAR
  || '/Npix=' || m.desi_npix::VARCHAR || '.parquet'
) AS d ON d.TARGETID = w.id;
```

## Tips

- Filter `want` first, join `master`, then join partner catalogs on IDs. Avoid scanning whole
  survey globs when you only need columns for a fixed ID list.
- Select only the columns you need; Parquet column projection keeps reads small.
