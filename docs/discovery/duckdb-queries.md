# DuckDB queries

### SQL builder from master (P3)

After column discovery, generate join SQL from ``<master>.meta.json`` and your column picks:

```python
from data_lake.query_from_master import build_select_from_master, parse_column_picks

plan = build_select_from_master(
    lake_root,
    lake_root / "associations" / "master_desi_euclid.parquet",
    primary_survey="DESI_DR1",
    columns={
        "DESI_DR1": ["Z", "MAG_G", "MAG_R"],
        "EUCLID_DR1": ["SOURCE_ID"],
    },
)
print(plan.all_sql())  # CREATE VIEW … + SELECT want → master → catalogs
```

CLI (same logic):

```bash
dl-build-query-from-master associations/master.parquet /data/lake \
  --primary-survey DESI_DR1 \
  --column DESI_DR1:Z,MAG_G,MAG_R \
  --column EUCLID_DR1:SOURCE_ID
```

Register ``want`` in DuckDB (or ``MultiCatalogAccessor._con.register("want", df)``), run
``plan.view_ddls`` then ``plan.sql``.

### Fast retrieval with DuckDB (ID list → master → catalogs)

Cross-matching is **by sky position**; partner catalogs may use **different**
`--norder` values. Retrieval is keyed on **native object IDs** in the master
and in each `catalogs/<survey>/` tree — not on matching HEALPix orders between
surveys.

**Master file** — flat Parquet, e.g. `<lake_root>/associations/master_desi_euclid.parquet`:

| Column | Purpose |
|--------|---------|
| `desi_targetid` (example) | Primary key for your science sample |
| `euclid_source_id`, … | Partner survey IDs from the matcher |
| `sep_arcsec` | Match separation (QA) |
| `desi_norder`, `desi_npix` | Optional; **that** survey’s tile path for Zarr / single-tile reads |

**Register catalogs** (same glob as `CatalogAccessor`):

```sql
CREATE VIEW desi AS
SELECT * FROM parquet_scan('catalogs/DESI_DR1/Norder=5/**/*.parquet', hive_partitioning=false);

CREATE VIEW euclid AS
SELECT * FROM parquet_scan('catalogs/EUCLID_DR1/Norder=6/**/*.parquet', hive_partitioning=false);

CREATE VIEW master AS
SELECT * FROM read_parquet('associations/master_desi_euclid.parquet');
```

**ID list** — prefer a small table over a huge literal `IN (...)`:

```sql
CREATE TEMP TABLE want (id BIGINT);
-- INSERT from read_csv('my_ids.csv') or register from Python (see notebook §9)
```

**Join** (only listed columns are read from Parquet):

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

Replace `TARGETID` / `SOURCE_ID` with the real ID columns (`CatalogAccessor(...).link_id_column`).

**Python** — single survey, batched `IN` queries:

```python
from data_lake.io.catalog import CatalogAccessor

cat = CatalogAccessor(lake_root, "DESI_DR1")
df = cat.get_sources_by_id(
    target_ids,
    columns=["TARGETID", "Z", "MAG_G", "MAG_R", "MAG_Z"],
)
```

**Multi-survey** — `MultiCatalogAccessor` + `want` + `master` (full example in
`notebooks/11_duckdb_catalog_query.ipynb` §9).

**Optional: one Parquet tile** when the master stores `desi_npix` at DESI’s order:

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

**Avoid** scanning full survey globs when you only need columns for a fixed ID
list — filter `want` first, join `master`, then partner catalogs on IDs.

