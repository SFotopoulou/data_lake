# Gathering multiwavelength products (`dl-gather`)

`dl-gather` materialises a **derived product catalog** by joining a base catalog
to one or more partners **via existing crossmatch trees**, restricted to a
selection. The result is stored under `catalogs/<name>/` with
`catalog_info.json` `kind: "product"` and full provenance — queryable like any
catalog and filterable with `dl-describe-lake --kind product`.

Prerequisite: the partner crossmatch trees already exist (run
[`dl-crossmatch`](crossmatch.md), e.g. `dl-crossmatch --from-area`). Gather never
recomputes matches; it reads the precomputed `A_x_B__r<radius>` tiles.

## Quick start

```bash
# From an area's gather block (base, columns, radii, multiplicity, output name)
dl-gather /data/lake --from-area Euclid_North

# Ad-hoc
dl-gather /data/lake \
  --base EUCLID \
  --columns '{"EUCLID":["ra","dec"],"DESI_DR1":["z"]}' \
  --radii   '{"DESI_DR1":1.0}' \
  --cone 150.1 2.2 --radius-arcsec 600 \
  --materialize-as EUCLID_desi_wide47
```

## Area JSON (`gather` block)

When you run `dl-gather --from-area`, column selection and output options come
from the area file (`areas/<area_id>.json`). The `columns` key is a mapping
**survey name → list of native catalog column names** (same shape as the CLI
`--columns` JSON):

```json
"gather": {
  "base": "EUCLID",
  "columns": {
    "EUCLID":   ["ra", "dec"],
    "DESI_DR1": ["z"],
    "ALLWISE":  ["w1mpro"]
  },
  "multiplicity": "nearest",
  "include_sep": true,
  "materialize_as": "EUCLID_north_joined",
  "where_joined": "DESI_DR1_z > 0.5"
}
```

| Field | Required | Meaning |
|-------|----------|---------|
| `base` | yes | Base catalog; defines the row set and HEALPix tiling. |
| `columns` | yes | `{survey: [col, …]}`. Every survey except `base` is joined as a partner. |
| `materialize_as` | yes | Product name under `catalogs/<name>/`. |
| `multiplicity` | no | `nearest` (default) or `all`. |
| `include_sep` | no | Default `true`; set `false` to omit `<survey>_sep_arcsec` columns. |
| `where_joined` | no | SQL predicate on **joined** (prefixed) partner columns, applied after the join. |
| `partners` | no | Optional `[{survey, radius_arcsec}]` override when radii differ from `crossmatch_plan`. |

**Column names** must match ingested catalog columns — check with
`dl-describe-survey <name> --modality catalog`. Survey keys must match
`dl-describe-lake` names.

**Match radii** for partners are taken from `crossmatch_plan.partners` on the
same area (each partner in `columns` needs a radius there, or in
`gather.partners`). Run [`dl-crossmatch --from-area`](crossmatch.md) before
gather so the `A_x_B__r<radius>` trees exist.

### Output column naming

- Base link ID is always written as **`_source_id`** (you do not list it in
  `columns[base]`).
- Base columns keep their native names (`ra`, `dec`, …).
- Partner columns are prefixed: `DESI_DR1_z`, `ALLWISE_w1mpro` (unless the
  name already starts with `SURVEY_`).
- Separations: `DESI_DR1_sep_arcsec`, etc., when `include_sep` is true.
- **`_healpix_norder<N>`** is added from the base catalog order.

Only columns listed under each survey are read and written; surveys omitted from
`columns` are not joined.

## Selection (one of)

`dl-gather` keeps only the base rows you select, then joins partner columns:

| Selector | Meaning |
|----------|---------|
| `--from-area AREA` | Spatial, from the area's region (tile-granular). |
| `--cone RA DEC --radius-arcsec R` / `--bbox …` / `--npix … --norder N` | Ad-hoc spatial region. |
| `--ids FILE` | Explicit base source-ids from a Parquet/CSV file. |
| `--where SQL` | A predicate over **base** columns (filter-then-crossmatch). |

The selection's npix always bound the crossmatch and partner tiles read, so id
and predicate selections stay tile-scoped (no full-tree scans).

## Match multiplicity

- **`nearest`** (default) — one partner match per base source (minimum
  separation); output is one wide row per base source.
- **`all`** (`--multiplicity all`) — keep every match (fan-out); base rows
  repeat per match.

Per-partner `<survey>_sep_arcsec` columns are included unless `--no-sep`.

## Partner-column predicates

`--where-joined` applies **after** the join (crossmatch-then-filter), e.g.
`--where-joined "DESI_DR1_z > 1.0"`. Use `--where` for base-only predicates
(applied during selection, before crossmatch).

## Extracting other modalities

`dl-gather` keeps only catalog rows in the product. To pull **spectra / cutouts**
for the product's sources into a portable bundle **outside** the lake (heavy data
is never duplicated in-lake):

```bash
dl-gather /data/lake --from-area Euclid_North \
  --extract-modalities spectra,cutout --output-dir /scratch/euclid_north_bundle
```

This drives the existing extractors (`dl-extract-spectra-subset`,
per-source cutout FITS) from the product's `_source_id` list. `--extract-survey`
overrides the source survey (default: the product's base catalog).

## Parallelism

Gather processes base tiles **sequentially**. Per tile it reads one base Parquet
file, one crossmatch file per partner, and **only the partner catalog tiles that
overlap that base pixel** (plus a one-pixel neighbour ring), not the full
partner survey.

Partner lookups use
:meth:`~data_lake.io.catalog.CatalogAccessor.get_sources_by_id_in_healpix_pixels`
instead of scanning the whole ``catalog`` DuckDB view. CPU use can still spike
when several DuckDB/Polars threads run per tile — cap with ``DUCKDB_THREADS`` and
``POLARS_MAX_THREADS`` if needed.
