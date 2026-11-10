# Gathering multiwavelength products (`dl-gather`)

`dl-gather` builds a derived product catalog by joining a base catalog to one or
more partner catalogs through existing crossmatch trees, restricted to a
selection of base sources. Use it when you want one wide table (for example
Euclid photometry plus DESI redshifts plus AllWISE magnitudes) to query or
export.

The product is written to `products/<name>/` with `kind: "product"` and full
provenance in `catalog_info.json`. You can query it like any catalog and list
products with `dl-describe-lake --kind product`. Products contain tiles and
`catalog_info.json` only.

Gather never computes matches itself; it reads precomputed crossmatch tiles.
Run [`dl-crossmatch`](crossmatch.md) first (for example
`dl-crossmatch --from-area`) so the partner trees exist.

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

Use `--from-area` for anything you will repeat; define the area's `gather` block
with [`dl-area set-gather`](areas-cli.md) or by editing `areas/<id>.json`. The
ad-hoc form needs `--base`, `--columns`, `--materialize-as` and one selection
flag; `--radii` gives the sky-match radius of each partner's crossmatch tree.
Pass `--overwrite` to replace an existing product.

## Area JSON (`gather` block)

With `dl-gather --from-area`, column selection and output options come from the
area file `areas/<area_id>.json`. The area id may be given with or without a
`.area` or `.json` suffix (`EDFF-test-01` and `EDFF-test-01.area` are
equivalent), and files named `areas/<id>.area.json` are also found. The
`columns` key maps each survey name to a list of native catalog column names,
the same shape as the CLI `--columns` JSON:

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
| `materialize_as` | yes | Product name (written to `products/<name>/`). Overridden by CLI `--materialize-as`. |
| `multiplicity` | no | `nearest` (default) or `all`. |
| `include_sep` | no | Default `true`; set `false` to omit `<survey>_sep_arcsec` columns. |
| `keep_all` | no | Default `true`; set `false` for matches only (same as `--matches-only`). |
| `where_joined` | no | SQL predicate on joined (prefixed) partner columns, applied after the join. |
| `partners` | no | `[{survey, radius_arcsec}]` override when radii differ from `crossmatch_plan`. |

Column names must match the ingested catalog columns; check them with
`dl-describe-survey <name> --modality catalog`. Survey keys must match the names
shown by `dl-describe-lake`.

Partner match radii come from `crossmatch_plan.partners` in the same area, so
each partner in `columns` needs a radius there or in `gather.partners`. Column
partners in the plan are resolved to their `__col_` trees by column pair.

Explicit CLI flags take precedence over the area file. For example,
`--materialize-as MyProduct` writes to `products/MyProduct/` even when the area
sets a different `gather.materialize_as`. At start, `dl-gather` prints the
resolved area file path (with `--from-area`), then the product name, base
survey, partner count and tile count.

### Output column naming

- The base link ID is always written as `_source_id`; you do not list it in
  `columns[base]`.
- Base columns keep their native names (`ra`, `dec`, …).
- Partner columns are prefixed with the survey name, for example `DESI_DR1_z`
  and `ALLWISE_w1mpro`, unless the name already starts with `SURVEY_`.
- Separations are written as `DESI_DR1_sep_arcsec` and so on, when
  `include_sep` is true.
- `_healpix_norder<N>` is added from the base catalog order.

Only the listed columns are read and written; surveys not in `columns` are not
joined.

## Selection

`dl-gather` keeps only the base rows you select, then joins partner columns.
Give exactly one selector:

| Selector | Meaning |
|----------|---------|
| `--from-area AREA` | The area's region (tile-granular). |
| `--cone RA DEC --radius-arcsec R` / `--bbox RA_MIN RA_MAX DEC_MIN DEC_MAX` / `--npix … --norder N` | An ad-hoc sky region. |
| `--ids FILE` | Explicit base source IDs from a Parquet or CSV file. |
| `--where SQL` | A predicate over base columns, applied before the join. |

The selected tiles always bound which crossmatch and partner tiles are read, so
ID and predicate selections stay tile-scoped rather than scanning whole trees.

## Match multiplicity and base rows

With `--multiplicity nearest` (the default), each base source gets at most one
match per partner, the one with the smallest separation, so the output has one
wide row per base source. With `--multiplicity all`, every match is kept and
base rows repeat once per match. Per-partner `<survey>_sep_arcsec` columns are
included unless you pass `--no-sep`.

By default every selected base source is kept, with partner columns set to
null when there is no match. Pass `--matches-only` to drop base rows that have
no match in any partner (area JSON: `"keep_all": false`). There is no
`--keep-all` flag, since that is the default.

Unmatched partner columns are filled with nulls typed from the partner catalog
schema (strings stay strings, floats stay floats), so all product tiles share
one schema.

## Partner-column predicates

`--where-joined` filters the joined rows, after the partner columns are added,
for example `--where-joined "DESI_DR1_z > 1.0"`. Use `--where` for predicates on
base columns only; it is part of the selection and applied before the join.

Filters are applied in this order: partner joins, then `--matches-only` (if
set), then `--where-joined`.

## Partner tile cache

Crossmatch tiles are read at the base survey's HEALPix pixel. Partner rows are
read from the partner's own tiles, choosing them as follows:

- Crossmatch trees with a `healpix_npix_b` column name the exact partner tile
  for each match, and gather opens only those tiles.
- Older trees without it fall back to rescaling the base pixel to the partner
  order plus a one-pixel neighbour ring. Re-run `dl-crossmatch --from-area` (or
  pair mode with `--overwrite`) to add the column.

Neighbouring base tiles usually need the same partner tiles, so gather keeps a
least-recently-used (LRU) cache of whole partner tiles for the run. Each partner
tile is read from disk once, with only the requested columns plus the link ID,
and is then filtered in memory for every later base tile that needs it. Empty
tiles are remembered too and do not count toward the tile limit. With the cache
disabled, each base tile's partner rows are fetched in one query over the
relevant partner files.

| Flag | Default | Meaning |
|------|---------|---------|
| `--partner-cache-mb` | `512` | Maximum cache size in MiB (in-memory frames); `0` disables the cache. |
| `--partner-cache-tiles` | `48` | Maximum number of cached partner tiles; `0` disables the cache. |
| `--no-partner-cache` | off | Disable the cache. |

The defaults come from `data_lake.knobs.PARTNER_TILE_CACHE_*`, shared with the
`dl-crossmatch` survey-B cache. On full-sky gathers the cache evicts the least
recently used tiles to stay within budget, so memory stays bounded; an evicted
tile is read again if a later base tile needs it. Keep the defaults unless
memory is tight (lower them) or partner tiles are large and reused across a wide
area (raise them).

## Parallelism

Gather processes base tiles one at a time. CPU use can still spike because
DuckDB and Polars use many threads per tile; cap them with
`DATA_LAKE_DUCKDB_THREADS` and `POLARS_MAX_THREADS` if needed.

## Extracting other modalities

A product contains catalog rows only. To export spectra or cutouts for the
product's sources into a portable bundle outside the lake (heavy data is never
duplicated inside the lake):

```bash
dl-gather /data/lake --from-area Euclid_North \
  --extract-modalities spectra,cutout --output-dir /scratch/euclid_north_bundle
```

This runs the existing extractors (`dl-extract-spectra-subset` and per-source
cutout FITS) on the product's `_source_id` list. `--output-dir` is required with
`--extract-modalities`. `--extract-survey` chooses the survey to extract from
(default: the product's base catalog).

## Exporting the product catalog

Use [`dl-extract-catalog`](../export-and-sharing.md#catalog-by-survey-name-then-spectra--cutouts)
with the product name like any lake catalog. `-o file.parquet` (or `.fits`,
`.csv`, `.votable`) writes one merged file; `--output-dir` writes a HATS tile
tree (Parquet only).

```bash
dl-extract-catalog --survey EDFF-test-01 \
  --all-columns -o EDFF-test-01.fits --format fits
```

With `$DATA_LAKE_CONFIG` set (or `--config`), `--lake-root` is optional when
`--survey` is given. Use `-c` to export a subset of columns, or `--all-columns`
for the full wide schema; the two are mutually exclusive.
