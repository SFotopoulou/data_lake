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

Gather is tile-bounded and processes the selected base tiles; counting and
tile-index builds use a thread pool, crossmatch uses a process pool. All
honour the global `max_workers ≤ 64` cap.
