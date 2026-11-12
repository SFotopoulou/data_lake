# Regions, areas, and discovery (`dl-region`)

Use this page to find out which surveys and data types cover a patch of sky, and to save
that patch as a reusable **area** for crossmatching and joining. It covers the `Region`
selector, the `dl-region` command, MOC export, and the area file format. To attach
crossmatch, gather, or homogenize plans to an area without editing JSON, see
[`dl-area`](areas-cli.md).

## Quick start

```bash
# What catalogs overlap a 10-arcmin cone?
dl-region /data/lake --cone 150.1 2.2 --radius-arcsec 600

# Include spectra and crossmatch trees, with exact counts
dl-region /data/lake --cone 150.1 2.2 --radius-arcsec 600 \
  --modalities catalog,spectra,crossmatch --count

# Save the cone as an area for later steps
dl-region /data/lake --cone 150.1 2.2 --radius-arcsec 600 --save-as MyCone
```

The lake root positional argument is optional when `$DATA_LAKE_CONFIG` is set.

## Terms

| Term | Meaning |
|------|---------|
| Modality | A data type stored as its own top-level tree: `catalog`, `spectra`, `cutout`, `crossmatch`. |
| Region | A sky selector (`npix`, `cone`, `bbox`, or `moc`) that resolves to HEALPix NESTED pixels at any order. |
| Area | A metadata-only file `areas/<id>.json` holding a region plus optional plans. It spans surveys and modalities and contains no tile data. |
| Selection | The base-source input to `dl-gather`: a region (tile-granular), an ID list, or a base-catalog predicate. |
| Product | A derived joined catalog (`kind: "product"`) written by `dl-gather`. |
| npix | A HEALPix NESTED pixel index at a stated order; the on-disk tile key (`Npix=*`). |
| Tile index | The cached list of populated pixels per survey and modality, `shared/registry/tile_index/<survey>.<modality>.json`, so discovery does not walk the directory tree. |

## `dl-region` — discover what's in a region

Pass exactly one region selector. The output is a survey × modality table with the HEALPix
order, the number of overlapping tiles, and a row count.

```bash
# A saved area
dl-region /data/lake --from-area Wide_Field_47

# Ad-hoc selectors
dl-region /data/lake --cone 150.1 2.2 --radius-arcsec 600 --modalities catalog,spectra
dl-region /data/lake --npix 1002198,1002199,1003000-1003010 --norder 5
dl-region /data/lake --bbox 149.5 150.5 1.8 2.6 --count
dl-region /data/lake --moc footprint.moc.fits

# Save an ad-hoc region as a reusable area
dl-region /data/lake --npix 1002198-1003000 --norder 5 --save-as Wide_Field_47
```

By default, row counts are rounded estimates (`~12k`, `~750M`) computed from the registry and
tile index without reading any tiles. Pass `--count` for exact counts over the overlapping
tiles only: Parquet footer row counts for catalog and crossmatch, and Zarr `_source_id`
lengths for spectra and cutouts. Neither mode reads row data.

| Flag | Meaning |
|------|---------|
| `--from-area ID` | Use the region of `areas/<ID>.json`. |
| `--cone RA DEC` + `--radius-arcsec R` | Cone in degrees; radius in arcsec. |
| `--npix LIST` + `--norder N` | Comma-separated pixels and `a-b` ranges at source order `N`. |
| `--bbox RA_MIN RA_MAX DEC_MIN DEC_MAX` | Box in degrees; RA may wrap through 0. |
| `--moc FILE` | MOC FITS file (needs the `moc` extra). |
| `--surveys LIST` | Comma-separated survey names, or `all` (default). |
| `--modalities LIST` | Any of `catalog,spectra,cutout,crossmatch`. |
| `--count` | Exact overlap counts (see above). |
| `--save-as ID` | Write the region as `areas/<ID>.json`. |
| `--overwrite` | Replace an existing area (`--save-as`) or MOC file (`--export-moc`). |
| `--export-moc FILE` + `--moc-order N` | Write the region as a MOC (see below). |
| `--moc-format` | `fits` (default), `json`, or `ascii`, for `--export-moc`. |

**Default scope.** Explicit `--surveys` / `--modalities` always win. Otherwise an ad-hoc
selector searches all surveys in the `catalog` modality only. With `--from-area`, the area's
`discover` block sets the scope; if the area has no `discover` block (or it omits
`modalities`), the default is `catalog,spectra,cutout`. Crossmatch trees are only listed
when you include `crossmatch` explicitly.

**What `--save-as` writes.** Only the `region`, plus a `discover` block when you passed a
non-default survey or modality scope. Attach plans afterwards with [`dl-area`](areas-cli.md).

### Export as MOC

Any region selector can be written as an IVOA Multi-Order Coverage (MOC) map (HEALPix
NESTED, ICRS) for use in Aladin, TOPCAT, or TAP services. `--moc-order` is required and
sets the resolution: higher orders give a finer footprint and larger files.

```bash
dl-region /data/lake --from-area MyCone \
  --export-moc /scratch/my_cone.moc.fits --moc-order 8

# JSON or ASCII (IVOA string) instead of FITS
dl-region /data/lake --from-area MyCone \
  --export-moc footprint.json --moc-order 6 --moc-format json
```

A FITS MOC can be used as a selector again with `dl-region --moc footprint.moc.fits`.

To export the footprint of a survey (its populated tiles from the tile index), use
`dl-export-moc`. It takes `--survey`, `--moc-order`, and `-o` (all required), an optional
`--modality` (`catalog` default, `spectra`, or `cutout`), and at most one clip selector
(`--from-area`, `--cone`, `--bbox`, `--npix`, or `--moc`):

```bash
dl-export-moc /data/lake --survey SDSS_DR17 --moc-order 5 -o sdss_footprint.fits

# Clip to an area
dl-export-moc --survey DESI_DR1 --modality spectra \
  --from-area MyCone --moc-order 8 -o desi_in_cone.moc.fits
```

MOC import and export need the optional `moc` extra: `uv sync --extra moc` (installs `mocpy`).

## Export as DS9 region

`dl-export-ds9-region` writes a DS9 `.reg` (fk5) for a catalog survey: HEALPix
tile polygons (`--mode footprint`), source circles (`--mode sources`, default
radius 0.5″ via `--radius-arcsec`), or both (default). Source rows accept
`--where` / `--limit` (same DuckDB predicate rules as `dl-extract-*`). Clip with
at most one of `--from-area`, `--cone` (+ `--cone-radius-arcsec`), `--bbox`,
`--npix`, or `--moc`:

```bash
dl-export-ds9-region /data/lake --survey SDSS_DR17 --mode both \
  -o sdss.reg --overwrite

dl-export-ds9-region --survey DESI_DR1 --mode sources \
  --from-area MyCone --where "Z > 0.5" --limit 5000 -o desi.reg
```

## Region selector (Python)

A `Region` resolves to pixels at a target order, because catalog, spectra, and cutout layers
may each use a different `hats_order`:

```python
from data_lake.discovery.region import Region

Region.from_npix([1002198, 1002199], source_norder=5)   # explicit tiles
Region.cone(ra_deg=150.1, dec_deg=2.2, radius_arcsec=600)
Region.bbox(ra_min=149.5, ra_max=150.5, dec_min=1.8, dec_max=2.6)  # RA wrap-aware
Region.from_moc(path="footprint.moc.fits")              # needs the moc extra

region.to_npix(target_norder=6)   # -> set of NESTED pixels
```

`npix` regions rescale exactly between orders (children or parent in the NESTED scheme).
`bbox` treats `ra_min > ra_max` as wrapping through RA 0 and clamps Dec to ±90°. MOC regions
need `mocpy` (`pip install mocpy` or `uv sync --extra moc`).

## Areas (`areas/<area_id>.json`)

An area is a single self-contained JSON file. Only `area_id` and `region` are required; the
other blocks are optional and can be added later with [`dl-area`](areas-cli.md):

```json
{
  "area_id": "Euclid_North",
  "region": { "type": "npix", "npix": ["1002198-1003000"], "source_norder": 5 },
  "discover": { "surveys": "all", "modalities": ["catalog", "spectra", "cutout"] },
  "crossmatch_plan": {
    "base_catalog": "EUCLID_DR1",
    "partners": [
      { "survey": "DESI_DR1", "radius_arcsec": 1.0 },
      { "survey": "ALLWISE",  "radius_arcsec": 2.0 }
    ],
    "reuse_existing": true
  },
  "gather": {
    "base": "EUCLID_DR1",
    "columns": { "EUCLID_DR1": ["ra", "dec"], "DESI_DR1": ["z"], "ALLWISE": ["w1mpro"] },
    "multiplicity": "nearest",
    "materialize_as": "EUCLID_north_joined"
  }
}
```

Region types and their fields: `npix` (`npix` list of integers or `"a-b"` ranges,
`source_norder`), `cone` (`ra_deg`, `dec_deg`, `radius_arcsec`), `bbox` (`ra_min`, `ra_max`,
`dec_min`, `dec_max`), and `moc` (`moc_path` or `moc_string`).

Areas may overlap, so one tile can belong to several areas. `dl-describe-lake --areas` lists
them in a separate block so the survey table stays short.

### `crossmatch_plan` fields

| Field | Required | Meaning |
|-------|----------|---------|
| `base_catalog` | yes | Survey A; its HEALPix tiling partitions the match. |
| `partners` | yes | List of partners. Sky partners: `{survey, radius_arcsec}`. Column-equality partners: `{survey, match_mode: "column", match_col_a, match_col_b}`. |
| `reuse_existing` | no | Written as `true` by default. `dl-crossmatch` does not currently read it: plan runs always skip existing tiles unless you pass `--overwrite` with `--plan`. |

`dl-crossmatch --from-area` runs this block and restricts base tiles to the area's `region`.
A standalone plan file (`dl-crossmatch --plan file.json`) runs on the full base survey with
no region cut.

See [Crossmatch — region-bounded plans](crossmatch.md#region-bounded-plans---from-area----plan)
and [Gather — area JSON](gather.md#area-json-gather-block) for column selection and output
naming.

## End-to-end workflow (cone → crossmatch → gather)

Each of the four selectors (npix, MOC, cone, box) can be saved with `--save-as` and then
matched with `dl-crossmatch --from-area` and joined with `dl-gather --from-area`:

```bash
# 1. Save the sky selection
dl-region /data/lake --cone 150.1 2.2 --radius-arcsec 600 --save-as MyCone

# 2. Attach plans
dl-area set-crossmatch MyCone --base EUCLID_DR1 \
  --partner DESI_DR1:1.0 --partner ALLWISE:2.0
dl-area set-gather MyCone --base EUCLID_DR1 \
  --columns '{"EUCLID_DR1":["ra","dec"],"DESI_DR1":["z"]}' \
  --materialize-as euclid_native_v1
dl-area set-homogenize MyCone --from-product euclid_native_v1 \
  --transform phot_ab_v1 --materialize-as euclid_ab_v1

# 3. Match base × partners inside the cone
dl-crossmatch /data/lake --from-area MyCone --n-workers 8

# 4. Materialise the joined product catalog
dl-gather /data/lake --from-area MyCone

# 5. Homogenize, then export for ML
dl-homogenize --from-area MyCone
dl-extract-catalog --from-product euclid_ab_v1 -o /scratch/training.parquet
```

Instead of step 2 you can merge plans from a template with
`dl-area import MyCone --from-file plan.json`. Full command reference:
[areas-cli.md](areas-cli.md). Step-by-step guide including spectra export:
[workflow.md](workflow.md), with the worked notebook
[`notebooks/14_discovery_workflow.ipynb`](../../notebooks/14_discovery_workflow.ipynb) and
template [`examples/areas/multi_survey_cone.example.json`](../../examples/areas/multi_survey_cone.example.json).

`dl-region` is named that way, rather than `dl-discover`, to avoid confusion with other
"discovery" features (registry inventory, config lookup, column discovery).
