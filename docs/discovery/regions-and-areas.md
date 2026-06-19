# Regions, areas, and discovery (`dl-region`)

This page covers the spatial **region** selector, metadata-only **areas**, and the
`dl-region` discovery command. To attach crossmatch/gather/homogenize plans to an
area without editing JSON, see [`dl-area`](areas-cli.md). For the join/materialise
step see [`dl-gather`](gather.md); for matching see [Crossmatch](crossmatch.md).

## Glossary

| Term | Meaning |
|------|--------|
| **Modality** | A data type stored as its own top-level tree: `catalog`, `spectra`, `cutout`, `crossmatch`. |
| **Region** | A sky selector that resolves to HEALPix **NESTED** pixels at any order: `npix`, `cone`, `bbox`, or `moc`. |
| **Area** | A metadata-only file `areas/<id>.json` bundling a region plus optional crossmatch/gather plans. Spans surveys/modalities; holds no tile data. |
| **Selection** | The base-source input to `dl-gather`: a region (tile-granular), an id list, or a base-catalog predicate. |
| **Product** | A derived joined catalog (`kind: "product"`) materialised by `dl-gather`. |
| **npix** | A HEALPix NESTED pixel index at a stated order; the on-disk tile key (`Npix=*`). |
| **Tile index** | Cached `shared/registry/tile_index/<survey>.<modality>.json` of populated npix, so discovery avoids walking the tree. |

## Region selector

A `Region` resolves to pixels **at a target order** (each modality may use a
different `hats_order`):

```python
from data_lake.discovery.region import Region

Region.from_npix([1002198, 1002199], source_norder=5)   # explicit tiles
Region.cone(ra_deg=150.1, dec_deg=2.2, radius_arcsec=600)
Region.bbox(ra_min=149.5, ra_max=150.5, dec_min=1.8, dec_max=2.6)  # RA wrap-aware
Region.from_moc(path="footprint.moc.fits")              # needs the [moc] extra

region.to_npix(target_norder=6)   # -> set of NESTED pixels
```

`npix` regions rescale exactly between orders (children/parent in the NESTED
scheme). `bbox` handles RA wrap-around (`ra_min > ra_max`) and clamps Dec at the
poles. MOC support requires the optional `mocpy` dependency (`pip install
mocpy` or `uv sync --extra moc`).

## `dl-region` — discover what's in a region

```bash
# A saved area
dl-region /data/lake --from-area Wide_Field_47

# Ad-hoc selectors (exactly one)
dl-region /data/lake --cone 150.1 2.2 --radius-arcsec 600 --modalities catalog,spectra
dl-region /data/lake --npix 1002198,1002199,1003000-1003010 --norder 5
dl-region /data/lake --bbox 149.5 150.5 1.8 2.6 --count   # exact catalog counts

# Save an ad-hoc region as a reusable area
dl-region /data/lake --npix 1002198-1003000 --norder 5 --save-as Wide_Field_47

# Export the region as an IVOA MOC (requires uv sync --extra moc)
dl-region /data/lake --cone 150.1 2.2 --radius-arcsec 600 \
  --export-moc /scratch/cone.moc.fits --moc-order 8
```

### Export as MOC

Any region selector can be written to an **IVOA Multi-Order Coverage** map
(HEALPix NESTED, ICRS) for sharing with Aladin, TOPCAT, or TAP services:

```bash
dl-region /data/lake --from-area MyCone \
  --export-moc /scratch/my_cone.moc.fits --moc-order 8

# JSON or ASCII (IVOA string) instead of FITS
dl-region ... --export-moc footprint.json --moc-order 6 --moc-format json
```

`--moc-order` sets the HEALPix resolution (higher → finer footprint, larger
files). Re-import with `dl-region --moc footprint.moc.fits`.

**Survey tile footprint** (populated HEALPix tiles from the tile index):

```bash
dl-export-moc /data/lake --survey SDSS_DR17 --moc-order 5 -o sdss_footprint.fits

# Clip to an area or cone
dl-export-moc --survey DESI_DR1 --modality spectra \
  --from-area MyCone --moc-order 8 -o desi_in_cone.moc.fits
```

Requires the optional **`moc` extra**: `uv sync --extra moc` (`mocpy`).

Output is a `survey × modality` table with **rounded** row estimates
(`~12k`, `~750M`) by default — fast and index-driven, no tile reads. Pass
`--count` for **exact** catalog counts (Parquet footer sums over the overlap
tiles only; spectra/cutout stay estimates).

> Named `dl-region` (not `dl-discover`) to avoid colliding with existing
> "discovery" vocabulary (registry inventory, config lookup, column discovery).

## Areas (`areas/<area_id>.json`)

An area is flat and self-contained — the whole definition lives in one file,
even blocks added later:

```json
{
  "area_id": "Euclid_North",
  "region": { "type": "npix", "npix": ["1002198-1003000"], "source_norder": 5 },
  "discover": { "surveys": "all", "modalities": ["catalog", "spectra", "cutout"] },
  "crossmatch_plan": {
    "base_catalog": "EUCLID",
    "partners": [
      { "survey": "DESI_DR1", "radius_arcsec": 1.0 },
      { "survey": "ALLWISE",  "radius_arcsec": 2.0 }
    ],
    "reuse_existing": true
  },
  "gather": {
    "base": "EUCLID",
    "columns": { "EUCLID": ["ra", "dec"], "DESI_DR1": ["z"], "ALLWISE": ["w1mpro"] },
    "multiplicity": "nearest",
    "materialize_as": "EUCLID_north_joined"
  }
}
```

An `Npix` may belong to **more than one area** (areas can overlap). Areas are
surfaced in their own block by `dl-describe-lake --areas`, keeping the survey
table short even with many live tiles.

## User stories

- **A. Given npix** — `dl-region --npix … --norder N` → all available data;
  optionally crossmatch via an area `crossmatch_plan`.
- **B. Given a MOC** — `dl-region --moc footprint.fits`.
- **C. Given RA/Dec + radius** — `dl-region --cone RA DEC --radius-arcsec R`.
- **D. Given RA/Dec box** — `dl-region --bbox RA_MIN RA_MAX DEC_MIN DEC_MAX`.

Each can be saved with `--save-as` and then matched (`dl-crossmatch --from-area`)
and joined (`dl-gather --from-area`).

> **Worked example:** [`docs/discovery/workflow.md`](workflow.md) and
> [`notebooks/14_discovery_workflow.ipynb`](../../notebooks/14_discovery_workflow.ipynb)
> with template [`examples/areas/multi_survey_cone.example.json`](../../examples/areas/multi_survey_cone.example.json).

## End-to-end workflow (cone → crossmatch → gather)

`dl-region --save-as` writes **only the region** (and optional `discover` block).
Attach `crossmatch_plan`, `gather`, and `homogenize` with **[`dl-area`](areas-cli.md)**:

```bash
dl-area set-crossmatch MyCone --base EUCLID_DR1 \
  --partner DESI_DR1:1.0 --partner ALLWISE:2.0
dl-area set-gather MyCone --base EUCLID_DR1 \
  --columns '{"EUCLID_DR1":["ra","dec"],"DESI_DR1":["z"]}' \
  --materialize-as euclid_native_v1
dl-area set-homogenize MyCone --from-product euclid_native_v1 \
  --transform phot_ab_v1 --materialize-as euclid_ab_v1
```

Or merge from a template: `dl-area import MyCone --from-file plan.json`.
Full command reference: [areas-cli.md](areas-cli.md).

```bash
# 1. Save the sky selection
dl-region /data/lake --cone 150.1 2.2 --radius-arcsec 600 --save-as MyCone

# 2. Attach plans (dl-area) — see commands above

# 3. Match base × partners inside the cone
dl-crossmatch /data/lake --from-area MyCone --n-workers 8 --progress

# 4. Materialise the joined product catalog
dl-gather /data/lake --from-area MyCone

# 5. Homogenize + ML export (see workflow guide)
dl-homogenize --from-product MyCone_native --transform phot_ab_v1 --materialize-as MyCone_ab
dl-extract-catalog --from-product MyCone_ab -o /scratch/training.parquet
```

Full step-by-step with homogenize and spectra export:
[workflow.md](workflow.md).

### `crossmatch_plan` fields

| Field | Required | Meaning |
|-------|----------|---------|
| `base_catalog` | yes | Survey A (defines HEALPix partitioning for the match). |
| `partners` | yes | `[{survey, radius_arcsec, …}]` — one tree per partner. |
| `reuse_existing` | no | Default `true`; skip finished tiles and gap-fill on re-run. |

`dl-crossmatch --from-area` loads this block **and** bounds tiles to the area's
`region` (cone, npix, bbox, or MOC). A standalone plan file
(`dl-crossmatch --plan file.json`) runs on the full base survey with no region
cut.

See [Crossmatch — region-bounded plans](crossmatch.md#region-bounded-plans---from-area--plan)
and [Gather — area JSON](gather.md#area-json-gather-block) for column selection
and output naming.
