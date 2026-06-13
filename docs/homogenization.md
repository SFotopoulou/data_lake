# Homogenization

Native ingest preserves survey column names and photometric systems. Homogenization is a **downstream, opt-in** step that materialises **product catalogs** with comparable columns (e.g. AB magnitudes) using a versioned transform registry.

## When homogenization runs

| Stage | Homogenize? |
|-------|-------------|
| Ingest (`dl-ingest-catalog*`) | No — archival native truth |
| Crossmatch (`dl-crossmatch`) | No — geometry only |
| **Homogenize (`dl-homogenize`)** | **Yes** — pick `transform_id` + survey + region |

Recipes live in `data_lake/homogenize/transforms/` (defaults) or `shared/registry/transforms/` (lake overrides). Column matching uses **`dl-describe-survey`** manifests at apply time — no manual survey inventory required.

## Primary workflow: single survey + region

```bash
dl-describe-lake /data/lake --modality catalog
dl-describe-survey ALLWISE --modality catalog

dl-homogenize /data/lake \
  --survey ALLWISE \
  --from-area Euclid_North \
  --transform phot_ab_v1 \
  --materialize-as ALLWISE_euclid_north_ab_v1

# Dry run: rule resolution only
dl-homogenize /data/lake --from-area Euclid_North --check-only

# Or ad-hoc flags with area region:
dl-homogenize /data/lake --survey ALLWISE --cone 150.1 2.2 --radius-arcsec 600 \
  --transform phot_ab_v1 --materialize-as tmp --check-only
```

Selection selectors (exactly one): `--from-area`, `--cone`, `--bbox`, `--npix` + `--norder`, `--ids`, `--where`.

Output: `catalogs/<materialize-as>/` with `kind: product`, `product_subtype: homogenized`, and provenance (`transform_id`, column lineage).

## Transform packs

| Pack | Modality | Purpose |
|------|----------|---------|
| `phot_ab_v1` | catalog | Vega/mm mag → AB via offsets and scales |
| `spec_observed_v1` | spectra | Phase F — flux unit normalization |
| `cutout_njy_v1` | cutout | Phase F — nJy/pixel calibration |

Bandpass metadata for FM conditioning: `data_lake/homogenize/bandpass.json`.

## Query-time (exploration)

For ad-hoc SQL without materialising a product, use registry-driven view SQL from `build_homogenized_view_sql` (see notebook §8 migration path in `notebooks/11_duckdb_catalog_query.ipynb`).

## Areas

Top-level `homogenize` block in `areas/<id>.json`:

```json
"homogenize": {
  "survey": "ALLWISE",
  "region": { "from_area": "Euclid_North" },
  "transform": "phot_ab_v1",
  "materialize_as": "ALLWISE_euclid_north_ab_v1"
}
```

Multi-survey: use `gather` first, then `homogenize.from_product` (planned).

Run from an area file (uses the top-level `homogenize` block):

```bash
dl-homogenize /data/lake --from-area Euclid_North
```

See also: [Regions and areas](discovery/regions-and-areas.md), [Gather](discovery/gather.md), [Column overlays](../shared/registry/overlays/README.md).
