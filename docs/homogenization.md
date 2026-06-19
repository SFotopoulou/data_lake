# Homogenization

Native ingest preserves survey column names and photometric systems. Homogenization is a **downstream, opt-in** step that materialises **product catalogs** with comparable columns (e.g. AB magnitudes) using a versioned transform registry.

## When homogenization runs

| Stage | Homogenize? |
|-------|-------------|
| Ingest (`dl-ingest-catalog*`) | No — archival native truth |
| Crossmatch (`dl-crossmatch`) | No — geometry only |
| **Homogenize (`dl-homogenize`)** | **Yes** — pick `transform_id` + survey + region |

Recipes live in **`shared/registry/homogenize/<SURVEY>.json`** (lake overrides) or bundled **`data_lake/homogenize/surveys/<SURVEY>.json`**. Transform packs (`phot_ab_v1`, etc.) define profile semantics only — **executable rules live only in per-survey files**.

Check coverage with `dl-validate-homogenization --ab-coverage` (requires lake root / `$DATA_LAKE_CONFIG`).

Column matching uses **`dl-describe-survey`** manifests at apply time — no manual survey inventory required.

## Primary workflow: single survey + region

``dl-homogenize`` honours ``--config`` and ``$DATA_LAKE_CONFIG`` like other lake
CLIs: when a deployment config is set, omit the ``OUTPUT_ROOT`` positional (it
defaults to ``lake.root``).

```bash
export DATA_LAKE_CONFIG=/path/to/lake_config.toml

dl-describe-lake --modality catalog
dl-describe-survey ALLWISE --modality catalog

dl-homogenize \
  --survey ALLWISE \
  --from-area Euclid_North \
  --transform phot_ab_v1 \
  --materialize-as ALLWISE_euclid_north_ab_v1

# Dry run: rule resolution only
dl-homogenize --from-area Euclid_North --check-only

# Or ad-hoc flags with area region:
dl-homogenize --survey ALLWISE --cone 150.1 2.2 --radius-arcsec 600 \
  --transform phot_ab_v1 --materialize-as tmp --check-only
```

Selection selectors (exactly one): `--from-area`, `--cone`, `--bbox`, `--npix` + `--norder`, `--ids`, `--where`.

Output: `catalogs/<materialize-as>/` with `kind: product`, `product_subtype: homogenized`, and provenance (`transform_id`, column lineage).

## Transform packs

| Pack | Modality | Purpose |
|------|----------|---------|
| `phot_ab_v1` | catalog | Vega/mm mag → AB via `mag_offset`, `scale`, or `flux_to_ab` (Jy flux) |
| `spec_observed_v1` | spectra | Observed-frame flux unit normalization (`flux_scale`) |
| `cutout_njy_v1` | cutout | nJy/pixel calibration via `flux_scale` on image stamps |

Bandpass metadata for FM conditioning: `data_lake/homogenize/bandpass.json`.

### Catalog rule types (`phot_ab_v1`)

| Type | Use when | Example |
|------|----------|---------|
| `mag_offset` | Native column is Vega magnitude | `ALLWISE` `w1mpro` + 2.699 |
| `scale` | Native column needs unit scaling (e.g. mmag) | `GAIA_DR3_source` G band |
| `flux_to_ab` | Native column is flux in Jy | `UNWISE_W1` `flux` with `zp: 8.906` |

`flux_to_ab` uses `m_AB = -2.5 log10(f_Jy) + zp` with WISE zero points (W1: 8.906, W2: 16.415).

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

Multi-survey: use `gather` first, then homogenize the native wide product:

```bash
dl-homogenize \
  --from-product EUCLID_wise_native \
  --transform phot_ab_v1 \
  --materialize-as EUCLID_wise_ab_v1
```

Run from an area file (uses the top-level `homogenize` block):

```bash
dl-homogenize --from-area Euclid_North
```

## Spectra and cutouts (Phase F)

```bash
dl-homogenize --modality spectra --survey SDSS_DR17 \
  --from-area Euclid_North --transform spec_observed_v1 \
  --materialize-as SDSS_DR17_spec_obs_v1

dl-homogenize --modality cutout --survey DESI_DR1 \
  --cone 150.1 2.2 --radius-arcsec 600 --transform cutout_njy_v1 \
  --materialize-as DESI_DR1_cutout_njy_v1
```

## Validation

```bash
dl-validate-homogenization --golden --transform phot_ab_v1
dl-validate-homogenization --product ALLWISE_ab_test --transform phot_ab_v1
```

## ML export from homogenized products

```bash
dl-extract-catalog --lake-root /data/lake \
  --from-product ALLWISE_euclid_north_ab_v1 \
  -c _source_id -c ra -c dec -c phot_ab_w1 \
  -o /scratch/allwise_ab.parquet
```

`--from-product` implies `product_subtype: homogenized` and writes `extract_provenance.json` (or `*.homogenize_provenance.json` for single-file exports) with `transform_id` and lineage for ML training configs.

**End-to-end workflow** (region → crossmatch → gather → homogenize → export):
[discovery/workflow.md](discovery/workflow.md) and
[`notebooks/14_discovery_workflow.ipynb`](../notebooks/14_discovery_workflow.ipynb).

See also: [Regions and areas](discovery/regions-and-areas.md), [Gather](discovery/gather.md), [Column overlays](../shared/registry/overlays/README.md).
