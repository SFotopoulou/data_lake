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

# Dry run: rule resolution only (--survey must be set here or in the area's homogenize block)
dl-homogenize --from-area Euclid_North --check-only

# Or ad-hoc flags with area region:
dl-homogenize --survey ALLWISE --cone 150.1 2.2 --radius-arcsec 600 \
  --transform phot_ab_v1 --materialize-as tmp --check-only
```

Region selectors (at most one required): `--from-area`, `--cone`, `--bbox`, `--npix` + `--norder`, `--ids`, `--where`. When `--from-product` is used (catalog modality only), no region selector is needed — all tiles in the product are processed automatically.

Output directory depends on modality: `catalogs/<materialize-as>/` (catalog), `spectra/<materialize-as>/` (spectra), `cutouts/<materialize-as>/` (cutout). All outputs carry `kind: product`, `product_subtype: homogenized`, and provenance (`transform_id`, column lineage).

## Transform packs

| Pack | Modality | Purpose |
|------|----------|---------|
| `phot_ab_v1` | catalog | Photometric unit conversion to AB magnitudes (`mag_offset`, `scale`, `identity`, `null_if_sentinel`, `flux_to_ab`) |
| `spec_observed_v1` | spectra | Observed-frame flux unit normalisation (`flux_scale`) |
| `cutout_njy_v1` | cutout | nJy/pixel calibration via `flux_scale` on image stamps |

Bandpass metadata for SED conditioning: `data_lake/homogenize/bandpass.json`.

## Rule types reference

### Catalog rules (`phot_ab_v1`)

Each rule operates on one source column and writes one target column. All catalog rules apply **built-in sentinel cleaning** automatically before the formula: values equal to `-9999`, `9999`, `-999`, or `999`, and IEEE NaN, are replaced with `null` before any arithmetic.

| Type | Required params | Behaviour | Uncertainty propagation |
|------|-----------------|-----------|------------------------|
| `mag_offset` | `delta` | `target = source + delta` | copied (nulled when source is null) |
| `scale` | `factor` | `target = source × factor` | `err × factor` |
| `identity` | — | copy after built-in sentinel clean | copied (nulled when source is null) |
| `null_if_sentinel` | `values` (optional list) | copy; additionally null values in `values` and NaN | copied (nulled when source or uncertainty is null) |
| `flux_to_ab` | `zp` | `target = −2.5 log₁₀(source) + zp`; source ≤ 0 → null | `2.5 / ln(10) × dflux / flux` |

**JSON fields used on every rule:**

| Field | Required | Description |
|-------|----------|-------------|
| `source_column` | yes | Native column name in the survey tile |
| `target_column` | yes | Output column name in the homogenized product |
| `uncertainty_column` | no | Native uncertainty column paired with `source_column` |
| `target_uncertainty_column` | no | Output uncertainty column; must accompany `uncertainty_column` |
| `native_system` | no | Metadata only (e.g. `"Vega"`); does not affect computation |

#### `flux_to_ab` — flux units and zero points

The formula `m_AB = −2.5 log₁₀(f) + zp` expects `f` in whatever units the zero point was derived for:

| Flux unit | Zero point (`zp`) | Example survey |
|-----------|-------------------|----------------|
| Jy | 8.906 (W1), 16.415 (W2) | `UNWISE_W1` |
| µJy | 23.9 | EUCLID-style columns |

Use `zp=8.906` for Jy flux; use `zp=23.9` for µJy flux (e.g. EUCLID `flux_vis_2fwhm_aper`). Do **not** mix units and zero points.

#### `null_if_sentinel` — custom sentinel values

Surveys sometimes encode missing data with survey-specific values (e.g. `99.0` for saturated detections) not covered by the built-in set. Use `null_if_sentinel` with an explicit `values` list:

```json
{
  "type": "null_if_sentinel",
  "values": [99.0, -99.0]
}
```

An empty or omitted `values` list behaves identically to `identity` (built-in sentinels only).

#### Query-time SQL

`build_homogenized_view_sql` supports all five catalog rule types and generates `CASE WHEN` expressions for sentinel nulling. It does **not** propagate uncertainty columns in SQL — for production use, always materialise with `dl-homogenize` instead.

### Spectra and cutout rules

| Rule type | Used in pack | Parameters | Effect |
|-----------|-------------|------------|--------|
| `flux_scale` | `spec_observed_v1`, `cutout_njy_v1` | `factor` | Scales flux by `factor`; ivar by `1/factor²` |
| `flux_calibration` | survey recipe only | `flux_scale` | Applied at ingest/export time (SDSS/DESI); **not** run by `dl-homogenize` |

For `spec_observed_v1`: the transform block in the survey recipe sets `transform.type = "flux_scale"` and `transform.factor`. For `cutout_njy_v1`: per-band overrides via `bands.<name>.flux_scale` are also supported.

Cross-reference: [Per-survey recipe schema](../shared/registry/homogenize/README.md).

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

Run from an area file (uses the top-level `homogenize` block — set it with
[`dl-area set-homogenize`](discovery/areas-cli.md)):

```bash
dl-homogenize --from-area Euclid_North
```

## Spectra and cutouts (Phase F)

Spectra and cutout homogenization requires a `spec_observed_v1` or `cutout_njy_v1` block in the survey recipe. The bundled `SDSS_DR17.json` and `DESI_DR1.json` do **not** yet include these blocks — add a lake-level override in `shared/registry/homogenize/<SURVEY>.json` (see the schema in that directory's `README.md`). The `synthetic` recipe is a working reference.

```bash
# After adding spec_observed_v1 block to the survey recipe:
dl-homogenize --modality spectra --survey SDSS_DR17 \
  --from-area Euclid_North --transform spec_observed_v1 \
  --materialize-as SDSS_DR17_spec_obs_v1

# After adding cutout_njy_v1 block to the survey recipe:
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

See also: [Area plans (`dl-area`)](discovery/areas-cli.md), [Regions and areas](discovery/regions-and-areas.md), [Gather](discovery/gather.md), [Column overlays](../shared/registry/overlays/README.md), [SED + spectrum plot](plotting.md).
