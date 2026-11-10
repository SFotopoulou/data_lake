# Homogenization

Ingest keeps each survey's native column names and photometric systems (Vega magnitudes, Jy fluxes, and so on). Homogenization is a separate, opt-in step that writes a new **product catalog** whose columns are comparable across surveys, for example AB magnitudes in `phot_ab_*` columns. Use this page when you need consistent photometry for SED fitting, photometric redshifts or ML training.

## When homogenization runs

Only `dl-homogenize` changes units. Ingest and crossmatch never do.

| Stage | Homogenizes? |
|-------|--------------|
| Ingest (`dl-ingest-catalog*`) | No. Native values are kept as archived. |
| Crossmatch (`dl-crossmatch`) | No. It matches positions only. |
| Homogenize (`dl-homogenize`) | Yes. You choose a transform pack, a survey (or product) and a region. |

## Primary workflow: single survey + region

Set your deployment config, inspect the survey's columns, then homogenize a region into a named product. When `--config` or `$DATA_LAKE_CONFIG` is set you can omit the `OUTPUT_ROOT` positional argument, which defaults to `lake.root`.

```bash
export DATA_LAKE_CONFIG=/path/to/lake_config.toml

dl-describe-lake --modality catalog
dl-describe-survey ALLWISE --modality catalog

dl-homogenize \
  --survey ALLWISE \
  --from-area Euclid_North \
  --transform phot_ab_v1 \
  --materialize-as ALLWISE_euclid_north_ab_v1

# Dry run: resolve rules and the selection, write nothing
# (--survey must be given here or in the area's homogenize block)
dl-homogenize --from-area Euclid_North --check-only

# Same dry run with a cone instead of an area
dl-homogenize --survey ALLWISE --cone 150.1 2.2 --radius-arcsec 600 \
  --transform phot_ab_v1 --materialize-as tmp --check-only
```

`--check-only` prints a JSON summary of which rules would be applied and which were skipped because the source column is missing. Run it before a large job.

Before you start, check which ingested catalogs have a usable `phot_ab_v1` recipe:

```bash
dl-validate-homogenization --ab-coverage
```

This needs a lake root, from `OUTPUT_ROOT` or `$DATA_LAKE_CONFIG`.

### Options

You must give exactly one region selector. The exception is `--from-product` in catalog modality, where you can omit the selector to process every tile of the product.

| Option | Meaning |
|--------|---------|
| `--survey NAME` | Native survey to homogenize. Cannot be combined with `--from-product` for catalogs. |
| `--from-product NAME` | Homogenize a gathered product (catalog modality). For spectra or cutouts it limits the tiles to those covered by the product. |
| `--from-area ID` | Use the area's region, and its `homogenize` block if present. |
| `--cone RA DEC` + `--radius-arcsec R` | Cone selection (degrees, arcsec). |
| `--bbox RA_MIN RA_MAX DEC_MIN DEC_MAX` | Box selection (four values). |
| `--npix LIST` + `--norder N` | Explicit HEALPix pixels (comma-separated values and `a-b` ranges). |
| `--ids FILE` | Select sources from an ID file. |
| `--where SQL` | SQL predicate on the base catalog. |
| `--transform ID` | Transform pack. The default is `phot_ab_v1` (catalog), `spec_observed_v1` (spectra) or `cutout_njy_v1` (cutout). |
| `--materialize-as NAME` | Output product name. Required unless `--check-only` is set or the area provides it. |
| `--modality` | `catalog` (default), `spectra` or `cutout`. |
| `--columns A,B,…` | Restrict to rules for these native columns. By default, rules are applied to columns whose manifest role is photometry. |
| `--overwrite` | Replace an existing product. Without it, an existing output is an error. |
| `--n-workers N` | Parallel workers (default 8). |
| `--check-only` | Validate rules and the selection without writing. |

The output location depends on the modality: `products/<name>/` for catalogs, `spectra/<name>/` for spectra and `cutouts/<name>/` for cutouts. A catalog product name must not match an ingested survey under `catalogs/`. Every output records `kind: product`, `product_subtype: homogenized` and provenance (`transform_id` and column lineage).

Column matching uses the survey manifest (what `dl-describe-survey` shows) at run time, so you do not need to list the survey's columns yourself.

## Areas

An area file (`areas/<id>.json`) can hold a top-level `homogenize` block so that `dl-homogenize --from-area <id>` needs no other flags:

```json
"homogenize": {
  "survey": "ALLWISE",
  "region": { "from_area": "Euclid_North" },
  "transform": "phot_ab_v1",
  "materialize_as": "ALLWISE_euclid_north_ab_v1"
}
```

Set the block with [`dl-area set-homogenize`](discovery/areas-cli.md), then run:

```bash
dl-homogenize --from-area Euclid_North
```

Command-line flags override values from the block.

For several surveys at once, run `dl-gather` first to build a wide native product, then homogenize that product:

```bash
dl-homogenize \
  --from-product EUCLID_wise_native \
  --transform phot_ab_v1 \
  --materialize-as EUCLID_wise_ab_v1
```

## Spectra and cutouts

Spectra and cutout homogenization need a `spec_observed_v1` or `cutout_njy_v1` block in the survey recipe. The bundled `SDSS_DR17.json` and `DESI_DR1.json` recipes do not include these blocks yet. Add a lake-level override in `shared/registry/homogenize/<SURVEY>.json` (the schema is described in that directory's `README.md`). The bundled `synthetic` recipe is a working example of both blocks.

```bash
# After adding a spec_observed_v1 block to the survey recipe:
dl-homogenize --modality spectra --survey SDSS_DR17 \
  --from-area Euclid_North --transform spec_observed_v1 \
  --materialize-as SDSS_DR17_spec_obs_v1

# After adding a cutout_njy_v1 block to the survey recipe:
dl-homogenize --modality cutout --survey DESI_DR1 \
  --cone 150.1 2.2 --radius-arcsec 600 --transform cutout_njy_v1 \
  --materialize-as DESI_DR1_cutout_njy_v1
```

## Validation

```bash
# Golden input/output spot checks for a transform pack
dl-validate-homogenization --golden --transform phot_ab_v1

# Check a materialized product (needs a lake root)
dl-validate-homogenization --product ALLWISE_ab_test --transform phot_ab_v1
```

Add `--strict` to treat warnings as errors.

## ML export from homogenized products

```bash
dl-extract-catalog --lake-root /data/lake \
  --from-product ALLWISE_euclid_north_ab_v1 \
  -c _source_id -c ra -c dec -c phot_ab_w1 \
  -o /scratch/allwise_ab.parquet
```

`--from-product` checks that the catalog is a homogenized product and writes provenance (`transform_id` and lineage) for your training config. Single-file exports write it to `<output>.homogenize_provenance.json`, and `--output-dir` exports write `extract_provenance.json`. Pass `--no-homogenize-provenance` to skip it.

For the full end-to-end workflow (region, crossmatch, gather, homogenize, export), see [discovery/workflow.md](discovery/workflow.md) and [`notebooks/14_discovery_workflow.ipynb`](../notebooks/14_discovery_workflow.ipynb).

## Recipe files

Each survey has a recipe file. `dl-homogenize` first looks for a lake override in `shared/registry/homogenize/<SURVEY>.json`, then falls back to the bundled `data_lake/homogenize/surveys/<SURVEY>.json`. Transform packs (`phot_ab_v1` and others) only define what the output means. The executable rules are always in the per-survey recipe files.

## Transform packs

| Pack | Modality | Purpose |
|------|----------|---------|
| `phot_ab_v1` | catalog | Convert photometry to AB magnitudes (rule types `mag_offset`, `scale`, `identity`, `null_if_sentinel`, `flux_to_ab`, `inverse`) |
| `spec_observed_v1` | spectra | Normalise observed-frame flux units (`flux_scale`) |
| `cutout_njy_v1` | cutout | Calibrate image stamps to nJy/pixel (`flux_scale`) |

Band metadata used for SED plots (effective wavelength, width) is in `data_lake/homogenize/bandpass.json`. See [plotting.md](plotting.md).

## Rule types reference

### Catalog rules (`phot_ab_v1`)

Each rule reads one native column and writes one target column. Before the formula is applied, values equal to `-9999`, `9999`, `-999` or `999`, and NaN, are set to null. Rules whose source column is not numeric are skipped with a warning.

| Type | Required params | Behaviour | Default uncertainty propagation |
|------|-----------------|-----------|------------------------|
| `mag_offset` | `delta` | `target = source + delta` | copied (null when the target is null) |
| `scale` | `factor` | `target = source × factor` | `err × factor` |
| `identity` | none | copy after sentinel cleaning | copied (null when the target is null) |
| `null_if_sentinel` | `values` (optional list) | copy, also nulling the listed values | copied (null when the target is null) |
| `flux_to_ab` | `zp` | `target = −2.5 log₁₀(source) + zp`; source ≤ 0 becomes null | `2.5 / ln(10) × dflux / flux` |
| `inverse` | none | `target = 1 / source`; source = 0 becomes null | copied; set `uncertainty_transform: inverse` to get `1/err` |

Fields used on every rule:

| Field | Required | Description |
|-------|----------|-------------|
| `source_column` | yes | Native column name in the survey tile |
| `target_column` | yes | Output column name in the homogenized product |
| `transform` | yes | An object with `type` and its parameters, or a non-empty list of such objects applied left to right (a chain) |
| `uncertainty_column` | no | Native uncertainty column paired with `source_column` |
| `target_uncertainty_column` | no | Output uncertainty column; needed together with `uncertainty_column` |
| `uncertainty_transform` | no | Explicit transform for the uncertainty (same `type` vocabulary). When omitted, the uncertainty is propagated from the value transform as in the table above |
| `native_system` | no | Metadata only (for example `"Vega"` or `"uJy"`); it does not affect the computation |

#### Chained transforms

A rule can apply several steps in order if `transform` is a list:

```json
{
  "source_column": "flux_ujy",
  "target_column": "phot_ab_x",
  "uncertainty_column": "flux_ujy_err",
  "target_uncertainty_column": "phot_ab_x_err",
  "transform": [
    {"type": "scale", "factor": 1e-6},
    {"type": "flux_to_ab", "zp": 8.906}
  ]
}
```

Each step works on the result of the previous one. Built-in sentinel cleaning happens only on the first step, which reads the native column. `uncertainty_transform` can also be a list. When it is omitted, propagation follows the chain: `scale` multiplies the error, and `flux_to_ab` uses the flux after any preceding scales.

#### Explicit `uncertainty_transform`

Set `uncertainty_transform` when the default propagation is wrong for your data. For example, use it when native errors are in millimag but the magnitudes are already in mag:

```json
{
  "source_column": "phot_g_mean_mag",
  "target_column": "phot_ab_g",
  "uncertainty_column": "phot_g_mean_mag_error",
  "target_uncertainty_column": "phot_ab_g_err",
  "transform": {"type": "identity"},
  "uncertainty_transform": {"type": "scale", "factor": 0.001}
}
```

It requires both `uncertainty_column` and `target_uncertainty_column`. Supported types:

| Type | Effect on the uncertainty |
|------|--------------------------|
| `scale` | `err × factor` |
| `mag_offset` | `err + delta` (only when set explicitly; a value `mag_offset` never changes the error) |
| `identity` / `null_if_sentinel` | copy, null when the value target is null |
| `flux_to_ab` | `2.5 / ln(10) × dflux / flux`, using `source_column` as the flux (`zp` is not used) |
| `inverse` | `1 / err`; zero becomes null |

#### `flux_to_ab` — flux units and zero points

The formula `m_AB = −2.5 log₁₀(f) + zp` only gives AB magnitudes if `zp` matches the unit of `f`. For a flux density in Jy, the AB zero point is 8.90 (8.906 is used in recipes). For µJy it is 23.9, as in EUCLID-style columns such as `flux_vis_2fwhm_aper`. Do not mix units and zero points.

The bundled `UNWISE_W1` recipe uses `zp = 8.906` and `UNWISE_W2` uses `zp = 16.415`. Both recipes describe the native system as "Vega Jy". Check the native flux unit before you reuse these values for another survey.

#### `null_if_sentinel` — custom sentinel values

Some surveys mark missing data with their own values (for example `99.0` for saturated detections). List them in `values`:

```json
{
  "type": "null_if_sentinel",
  "values": [99.0, -99.0]
}
```

With an empty or missing `values` list it behaves like `identity` (built-in sentinels only).

### Spectra and cutout rules

| Rule type | Used in pack | Parameters | Effect |
|-----------|-------------|------------|--------|
| `flux_scale` | `spec_observed_v1`, `cutout_njy_v1` | `factor` | Multiplies flux by `factor` (and ivar by `1/factor²`) |
| `flux_calibration` | survey recipe only | `flux_scale` | Not run by `dl-homogenize`. It is applied at export time by `dl-extract-spectra-subset --apply-survey-calibration` |

In a `spec_observed_v1` block, set `transform.type = "flux_scale"` and `transform.factor`. A `cutout_njy_v1` block also accepts per-band factors in `bands.<name>.flux_scale`, which override the default factor for that band.

Recipe schema: [Per-survey recipe schema](../shared/registry/homogenize/README.md).

## Query-time (exploration)

To try a conversion in SQL without writing a product, generate the view SQL with `build_homogenized_view_sql` (`data_lake.homogenize.transforms`). Section 8 of [`notebooks/11_duckdb_catalog_query.ipynb`](../notebooks/11_duckdb_catalog_query.ipynb) shows an example. This SQL is for exploration only, because it does not reproduce the materialized results exactly:

- It does not compute uncertainty columns.
- Built-in sentinel values are nulled only in `null_if_sentinel` rules (and `flux_to_ab` nulls values ≤ 0).

For science or training data, materialize with `dl-homogenize`.

## Troubleshooting

**`AppendRowGroups requires equal schemas` during finalize.** This comes from older products built from sparse gather tiles, where a partner band was missing on some HEALPix tiles. Current versions write the missing targets as null columns so every tile shares one Parquet schema. Re-run `dl-homogenize … --overwrite`.

**`Provide exactly one selection selector`.** Give exactly one of `--from-area`, `--ids`, `--where`, `--cone`, `--bbox` or `--npix`. Omit them only with `--from-product` in catalog modality.

**`product catalog already exists`.** Add `--overwrite`, or choose a new `--materialize-as` name.

See also: [Area plans (`dl-area`)](discovery/areas-cli.md), [Regions and areas](discovery/regions-and-areas.md), [Gather](discovery/gather.md), [Column overlays](../shared/registry/overlays/README.md), [SED + spectrum plot](plotting.md).
