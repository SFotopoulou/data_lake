# Per-survey homogenization registry

Homogenization recipes for **catalog**, **spectra**, and **cutout** live in one JSON file per survey. The survey name is the same across modalities at ingest (`catalogs/ALLWISE/`, `spectra/ALLWISE/`, `cutouts/ALLWISE/`), so one file drives all applicable transforms.

## Layout

| Location | Purpose |
|----------|---------|
| `data_lake/homogenize/surveys/<SURVEY>.json` | Bundled defaults (ship with the package) |
| `shared/registry/homogenize/<SURVEY>.json` | Lake-local overrides (steward edits) |

Lake overrides win over bundled defaults. Omit a modality block when the survey has no data for it.

## Schema (catalog — phot_ab_v1)

Catalog homogenization converts native photometry columns to AB magnitudes.

```json
{
  "survey": "ALLWISE",
  "version": 1,
  "catalog": {
    "phot_ab_v1": {
      "output_system": "AB_mag",
      "rules": [
        {
          "source_column": "w1mpro",
          "target_column": "phot_ab_w1",
          "transform": {"type": "mag_offset", "delta": 2.699}
        },
        {
          "source_column": "w1sigmpro",
          "target_column": "phot_ab_w1_err",
          "transform": {"type": "scale", "factor": 1.0}
        }
      ]
    }
  }
}
```

### Catalog rule types

All five catalog transform types are supported. Built-in sentinels (`-9999`, `9999`, `-999`, `999`, NaN) are cleaned automatically before any formula.

| Type | Parameters | Use when |
|------|------------|----------|
| `mag_offset` | `delta` | Native column is a Vega magnitude; add AB offset |
| `scale` | `factor` | Native column needs unit scaling (e.g. mmag → mag) |
| `identity` | — | Pass-through with sentinel clean (rename only) |
| `null_if_sentinel` | `values` (optional list) | Survey uses non-standard sentinel values (e.g. `99.0`) |
| `flux_to_ab` | `zp` | Native column is flux; `zp=8.906` for Jy, `zp=23.9` for µJy |
| `inverse` | — | Reciprocal: `1 / source` (zero → null). No auto uncertainty — pair with `uncertainty_transform: {type: inverse}` for `1/err` if needed. |

**Pairing uncertainty columns:** set `uncertainty_column` + `target_uncertainty_column` on the same rule entry (not as a separate rule). By default the engine auto-propagates uncertainty from the value `transform` type. Override with an explicit `uncertainty_transform` (same `type` vocabulary) when the error needs a different formula than the value.

`transform` and `uncertainty_transform` may each be a **single object** or a **non-empty list of steps** applied left-to-right (e.g. `scale` then `flux_to_ab`).

```json
{
  "source_column": "w1mpro",
  "target_column": "phot_ab_w1",
  "uncertainty_column": "w1sigmpro",
  "target_uncertainty_column": "phot_ab_w1_err",
  "transform": {"type": "mag_offset", "delta": 2.699},
  "native_system": "Vega"
}
```

**Explicit uncertainty transform** (e.g. mmag errors → mag while values pass through):

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

**`identity` example** (rename + sentinel clean, no arithmetic):

```json
{
  "source_column": "mag_r_auto",
  "target_column": "phot_ab_r",
  "uncertainty_column": "magerr_r_auto",
  "target_uncertainty_column": "phot_ab_r_err",
  "transform": {"type": "identity"}
}
```

**`null_if_sentinel` example** (custom survey sentinels `99.0` and `-99.0`):

```json
{
  "source_column": "mag_aper_3",
  "target_column": "phot_ab_r",
  "uncertainty_column": "magerr_aper_3",
  "target_uncertainty_column": "phot_ab_r_err",
  "transform": {"type": "null_if_sentinel", "values": [99.0, -99.0]}
}
```

## Schema (spectra — two shapes)

There are two distinct spectra recipe shapes used in practice:

### Shape A — extract-time calibration (`flux_calibration`)

Used by SDSS_DR17 and DESI_DR1. Applied at **spectrum extraction time** (not by `dl-homogenize`). The Zarr engine reads the calibration factor from this block and scales flux during `dl-ingest-spectra`. **No `spec_observed_v1` block is needed** for these surveys.

```json
{
  "survey": "SDSS_DR17",
  "version": 1,
  "spectra": {
    "flux_calibration": {
      "native_flux_unit": "1e-17 erg/s/cm2/Angstrom",
      "output_flux_unit": "erg/s/cm2/Angstrom",
      "flux_scale": 1e-17,
      "reference": "SDSS spectroscopic pipeline (BOSS DR17)"
    }
  }
}
```

### Shape B — homogenize transform (`spec_observed_v1`)

For surveys where `dl-homogenize --modality spectra` applies a uniform scale factor across all spectra. The Zarr engine scales the fixed internal flux arrays (`flux`, `ivar`) by the factor — the field names `flux_array`/`ivar_array` in the recipe are **not used** by the engine (internal array names are fixed by the ingest schema).

```json
{
  "survey": "MY_SURVEY",
  "version": 1,
  "spectra": {
    "spec_observed_v1": {
      "transform": {"type": "flux_scale", "factor": 1e-20},
      "wavelength_unit": "Angstrom"
    }
  }
}
```

## Schema (cutout — cutout_njy_v1)

```json
{
  "survey": "MY_SURVEY",
  "version": 1,
  "cutout": {
    "cutout_njy_v1": {
      "image_array": "images",
      "transform": {"type": "flux_scale", "factor": 1.0},
      "bands": {
        "W1": {"flux_scale": 2.5e10, "reference": "WISE docs ZP"}
      }
    }
  }
}
```

`flux_scale` is the only Zarr rule type (used by both `spec_observed_v1` and `cutout_njy_v1`). The catalog engine has five rule types (`mag_offset`, `scale`, `identity`, `null_if_sentinel`, `flux_to_ab`).

Transform profile IDs (`phot_ab_v1`, `spec_observed_v1`, `cutout_njy_v1`) match the global transform packs under `data_lake/homogenize/transforms/`. Those packs define semantics only (no per-survey rules). **Per-survey files are the sole source of executable rules**; lake overrides replace bundled defaults.

See [homogenization.md](../../../docs/homogenization.md) for the full workflow.
