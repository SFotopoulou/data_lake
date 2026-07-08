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

Transform types: `mag_offset` (Vega → AB), `scale` (unit change), `flux_to_ab` (Jy flux → AB mag).

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

Transform profile IDs (`phot_ab_v1`, `spec_observed_v1`, `cutout_njy_v1`) match the global transform packs under `data_lake/homogenize/transforms/`. Those packs define semantics only (no per-survey rules). **Per-survey files are the sole source of executable rules**; lake overrides replace bundled defaults.

See [homogenization.md](../../../docs/homogenization.md) for the full workflow.
