# Per-survey homogenization registry

Homogenization recipes for **catalog**, **spectra**, and **cutout** live in one JSON file per survey. The survey name is the same across modalities at ingest (`catalogs/ALLWISE/`, `spectra/ALLWISE/`, `cutouts/ALLWISE/`), so one file drives all applicable transforms.

## Layout

| Location | Purpose |
|----------|---------|
| `data_lake/homogenize/surveys/<SURVEY>.json` | Bundled defaults (ship with the package) |
| `shared/registry/homogenize/<SURVEY>.json` | Lake-local overrides (steward edits) |

Lake overrides win over bundled defaults.

## Schema (sketch)

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
        }
      ]
    }
  },
  "spectra": {
    "spec_observed_v1": {
      "flux_array": "flux",
      "ivar_array": "ivar",
      "transform": {"type": "flux_scale", "factor": 1.0}
    }
  },
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

Omit a modality block when the survey has no data for it.

Transform profile ids (`phot_ab_v1`, `spec_observed_v1`, `cutout_njy_v1`) match the global transform packs under `data_lake/homogenize/transforms/`. Those packs define semantics; **per-survey files hold the numeric calibration**.

See [homogenization.md](../../../docs/homogenization.md).
