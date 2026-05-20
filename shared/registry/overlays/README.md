# Column overlays (optional)

Analyst-authored JSON merged at `dl-describe-survey` time. Files are searched in order:

1. `shared/registry/overlays/<survey>.<modality>.json`
2. `shared/registry/overlays/<survey>.json`

Each file maps column names to metadata:

```json
{
  "columns": {
    "w1mpro": {
      "unit": "mag (Vega)",
      "description": "WISE W1 profile-fit magnitude",
      "homogenized_ab_offset": 2.699
    }
  }
}
```

`homogenized_ab_offset` documents the constant added to Vega magnitudes for AB (see notebook §8).
