# Data lake backlog

Last updated: 2026-06-19.

## Homogenization — recipe content

### Ready catalog recipes (bundled)

`ALLWISE`, `2MASS_PSC`, `2MASS_XSC`, `ASSEF18_*`, `GAIA_DR3_source`, `VHS_DR3`, `VIDEO_DR5`, `VIKING_DR4`, `ultraVISTA_DR6`, `UNWISE_W1`, `UNWISE_W2`.

### Spectra extract calibration (SDSS, DESI)

Constant flux unit normalization is applied at **extract** time, not via
`dl-homogenize --modality spectra`.  Per-survey factors live in
`spectra.flux_calibration` inside `homogenize/surveys/<SURVEY>.json`:

```json
"spectra": {
  "flux_calibration": {
    "native_flux_unit": "1e-17 erg/s/cm2/Angstrom",
    "output_flux_unit": "erg/s/cm2/Angstrom",
    "flux_scale": 1e-17
  }
}
```

Use `dl-extract-spectra-subset --apply-survey-calibration` or `--flux-scale`.
See [export-and-sharing.md](export-and-sharing.md).

### Spectra and cutout homogenize recipes (zarr engine)

Code supports `--modality spectra|cutout` for tile-level product transforms;
only `synthetic.json` has production-shaped blocks today.  **Do not** use this
path for SDSS/DESI constant flux scaling — use extract calibration above.

- [ ] Add `cutout_njy_v1` rules for ingested cutout surveys
- [ ] End-to-end test on real lake tiles (not just `tests/test_homogenize.py` fixtures)

### Other ingested catalogs needing `phot_ab_v1`

Run `dl-validate-homogenization --ab-coverage` on your lake for the live list (e.g. `DESI_DR1`, `EUCLID_DR1`, `SDSS_DR17`, …).

## Discovery & areas — UX

- [ ] CLI to attach `crossmatch_plan` and `gather` blocks to areas (today: `dl-region --save-as` then hand-edit `areas/<id>.json`)
- [ ] Reference workflow notebook or example area: region → crossmatch → gather → homogenize → ML extract

## Export & ML

- [x] `dl-extract-spectra-subset`: flux calibration (`--flux-scale`, `--apply-survey-calibration`; SDSS/DESI bundled)
- [ ] `dl-extract-spectra-subset`: support `wavelength_mode != "shared"` (per-source grids; currently `NotImplementedError` in `SpectrumAccessor`)
- [ ] Promote query-time homogenization (`build_homogenized_view_sql`) — e.g. `dl-homogenize --check-only` view SQL or DuckDB notebook §8 as first-class recipe

## Schema registry

- [ ] Column overlays for surveys beyond `ALLWISE.catalog.json` (`shared/registry/overlays/`)

## MCP & security

- [ ] Document MCP read-only boundary (no ingest, no job submission) for team deployments
- [ ] Future: proper authentication beyond ingest token (`docs/quickstart.md` notes ACLs + token are minimal)
