# Data lake backlog

Last updated: 2026-06-18.

## Homogenization — recipe content

Catalog photometry recipes ship with the package; many still need real calibration values.

### Skeleton `phot_ab_v1` recipes (fill Vega→AB offsets)

These files have `"recipe_status": "skeleton"` and `delta: 0.0` placeholders:

- [ ] `VHS_DR3.json`
- [ ] `VIDEO_DR5.json`
- [ ] `VIKING_DR4.json`
- [ ] `ultraVISTA_DR6.json`
- [ ] `UNWISE_W1.json`
- [ ] `UNWISE_W2.json`

Validate with: `dl-validate-homogenization --ab-coverage` (requires lake root / `$DATA_LAKE_CONFIG`).

### Ready catalog recipes (no action unless lake overrides needed)

`ALLWISE`, `2MASS_PSC`, `2MASS_XSC`, `ASSEF18_*`, `GAIA_DR3_source`.

### Spectra and cutout homogenize recipes

Code supports `--modality spectra|cutout`; only `synthetic.json` has production-shaped blocks today.

- [ ] Add `spec_observed_v1` rules for ingested spectrum surveys (e.g. SDSS, DESI)
- [ ] Add `cutout_njy_v1` rules for ingested cutout surveys
- [ ] End-to-end test on real lake tiles (not just `tests/test_homogenize.py` fixtures)

## Discovery & areas — UX

- [ ] CLI to attach `crossmatch_plan` and `gather` blocks to areas (today: `dl-region --save-as` then hand-edit `areas/<id>.json`)
- [ ] Reference workflow notebook or example area: region → crossmatch → gather → homogenize → ML extract

## Export & ML

- [ ] `dl-extract-spectra-subset`: support `wavelength_mode != "shared"` (per-source grids; currently `NotImplementedError` in `SpectrumAccessor`)
- [ ] Promote query-time homogenization (`build_homogenized_view_sql`) — e.g. `dl-homogenize --check-only` view SQL or DuckDB notebook §8 as first-class recipe

## Schema registry

- [ ] Column overlays for surveys beyond `ALLWISE.catalog.json` (`shared/registry/overlays/`)

## MCP & security

- [ ] Document MCP read-only boundary (no ingest, no job submission) for team deployments
- [ ] Future: proper authentication beyond ingest token (`docs/quickstart.md` notes ACLs + token are minimal)
