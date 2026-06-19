# Data lake backlog

Last updated: 2026-06-19.

## Homogenization — recipe content

### Ready catalog recipes (bundled)

`ALLWISE`, `2MASS_PSC`, `2MASS_XSC`, `ASSEF18_*`, `GAIA_DR3_source`, `VHS_DR3`, `VIDEO_DR5`, `VIKING_DR4`, `ultraVISTA_DR6`, `UNWISE_W1`, `UNWISE_W2`.

### Spectra and cutout homogenize recipes

Code supports `--modality spectra|cutout`; only `synthetic.json` has production-shaped blocks today.

- [ ] Add `spec_observed_v1` rules for ingested spectrum surveys (e.g. SDSS, DESI)
- [ ] Add `cutout_njy_v1` rules for ingested cutout surveys
- [ ] End-to-end test on real lake tiles (not just `tests/test_homogenize.py` fixtures)

### Other ingested catalogs needing `phot_ab_v1`

Run `dl-validate-homogenization --ab-coverage` on your lake for the live list (e.g. `DESI_DR1`, `EUCLID_DR1`, `SDSS_DR17`, …).

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
