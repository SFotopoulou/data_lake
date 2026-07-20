# Data lake backlog

Last updated: 2026-07-06.

**Active sprint (Lake COMO):** [`docs/sprints/como-2026-07.md`](docs/sprints/como-2026-07.md) — MCP snapshot + prioritized backlog.

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

- [ ] Add `cutout_njy_v1` rules for ingested cutout surveys (no COMO cutouts yet — defer)
- [ ] End-to-end test on real lake tiles (not just `tests/test_homogenize.py` fixtures) — **COMO sprint C2**

### Other ingested catalogs needing `phot_ab_v1`

Run `dl-validate-homogenization --ab-coverage` on your lake for the live list.

**COMO (2026-07-06):** 24 catalogs missing recipes — sprint priority:
`EUCLID_DR1`, `DESI_DR1`, `SDSS_DR17` (see sprint doc). Bundled ready:
`ALLWISE`, `GAIA_DR3_source`, VISTA family, `UNWISE_W1/W2`, `2MASS_*`, `ASSEF18_*`.

## Discovery & areas — UX

- [x] Reference workflow notebook + docs ([`notebooks/14_discovery_workflow.ipynb`](notebooks/14_discovery_workflow.ipynb), [`docs/discovery/workflow.md`](docs/discovery/workflow.md); example area in [`examples/areas/`](examples/areas/))
- [x] CLI to attach `crossmatch_plan` and `gather` blocks to areas — **`dl-area`** (`set-crossmatch`, `set-gather`, `set-homogenize`, `import`)

## Visualisation

- [x] `dl-plot-source`: two-panel SED + 1D spectrum figure (`--id`, `--product`, `--spectra-survey`, `-o`); requires `--extra viz` — **v0.5.0**
- [ ] Replace illustrative bandpass ECSV files with authoritative SVO curves (all 13 `phot_ab_*` bands); add EUCLID/DESI/SDSS optical bands once `phot_ab_v1` recipes exist
- [ ] `--y-unit fnu|flam` overlay mode: convert SED points to flux density units for direct overlay on the spectrum panel (single axis)

## Export & ML

- [x] `dl-extract-spectra-subset`: flux calibration (`--flux-scale`, `--apply-survey-calibration`; SDSS/DESI bundled)
- [x] MOC export: `dl-region --export-moc` and `dl-export-moc` (`--moc-order`; requires `--extra moc`)
- [x] `dl-extract-spectra-subset`: support `wavelength_mode != "shared"` — **blocks COMO 2df/6df** (`NotImplementedError` in `SpectrumAccessor`)
- [ ] Promote query-time homogenization (`build_homogenized_view_sql`) — function exists in `homogenize/transforms.py`; not yet a CLI path (**sprint C3**)

## Schema registry

- [ ] Column overlays for surveys beyond `ALLWISE.catalog.json` (`shared/registry/overlays/`)

## MCP & security

- [ ] Document MCP read-only boundary for team deployments — **partial:** `docs/mcp-lake.md` § Out of scope; need runbook (**sprint D1**)
- [ ] Fix MCP `get_area` for legacy `areas/*.area.json` filenames (**sprint C1**)
- [ ] Future: proper authentication beyond ingest token (`docs/quickstart.md` notes ACLs + token are minimal)
