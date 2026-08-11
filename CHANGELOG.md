# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`dl-ingest-catalog --set-column NAME=VALUE`** — inject a constant column into every row *before* identity resolution, so composite `--link-id-col` specs can reference values that are not present in the source file (e.g. `VISIT`). Repeatable; auto-infers int64/float64/string, or use `NAME:TYPE=VALUE` to force a type (`int`, `float`, `str`, `bool`). Provenance is stored in `catalog_info.json` as `set_columns: {NAME: VALUE}`. `--tile-mode append` raises a clear error when the on-disk tile schema does not match (all runs for a survey must use the same injected columns from the first write).

### Changed

- **Breaking: `dl-describe-lake --from-area` removed** — describe-lake is lake-wide only. Use `dl-region --from-area AREA --modalities …,crossmatch` (optionally `--count`) for area-scoped inventory including crossmatch trees.
- **`dl-region` crossmatch discovery** — `crossmatch` is a discoverable modality (`--modalities catalog,spectra,crossmatch`); tile indices include `*.crossmatch.json`; `--count` sums Parquet footers for catalog and crossmatch overlap tiles. Default modalities remain catalog-only.

### Fixed

- **`dl-region` spectra row estimates always `~0`** — `spectrum_info.json` from ingest often omits `total_rows` / `total_spectra`, so estimates fell through to 0 even when overlap tiles existed. Discovery now accepts `total_spectra` (and cutout equivalents), falls back to the same Zarr `_source_id` length sum used by `dl-describe-lake`, and `--count` sums those lengths over overlap tiles. New spectrum ingest writes `total_rows` into the sidecar when known.
- **`dl-describe-lake --kind ingested` omitted spectra/cutouts** — registry rows for those modalities now set provenance `kind` (default `ingested`; homogenized products keep `product`). Stale `surveys.parquet` with null `kind` on spectra/cutouts is coalesced to `ingested` at filter/display time, so `--kind ingested` lists catalogs and spectra without requiring a refresh (refresh still rewrites the on-disk column).
- **`dl-crossmatch --from-area` sidecar shrink** — `crossmatch_info.json` `total_rows` / `n_tiles` are always rewritten from a full on-disk Parquet recount after every run (sky and column), so a region-bounded gap-fill cannot make `dl-describe-lake` look like full-sky matches vanished. `--overwrite` is rejected with `--from-area` (gap-fill only; existing tiles outside the area are left untouched).
- **`dl-describe-lake` crossmatch counts** — registry / describe prefer on-disk Parquet footers over the sidecar (lake-wide totals).
- **`dl-gather` unmatched partner padding** — null-padded partner columns now use the partner catalog's real Arrow/Polars dtypes (string, int, float, …) instead of hard-coded `Float64`. Fixes schema conflicts such as `SDSS_DR17_CLASS` (`large_string` on matched tiles vs float on unmatched).
- **`dl-homogenize --from-product` float vs string** — product schema union prefers string/binary over float when tiles disagree, so already-gathered products with legacy float64 null pads can homogenize without re-gathering.
- **`dl-extract-catalog` FITS string columns** — nullable Arrow strings are written as fixed-width unicode (null → `""`) so Astropy no longer raises `unsupported object types or mixed types`. Schema unify also prefers string over float for legacy gather pads.
- **Homogenize transform chains** — catalog recipe `transform` / `uncertainty_transform` may be a single object (unchanged) or a non-empty list of steps applied left-to-right (e.g. `scale` then `flux_to_ab`).
- **`inverse` catalog transform** — `target = 1 / source` (null when source is 0); no auto uncertainty propagation (optional `uncertainty_transform: {type: inverse}` for `1/err`).

## [0.6.1] - 2026-07-20

### Added

- **Per-source wavelength support in `dl-extract-spectra-subset`** — surveys ingested with `wavelength_mode="per_source"` (2dF, 6dF, SDSS spPlate) can now be subset-extracted to Zarr, Parquet, HDF5, and FITS (per-file layout). Zarr output stores `wavelength` as `(N_written, N_pix) float32` with `wavelength_mode="per_source"` in attrs; Parquet adds a per-row `wavelength` list column. Cross-tile width differences are handled by padding shorter tiles to the maximum `N_pix` with `nan`/`0`. FITS catalog layout raises a `ValueError` with a clear message directing users to `--fits-layout per-file`.

### Fixed

- `SpectrumAccessor._iter_subset_tile_batches` now delegates to `SpectrumTileStore.get_spectra` for both shared and per-source reads, removing duplicate orthogonal-selection code and the hard-coded shared-only path.

## [0.6.0] - 2026-07-20

### Changed

- **Breaking on-disk layout: derived products move to `products/`** — `dl-gather` and `dl-homogenize` now write catalog products to `<lake_root>/products/<name>/` instead of `catalogs/<name>/`. No migration is provided; re-run `dl-gather` and `dl-homogenize --overwrite` to regenerate under the new location. Ingested surveys stay in `catalogs/` and are unaffected.
- **Name resolver** — all read-side code (`CatalogAccessor`, `CatalogExtractor`, tile index, homogenize engine, crossmatch, MCP server, pack-tile, query-from-master, validate-homogenization) now calls `resolve_catalog_root(lake_root, name)`, which checks `catalogs/<name>` first and falls back to `products/<name>`. References by name continue to work transparently.
- **Collision guard** — `dl-gather` and `dl-homogenize` refuse to write a product if a same-named ingested survey already exists under `catalogs/`, preventing resolver ambiguity.
- **Lake registry scans both trees** — `dl-describe-lake --refresh` and `build_lake_registry_table` now iterate both `catalogs/` and `products/`, so products appear with `kind=product` and `path` starting with `products/`.
- **`dl-init` creates `products/`** — new lake deployments include a `products/` subdirectory alongside `catalogs/`, `spectra/`, `cutouts/`, and `shared/`.
- `[paths]` section of `lake_config.toml` now includes a `products` key (default: `"products"`); existing configs without the key continue to work via the default.

### Added

- **`resolve_catalog_root` / `catalog_write_root`** in `data_lake/schema_registry.py` — central path-resolution helpers; use these instead of hard-coding `lake_root / "catalogs" / name` in any new code.
- **`CATALOGS_LAYER` / `PRODUCTS_LAYER`** constants in `schema_registry.py`.

### Fixed

- **`dl-homogenize --from-product` schema mismatch** — tiles with missing partner photometry (or null-typed partner columns) no longer fail finalize with `AppendRowGroups requires equal schemas`. Output tiles are aligned to a canonical schema before write.
- **`dl-describe-lake` / lake registry** — crossmatch fields (`match_mode`, `match_col_a`, …) are no longer dropped when catalogs are present (PyArrow `from_pylist` takes the first-row schema). Column trees show `detail=col:<col_a>:<col_b>`; survey column width adapts for long `__col_` names.

### Added (carried forward from Unreleased)

- **`uncertainty_transform` on catalog homogenize rules** — optional per-rule transform for uncertainty columns (same types as value transforms: `scale`, `mag_offset`, `identity`, `null_if_sentinel`, `flux_to_ab`). When omitted, auto-propagation from the value `transform` type is unchanged.
- **Column crossmatch tree naming simplified** — tree directories are now `{A}_x_{B}__col_{col_a}__{col_b}` (column names embedded, `__` separator). The previous `--match-id` label and Blake2 hex token are dropped. `dl-crossmatch` no longer requires `--match-id`; partner specs change from `SURVEY:col:MATCH_ID:COL_A:COL_B` to `SURVEY:col:COL_A:COL_B`. `dl-describe-lake` detail now shows `col:COL_A:COL_B` — paste directly into a partner spec. No backwards compatibility with `__col_<id>_<hex>` trees; re-run crossmatch.

## [0.5.0] - 2026-07-07

### Added

- **`dl-plot-source`** — two-panel SED + 1D spectrum figure for any source identified by its lake `_source_id`.  Reads photometry from a homogenized product catalog and the spectrum from a Zarr spectra store; shares a log-wavelength x-axis in micron.  Requires `pip install "data-lake[viz]"` (new `viz` optional extra that pulls in `matplotlib>=3.7`).
- **`BandpassRegistry`** (`data_lake/homogenize/bandpass.py`) — resolves per-band effective wavelength, FWHM, and optional transmission-curve files with lake-override semantics (lake `shared/registry/bandpasses/` beats bundled defaults).
- **`bandpass.json` extended** — all 13 `phot_ab_*` bands now carry `lambda_eff_um`, `fwhm_um`, and (where available) a `"curve"` filename: WISE W1–W4, Gaia G/BP/RP, VISTA Z/Y/Ks, 2MASS/VISTA J/H, 2MASS K.
- **`data_lake/homogenize/bandpasses/`** — new folder for per-band transmission-curve ECSV files (`wavelength` Å, `throughput`).  Bundled illustrative curves for WISE W1 and Gaia G; see `bandpasses/README.md` for the format and SVO fetch instructions.
- **`data_lake/plot/sed.py`** — `assemble_sed()` dynamically discovers `phot_ab_*` columns in a product catalog row, intersects with `BandpassRegistry`, skips NaN/unknown bands, returns `SEDPoints` sorted by wavelength.
- **`data_lake/plot/source_figure.py`** — `plot_source_sed_spectrum()` renders the two-panel figure; lazy `import matplotlib` with a helpful error when the `viz` extra is missing; optional bandpass shading on a twin axis.

### Documentation

- **`docs/plotting.md`** — usage, ID semantics, bandpass registry, Python API, follow-up roadmap.
- `docs/cli-reference.md` — new Visualisation section with `dl-plot-source`.
- `docs/layout/data-on-disk.md` — `shared/registry/bandpasses/` lake-override directory added to layout diagram.
- `docs/layout/repository.md` — `homogenize/bandpasses/`, `homogenize/bandpass.py`, and `plot/` subpackage listed.
- `docs/homogenization.md` — cross-link to plotting guide.
- `README.md` — SED + spectrum plot row in discovery table; `viz` extra in dependencies note.

## [0.4.3] - 2026-06-19

### Added

- **`dl-area`** — manage `areas/<id>.json`: `set-crossmatch`, `set-gather`, `set-homogenize`, `import`, `validate` (no hand-editing of plan blocks).
- **MOC export** — `dl-region --export-moc` and **`dl-export-moc`** write IVOA Multi-Order Coverage maps at a user-chosen HEALPix order (`uv sync --extra moc`).
- **Reference workflow** — `notebooks/14_discovery_workflow.ipynb`, `examples/areas/multi_survey_cone.example.json`, and `docs/discovery/workflow.md`.

### Documentation

- **`docs/discovery/areas-cli.md`** — full `dl-area` command reference; cross-links from workflow, regions, gather, and homogenization guides.
- MOC export section in regions-and-areas guide.

## [0.4.2] - 2026-06-18

### Added

- **Extract-time spectrum flux calibration** — `dl-extract-spectra-subset` accepts `--flux-scale` and `--apply-survey-calibration`; scales flux and ivar at export and writes a `*.calibration.json` sidecar.
- **Bundled SDSS/DESI calibration** — `spectra.flux_calibration` in `homogenize/surveys/SDSS_DR17.json` and `DESI_DR1.json` (native 10⁻¹⁷ → cgs erg/s/cm²/Å).

### Documentation

- Export guide: flux calibration flags and provenance sidecars.
- TODO: extract calibration is the primary path for SDSS/DESI flux units (not `dl-homogenize --modality spectra`).

## [0.4.1] - 2026-06-19

### Added

- **`flux_to_ab` catalog transform** — convert native flux (Jy) columns to AB magnitudes (`UNWISE_W1`, `UNWISE_W2`).
- **Ready `phot_ab_v1` recipes** for `VHS_DR3`, `VIDEO_DR5`, `VIKING_DR4`, `ultraVISTA_DR6` (VISTA Vega → AB offsets; VIDEO uses `Z_MAG_AUTO` … `KS_MAG_AUTO` column names).

### Documentation

- Homogenization guide: catalog rule types and `flux_to_ab` zero points.

## [0.4.0] - 2026-06-18

### Added

- **Discovery model** — `Region` selectors (npix, cone, bbox, MOC), metadata-only **areas** (`areas/<id>.json`), cached **tile index**, and **base-source selection** for spatial queries.
- **`dl-region`** — save sky selections as reusable areas; **`dl-gather`** — materialise derived multi-survey product catalogs from crossmatch trees; **`dl-gather --extract-modalities`** — portable spectra/cutout bundles alongside gathered catalogs.
- **`dl-crossmatch --from-area` / `--plan`** — region-bounded crossmatch execution with gap-fill reuse.
- **`dl-homogenize`** — opt-in homogenized **product catalogs** (`phot_ab_v1`), **spectra**, and **cutouts** via per-survey recipes under `data_lake/homogenize/surveys/` and lake overrides in `shared/registry/homogenize/`.
- **`dl-validate-homogenization`** — golden tests, product lint, and `--ab-coverage` reporting for missing photometry recipes.
- **`dl-mcp-lake`** — read-only MCP explorer (region discovery, provenance, ingest recommendations); **`ingest_advisor`** module for batch sizing hints.
- **MOC region support** and **deferred-finalize live catalog ingest** (`lifecycle=live`).
- **`dl-extract-catalog --from-product`** with homogenization provenance sidecars for ML export.
- **`DATA_LAKE` environment variable** — alternate lake-root discovery alongside `$DATA_LAKE_CONFIG`.

### Performance

- Gather and crossmatch tile iteration optimisations; partner tile cache for repeated region overlap.
- Parallel catalog index rebuild improvements.

### Fixed

- Arrow type handling in `dl-extract-catalog`; env-based lake root resolution.
- Area naming resolution and gather product naming edge cases.
- FITS column names: strip leading/trailing whitespace during catalog ingest and repair.

### Documentation

- Areas, regions, gather, crossmatch, homogenization, and MCP lake guides.
- On-disk layout updates for crossmatch trees, areas, and product catalogs.

## [0.3.0] - 2026-06-05

### Added

- **`dl-check-fits-table-format`** — header-only FITS layout probe (`standard-bintable` vs packed-vector / STILTS colfits) with estimated source counts; use before large catalog ingests.
- **`dl-widen-spectrum-tiles`** — pad narrower spectrum Zarr tiles to the survey `n_pix` in `spectrum_info.json`.
- **`dl-mcp-docs`** — stdio MCP server for agent access to lake docs and inventory (optional `[mcp]` extra).
- **RAPIDS GPU crossmatch** — `--match-backend rapids --gpu-id N` on `dl-crossmatch` (optional `[rapids]` extra).
- **GAMA 1-D spectra loader**.
- **`--allow-incomplete-link-id`** — keep catalog rows with missing composite/label link parts (`_source_id` null, `_spectrum_index` -1).
- **Catalog parallel tuning** — `--files-per-worker`, `--partition-by-dir`, `--streaming-parallel N`, and `--columns` projection on streaming ingest.
- **Cutout parallel file-list ingest** — `dl-ingest-cutouts-from-list --n-workers N`.
- **Validation `--all`** — run validate commands across every survey in the lake.
- **Parallel validate/rebuild** — `--n-workers` on `dl-rebuild-catalog-indices` and `dl-validate-catalog-spectra-link`.
- **Benchmark harness** — `python -m data_lake.bench` for ingest/lookup/crossmatch timing records.
- **`dl-describe-lake`** — `--modality`, summary row counts; richer `dl-describe-survey` metadata.
- **Package version** — `data_lake.__version__` and `--version` on describe commands.

### Performance

- Bulk HEALPix tile index lookup for spectra/cutout `get_batch()` (replaces per-ID tile resolution hot loops).
- Contiguous Zarr slice reads in spectrum and cutout tile stores.
- FITS reader and memmap policy optimizations.
- Crossmatch `--tiles-per-worker` batching and cached catalog ID column resolution.
- Vectorized spPlate ingest; parallel catalog/spectrum file-list decode improvements.

### Fixed

- **`--streaming-parallel` schema mismatch** on finalize (resume/append could fail or produce wrong schema).
- Catalog streaming logic and `--streaming` option handling.
- Index rebuild bug in `dl-rebuild-catalog-indices`.
- Validation failures on null link IDs.
- Survey ID resolution for 2dF, 6dF, and WiggleZ.
- spPlate parallel ingest length-mismatch errors.
- Spectrum tile Npix size validation edge cases.

### Changed

- **Stricter ingest validation** — `--link-id-col` required; invalid RA/Dec rows abort before HEALPix assignment.
- **Decoupled catalog vs spectrum HEALPix order** — linkage uses `_source_id`; catalog tiles store modality-specific pixel indices.

### Documentation

- Refactored `docs/` tree (ingest, discovery, performance, troubleshooting, MCP).
- Pre-ingest FITS check workflow (`dl-check-fits-table-format`) in catalog and batch guides.
- Performance tuning guide and colfits / row-count troubleshooting notes.

## [0.2.0] - 2026-05-31

### Changed

- **Breaking:** Renamed `--source-id-col` to `--link-id-col`; dropped legacy join-column migration paths.
- Require `_source_id` on all catalog tiles; simplified ingest and repair APIs.

## [0.1.0] - Initial release

- Multi-survey catalog (Parquet/HATS), cutout, and 1-D spectrum (Zarr v3) ingest.
- Lake registry, crossmatch, validation, and export tooling.

[0.4.1]: https://github.com/SFotopoulou/data_lake/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/SFotopoulou/data_lake/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/SFotopoulou/data_lake/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/SFotopoulou/data_lake/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/SFotopoulou/data_lake/releases/tag/v0.1.0
