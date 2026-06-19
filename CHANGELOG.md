# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
