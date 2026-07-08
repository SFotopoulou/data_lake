# Data layout on disk

```
<lake_root>/
  catalogs/
    <survey>/
      Norder=5/Dir=0/Npix=0.parquet
      Norder=5/Dir=0/Npix=1.parquet
      ...
      _metadata             ← Parquet aggregate footer
      catalog_info.json     ← HATS descriptor (kind: ingested|product, lifecycle, finalized)
      schema_manifest.json  ← column manifest (name, dtype, role) written by dl-describe-survey --rebuild
  crossmatch/                ← top-level modality (was catalogs/crossmatch/)
    <surveyA>_x_<surveyB>__r<radius>/   ← match radius is part of the name
      Norder=5/Dir=0/Npix=0.parquet
      crossmatch_info.json   ← radius, survey-A/B hats_order, backend
  areas/                     ← metadata-only logical groupings (no tile data)
    <area_id>.json           ← region + optional crossmatch_plan / gather blocks
  shared/
    registry/
      surveys.parquet        ← survey × modality index (dl-describe-lake)
      tile_index/            ← cached <survey>.<modality>.json populated-npix lists
      homogenize/            ← per-survey homogenization recipe overrides (<SURVEY>.json)
      bandpasses/            ← lake-local bandpass transmission-curve overrides (<file>.ecsv)
      overlays/              ← per-survey column overlay definitions (<SURVEY>.json)
      transforms/            ← global transform pack definitions (phot_ab_v1.json, etc.)
  cutouts/
    <survey>/
      Norder=5/Dir=0/Npix=0.zarr/   ← one Zarr group per tile
        images/   (N, B, H, W) float32, sharded
        _source_id/ (N,) int64
        wcs/      (N,) structured bytes
      cutout_info.json
  spectra/
    <survey>/
      Norder=5/Dir=0/Npix=0.zarr/   ← one Zarr group per tile
        flux/       (N, N_pix) float32, sharded
        ivar/       (N, N_pix) float32, sharded
        mask/       (N, N_pix) uint8,   sharded
        wavelength/ (N_pix,)   float64  (shared) or (N, N_pix) float32 (per-source)
        _source_id/  (N,) int64
        meta/       (N,) structured bytes (z, z_err, snr, exptime, R, instr, ra_key, dec_key, ra, dec, source_file)
      spectrum_info.json
```

Each catalog row carries:
- `_source_id` — stable int64 join key for Zarr/cross-match (sequential 0…N−1, copy of native int ID, or hash of a label column)
- native survey ID columns (e.g. `TARGETID`, `SOURCE_ID`) when ``link_id_mode`` is ``column:…`` or ``label:…``
- `_healpix_norder{N}` — HEALPix tile pixel at the **catalog** partition order (partitioning key only)
- `_cutout_index` — local row offset inside the Zarr cutout tile identified by `_cutout_npix` (-1 = not ingested)
- `_cutout_npix` — HEALPix pixel at the **cutout** `hats_order` that identifies the Zarr tile (-1 = not ingested)
- `_spectrum_index` — local row offset inside the Zarr spectrum tile identified by `_spectrum_npix` (-1 = not ingested)
- `_spectrum_npix` — HEALPix pixel at the **spectrum** `hats_order` that identifies the Zarr tile (-1 = not ingested)

The catalog and spectrum/cutout layers may use **different** `hats_order` values and different sky coordinates; linkage is via `_source_id` + the modality-specific npix column.

## Derived products and areas

- **Product catalogs** are derived tables under `catalogs/<name>/` with `catalog_info.json` having `kind: "product"`. They are produced by [`dl-gather`](../discovery/gather.md) (joined multi-survey wide tables, `product_subtype: joined`) or by [`dl-homogenize`](../../docs/homogenization.md) (standardised photometry, `product_subtype: homogenized`). Filter them with `dl-describe-lake --kind product`.
- **Areas** (`areas/<area_id>.json`) are metadata-only: a `region` selector (npix / cone / bbox / MOC) plus optional `crossmatch_plan` and `gather` blocks. They span surveys and modalities and never hold tile data. See [Regions and areas](../discovery/regions-and-areas.md) and [`dl-area`](../discovery/areas-cli.md).
- **Lifecycle**: live (incrementally ingested) catalogs can be ingested with `dl-ingest-catalog --defer-finalize` (records `lifecycle: "live"`, `finalized: false`), skipping the per-ingest `_metadata` rebuild; run `dl-finalize-catalog` once at the end.

