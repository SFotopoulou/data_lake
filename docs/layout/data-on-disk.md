# Data layout on disk

```
<lake_root>/
  catalogs/
    <survey>/
      Norder=5/Dir=0/Npix=0.parquet
      Norder=5/Dir=0/Npix=1.parquet
      ...
      _metadata             ← Parquet aggregate footer
      catalog_info.json     ← HATS descriptor
    crossmatch/
      <surveyA>_x_<surveyB>/
        Norder=5/Dir=0/Npix=0.parquet
        catalog_info.json
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

