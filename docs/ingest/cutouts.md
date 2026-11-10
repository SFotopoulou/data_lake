# Cutout ingest

This page explains how to load image cutouts (postage stamps) from FITS (Flexible Image
Transport System) files into the lake and link them to an existing catalog. Read it when
you have one stamp per object, or full-field images you want to cut into stamps first.

Cutouts are stored as Zarr v3 stacks, one group per HEALPix (Hierarchical Equal Area
isoLatitude Pixelization) tile under `cutouts/<survey>/`. Each ingested FITS file adds one
or more rows to the tile's `_source_id`, `images` and `wcs` arrays. With the default
`--update-catalog`, the matching catalog rows get `_cutout_index` (and `_cutout_npix`) set
to that row's position, so retrieval can find the stamp without scanning.

## Typical workflow: one FITS file per catalog row

This matches pipelines that write one stamp per object (for example DESI targets carrying
the same `TARGETID` as the catalog). Ingest the catalog first, then the cutouts with the
same survey name and ID column:

```bash
# 1. Catalog already ingested with native IDs
dl-ingest-catalog zall-pix-iron.fits --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID --streaming

# 2. One cutout FITS per object (paths in cutout_files.txt)
dl-ingest-cutouts-from-list cutout_files.txt --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID \
  --on-duplicate skip
```

Use the same `--survey`, sky columns and `--link-id-col` as the catalog so the cutout
`_source_id` values match and `_cutout_index` can be patched. If no catalog exists yet for
the survey, the patch is skipped; run `dl-rebuild-catalog-indices --survey <name> --kind cutout`
after ingesting the catalog.

For a single file, `dl-ingest-cutouts FILE --survey ...` takes the same core flags.
`dl-ingest-cutouts-from-list` adds checkpointing (`--checkpoint`, `--no-skip-completed`),
`--failures-log` and `--heartbeat-interval`; see
[Batch ingest and checkpoints](batch-and-checkpoints.md) and [Logging](logging.md).

## FITS header requirements (per cutout file)

| Purpose | CLI flag | Header keyword(s) | Notes |
|--------|----------|-------------------|--------|
| Object ID (join to catalog) | `--link-id-col` | e.g. `TARGETID`, `NAME`, or `TARGETID,SURVEY,PROGRAM` | Required in practice. A single keyword or a comma-separated composite spec (same rule as catalog `--link-id-col` and DESI ingest; the hash is identical). Must match catalog ingest. |
| Sky position (tile routing) | `--ra-col` / `--dec-col` | e.g. `TARGET_RA`, `TARGET_DEC` | Degrees; default `RA` / `DEC`. Should match catalog coordinates. |
| Astrometry (export) | — | Standard 2-D WCS | `CTYPE*`, `CRVAL*`, `CRPIX*`, `CD*_*` (or `CDELT`/`CROTA`); stored in `wcs` for FITS round-trip. |
| Image data | — | Primary or image HDU | 2-D `(H, W)` is one band; 3-D needs `--band-axis` (default axis 0). All stamps in a survey tile share the `(H, W)` of the first file. |

WCS is the World Coordinate System; an HDU is a FITS header-data unit (extension).

If `--link-id-col` is omitted, ingest tries `SOURCE_ID`, `OBJ_ID`, `OBJID`, `TARGETID`,
`FIBERID` (and lower-case variants) and finally falls back to the HDU index with a warning.
The HDU index will not match your catalog, so always pass `--link-id-col` for production.

If the `--ra-col` / `--dec-col` keywords are missing, ingest falls back to
`RA_TARG`/`DEC_TARG`, then `TARGET_RA`/`TARGET_DEC`, then `CRVAL1`/`CRVAL2`.

## Options

| Flag | Default | Notes |
|------|---------|-------|
| `--band-names r,i,z` | none | Stored on the Zarr group as `band_names` |
| `--dtype` | `float32` | NumPy dtype for stored images |
| `--image-hdu N` | `0` | `0` reads every HDU that has image data, each as its own cutout row; `N > 0` reads only HDU `N` |
| `--band-axis` | axis 0 for 3-D data | Band axis of a 3-D image |
| `--on-duplicate` | `skip` | `skip`, `error` or `append` when a `_source_id` is already in the tile |
| `--update-catalog / --no-update-catalog` | on | Patch `_cutout_index` in the catalog after ingest |
| `--norder` | config, else 5 | HEALPix order |
| `--n-workers N` | `1` | `dl-ingest-cutouts-from-list` only: decode files in `N` processes with one Zarr writer; `--files-per-worker` and `--max-in-flight` tune batching |
| `--fits-memmap` | `auto` | FITS read policy; see [FITS I/O tuning](batch-and-checkpoints.md#fits-io-tuning-memmap-and-workers) |

## Minimal DESI-like cutout FITS (example)

A Python sketch that writes a single-object stamp in the expected format:

```python
from astropy.io import fits
import numpy as np

data = np.zeros((64, 64), dtype=np.float32)  # flux stamp
hdu = fits.PrimaryHDU(data)
h = hdu.header
tid = 9876543210123456  # same int64 as catalog TARGETID
h["TARGETID"] = tid
h["TARGET_RA"] = 150.123
h["TARGET_DEC"] = 2.456
h["CTYPE1"] = "RA---TAN"
h["CTYPE2"] = "DEC--TAN"
h["CRVAL1"] = 150.123
h["CRVAL2"] = 2.456
h["CRPIX1"] = 32.0
h["CRPIX2"] = 32.0
h["CD1_1"] = -0.262 / 3600.0   # ~0.262 arcsec/pix
h["CD1_2"] = 0.0
h["CD2_1"] = 0.0
h["CD2_2"] = 0.262 / 3600.0
hdu.writeto("cutout_9876543210123456.fits", overwrite=True)
```

Ingest it:

```bash
dl-ingest-cutouts cutout_9876543210123456.fits --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID
```

## Generate cutout FITS from a catalog + band images

If you start from full-field (or tile) images rather than pre-cut stamps, use
`dl-generate-cutout-fits` to write one multi-band FITS per catalog row. Each file has shape
`N_bands × N_pix × N_pix`, with bands in the order of the image list.

```bash
# bands.txt: one path per line (r.fits, then i.fits, then z.fits)
dl-generate-cutout-fits targets.parquet /data/stamps \
  --images-file bands.txt \
  --size 64 \
  --id-col TARGETID,SURVEY,PROGRAM \
  --ra-col TARGET_RA --dec-col TARGET_DEC \
  --id-hdu-key SOURCE_ID --ra-hdu-key TARGET_RA --dec-hdu-key TARGET_DEC \
  --band-names r,i,z

find /data/stamps -name 'cutout_*.fits' | sort > cutout_files.txt
dl-ingest-cutouts-from-list cutout_files.txt --survey desi_dr1 \
  --link-id-col SOURCE_ID \
  --ra-col TARGET_RA --dec-col TARGET_DEC \
  --band-names r,i,z
```

`--id-col` accepts a single column name (e.g. `TARGETID`) or a comma-separated composite
spec (e.g. `TARGETID,SURVEY,PROGRAM`). The generator computes the same `_source_id` as
catalog and DESI spectra ingest and writes that integer to the `--id-hdu-key` header
keyword (default `SOURCE_ID`). At ingest, pass that keyword as `--link-id-col`, not the
original composite columns, which are not copied into the stamp headers.

Band images must share a consistent astrometric grid (a 2-D WCS per FITS). Stamps are cut
with `astropy.nddata.Cutout2D` in `mode='partial'`, so sources near an image edge are
padded with `NaN`.

Other generator options: `--images` (comma-separated image paths instead of `--images-file`),
`--image-hdu`, `--filename-template` (default `cutout_{source_id}.fits`),
`--max-sources` and `--skip-existing`.

## Troubleshooting

- **`_cutout_index` stays -1 or unset.** The cutout IDs do not match the catalog
  `_source_id`. Check that `--link-id-col` names the same ID (or, for generated stamps, the
  `--id-hdu-key` keyword), then run `dl-rebuild-catalog-indices --survey <name> --kind cutout`.
- **Stamps land in the wrong tile or near RA = 0, Dec = 0.** None of the sky keywords were
  found, so the position fell back to `CRVAL1`/`CRVAL2` or to 0. Pass `--ra-col` /
  `--dec-col` matching your headers.
- **Ingest fails when appending to a tile.** Every stamp in a tile must have the same
  `(H, W)` and band count as the first one written; regenerate stamps at a fixed `--size`.
