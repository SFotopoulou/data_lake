# Cutout ingest

### Ingest cutouts

Cutouts are stored as **Zarr v3** stacks (one group per HEALPix tile). Each ingested
FITS contributes one or more rows in the tile's ``_source_id/``, ``images/``, and
``wcs/`` arrays. With default ``--update-catalog``, matching Parquet rows get
``_cutout_index`` set to that row offset.

#### Typical workflow: one FITS file per catalog row

This matches pipelines that write **one stamp per object** (e.g. DESI targets with
the same ``TARGETID`` as the catalog):

```bash
# 1. Catalog already ingested with native IDs
dl-ingest-catalog zall-pix-iron.fits --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID --streaming

# 2. One cutout FITS per object (paths in cutout_files.txt)
dl-ingest-cutouts-from-list cutout_files.txt --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID \
  --on-duplicate skip
```

Use the **same** ``--survey``, sky columns, and ``--link-id-col`` as the catalog
so ``update_index_column`` can patch ``_cutout_index``.

#### FITS header requirements (per cutout file)

| Purpose | CLI flag | Header keyword(s) | Notes |
|--------|----------|-------------------|--------|
| Object ID (join to catalog) | ``--link-id-col`` | e.g. ``TARGETID``, ``NAME`` | **Required** for production. Must match catalog ingest (int64 or hashed label). See [Object identifiers](#object-identifiers---link-id-col). If omitted, tries ``SOURCE_ID``, ``OBJ_ID``, ``TARGETID``, … then HDU index. |
| Sky position (tile routing) | ``--ra-col`` / ``--dec-col`` | e.g. ``TARGET_RA``, ``TARGET_DEC`` | Degrees; fallbacks include ``RA_TARG``/``DEC_TARG``, ``CRVAL1``/``CRVAL2``. Should match catalog coordinates. |
| Astrometry (export) | — | Standard 2-D WCS | ``CTYPE*``, ``CRVAL*``, ``CRPIX*``, ``CD*_*`` (or CDELT/CROTA); stored in ``wcs/`` for FITS round-trip. |
| Image data | — | Primary or image HDU | 2-D ``(H,W)`` → one band; 3-D → set ``--band-axis``. Fixed ``(H,W)`` per survey tile after the first file. |

Optional: ``--band-names r,i,z``, ``--dtype float32``, ``--image-hdu N`` (select one
extension in multi-HDU files), ``--on-duplicate skip|error|append`` (default ``skip``).

#### Minimal DESI-like cutout FITS (example)

Python sketch for a single-object stamp (same pattern as the test suite):

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

Ingest:

```bash
dl-ingest-cutouts cutout_9876543210123456.fits --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID
```

Single-file and file-list CLIs accept the same flags.

#### Generate cutout FITS from a catalog + band images

If you start from full-field (or tile) images rather than pre-cut stamps, use
``dl-generate-cutout-fits`` to write one multi-band FITS per catalog row
(shape ``N_bands × N_pix × N_pix``, band order = image list order):

```bash
# bands.txt: one path per line (r.fits, then i.fits, then z.fits)
dl-generate-cutout-fits targets.parquet /data/stamps \\
  --images-file bands.txt \\
  --size 64 \\
  --id-col TARGETID --ra-col TARGET_RA --dec-col TARGET_DEC \\
  --id-hdu-key TARGETID --ra-hdu-key TARGET_RA --dec-hdu-key TARGET_DEC \\
  --band-names r,i,z

find /data/stamps -name 'cutout_*.fits' | sort > cutout_files.txt
dl-ingest-cutouts-from-list cutout_files.txt --survey desi_dr1 \\
  --link-id-col TARGETID --ra-col TARGET_RA --dec-col TARGET_DEC \\
  --band-names r,i,z
```

Band images must share a consistent astrometric grid (2-D WCS per FITS). Cutouts
use ``astropy.nddata.Cutout2D`` with ``mode='partial'`` (edge sources may include
``NaN`` fills).

