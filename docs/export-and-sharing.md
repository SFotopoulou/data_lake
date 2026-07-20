# Export and sharing

### Extract a curated subset into one flat Zarr

Once spectra are ingested, you can materialise a self-contained Zarr
group containing only a user-specified subset of sources (e.g. 100k
DESI targets out of a fully-ingested release). Reads are batched per
HEALPix tile via Zarr orthogonal indexing so each source shard is
decompressed at most once.

Python API:

```python
import numpy as np
from astropy.table import Table
from data_lake.io.catalog import CatalogAccessor
from data_lake.io.spectra import SpectrumAccessor

target_ids = np.asarray(Table.read("zall-pix-iron-qso.fits")["TARGETID"])

lake = "/data/lake"
survey = "DESI_DR1"
cat = CatalogAccessor(lake, survey)   # fast _spectrum_index lookup + catalog Z
acc = SpectrumAccessor(lake, survey, catalog_accessor=cat)
result = acc.extract_subset_to_zarr(
    source_ids=target_ids,
    output_zarr="/scratch/qso_subset.zarr",
    missing="skip",          # report-and-skip IDs not in the lake
    overwrite=False,
)
print(result["n_written"], "rows written;",
      len(result["missing_ids"]), "missing")
# Redshift in the output uses cat.redshift_column (e.g. Z), not spectrum-tile meta.
```

Or the CLI:

```bash
# Default: single flat Zarr group (--with-catalog: fast index + catalog Z for redshift)
dl-extract-spectra-subset \
    --survey DESI_DR1 \
    --target-list zall-pix-iron-qso.fits \
    --target-id-col TARGETID \
    --format zarr \
    --output /scratch/qso_subset.zarr

# One Parquet file (one row per spectrum; wavelength in file metadata)
dl-extract-spectra-subset ... --format parquet --output /scratch/qso_subset.parquet

# One HDF5 file (stacked flux/ivar/mask + shared wavelength; gzip-compressed)
dl-extract-spectra-subset ... --format hdf5 --output /scratch/qso_subset.h5

# One FITS per spectrum (directory)
dl-extract-spectra-subset ... --format fits --fits-layout per-file \
    --output /scratch/qso_fits/

# One multi-row FITS catalog (BINTABLE: TARGETID, Z, FLUX, IVAR, MASK + WAVELENGTH HDU)
dl-extract-spectra-subset ... --format fits --fits-layout catalog \
    --output /scratch/qso_spectra.fits

# Apply bundled SDSS/DESI flux calibration (native 10^-17 → cgs erg/s/cm²/Å)
dl-extract-spectra-subset ... --survey SDSS_DR17 --apply-survey-calibration \
    --output /scratch/qso_ab.zarr

# Explicit scale (overrides registry); writes subset.calibration.json by default
dl-extract-spectra-subset ... --flux-scale 1e-17 --output /scratch/qso_ab.zarr
```

**Flux calibration** (optional): multiply extracted flux by a constant factor
and divide ivar by factor².  Use `--apply-survey-calibration` to load
`spectra.flux_calibration` from `data_lake/homogenize/surveys/<SURVEY>.json`
(bundled for `SDSS_DR17`), or `--flux-scale FACTOR` for an explicit value
(explicit wins).  A sidecar `*.calibration.json` records provenance unless
`--no-calibration-sidecar` is set.  Zarr/HDF5/Parquet exports also store
`flux_scale` in group/file metadata when scaling is applied.

`--format` choices: `zarr` (default), `parquet`, `hdf5`, `fits`.  For FITS,
`--fits-layout` is `per-file` (default) or `catalog`.  All formats support
both `wavelength_mode="shared"` and `wavelength_mode="per_source"` surveys,
except FITS catalog layout, which requires a shared wavelength grid (use
`--fits-layout per-file` for per-source surveys such as 2dF/6dF).

Output layout for **zarr** (shared wavelength):

```
qso_subset.zarr/
  flux/        (N_written, N_pix) float32 sharded
  ivar/        (N_written, N_pix) float32 sharded
  mask/        (N_written, N_pix) uint8 or uint16 sharded  (dtype follows source survey)
  wavelength/  (N_pix,)           float64 shared grid
  _source_id/  (N_written,)       int64
  redshift/    (N_written,)       float32  (from catalog ``Z`` when catalog is used)
```

Output layout for **zarr** (per-source wavelength, e.g. 2dF/6dF):

```
2df_subset.zarr/
  flux/        (N_written, N_pix) float32 sharded
  ivar/        (N_written, N_pix) float32 sharded
  mask/        (N_written, N_pix) uint8 or uint16 sharded
  wavelength/  (N_written, N_pix) float32 sharded  ← one row per source
  _source_id/  (N_written,)       int64
  redshift/    (N_written,)       float32
```

`N_pix` is the maximum pixel count across all tiles in the subset; shorter
rows are right-padded with `nan` (flux), `0` (ivar/mask/wavelength).  The
group attr `wavelength_mode` is `"per_source"`.  Parquet exports add a
`wavelength` list column with the same per-row content.

Rows are written in HEALPix-tile-traversal order for fast contiguous
writes; the returned `id_to_row` mapping lets you reorder if needed.
Lookup uses the catalog's `_spectrum_index` column when available
(O(catalog SQL) batched), otherwise falls back to a vectorised tile
scan (one `_source_id` array read per tile + `np.isin`).

**Redshift provenance** (Zarr `redshift/`, Parquet `redshift`, FITS `Z`):
when a Parquet catalog is attached (`catalog_accessor` / CLI
`--with-catalog`, default on), values come from the survey catalog column
(auto-detected: `Z`, `ZCOSMO`, `Z_HP`, `Z_PHOT`, `REDSHIFT`, …), **not**
from spectrum-tile `meta.z` (which is only a copy from ingest-time FITS).
Pass `--no-with-catalog` to skip the catalog entirely (tile scan for IDs,
`meta.z` for redshift).  If the catalog exists but has no redshift column,
the code warns and falls back to `meta.z`.

### Update catalog with spectrum / cutout index

The `dl-ingest-spectra`, `dl-ingest-cutouts`, and `dl-ingest-cutouts-from-list`
CLIs patch the catalog automatically after each ingest run (`--no-update-catalog`
to skip).  The ID column is resolved from `catalog_info.json`, so surveys
ingested with `--link-id-col TARGETID` (or any other native column) work
without extra configuration.

To backfill an existing lake where spectra or cutouts were ingested without
catalog patching (no FITS re-ingestion needed):

```bash
# Register console scripts after pulling (once per env):
uv sync --extra desi --extra dev

dl-rebuild-catalog-indices --survey DESI_DR1 --kind spectrum
dl-rebuild-catalog-indices --survey DESI_DR1 --kind cutout   # if cutouts exist

``--norder`` defaults to ``hats_order`` in ``catalogs/<survey>/catalog_info.json``.
Pass ``--norder`` only to override that metadata value.

# Without reinstalling, use the module directly:
uv run python -m data_lake.ingest.update_catalog_indices --survey DESI_DR1 --kind spectrum
```

For manual / Python-API use:

```python
from data_lake.ingest.update_catalog_indices import update_index_column
update_index_column(lake_root="/data/lake", survey_name="SDSS_DR17",
                    source_id_to_index=index_map, kind="spectrum")
```

### Pack a tile for sharing

```bash
dl-pack-tile /data/lake /data/share --norder 5 --npix 1234 --survey des_dr2 --manifest
# Exclude spectra if not needed:
dl-pack-tile /data/lake /data/share --norder 5 --npix 1234 --survey des_dr2 --no-spectra
```

