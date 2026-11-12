# Export and sharing

This page shows how to get data out of the lake: catalog rows, spectra and
cutouts for a chosen set of sources, in Zarr, Parquet, HDF5 or FITS. It also
covers backfilling the catalog index columns that make those extracts fast, and
packing one sky tile to hand to a collaborator.

## Catalog by survey name, then spectra / cutouts

The usual workflow has two steps. First, pull the rows you want from the
ingested catalog with `dl-extract-catalog --survey NAME` (this reads the lake,
not a raw file). Second, pass that file to the spectrum or cutout extractor.

Keep `_source_id` in the catalog extract: it is the join key between a survey's
catalog, spectra and cutouts. To fetch spectra or cutouts from a *different*
(partner) survey, positionally match the two catalogs first (`dl-crossmatch` or
STILTS) and use the partner's `_source_id`, which is `source_id_b` in a
crossmatch tree.

```bash
# $DATA_LAKE_CONFIG supplies --lake-root when set

# 1. Catalog from the lake survey name
dl-extract-catalog --survey DESI_DR1 \
  -c _source_id,TARGET_RA,TARGET_DEC,SPECTYPE,Z \
  --where "upper(trim(SPECTYPE)) = 'STAR'" --limit 1000 \
  -o /scratch/desi_stars.parquet

# 2a. Same survey — spectra and cutouts via _source_id
dl-extract-spectra-subset --survey DESI_DR1 \
  --target-list /scratch/desi_stars.parquet --target-id-col _source_id \
  --format zarr --output /scratch/desi_stars.zarr \
  --apply-survey-calibration

dl-extract-cutout-subset --survey DESI_DR1 \
  --target-list /scratch/desi_stars.parquet --target-id-col _source_id \
  --output /scratch/desi_stars_cutouts.zarr

# 2b. Partner survey in one go (existing crossmatch tree)
# --survey = modality/catalog to extract; --target-list IDs are the *other* side
dl-extract-catalog --survey Legacy_DR11_north \
  --via-crossmatch DESI_DR1_x_Legacy_DR11_north__r1.0 \
  --target-list DESI-subset.fits --target-id-col TARGETID,SURVEY,PROGRAM \
  -c _source_id,ra,dec,type,flux_g,flux_r,flux_z \
  -o /scratch/legacy_for_desi_subset.parquet

# Same flag on spectra / cutout extract (partner must have that modality)
dl-extract-cutout-subset --survey Legacy_DR11_north \
  --via-crossmatch DESI_DR1_x_Legacy_DR11_north__r1.0 \
  --target-list DESI-subset.fits --target-id-col TARGETID,SURVEY,PROGRAM \
  --output /scratch/legacy_cutouts_for_desi.zarr

dl-extract-spectra-subset --survey EUCLID_DR1 \
  --via-crossmatch DESI_DR1_x_EUCLID_DR1__r1.0 \
  --target-list DESI-subset.fits --target-id-col TARGETID,SURVEY,PROGRAM \
  --output /scratch/euclid_spectra_for_desi.zarr

# --target-id-col columns are kept for join-back to the target-list:
# catalog extract merges them into the output table; spectra/cutout write
# <output>.target_ids.parquet (_source_id + TARGETID / SURVEY / PROGRAM / …).

# 2c. Partner modalities after a manual ID map (same as before)
dl-extract-catalog --survey EUCLID_DR1 \
  -c _source_id,ra,dec -o /scratch/euclid_sky.parquet

dl-crossmatch DESI_DR1 EUCLID_DR1 "$LAKE" --radius-arcsec 1.0
# Tree: crossmatch/DESI_DR1_x_EUCLID_DR1__r1.0/  (source_id_a, source_id_b)

duckdb -c "
COPY (
  SELECT m.source_id_b
  FROM read_parquet('/scratch/desi_stars.parquet') s
  JOIN read_parquet('$LAKE/crossmatch/DESI_DR1_x_EUCLID_DR1__r1.0/**/*.parquet',
                    hive_partitioning=false) m
    ON s._source_id = m.source_id_a
) TO '/scratch/desi_x_euclid.parquet' (FORMAT PARQUET);
"

dl-extract-cutout-subset --survey EUCLID_DR1 \
  --target-list /scratch/desi_x_euclid.parquet --target-id-col source_id_b \
  --output /scratch/euclid_matched_cutouts.zarr

dl-extract-spectra-subset --survey EUCLID_DR1 \
  --target-list /scratch/desi_x_euclid.parquet --target-id-col source_id_b \
  --output /scratch/euclid_matched_spectra.zarr
```

If you do not want an in-lake crossmatch tree, run STILTS `tskymatch2` on the two
catalog extracts (`TARGET_RA`/`TARGET_DEC` against `ra`/`dec`) instead; see
[Crossmatch](discovery/crossmatch.md#associations-with-stilts).

Note the different `--target-id-col` defaults: `dl-extract-spectra-subset`
defaults to `TARGETID` (a survey-native ID list), while
`dl-extract-cutout-subset` defaults to `_source_id`. Pass the flag explicitly
when in doubt.

Worked notebooks: [`02_spectrum_workflow.ipynb`](../notebooks/02_spectrum_workflow.ipynb)
§6, [`03_cutout_ingest.ipynb`](../notebooks/03_cutout_ingest.ipynb) §6b,
[`14_discovery_workflow.ipynb`](../notebooks/14_discovery_workflow.ipynb) §8.

## Extract a curated subset into one flat Zarr

Once spectra are ingested, `dl-extract-spectra-subset` writes a self-contained
file containing only the sources you ask for (for example 100k DESI quasars
out of a full release). Reads are grouped by HEALPix sky tile, and each tile is
read with one Zarr orthogonal-indexing call, so every source shard is
decompressed at most once.

### Command line

```bash
# Default output: one flat Zarr group
dl-extract-spectra-subset \
    --survey DESI_DR1 \
    --target-list zall-pix-iron-qso.fits \
    --target-id-col TARGETID,SURVEY,PROGRAM \
    --format zarr \
    --output /scratch/qso_subset.zarr
```

`--target-id-col` must match the `--link-id-col` used when the spectra and
catalog were ingested. DESI coadds ingested with the default composite key need
`TARGETID,SURVEY,PROGRAM`, not `TARGETID` alone.

Instead of an ID list, you can filter the ingested catalog with `--where` and
`--limit`. `--where` is a DuckDB predicate for **trusted operators** (do not
pass untrusted input; see [security.md](security.md)). Rows without a stored
spectrum (`_spectrum_index < 0`) are dropped automatically, so `--limit` counts
spectra that can actually be read:

```bash
dl-extract-spectra-subset \
    --survey DESI_DR1 \
    --where "upper(trim(SPECTYPE)) = 'STAR'" \
    --limit 1000 \
    --format zarr \
    --output /scratch/desi_stars_1000.zarr \
    --apply-survey-calibration
```

Other output formats:

```bash
# One Parquet file (one row per spectrum; wavelength in file metadata)
dl-extract-spectra-subset ... --format parquet --output /scratch/qso_subset.parquet

# One HDF5 file (stacked flux/ivar/mask + shared wavelength; gzip-compressed)
dl-extract-spectra-subset ... --format hdf5 --output /scratch/qso_subset.h5

# One FITS file per spectrum (written into a directory)
dl-extract-spectra-subset ... --format fits --fits-layout per-file \
    --output /scratch/qso_fits/

# One multi-row FITS catalog (BINTABLE: TARGETID, Z, FLUX, IVAR, MASK + WAVELENGTH HDU)
dl-extract-spectra-subset ... --format fits --fits-layout catalog \
    --fits-chunk-rows 50000 --yes \
    --output /scratch/qso_spectra.fits
```

| Flag | Values / default | Notes |
|------|------------------|-------|
| `--format` | `zarr` (default), `parquet`, `hdf5`, `fits` | |
| `--fits-layout` | `per-file` (default), `catalog` | FITS only |
| `--fits-chunk-rows` | 50000 | Rows per temporary part file for the FITS catalog layout |
| `--yes` / `-y` | off | Skip the part-count confirmation prompt |
| `--keep-part-files` | off | Keep the temporary FITS part files after merging |
| `--force-many-fits-parts` | off | Allow more than 100 FITS parts (hard maximum 10,000) |
| `--with-catalog/--no-with-catalog` | on | Use the survey catalog for ID lookup and redshift |
| `--missing` | `skip` (default), `error` | What to do with IDs not in the lake |
| `--overwrite/--no-overwrite` | off | Replace an existing output |

**FITS catalog layout.** Astropy and CFITSIO can crash on vector BINTABLEs
larger than about 4 GiB, so the catalog layout is written as temporary
`*_partNNNNN.fits` shards of `--fits-chunk-rows` rows and then merged into
`--output` with `fitsio`. The command prints the expected number of parts and
asks for confirmation unless you pass `--yes`. It refuses more than 100 parts
unless you add `--force-many-fits-parts`. If a run crashes, re-run the same
command: completed parts are reused. The catalog layout needs a shared
wavelength grid; for per-source-wavelength surveys such as 2dF and 6dF use
`--fits-layout per-file`. All other formats support both shared and per-source
wavelengths.

### Flux calibration

Extracted flux can be multiplied by a constant (ivar is divided by its square).
Use `--apply-survey-calibration` to apply the `spectra.flux_calibration` entry
from `data_lake/homogenize/surveys/<SURVEY>.json` (bundled for `DESI_DR1` and
`SDSS_DR17`: native \(10^{-17}\) erg/s/cm²/Å to cgs), or `--flux-scale FACTOR`
for an explicit value. An explicit `--flux-scale` wins over the survey file.

```bash
dl-extract-spectra-subset ... --survey SDSS_DR17 --apply-survey-calibration \
    --output /scratch/qso_ab.zarr

dl-extract-spectra-subset ... --flux-scale 1e-17 --output /scratch/qso_ab.zarr
```

A sidecar `*.calibration.json` records where the factor came from unless you
pass `--no-calibration-sidecar`. Zarr, HDF5 and Parquet outputs also store
`flux_scale` in their metadata when scaling is applied.

### Python API

For many IDs, call `extract_subset_to_zarr` once with the whole list; it
resolves all IDs in a single batched lookup. Attaching a `CatalogAccessor`
lets that lookup use the catalog's `_spectrum_index` column and take redshift
from the catalog.

```python
import numpy as np
from astropy.table import Table
from data_lake.io.catalog import CatalogAccessor
from data_lake.io.spectra import SpectrumAccessor

target_ids = np.asarray(Table.read("zall-pix-iron-qso.fits")["TARGETID"])

lake = "/data/lake"
survey = "DESI_DR1"
cat = CatalogAccessor(lake, survey)   # batched _spectrum_index lookup + catalog redshift
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

The result dictionary also contains `n_requested`, `id_to_row` and
`output_zarr`.

### Reading many spectra or cutouts in Python

To read spectra or cutouts yourself, resolve all IDs first with `locate()`,
then read each one by position. `locate()` returns two arrays, `npix` (tile)
and `local` (row in the tile), with `-1` for IDs that are not in the lake. This
replaces one catalog query per ID with one query per million IDs.

```python
from data_lake.io.catalog import CatalogAccessor
from data_lake.io.cutouts import CutoutAccessor
from data_lake.io.spectra import SpectrumAccessor

cat = CatalogAccessor(lake, survey)

with SpectrumAccessor(lake, survey, catalog_accessor=cat) as spec:
    npix, local = spec.locate(target_ids)
    for sid, p, i in zip(target_ids, npix.tolist(), local.tolist()):
        if p < 0:
            continue                      # not in the lake
        sp = spec.get_spectrum_at(p, i, source_id=int(sid))

with CutoutAccessor(lake, survey, catalog_accessor=cat) as cut:
    npix, local = cut.locate(target_ids)
    for p, i in zip(npix.tolist(), local.tolist()):
        if p >= 0:
            image, wcs = cut.get_cutout_at(p, i)   # or get_image_at(p, i) for pixels only
```

`get_spectrum(source_id)` and `get_cutout(source_id)` still exist, but they look
up one ID at a time; use them only for a handful of sources. `get_batch(ids)`
also resolves IDs in one pass and returns stacked arrays, but raises `KeyError`
if any ID is missing.

To write one FITS file per source from Python, use
`data_lake.export.to_spectrum_fits.export_spectra_batch` or
`data_lake.export.to_fits.export_cutouts_batch`; both use `locate()` internally
and log, rather than raise, for IDs they cannot find. The single-source helpers
`export_spectrum` and `export_cutout` accept an existing `accessor=` to avoid
reopening stores when called in a loop.

### Output layout

Zarr output with a shared wavelength grid:

```
qso_subset.zarr/
  flux/        (N_written, N_pix) float32 sharded
  ivar/        (N_written, N_pix) float32 sharded
  mask/        (N_written, N_pix) uint8 or uint16 sharded  (dtype follows source survey)
  wavelength/  (N_pix,)           float64 shared grid
  _source_id/  (N_written,)       int64
  redshift/    (N_written,)       float32  (from catalog ``Z`` when catalog is used)
```

Zarr output with per-source wavelengths (for example 2dF and 6dF):

```
2df_subset.zarr/
  flux/        (N_written, N_pix) float32 sharded
  ivar/        (N_written, N_pix) float32 sharded
  mask/        (N_written, N_pix) uint8 or uint16 sharded
  wavelength/  (N_written, N_pix) float32 sharded  ← one row per source
  _source_id/  (N_written,)       int64
  redshift/    (N_written,)       float32
```

For per-source wavelengths, `N_pix` is the longest spectrum in the subset;
shorter rows are right-padded with `nan` (flux) and `0` (ivar, mask,
wavelength), and the group attribute `wavelength_mode` is `"per_source"`.
Parquet output adds a `wavelength` list column with the same per-row content.

Rows are written in HEALPix tile order, not in the order of your ID list, so
writes stay contiguous. Use the returned `id_to_row` mapping (or the
`_source_id` array) to reorder.

### Details: ID lookup and redshift source

With a catalog attached (`catalog_accessor=` in Python, `--with-catalog` on the
command line, the default), IDs are resolved through the catalog's
`_spectrum_index` column in batched DuckDB queries; any IDs the catalog cannot
resolve fall back to a vectorised scan of the spectrum tiles (one `_source_id`
array read per tile). With `--no-with-catalog`, only the tile scan is used.

The redshift written to the output (Zarr `redshift/`, Parquet `redshift`, FITS
`Z`) comes from the survey catalog when one is attached. The column is
auto-detected; the first match among `Z`, `ZCOSMO`, `Z_HP`, `Z_PHOT`,
`REDSHIFT`, `Z_QSO` and `Z_RED` is used. It does **not** come from the
spectrum tile's `meta.z`, which is only a copy taken from the FITS files at
ingest. With `--no-with-catalog`, or when the catalog has no redshift column
(a warning is logged), `meta.z` is used instead.

## Update catalog with spectrum / cutout index

`dl-ingest-spectra`, `dl-ingest-cutouts` and `dl-ingest-cutouts-from-list`
write the `_spectrum_index` / `_cutout_index` columns into the catalog
automatically after each run (pass `--no-update-catalog` to skip). The ID
column is read from `catalog_info.json`, so surveys ingested with
`--link-id-col TARGETID` or a composite key such as `TARGETID,SURVEY,PROGRAM`
need no extra configuration.

If spectra or cutouts were ingested without this step, backfill the index from
the Zarr tiles already on disk (no FITS re-ingestion needed):

```bash
# Register console scripts after pulling (once per env):
uv sync --extra desi --extra dev

dl-rebuild-catalog-indices --survey DESI_DR1 --kind spectrum
dl-rebuild-catalog-indices --survey DESI_DR1 --kind cutout   # if cutouts exist

# Without reinstalling, use the module directly:
uv run python -m data_lake.ingest.update_catalog_indices --survey DESI_DR1 --kind spectrum
```

`--norder` defaults to `hats_order` in `catalogs/<survey>/catalog_info.json`;
pass it only to override that value. See
[Validate and repair](validate-and-repair.md) for checking the result.

From Python, if you already have a `source_id → index` mapping (as returned by
the ingest functions):

```python
from data_lake.ingest.update_catalog_indices import update_index_column
update_index_column(lake_root="/data/lake", survey_name="SDSS_DR17",
                    source_id_to_index=index_map, kind="spectrum")
```

`update_index_column` also takes `norder` (default 5) and `link_id_col`; pass
`norder` explicitly if the catalog uses a different `hats_order`.

## Pack a tile for sharing

`dl-pack-tile` bundles one HEALPix tile of one or more surveys (catalog
Parquet plus, by default, spectrum and cutout Zarr stores and their info JSON
files) into a `.tar` archive in the output directory.

```bash
dl-pack-tile /data/lake /data/share --norder 5 --npix 1234 --survey des_dr2 --manifest
# Exclude spectra if not needed:
dl-pack-tile /data/lake /data/share --norder 5 --npix 1234 --survey des_dr2 --no-spectra
```

`--survey` can be repeated. Without `--npix`, every tile is packed into its own
archive. `--no-cutouts` excludes cutouts, `--manifest` (re)generates
`MANIFEST.json` (tile, surveys, sizes and SHA-256 checksums for each archive),
and `--norder` defaults to 5.
