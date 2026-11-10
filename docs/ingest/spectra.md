# Spectrum ingest

This page explains how to load 1-D spectra from FITS (Flexible Image Transport System) files
into the lake, where they are stored as Zarr v3 stacks with one tile per HEALPix
(Hierarchical Equal Area isoLatitude Pixelization) pixel. Read it when you add a spectroscopic
survey, or when spectra and catalog rows do not link up.

Every spectrum gets an integer `_source_id`, and the catalog row it belongs to must have the
same `_source_id`. The most important decision is therefore which link key you use, and you
must use the same key on both the catalog side and the spectrum side. Ingest the catalog
first; spectrum ingest then fills the catalog's `_spectrum_index` column (see
[Catalog vs spectrum CLI flags](#catalog-vs-spectrum-cli-flags)).

## Ingest spectra

For one file, use `dl-ingest-spectra`. The reader format is auto-detected; pass `--fmt` only
to force it.

```bash
# Single file (debug / smoke-testing).
# With $DATA_LAKE_CONFIG set, OUTPUT_ROOT is taken from the config.
# DESI coadds: object ID defaults to TARGETID|SURVEY|PROGRAM
# (fibermap TARGETID + PRIMARY SURVEY/PROGRAM, whitespace stripped).
# Catalog must use the same --link-id-col. Legacy native TARGETID: pass it explicitly.
dl-ingest-spectra coadd-1-b0-0000p005-thru20210801.fits --survey desi_dr1
# Catalog side (match spectrum composite):
# dl-ingest-catalog zall-pix-iron.fits --survey desi_dr1 \
#   --ra-col TARGET_RA --dec-col TARGET_DEC --link-id-col TARGETID,SURVEY,PROGRAM --streaming

# With resolution matrix (needed for redshift fitting / SPS / kinematic measurements)
# Storage cost: ~3× flux+ivar footprint (~170–200 GB per million coadded BRZ spectra)
dl-ingest-spectra coadd-1-b0-0000p005-thru20210801.fits \
    --survey desi_edr --with-resolution

# Without a config, pass OUTPUT_ROOT (the lake root) explicitly:
dl-ingest-spectra coadd-1-b0-0000p005-thru20210801.fits /data/lake --survey desi_edr

# SDSS/BOSS (no extra dependency needed)
# Match catalog specObj: SPECOBJID lives in the SPALL HDU (HDU 2), not the primary header.
# Pixel count varies slightly file-to-file; ingest uses per_source wavelength and pads
# to the tile's n_pix (override with --on-length-mismatch truncate).
dl-ingest-spectra spec-3586-55181-0001.fits --survey sdss_dr17 \
  --link-id-col SPECOBJID
```

DESI (Dark Energy Spectroscopic Instrument) coadds need the `desi` optional extra
(`pip install "data-lake[desi]"`, which installs `desispec`). The reader uses `desispec`'s
`read_spectra` and `coadd_cameras` so the B, R and Z camera arms are combined with
inverse-variance (IVAR) weighting onto one wavelength grid.

`--with-resolution` stores the DESI banded resolution matrix. It works only for DESI coadds
and only with `--wavelength-mode shared`; see [Resolution matrix usage](#resolution-matrix-usage).

For SDSS (Sloan Digital Sky Survey) and BOSS (Baryon Oscillation Spectroscopic Survey)
files, ingest switches to per-source wavelength storage, and an `--on-length-mismatch` of
`error` (the default) is treated as `pad`.

### Choosing a command

Use `dl-ingest-spectra` for single files and `dl-ingest-spectra-from-list` for everything
else, unless your files are DESI coadds or SDSS `spPlate` files, which have dedicated batch
commands.

| Input | Command |
|-------|---------|
| One FITS file | `dl-ingest-spectra FILE --survey NAME` |
| A text file listing many FITS files | `dl-ingest-spectra-from-list LIST --survey NAME` (add `--n-workers N` for parallel decode) |
| Many DESI coadd files | `dl-ingest-spectra-batch-desi-coadds` (see [Parallel batch ingest](#parallel-batch-ingest-many-coadd-files)) |
| Many SDSS `spPlate` files | `dl-ingest-spectra-batch-spplate` (see [SDSS spPlate ingest](#sdss-spplate-ingest-640-or-1000-fibers-per-file)) |

`dl-ingest-spectra-from-list` rejects `--fmt desi_coadd` when `--n-workers` is greater than 1.
None of the file-list or batch commands accept `--with-resolution`; use `dl-ingest-spectra`
per file if you need the resolution matrix. File-list checkpoints and duplicate handling are
covered in [Batch ingest, checkpoints, and duplicate policy](batch-and-checkpoints.md).

Common `dl-ingest-spectra` options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--survey` | required | Survey name; spectra go to `spectra/<survey>/` under the lake root |
| `--fmt` | auto-detect | Force a reader (see the table below) |
| `--link-id-col` | reader-specific | Object-ID keyword(s); ignored by readers that derive the ID themselves |
| `--ra-col` / `--dec-col` | `RA` / `DEC` | Sky keywords or columns for readers that honour them |
| `--norder` | config, else 5 | HEALPix order of the tiles |
| `--wavelength-mode` | config, else `shared` | `shared` (one grid per tile) or `per_source` |
| `--on-length-mismatch` | `error` | `error`, `pad` or `truncate` when a spectrum's pixel count differs from the tile's |
| `--on-duplicate` | `skip` | `skip`, `error` or `append` when a `_source_id` already exists in a tile |
| `--update-catalog` / `--no-update-catalog` | on | Patch `_spectrum_index` in the catalog after ingest |

### 1-D spectrum readers reference

Quick index of all supported `--fmt` values, the **catalog** ingest flag required to match
them, how each reader derives the spectrum link ID, and which coordinates are used for
HEALPix routing. Do not pass `--link-id-col` on `dl-ingest-spectra` for readers whose ID
comes from a fixed header or the filename; the reader resolves the ID internally.

| `--fmt` | Catalog `--link-id-col` | Spectrum link source | Spectrum sky source | Example / note |
|---------|------------------------|----------------------|---------------------|----------------|
| `desi_coadd` | `TARGETID,SURVEY,PROGRAM` (default) | Fibermap `TARGETID` plus PRIMARY `SURVEY`/`PROGRAM` (stripped), hashed as a composite; `--link-id-col TARGETID` for the legacy native integer | Fibermap `TARGET_RA` / `TARGET_DEC` (fallbacks: `RA_TARGET`, `FIBER_RA`; `DEC_TARGET`, `FIBER_DEC`) | Auto-detected from a `coadd-` filename or `B_FLUX`/`R_FLUX`/`Z_FLUX` HDUs; `TARGETID` alone is not unique in DR1 |
| `sdss_boss` | `SPECOBJID` | `--link-id-col` if given, then `SPECOBJID`, `SPEC_OBJID`, `OBJID`, `THING_ID`, `TARGETID`, `SOURCE_ID`; primary header first, then `SPALL` BINTABLE (HDU 2) | `--ra-col`/`--dec-col` (default `RA` / `DEC`), then `PLUG_RA` / `PLUG_DEC`, `RA_TARG` / `DEC_TARG`, `TARGET_RA` / `TARGET_DEC` | `spec-PLATE-MJD-FIBER.fits` (has a `COADD` HDU) |
| `sdss_spplate` | `PLATE,MJD,FIBERID` (default) or via sidecar / plate header | Plugmap `FIBERID` hashed with `PLATE` and `MJD` (default), or specObjID from sidecar/synthesis | Fiber table columns (default `RA` / `DEC`, configurable via `--ra-col/--dec-col`) | `spPlate-PLATE-MJD.fits`; see the spPlate section |
| `generic` | `--link-id-col` or auto | `--link-id-col` keyword(s); otherwise the first of `SOURCE_ID`, `OBJ_ID`, `OBJID`, `TARGETID`, `FIBERID`; otherwise the HDU index (with a warning) | Header keywords from `--ra-col/--dec-col` (default `RA` / `DEC`) | Any 1-D FITS with spectral WCS (world coordinate system) |
| `2df` | `SPFILE,FIBRE` (+ `--allow-incomplete-link-id` if some rows have no filename) | Header `SPFILE` \| `FIBRE` per SPECTRUM HDU | `SRRA` / `SRDEC` (fallback `OBSRA` / `OBSDEC`, then PRIMARY `RA` / `DEC`) | `data/389442.fits` |
| `6df` | `targetname,obsid_v,obsid_r` | Filename stem + V header `OBSID_V` + R header `OBSID_R` per V/R/VR triple | VR header `OBSRA` / `OBSDEC` (fallback header `RA` / `DEC`, PRIMARY WCS, `OBJCTRA` / `OBJCTDEC`) | `data/g2302140-251235.fits` |
| `gama` | `SPECID` | Primary header `SPECID` (`--link-id-col` overrides) | Header keywords from `--ra-col/--dec-col` (default `RA` / `DEC`) | `data/G23_Y7_015_265.fit` |
| `ozdes` | filename column | Full filename (`normalize_object_id(path.name)`) | Header `RA` / `DEC` | `OzDES_*.fits` |
| `zcosmos` | filename column | Full filename (`normalize_object_id(path.name)`) | Container table or PRIMARY header `RA` / `DEC` | `zCOSMOS_*.fits`; BinTable `WAVE`/`FLUX_REDUCED`/`ERR` |
| `vandels` | filename column | Full filename (`normalize_object_id(path.name)`) | Header `PND OBJRA` / `PND OBJDEC` (fallback `RA` / `DEC`) | `sc_*.fits` (PRIMARY + NOISE) |
| `vipers` | filename column | Full filename (`normalize_object_id(path.name)`) | Header `RA` / `DEC` (table/PRIMARY fallback) | `VIPERS_*.fits` |
| `vuds` | filename column | Full filename (`normalize_object_id(path.name)`) | Header `LAM CESAM VO ALPHA` / `DELTA` (fallback `RA` / `DEC`) | `sc_*.fits` with `LAM CESAM VO` header |
| `vvds` | filename column | Full filename (`normalize_object_id(path.name)`) | Header `RA` / `DEC`, else `ESO INS REF1 OBJ RA` / `DEC`; error if missing | `sc_*.fits` without VUDS header |
| `wig` | filename column | Full filename (e.g. `wig225415.fits`; stem alone will not match) | Header `RA_OBJ` / `DEC_OBJ` | `wig*.fits`; auto-detect when basename starts with `wig` |

Filename-based readers hash the **full** spectrum filename, including `.fits`, not the stem
alone. Auto-detection runs before `--fmt` is needed, so try
`dl-ingest-spectra FILE --survey NAME` first.

### Catalog vs spectrum CLI flags

On **catalog** ingest, `--link-id-col`, `--ra-col` and `--dec-col` define how native columns
map to `_source_id` and to sky position in Parquet. On **spectrum** ingest, most
survey-specific readers find the ID and position themselves: the OzDES, zCOSMOS, VANDELS,
WiggleZ, VIPERS, VUDS and VVDS readers hash the spectrum filename, and the 2dF and 6dF readers
read link keys from FITS extension headers. Do not pass those flags to `dl-ingest-spectra` for
any of these formats.

For SDSS/BOSS, DESI coadds, generic 1-D FITS and GAMA, `--link-id-col` (and sky keywords when
headers differ from the defaults) selects which header value the reader uses, so set it to
match your catalog. After ingest, `--update-catalog` patches `_spectrum_index` by joining on
the catalog's `_source_id` (resolved from `catalog_info.json`); it does not reuse
`--link-id-col`.

| Stage | `--link-id-col` | `--ra-col` / `--dec-col` |
|-------|-----------------|--------------------------|
| Catalog ingest | Required for production (native column(s) to `_source_id`) | Survey sky columns in degrees |
| Spectrum ingest (2df, 6df) | **Not used**; reader resolves IDs internally (`SPFILE`/`FIBRE`; filename stem + `OBSID_V`/`OBSID_R`) | **Not used**; reader reads `OBSRA`/`OBSDEC` (2dF tries `SRRA`/`SRDEC` first) |
| Spectrum ingest (OzDES, zCOSMOS, VANDELS, WiggleZ, VIPERS, VUDS, VVDS) | **Not used**; reader hashes the filename | **Not used**; reader reads sky from header |
| Spectrum ingest (spPlate, default) | **Not used**; reader hashes `PLATE\|MJD\|FIBERID` per fiber (catalog must use `--link-id-col PLATE,MJD,FIBERID`) | Plugmap columns (default `RA` / `DEC`) |
| Spectrum ingest (SDSS, DESI, generic, GAMA) | DESI defaults to composite `TARGETID,SURVEY,PROGRAM` (omit flag; `--link-id-col TARGETID` for legacy); GAMA defaults to `SPECID`; SDSS and generic try a fixed keyword list when omitted (see the readers table) | FITS header keywords (DESI sky from fibermap) |
| Spectrum ingest (spPlate, legacy modes) | Used only with `--specobj-lookup-from-catalog` (catalog ID column after the join) | Plugmap columns (default `RA` / `DEC`) |
| Catalog patch after spectrum ingest | **Not used**; joins on `_source_id` | — |

## Survey-specific ingest

### SDSS spPlate ingest (640 or 1000 fibers per file)

`spPlate-PLATE-MJD.fits` holds the spectra of every fiber on one plate (SDSS: 640 rows; BOSS:
1000 plugmap rows, typically about 500 with non-zero flux). The file has **no `SPECOBJID`**
column, so ingest maps each `FIBERID` to a `_source_id` in one of three ways. Use the default
(triplet hash) unless your catalog is keyed on CAS (SDSS Catalog Archive Server) `specObjID`.

1. **Triplet hash (default, fast).** Builds `_source_id` from a composite BLAKE2b hash of
   `PLATE|MJD|FIBERID` per fiber. No catalog scan is needed, and the result is identical to
   what catalog ingest with `--link-id-col PLATE,MJD,FIBERID` stores. No extra flags are
   required.
2. **Sidecar or catalog join (legacy).** Joins on `(survey,) PLATE, MJD, FIBERID` against a
   sidecar file (`--specobj-lookup`) or the lake catalog (`--specobj-lookup-from-catalog`).
   The ID is taken from `source_id`, `specobjid` or `TARGETID`, or from `--link-id-col`; the
   photometric `objid` is never used unless you name it explicitly.
3. **Plate header synthesis (legacy).** Synthesizes the CAS `specObjID` from plate, MJD and
   fiber (`--specobj-lookup-from-plate`). DR7 and DR8+ use different 64-bit layouts; choose
   with `--specobj-id-layout auto|dr7|dr8plus`.

Pass only one of `--specobj-lookup`, `--specobj-lookup-from-catalog` and
`--specobj-lookup-from-plate`.

HDU (header data unit) layout, using BOSS `spPlate-3523-55144.fits` as the example: the primary
HDU holds flux with shape `(n_fiber, n_pix)`; `IVAR` holds inverse variance (not sigma);
`ANDMASK` / `ORMASK` hold masks; and the `PLUGMAP` BINTABLE has `FIBERID`, `RA` and `DEC`.

```bash
# Default (fast): composite PLATE|MJD|FIBERID hash — no catalog scan.
# Catalog must be ingested with --link-id-col PLATE,MJD,FIBERID.
dl-ingest-catalog specObj.parquet --survey SDSS_DR17 \
  --link-id-col PLATE,MJD,FIBERID --ra-col RA --dec-col DEC

# Single plate (sequential, per-fiber reader; triplet hash by default)
dl-ingest-spectra spPlate-4002-55645.fits --survey SDSS_DR17 --fmt sdss_spplate

# Recommended for large plate lists: vectorized batch ingest with N workers
# (architecture mirrors dl-ingest-spectra-batch-desi-coadds)
dl-ingest-spectra-batch-spplate /path/to/lake --survey SDSS_DR17 \
  --file-list spplate_paths.txt --n-workers 8

# Alternatively with a directory glob
dl-ingest-spectra-batch-spplate /path/to/lake --survey SDSS_DR17 \
  --spplate-root /data/spectro/redux/v5_13_2 --n-workers 8

# Mixed n_pix across plates (e.g. SDSS vs BOSS reductions): batch ingest defaults to
# --on-length-mismatch pad — tiles widen when a longer plate lands; shorter rows are
# right-padded (NaN flux, 0 ivar/mask). Use --on-length-mismatch error to reject mismatches.
# DESI batch ingest (dl-ingest-spectra-batch-desi-coadds) always uses error (no flag).

# dl-ingest-spectra-from-list uses the same vectorized spPlate decoder when --n-workers > 1
# (unless --specobj-lookup-from-catalog is set)
dl-ingest-spectra-from-list spplate_paths.txt --survey SDSS_DR17 --fmt sdss_spplate \
  --n-workers 8

# Legacy: synthesize CAS specObjID from header (DR8+/BOSS, no catalog scan)
dl-ingest-spectra data/spPlate-3523-55144.fits --survey boss_dr12 \
  --fmt sdss_spplate --specobj-lookup-from-plate --specobj-id-layout dr8plus

# Legacy: SDSS-II / DR7 plates (low bits; no RUN2D) — force DR7 packing:
dl-ingest-spectra spPlate-287-52251.fits --survey sdss_dr7 \
  --fmt sdss_spplate --specobj-lookup-from-plate --specobj-id-layout dr7

# Legacy: sidecar Parquet/CSV (survey, PLATE, MJD, FIBERID, ID column)
dl-ingest-spectra spPlate-1960-53289.fits --survey sdss_dr17 \
  --fmt sdss_spplate --specobj-lookup /path/to/specobj_lookup.parquet

# Legacy: lake catalog scan — join on plate/mjd/fiber (slow for large catalogs)
dl-ingest-spectra spPlate-1960-53289.fits --survey sdss_dr17 \
  --fmt sdss_spplate --specobj-lookup-from-catalog
```

`dl-ingest-spectra-batch-spplate` supports the sidecar and plate-synthesis modes but not
`--specobj-lookup-from-catalog`. Its `--on-length-mismatch` defaults to `pad`. Catalogs
without plate, MJD and fiber columns cannot drive spPlate ingest; photometry-only tables need
`--specobj-lookup-from-plate` or a specObj export with spectroscopic keys. If a lookup maps
zero fibers, see [spPlate lookup maps zero fibers](#troubleshooting).

When you merge several releases into one sidecar lookup file, export the specObj rows with a
constant `survey` column. spPlate wavelength grids (about 3859 pixels, from `COEFF0`/`COEFF1`)
differ from the per-object `spec-*.fits` coadds (about 4628 pixels), so do not expect
pixel-identical spectra between the two products.

**Plates missing from your SDSS catalog.** `dl-extract-spplate-catalog` exports plugmap
positions from spPlate files, optionally keeping only fibers that are not already in the lake
catalog:

```bash
# All active fibers (non-zero flux) from one or more plates
dl-extract-spplate-catalog /data/SDSS/spPlate/spPlate-3523-55144.fits \
  -o spplate_3523_plugmap.parquet

# Many files
dl-extract-spplate-catalog --file-list spplate_paths.txt -o all_plugmaps.parquet

# Only fibers absent from catalogs/SDSS_DR17/ (plate+mjd+fiber anti-join)
dl-extract-spplate-catalog --file-list spplate_paths.txt \
  --subtract-catalog /path/to/lake --survey SDSS_DR17 \
  -o missing_from_specobj.parquet

# Plugmap predicate + cap
dl-extract-spplate-catalog --file-list spplate_paths.txt \
  --where "upper(trim(objtype)) = 'GALAXY'" --limit 1000 \
  -o galaxies.parquet
```

Output columns are `plate`, `mjd`, `fiberid`, `ra` and `dec`, plus `spplate_file`, `holetype`
and `objtype` when present. Use `--all-fibers` to keep fibers with all-zero flux. Ingest the
Parquet as a supplemental catalog (with `ra`/`dec` for HEALPix), or use it as a
`--specobj-lookup` sidecar after adding a `source_id` or `specobjid` column:

```bash
dl-ingest-spectra-from-list spPlate_files.txt --survey boss_dr12 \
  --fmt sdss_spplate --specobj-lookup /path/to/lookup.parquet \
  --on-duplicate skip
```

### 2dFGRS 1-D spectra ingest

2dFGRS (2dF Galaxy Redshift Survey) 1-D FITS files contain one **SPECTRUM** HDU per
observation. HDU 0 carries object metadata (`SEQNUM`, which is the catalog `serial`, plus
`NAME`, `RA`, `DEC`). Each SPECTRUM extension holds a `(3, 1024)` image with rows
`[flux, variance, sky]` and per-observation headers (`SPFILE`, `Z`, `SNR`, `OBSRA`, `OBSDEC`).
Wavelength is reconstructed from `CRVAL1`, `CRPIX1` and `CDELT1` in each extension.

**Link key.** Use the composite `SPFILE|FIBRE` from each SPECTRUM extension header, which is
unique per observation, rather than `serial`: many catalog rows share a `serial` because one
target can be observed several times. Keep `serial` as the science ID, and ingest the catalog
with `--link-id-col SPFILE,FIBRE` so `_source_id` matches on both sides (for example
`sgp805_001203_2z.fits|132`).

When some catalog rows have no spectrum filename (blank `SPFILE`), pass
`--allow-incomplete-link-id`. Those rows then stay in Parquet with a null `_source_id`
(unlinked, `_spectrum_index` stays `-1`) instead of getting a partial `FIBRE`-only key that
would never match a spectrum.

```bash
# Catalog: serial kept; _source_id built from SPFILE,FIBRE composite
dl-ingest-catalog 2dfgrs_catalog.fits --survey 2DFGRS_DR3 \
  --link-id-col SPFILE,FIBRE --ra-col RA --dec-col DEC \
  --allow-incomplete-link-id

# Single file (smoke test) — one Zarr row per SPECTRUM HDU
dl-ingest-spectra 389442.fits --survey 2DFGRS_DR3 --fmt 2df

# Auto-detection also works (SPECTRUM HDU + SEQNUM/BJSEL triggers 2df format)
dl-ingest-spectra 389442.fits --survey 2DFGRS_DR3

# File list (sequential default, with checkpoint for restarts)
dl-ingest-spectra-from-list 2df_files.txt \
  --survey 2DFGRS_DR3 \
  --fmt 2df \
  --on-duplicate skip \
  --on-length-mismatch pad \
  --checkpoint /path/to/lake/ingest_state/2df/checkpoint.json \
  --failures-log /path/to/lake/ingest_state/2df/failures.jsonl

# Parallel decode (2dF, 6dF, GAMA, …) — same checkpoint format as catalog-from-list
dl-ingest-spectra-from-list 2df_files.txt \
  --survey 2DFGRS_DR3 --fmt 2df --n-workers 8 \
  --on-duplicate skip --on-length-mismatch pad
```

Files with more than one SPECTRUM HDU (for example `389442.fits`, which has two), or whose
extensions have different wavelength grids, automatically use `wavelength_mode='per_source'`.

If the catalog was previously linked on `serial`, rebuild the IDs and indices:

```bash
dl-repair-catalog-metadata /path/to/lake --survey 2DFGRS_DR3 --rebuild-link-id SPFILE,FIBRE
dl-rebuild-catalog-indices --survey 2DFGRS_DR3 --kind spectrum
dl-validate-catalog-spectra-link --survey 2DFGRS_DR3 --strict
```

#### 2dFGRS Slurm batch ingest (300 k files)

For large runs, use `scripts/slurm_ingest_2df_spectra.sh`:

```bash
# 1. Build the file list (all 1-D FITS under your data tree)
find /data/2dFGRS/spectra -name "*.fits" | sort > 2df_files.txt

# 2. Submit
mkdir -p logs
export DATA_LAKE_CONFIG=/path/to/lake_config.toml
export LAKE_INGEST_TOKEN='your-secret'
export FILE_LIST="$(pwd)/2df_files.txt"
export SURVEY=2DFGRS_DR3
sbatch scripts/slurm_ingest_2df_spectra.sh
```

The script runs `dl-ingest-spectra-from-list --fmt 2df` with a checkpoint and failures log,
then runs `dl-finalize-catalog` and `dl-validate-spectra-ingest`. It decodes sequentially by
default; export `N_WORKERS` (for example `export N_WORKERS=8`) for parallel decode. Optional
variables `SURVEY`, `NORDER` and `WAVELENGTH_MODE` are documented at the top of the script.
The checkpoint and failures log are written to `ingest_state/2df/` in the directory that
contains `$DATA_LAKE_CONFIG`.

To restart after preemption or a timeout, re-submit the same `sbatch` command; paths recorded
in the checkpoint are skipped.

**Stamp-only FITS files are not spectra.** Some paths under the 2dFGRS final data release tree
are 49×49 postage-stamp images with only a PRIMARY HDU (`SEQNUM`, `RA`, `DEC`) and **no**
`SPECTRUM` extension (for example `161216.fits`). Ingest rejects these with
`2dF stamp-only FITS`. Remove them from the file list, or point ingest at the directory that
holds the matching `(3, n_pix)` spectrum files. To filter a list before submitting:

```bash
python scripts/filter_2df_spectrum_list.py 2df_files.txt -o 2df_spectra_only.txt
```

**Retrying failures:**

```bash
# List failed paths (failures.jsonl sits next to your lake config)
python -c "
import json, sys
for line in open('/path/to/config_dir/ingest_state/2df/failures.jsonl'):
    print(json.loads(line).get('path', ''))
" > retry_list.txt

# Re-run on failures only
export FILE_LIST=retry_list.txt
sbatch scripts/slurm_ingest_2df_spectra.sh
```

**QA after ingest:**

```bash
# Count spectra in Zarr vs files processed
dl-describe-survey 2DFGRS_DR3 --modality spectra

# Spot-check one spectrum (use SPFILE|FIBRE composite label from catalog)
python - <<'PY'
from data_lake.io.spectra import SpectrumAccessor
from data_lake.ingest.fits_to_parquet import composite_link_label, normalize_object_id
acc = SpectrumAccessor('/path/to/lake', '2DFGRS_DR3')
sp = acc.get_spectrum(normalize_object_id(composite_link_label('sgp805_001203_2z.fits', 132)))
print('flux shape:', sp.flux.shape, 'max_ivar:', sp.ivar.max())
PY

# Verify catalog linkage (SPFILE|FIBRE → _spectrum_index)
python - <<'PY'
import pyarrow.parquet as pq, pathlib
tiles = list(pathlib.Path('/path/to/lake/catalogs/2DFGRS_DR3').rglob('*.parquet'))
for t in tiles[:3]:
    tbl = pq.read_table(t, columns=['serial', 'SPFILE', 'FIBRE', '_spectrum_index'])
    unlinked = sum(1 for i in tbl['_spectrum_index'].to_pylist() if i < 0)
    print(t.name, 'rows:', len(tbl), 'unlinked:', unlinked)
PY
```

← [1-D readers reference](#1-d-spectrum-readers-reference) · [Catalog vs spectrum flags](#catalog-vs-spectrum-cli-flags) · [Validate linkage](../validate-and-repair.md#verify-catalog--spectra-linkage)

### 6dFGS spectra ingest (all VR extensions)

6dFGS (6dF Galaxy Survey) target FITS files are multi-extension products: postage-stamp image
HDUs followed by repeating **V / R / VR** spectral blocks. Ingest reads **every** combined
`SPECTRUM VR` extension (not `V` or `R` alone) and pairs each VR with the V and R extensions
immediately before it to build the link key. VR rows are `[flux, variance, sky, wavelength?]`;
when the fourth row (explicit wavelength) exists it is used, otherwise wavelength is
reconstructed from the WCS header keywords. Some targets have several VR versions in one file;
they share the filename stem and are told apart by their paired `OBSID_V` / `OBSID_R`.

**Link key.** The key is the filename stem (the target) plus `OBSID_V` from the paired V header
and `OBSID_R` from the paired R header, joined as `target|obsid_v|obsid_r` (for example
`g2302140-251235|UK-SCHM.20011021.121407|UK-SCHM.20011021.104720`). Ingest the catalog with a
**composite** link column so `_source_id` matches on both sides:

```bash
dl-ingest-catalog 6df_catalog.fits --survey SIXDF_DR3 \
  --link-id-col targetname,obsid_v,obsid_r --ra-col ra --dec-col dec
```

Spectrum ingest resolves the same composite label from the FITS headers itself (no sidecar
lookup, unlike SDSS spPlate). The catalog's `targetname` must equal the spectrum filename stem.

To recompute catalog IDs for this key shape:

```bash
dl-repair-catalog-metadata --survey SIXDF_DR3 \
  --rebuild-link-id targetname,obsid_v,obsid_r
dl-rebuild-catalog-indices --survey SIXDF_DR3 --kind spectrum
```

Then re-run failed spectrum paths from the checkpoint/failures log. Zarr tiles already written
with old `_source_id` keys must be re-ingested or cleaned up separately.

Sky coordinates for HEALPix assignment come from each VR extension's `OBSRA` / `OBSDEC`
keywords (degrees). When `OBSRA` is absent, the sexagesimal `OBJCTRA` / `OBJCTDEC` on the
PRIMARY stamp are parsed as a fallback.

```bash
# Single-file smoke test (two VR rows in data/g2302140-251235.fits)
dl-ingest-spectra g2302140-251235.fits --survey SIXDF_DR3 --fmt 6df

# File-list ingest
dl-ingest-spectra-from-list 6df_files.txt \
  --survey SIXDF_DR3 \
  --fmt 6df \
  --wavelength-mode shared \
  --on-duplicate skip \
  --checkpoint /path/to/lake/ingest_state/6df/checkpoint.json \
  --failures-log /path/to/lake/ingest_state/6df/failures.jsonl
```

As with 2dF, files with several VR extensions or differing wavelength grids switch to
`per_source` wavelength storage automatically.

#### 6dFGS Slurm batch ingest

Use `scripts/slurm_ingest_6df_spectra.sh`:

```bash
find /data/6dFGS/spectra -name "*.fits" | sort > 6df_files.txt
export DATA_LAKE_CONFIG=/path/to/lake_config.toml
export LAKE_INGEST_TOKEN='your-secret'
export FILE_LIST="$(pwd)/6df_files.txt"
export SURVEY=SIXDF_DR3
sbatch scripts/slurm_ingest_6df_spectra.sh
```

The script ingests sequentially with `--on-length-mismatch pad`, writes its checkpoint and
failures log to `ingest_state/6df/` next to `$DATA_LAKE_CONFIG`, and runs `dl-finalize-catalog`
at the end. Re-submit the same command to resume from the checkpoint after a timeout or
preemption.

← [1-D readers reference](#1-d-spectrum-readers-reference) · [Catalog vs spectrum flags](#catalog-vs-spectrum-cli-flags) · [Validate linkage](../validate-and-repair.md#verify-catalog--spectra-linkage)

### GAMA 1-D spectra ingest (stacked PRIMARY)

GAMA (Galaxy And Mass Assembly) AAOmega/2dF spectra are stored as a **2-D PRIMARY image**
`(n_row, n_pix)` with row labels in `ROW1`…`ROW5` (for example `Spectrum`, `Error`, sky rows).
Only the calibrated flux row and the 1σ error row are ingested (`ivar = 1/σ²`).

**Link key.** `SPECID` in the primary header (HDU 0), for example `G23_Y7_015_265`. The GAMA
reader uses `SPECID` by default; use the same column at catalog ingest so `_source_id` matches
on both sides.

```bash
dl-ingest-catalog gama_targets.fits --survey GAMA_DR4 \
  --link-id-col SPECID --ra-col RA --dec-col DEC

dl-ingest-spectra G23_Y7_015_265.fit --survey GAMA_DR4 --fmt gama

# Auto-detect (GAMA stacked layout + G23_/GAMA basename or ORIGIN=GAMA)
dl-ingest-spectra G23_Y7_015_265.fit --survey GAMA_DR4 --link-id-col SPECID
```

Auto-detection needs the GAMA stacked layout (for example `ROW1 = 'Spectrum'`) and either a
filename starting with `G23_` or `GAMA`, or `ORIGIN = 'GAMA'` in the header.

### OzDES spectra ingest (stacked only)

OzDES (the Australian spectroscopic follow-up of the Dark Energy Survey) target FITS files store
the **stacked** spectrum in the first three HDUs: PRIMARY (flux), `VARIANCE`, and `BADPIX`
(`0` = good, `1` = bad). Per-epoch `SPECTRUM_*` extensions are not ingested.

**Catalog linkage.** Ingest the catalog with `--link-id-col` set to the column that stores the
spectrum **filename** (for example `OzDES-DR2_00001.fits`). Spectrum ingest derives
`_source_id` from `normalize_object_id(path.name)`. The `SOURCE` header value (for example
`04D1qt`) stays a science column in the catalog.

```bash
dl-ingest-catalog ozdes_catalog.fits --survey OZDES_DR2 \
  --link-id-col filename --ra-col RA --dec-col DEC

dl-ingest-spectra OzDES-DR2_00001.fits --survey OZDES_DR2 --fmt ozdes

# Auto-detect when the HDU layout matches and the basename starts with OzDES
# (or the header has a SOURCE keyword)
dl-ingest-spectra OzDES-DR2_00001.fits --survey OZDES_DR2
```

### zCOSMOS spectra ingest

zCOSMOS 1-D spectra are stored in a spectral-container BinTable with columns `WAVE`,
`FLUX_REDUCED` and `ERR` (per-pixel arrays in one table row).

**Catalog linkage.** Ingest the catalog with `--link-id-col` set to the column that stores the
spectrum **filename** (for example `zCOSMOS_BRIGHT_DR3_000960004_ZCMRa65_M1_Q4_6_1.fits`).
Spectrum ingest derives `_source_id` from `normalize_object_id(path.name)`.

```bash
dl-ingest-catalog zcosmos_catalog.fits --survey zCOSMOS_DR3 \
  --link-id-col filename --ra-col ra --dec-col dec

dl-ingest-spectra zCOSMOS_BRIGHT_DR3_000960004_ZCMRa65_M1_Q4_6_1.fits \
  --survey zCOSMOS_DR3 --fmt zcosmos

# Auto-detect works when the filename starts with zCOSMOS and layout matches
dl-ingest-spectra zCOSMOS_BRIGHT_DR3_000960004_ZCMRa65_M1_Q4_6_1.fits \
  --survey zCOSMOS_DR3
```

### VANDELS spectra ingest (stacked only)

VANDELS multi-extension FITS files store the stacked 1-D spectrum in PRIMARY with a matching
`NOISE` extension (the 1σ noise estimate, converted to IVAR). Per-epoch `EXR2D`, `SKY` and
`THUMB` extensions are ignored.

**Catalog linkage.** Ingest the catalog with `--link-id-col` set to the column that stores the
spectrum **filename** (for example `sc_UDS313141_P3M1Q4_008_1.fits`). Spectrum ingest derives
the same `_source_id` from `normalize_object_id(path.name)`.

```bash
dl-ingest-catalog vandels_catalog.fits --survey VANDELS \
  --link-id-col <filename_column> --ra-col RA --dec-col DEC

dl-ingest-spectra sc_UDS313141_P3M1Q4_008_1.fits --survey VANDELS --fmt vandels

# Auto-detect works for PRIMARY + NOISE files named sc_*.fits (or with a PND OBJID header)
dl-ingest-spectra sc_UDS313141_P3M1Q4_008_1.fits --survey VANDELS
```

### VIPERS spectra ingest

VIPERS (VIMOS Public Extragalactic Redshift Survey) 1-D spectra are stored as a row-per-pixel
binary table with columns `WAVES`, `FLUXES`, `NOISE` and `MASK`. `MASK` values are stored as
they are, without remapping. Redshift is read from `REDSHIFT`.

**Catalog linkage.** Ingest the catalog with `--link-id-col` set to the column that stores the
spectrum **filename** (for example `VIPERS_406064719.fits`). Spectrum ingest derives
`_source_id` from `normalize_object_id(path.name)`. The `ID` value in the table header stays a
science column in the catalog.

```bash
dl-ingest-catalog vipers_catalog.fits --survey VIPERS \
  --link-id-col spectrum_filename --ra-col RA --dec-col DEC

dl-ingest-spectra VIPERS_406064719.fits --survey VIPERS --fmt vipers

# Auto-detect works for VIPERS_*.fits with the spectral table layout
dl-ingest-spectra VIPERS_406064719.fits --survey VIPERS
```

### VUDS spectra ingest

VUDS (VIMOS Ultra-Deep Survey) 1-D spectra are stored as a single PRIMARY image array with
spectral WCS. Object metadata uses `LAM CESAM VO IDENT`, `LAM CESAM VO ALPHA` / `DELTA`, and
`LAM CESAM VO Z`. No uncertainty or mask extensions are expected, so IVAR is set to 1 and the
mask to 0.

**Catalog linkage.** Ingest the catalog with `--link-id-col` set to the column that stores the
spectrum **filename** (for example `sc_5101243705_F51P006_join_A_10_1_atm_clean.fits`).
`LAM CESAM VO IDENT` in the FITS header is used only for format detection.

```bash
dl-ingest-catalog vuds_catalog.fits --survey VUDS \
  --link-id-col spectrum_filename --ra-col RA --dec-col DEC

dl-ingest-spectra sc_5101243705_F51P006_join_A_10_1_atm_clean.fits --survey VUDS \
  --fmt vuds

# Auto-detect works for sc_*.fits with LAM CESAM VO metadata
dl-ingest-spectra sc_5101243705_F51P006_join_A_10_1_atm_clean.fits --survey VUDS
```

### VVDS spectra ingest

VVDS (VIMOS VLT Deep Survey) 1-D spectra use a PRIMARY flux array (1-D or `(1, n_pix)`) with
spectral WCS. Sky coordinates are read from `RA` / `DEC` when present, otherwise from
`ESO INS REF1 OBJ RA` / `ESO INS REF1 OBJ DEC` (astropy stores hierarchical keywords without
the `HIERARCH` prefix). Missing or invalid coordinates raise an error; they are never defaulted
to (0°, 0°). No uncertainty or mask extensions are expected, so IVAR is set to 1 and the mask
to 0.

**Catalog linkage.** Ingest the catalog with `--link-id-col` set to the column that stores the
spectrum **filename** (for example `sc_000030078_CDFS005_vmM1_red_30_1_atm_clean.fits`). Files
with VUDS `LAM CESAM VO IDENT` metadata are routed to the `vuds` reader instead.

```bash
dl-ingest-catalog vvds_catalog.fits --survey VVDS \
  --link-id-col spectrum_filename --ra-col RA --dec-col DEC

dl-ingest-spectra sc_000030078_CDFS005_vmM1_red_30_1_atm_clean.fits --survey VVDS \
  --fmt vvds

# Auto-detect works for sc_*.fits without VUDS metadata
dl-ingest-spectra sc_000030078_CDFS005_vmM1_red_30_1_atm_clean.fits --survey VVDS
```

### WiggleZ spectra ingest

WiggleZ 1-D FITS files use a 1-D flux array (PRIMARY, or `EXTNAME='spectrum'`) plus a sibling
`VARIANCE` extension. Sky coordinates are in `RA_OBJ` / `DEC_OBJ`.

**Catalog linkage.** Ingest the catalog with `--link-id-col` set to the column that stores the
spectrum **filename** (for example `wig225415.fits`). Spectrum ingest derives the same
`_source_id` from the full filename (`normalize_object_id(path.name)`); the stem alone
(`wig225415`) will **not** match.

```bash
# Catalog (already ingested example)
dl-ingest-catalog wigglez_catalog.fits --survey WIGGLEZ \
  --link-id-col <filename_column> --ra-col RA --dec-col DEC

# Single spectrum
dl-ingest-spectra wig225415.fits --survey WIGGLEZ --fmt wig

# Auto-detect works when the basename starts with ``wig`` and layout matches
dl-ingest-spectra wig225415.fits --survey WIGGLEZ

dl-ingest-spectra-from-list wig_files.txt \
  --survey WIGGLEZ \
  --fmt wig \
  --wavelength-mode shared \
  --on-duplicate skip \
  --checkpoint /path/to/lake/ingest_state/wig/checkpoint.json \
  --failures-log /path/to/lake/ingest_state/wig/failures.jsonl
```

### Parallel batch ingest (many coadd files)

For survey-scale DESI jobs (for example about 10 000 coadd files), use
`dl-ingest-spectra-batch-desi-coadds`. It runs decoding and `coadd_cameras` in N worker
processes, while a single writer in the main process is the only one that touches the Zarr
store. This is the only safe multi-file pattern, because Zarr's `LocalStore` has no
cross-process locks for shard writes.

```bash
# By directory + glob
dl-ingest-spectra-batch-desi-coadds \
    --survey desi_dr1 \
    --coadd-root /data/desi/coadds \
    --coadd-glob 'coadd-*.fits' \
    --n-workers 16

# Or by explicit file list (one path per line)
ls /data/desi/coadds/coadd-*.fits > coadds.txt
dl-ingest-spectra-batch-desi-coadds \
    --survey desi_dr1 \
    --file-list coadds.txt \
    --n-workers 16 \
    --link-id-col TARGETID,SURVEY,PROGRAM
```

Pass exactly one of `--file-list` or `--coadd-root`. `--link-id-col` defaults to
`TARGETID,SURVEY,PROGRAM` (fibermap `TARGETID` plus PRIMARY `SURVEY`/`PROGRAM`, stripped and
hashed as a composite). Pass `TARGETID` alone only for the legacy native-integer path; the
catalog must use the same spec.

`--n-workers` is required, with no implicit default. A typical choice is `cpu_count - 1`, which
leaves one core for the writer and the operating system. Use processes, not threads:
`read_spectra` is mostly Python code that holds the GIL (global interpreter lock).

What the command does for you:

- **Resumes.** Completed file paths are appended, one per line, to
  `<lake>/spectra/<survey>/.ingest_checkpoint.json.jsonl`, so a restart skips them without
  rewriting a large JSON file. An older `.ingest_checkpoint.json` array is still read on
  resume. Override the location with `--checkpoint`.
- **Isolates errors.** Per-file failures are appended to a JSONL log (default
  `<lake>/spectra/<survey>/.ingest_failures.jsonl`, override with `--failures-log`) and the
  run continues. At 10 000-file scale, a small percentage of corrupt inputs is normal.
- **Logs to a file.** Logs go to `<lake>/spectra/<survey>/.ingest.log` (override with
  `--log-file`), so the terminal shows only the progress bar.
- **Patches the catalog.** After ingest it fills `_spectrum_index` in the Parquet catalog by
  scanning the Zarr tiles and joining on `_source_id`. Use `--no-update-catalog` to skip; the
  step is skipped automatically if the survey has no catalog yet.
- **Rejects length mismatches.** There is no `--on-length-mismatch` flag; a spectrum whose
  pixel count differs from its tile is an error.

Other options: `--max-in-flight` (default `n_workers + 2`), `--max-open-tiles` (default 64),
`--files-per-worker` (default 1; try 8–32 for small files), `--on-duplicate` (default `skip`)
and `--norder`.

Rough timing on a single workstation for 10 000 DESI coadd files (about 500 spectra each,
no resolution matrix):

| n_workers | Wall-clock estimate |
|---:|---|
|  1 |  6–11 h |
| 16 | 30–50 min |
| 32 | 20–35 min (SSD I/O may dominate beyond this) |

Disk footprint is roughly 120–170 GB compressed for about 5 million spectra at about 8 000
pixels. For memory problems on large runs, see [Troubleshooting](#troubleshooting).

## Resolution matrix usage

### Using the resolution matrix

When a tile was ingested with `--with-resolution`, each `Spectrum` returned by
`SpectrumAccessor.get_spectrum()` (or `get_spectrum_at()`) has `.resolution` and
`.resolution_offsets` populated. The resolution matrix describes the instrument's line-spread
function (LSF); forward-model templates through it before comparing them with the data.

```python
from data_lake.io.spectra import SpectrumAccessor
import numpy as np

acc = SpectrumAccessor("/data/lake", "desi_edr")
spec = acc.get_spectrum(source_id=39627190271009604)

# Rebuild the sparse banded LSF operator (requires scipy)
R = spec.resolution_operator()          # scipy.sparse.dia_matrix (N_pix, N_pix)

# Forward-model a template through the LSF before comparing to data
model_obs = R @ template_resampled_to_grid
chi2 = np.sum((spec.flux - model_obs) ** 2 * spec.ivar)

# Batched template fitting — single sparse matmul over a whole template library
templates_obs = np.column_stack([
    np.interp(spec.wavelength, tpl.wave * (1 + z), tpl.flux, 0, 0)
    for tpl in template_library
])
models = R @ templates_obs              # (N_pix, n_templates) in one call
chi2s  = ((spec.flux[:, None] - models) ** 2 * spec.ivar[:, None]).sum(axis=0)
best_template = template_library[np.argmin(chi2s)]
```

`resolution_operator()` raises `ValueError` if the spectrum was ingested without
`--with-resolution`, and `ImportError` if `scipy` is not installed.

If you have the banded diagonals and only need scipy (not the full `data-lake` package), you
can rebuild `R` directly:

```python
from scipy.sparse import dia_matrix
import numpy as np

diags   = spec.resolution           # (n_diag, N_pix)
offsets = spec.resolution_offsets   # (n_diag,)
N       = spec.flux.shape[0]
R       = dia_matrix((diags, offsets), shape=(N, N))
```

## Details

### Dynamic tile widening

Spectra that land in the same HEALPix tile may have different pixel counts (`n_pix`). When
the length policy is `pad` (set with `--on-length-mismatch pad`, and applied automatically
for SDSS `spec-*.fits` and `spPlate` files), an incoming spectrum that is **longer** than the
existing tile widens the tile instead of being truncated:

1. A temporary replacement tile is written next to the original
   (`Npix=<N>.zarr.__widening__`).
2. All existing arrays are copied with right-padding: `flux` with `NaN`, `ivar` with `0.0`,
   `mask` with `0`, `wavelength` with `0.0` (per-source rows or the shared 1-D vector), and
   `resolution` (if present) with `0.0` on the pixel axis.
3. The temporary tile is swapped into place using a backup rename.
4. Ingest continues and appends the new rows at the wider `n_pix`.

If widening fails mid-write, the original tile is left untouched. Widening is logged at INFO
level as `Widening spectrum tile Npix=N.zarr: n_pix OLD → NEW (K existing rows)`.

Incoming spectra that are **shorter** than the tile are right-padded. With
`--on-length-mismatch truncate`, widening is disabled and longer spectra are truncated.

A tile is only widened when a longer spectrum lands in it, so tiles filled earlier can stay
narrower than the survey width recorded in `spectrum_info.json`. To pad every narrow tile to
that width (for example after OzDES ingest):

```bash
dl-widen-spectrum-tiles --survey OzDES_DR2
dl-widen-spectrum-tiles --all --dry-run   # list tiles that would change
```

`--n-pix` sets a different target width and `--max-tiles` limits how many tiles per survey are
processed.

## Troubleshooting

**spPlate lookup maps zero fibers.** If a lookup finds 0 fibers, run the debug helper before
re-ingesting:

```bash
# Triplet hash probe — shows sample fiber → source_id without any catalog scan.
# Use this first to verify the IDs produced by the default mode.
dl-debug-specobj-lookup data/spPlate-3523-55144.fits --survey SDSS_DR17 --triplet-hash

# Plate synthesis + lake catalog scan (legacy BOSS / DR17 troubleshooting)
dl-debug-specobj-lookup data/spPlate-3523-55144.fits --survey SDSS_DR17 /path/to/lake

# Plate synthesis only (no catalog)
dl-debug-specobj-lookup spPlate-287-52251.fits --survey sdss_dr7 --no-catalog

# Sidecar + catalog
dl-debug-specobj-lookup spPlate-1960-53289.fits --survey sdss_dr17 /path/to/lake \
  --specobj-lookup /path/to/lookup.parquet
```

The tool reports the PLUGMAP column names (flagging `OBJID` as an imaging ID, not
`specObjID`), the plugmap fiber count, the resolved catalog columns, row counts for the plate
and MJD, sample fiber-to-`source_id` pairs, and the overlap with the plate file. With
`--triplet-hash` it also says whether the IDs will match a catalog ingested with
`--link-id-col PLATE,MJD,FIBERID`. It exits with code 1 when every mode maps zero fibers (the
same condition that makes ingest skip all spectra). Add `--json` for machine-readable output.

**`Spectrum length N != expected M`.** The file's pixel count differs from the tile's. Re-run
with `--on-length-mismatch pad` (widens or pads tiles; see
[Dynamic tile widening](#dynamic-tile-widening)) or `truncate`.

**`2dF stamp-only FITS`.** The file is a postage-stamp image, not a spectrum; filter the list
as described in [2dFGRS Slurm batch ingest](#2dfgrs-slurm-batch-ingest-300-k-files).

**The terminal dies during a large DESI batch run with no Python traceback.** Logs are in
`<lake>/spectra/<survey>/.ingest.log`, not the terminal. Check for **systemd-oomd**
(user-session memory pressure) as well as the kernel out-of-memory (OOM) killer:
`journalctl -u systemd-oomd --since today` or `grep -i oom /var/log/syslog`. To reduce memory:

- Lower `--n-workers`; this is the main lever for decoder process RAM.
- Lower `--max-in-flight` (default `n_workers + 2`), which caps queued decode results.
- Keep the default `--max-open-tiles 64` (`data_lake.knobs.WRITE_MAX_OPEN_TILES`) so the
  writer does not hold thousands of Zarr tiles open; use `0` (unlimited) only for small tests.
  Read and extract accessors use a tighter limit of 8 open tiles (`READ_MAX_OPEN_TILES`).
- Use `--on-duplicate skip` when resuming (the writer caches IDs per open tile).

Run long jobs under `tmux` or `nohup` so that oomd killing the terminal does not stop the
ingest. To rebuild `_spectrum_index` manually after ingest, use `dl-rebuild-catalog-indices`
(one Zarr scan plus one catalog pass; add `--n-workers` on large surveys).
