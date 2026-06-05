# Spectrum ingest

### Ingest spectra

**DESI** ingest requires the `desispec` optional extra (uses `read_spectra` +
`coadd_cameras` for correct IVAR-weighted B/R/Z camera combination):

```bash
# Single file (debug / smoke-testing).
# With $DATA_LAKE_CONFIG set, OUTPUT_ROOT is taken from the config.
# DESI coadds: object ID comes from fibermap TARGETID (default); override with --link-id-col.
dl-ingest-spectra coadd-1-b0-0000p005-thru20210801.fits --survey desi_edr \
  --link-id-col TARGETID

# With resolution matrix (needed for redshift fitting / SPS / kinematic measurements)
# Storage cost: ~3× flux+ivar footprint (~170–200 GB per million coadded BRZ spectra)
dl-ingest-spectra coadd-1-b0-0000p005-thru20210801.fits \
    --survey desi_edr --with-resolution

# Without a config, pass OUTPUT_ROOT explicitly:
dl-ingest-spectra coadd-1-b0-0000p005-thru20210801.fits /data/lake --survey desi_edr

# SDSS/BOSS (no extra dependency needed)
# Match catalog specObj: SPECOBJID lives in the SPALL HDU (HDU 2), not the primary header.
# Pixel count varies slightly file-to-file; ingest uses per_source wavelength and pads
# to the tile's n_pix (override with --on-length-mismatch truncate).
dl-ingest-spectra spec-3586-55181-0001.fits --survey sdss_dr17 \
  --link-id-col SPECOBJID
```

#### 1-D spectrum readers reference

Quick index of all supported `--fmt` values, the **catalog** ingest flag required to match them, how each reader derives the spectrum link ID, and which coordinates are used for HEALPix routing. Do **not** pass `--link-id-col` on `dl-ingest-spectra` for readers listed as "header" or "filename" — the reader resolves the ID internally.

| `--fmt` | Catalog `--link-id-col` | Spectrum link source | Spectrum sky source | Example / note |
|---------|------------------------|----------------------|---------------------|----------------|
| `desi` | `TARGETID` (or default) | Fibermap `TARGETID` | Fibermap `TARGET_RA` / `TARGET_DEC` (fallbacks: `RA_TARGET`, `FIBER_RA`; `DEC_TARGET`, `FIBER_DEC`) | Auto-detected from DESI coadd layout |
| `sdss_boss` | `SPECOBJID` | `SPALL` HDU header | Header `RA` / `DEC` (fallback `PLUG_RA` / `PLUG_DEC`) | `spec-PLATE-MJD-FIBER.fits` |
| `sdss_spplate` | `PLATE,MJD,FIBERID` (default) or via sidecar / plate header | plugmap `FIBERID` → composite hash (default), or specObjID from sidecar/synthesis | Fiber table columns (default `RA` / `DEC`, configurable via `--ra-col/--dec-col`) | `spPlate-PLATE-MJD.fits`; see spPlate section |
| `generic` | `--link-id-col` or auto | Header keyword chain | Header columns from `--ra-col/--dec-col` (default `RA` / `DEC`) | Any 1-D FITS with spectral WCS |
| `2df` | `SPFILE,FIBRE` (+ `--allow-incomplete-link-id` if some rows have no filename) | Header `SPFILE` \| `FIBRE` per SPECTRUM HDU | `SRRA` / `SRDEC` (fallback `OBSRA` / `OBSDEC`, then PRIMARY `RA` / `DEC`) | `data/389442.fits` |
| `6df` | `targetname,obsid_v,obsid_r` | Filename stem + V header `OBSID_V` + R header `OBSID_R` per V/R/VR triple | VR header `OBSRA` / `OBSDEC` (fallback header `RA` / `DEC`, PRIMARY WCS, `OBJCTRA` / `OBJCTDEC`) | `data/g2302140-251235.fits` |
| `gama` | `SPECID` | Primary header `SPECID` | Header columns from `--ra-col/--dec-col` (default `RA` / `DEC`) | `data/G23_Y7_015_265.fit` |
| `ozdes` | filename column | Basename (stem) | Header `RA` / `DEC` | `OzDES_*.fits` |
| `vandels` | filename column | Basename | Header `PND OBJRA` / `PND OBJDEC` (fallback `RA` / `DEC`) | `sc_*.fits` (PRIMARY + NOISE) |
| `vipers` | filename column | Basename | Header `RA` / `DEC` (table/PRIMARY fallback) | `VIPERS_*.fits` |
| `vuds` | filename column | Basename | Header `ALPHA` / `DELTA` (fallback `RA` / `DEC`) | `sc_*.fits` with `LAM CESAM VO` header |
| `vvds` | filename column | Basename | Header `RA` / `DEC`, else `ESO INS REF1 OBJ RA` / `DEC`; error if missing | `sc_*.fits` without VUDS header |
| `wigglez` | filename column | Basename (full, e.g. `wig225415.fits`) | Header `RA_OBJ` / `DEC_OBJ` | `wig*.fits`; stem alone will not match |

Auto-detection runs before `--fmt` is needed: try `dl-ingest-spectra FILE --survey NAME` first.

#### Catalog vs spectrum CLI flags

``--link-id-col``, ``--ra-col``, and ``--dec-col`` on **catalog** ingest define
how native columns map to ``_source_id`` and sky position in Parquet.  Format-specific
spectrum readers (OzDES, VANDELS, WiggleZ, VIPERS, VUDS, VVDS) resolve object IDs
**from the spectrum filename**; 2dF and 6dF read link keys **from FITS extension
headers** — you do **not** pass those flags on ``dl-ingest-spectra`` for any of
these formats.

For SDSS/BOSS, DESI coadds, generic 1-D FITS, and spPlate, pass ``--link-id-col``
(and sky columns when headers differ from defaults) so the reader matches your
catalog.  After ingest, ``--update-catalog`` patches ``_spectrum_index`` by joining
on catalog ``_source_id`` (resolved from ``catalog_info.json``), not by reusing
``--link-id-col``.

| Stage | ``--link-id-col`` | ``--ra-col`` / ``--dec-col`` |
|-------|---------------------|------------------------------|
| Catalog ingest | Required for production (native column → ``_source_id``) | Survey sky columns in degrees |
| Spectrum ingest (2df, 6df) | **Not used** — reader resolves IDs internally (`SPFILE`/`FIBRE`; filename stem + `OBSID_V`/`OBSID_R`) | **Not used** — reader reads `OBSRA`/`OBSDEC` from header |
| Spectrum ingest (OzDES, VANDELS, WiggleZ, VIPERS, VUDS, VVDS) | **Not used** — reader hashes the filename | **Not used** — reader reads sky from header |
| Spectrum ingest (spPlate, default) | **Not used** — reader hashes `PLATE\|MJD\|FIBERID` per fiber (catalog must use `--link-id-col PLATE,MJD,FIBERID`) | FITS plugmap `RA` / `DEC` |
| Spectrum ingest (SDSS, DESI, generic) | Header keyword / fibermap column | FITS header keywords |
| Spectrum ingest (spPlate, legacy modes) | `--link-id-col` used only with `--specobj-lookup-from-catalog` | FITS plugmap `RA` / `DEC` |
| Catalog patch after spectrum ingest | **Not used** — joins on ``_source_id`` | — |

#### SDSS spPlate ingest (640 or 1000 fibers per file)

``spPlate-PLATE-MJD.fits`` holds plate-run spectra for all fibers (SDSS: 640 rows;
BOSS: 1000 plugmap rows, typically ~500 with non-zero flux).  There is **no
SPECOBJID** column in the FITS file.  Ingest maps each **FIBERID** to
``source_id`` via one of:

1. **Triplet hash (default, fast)** — derives ``_source_id`` from a composite
   BLAKE2b hash of ``PLATE|MJD|FIBERID`` per fiber.  No catalog scan; identical
   to what catalog ingest with ``--link-id-col PLATE,MJD,FIBERID`` stores in
   ``_source_id``.  Use this mode (no extra flags required) when your catalog was
   ingested that way.  This is the recommended approach.
2. **Sidecar / catalog** — join on ``(survey,) PLATE, MJD, FIBERID``; ID from
   ``source_id``, ``specobjid``, ``TARGETID``, or ``--link-id-col`` (not
   photometric ``objid`` unless you set that column explicitly)
3. **Plate header** — synthesize CAS ``specObjID`` from plate/mjd/fiber
   (``--specobj-lookup-from-plate``).  **DR7 and DR8+ use different 64-bit layouts**
   (see below); use ``--specobj-id-layout auto|dr7|dr8plus``.

HDU layout (BOSS example ``spPlate-3523-55144.fits``): primary flux
``(n_fiber, n_pix)``; ``IVAR`` (inverse variance, not sigma); ``ANDMASK`` /
``ORMASK``; ``PLUGMAP`` BINTABLE with ``FIBERID``, ``RA``, ``DEC``.

##### Dynamic tile widening

Different ``spPlate`` files for the same sky tile may have different pixel
counts (``n_pix``).  When ``--on-length-mismatch pad`` is set (the default for
``sdss_spplate`` format), **incoming spectra that are longer than the existing
tile are handled by widening the tile** rather than being truncated:

1. A temporary replacement tile is written alongside the original
   (``Npix=<N>.zarr.__widening__``).
2. All existing arrays are copied row-by-row with right-padding:
   ``flux`` → ``NaN``, ``ivar`` → ``0.0``, ``mask`` → ``0``,
   ``wavelength`` → ``0.0`` (per-source rows or shared 1-D vector),
   ``resolution`` (if present) → ``0.0`` on the pixel axis.
3. The temp tile is atomically swapped into place (backup rename strategy).
4. Ingest continues normally: new rows are appended at the wider ``n_pix``.

If widening fails mid-write the original tile is left untouched.  Widening
is logged at INFO level: ``Widening spectrum tile Npix=N.zarr: n_pix OLD → NEW
(K existing rows)``.

Incoming spectra **shorter** than the current tile are still right-padded by
``_fix_length`` as before.  Using ``--on-length-mismatch truncate`` disables
widening and truncates longer spectra instead.

Tiles that were filled **before** a longer ``n_pix`` appeared in the survey never
get widened until a longer spectrum lands in that HEALPix tile.  To pad every
narrow tile to the ``spectrum_info.json`` width (e.g. after OzDES ingest):

```bash
dl-widen-spectrum-tiles --survey OzDES_DR2
dl-widen-spectrum-tiles --all --dry-run   # list tiles that would change
```

```bash
# Default (fast): composite PLATE|MJD|FIBERID hash — no catalog scan.
# Catalog must be ingested with --link-id-col PLATE,MJD,FIBERID.
dl-ingest-catalog specObj.parquet --survey SDSS_DR17 \
  --link-id-col PLATE,MJD,FIBERID --ra-col RA --dec-col DEC

# Single plate (sequential, uses fast vectorized decoder automatically)
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
# DESI batch ingest (dl-ingest-spectra-batch-desi-coadds) still defaults to error.

# dl-ingest-spectra-from-list also uses the fast vectorized path for spPlate
# (benefits from --n-workers without requiring the dedicated CLI)
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

If catalog lookup finds **0 fibers**, run the debug helper before re-ingesting:

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

The tool reports **PLUGMAP column names** (flags ``OBJID`` as imaging ID, not
``specObjID``), plugmap fiber count, resolved catalog columns, row counts for
plate/mjd, sample ``fiber → source_id`` pairs, and overlap with the plate file.
Exit code **1** when every mode maps zero fibers (same failure as ingest).

When using `--triplet-hash`, the tool notes whether the IDs will match a catalog
that was ingested with `--link-id-col PLATE,MJD,FIBERID`.

Catalogs without plate/mjd/fiber cannot drive spPlate ingest; photo-only tables
need ``--specobj-lookup-from-plate`` or a specObj export with spectroscopic keys.

**Plates missing from your SDSS catalog** — export plugmap positions from the
FITS files, optionally keeping only fibers not already in the lake catalog:

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
```

Output columns: ``plate``, ``mjd``, ``fiberid``, ``ra``, ``dec`` (plus
``spplate_file``, ``holetype``, ``objtype`` when present).  Ingest the Parquet
as a supplemental catalog (with ``ra``/``dec`` for HEALPix) or use it as a
``--specobj-lookup`` sidecar after adding a ``source_id`` / ``specobjid`` column.

```bash
dl-ingest-spectra-from-list spPlate_files.txt --survey boss_dr12 \
  --fmt sdss_spplate --specobj-lookup /path/to/lookup.parquet \
  --on-duplicate skip
```

#### 2dFGRS 1-D spectra ingest

2dFGRS 1-D FITS files contain one **SPECTRUM** HDU per observation.  HDU 0
carries object metadata (``SEQNUM`` = catalog ``serial``, ``NAME``, ``RA``,
``DEC``).  Each SPECTRUM extension holds a ``(3, 1024)`` image with rows
``[flux, variance, sky]`` plus per-observation headers (``SPFILE``, ``Z``,
``SNR``, ``OBSRA``, ``OBSDEC``).  Wavelength is reconstructed from
``CRVAL1 / CRPIX1 / CDELT1`` in each extension.

**Link key:** use composite ``SPFILE|FIBRE`` from each SPECTRUM extension header
(unique per observation), not ``serial``.  Many catalog rows share the same
``serial`` (multiple observations of one target).  Keep ``serial`` as the science
ID; ingest the catalog with ``--link-id-col SPFILE,FIBRE`` so ``_source_id`` matches
spectrum ingest on both sides (e.g. ``sgp805_001203_2z.fits|132``).

When the catalog has rows with no spectrum filename (blank ``SPFILE``), use
``--allow-incomplete-link-id`` so those rows stay in Parquet with null
``_source_id`` (unlinked; ``_spectrum_index`` stays ``-1``) instead of building
a partial ``FIBRE``-only key that will not match spectra.

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

For **DESI coadd** file lists use ``dl-ingest-spectra-batch-desi-coadds`` (not
``--n-workers`` on from-list).

Multi-observation files (e.g. ``389442.fits`` with two SPECTRUM HDUs) auto-use
``wavelength_mode='per_source'`` when extensions have different WCS grids.

If the catalog was previously linked on ``serial``:

```bash
dl-repair-catalog-metadata /path/to/lake --survey 2DFGRS_DR3 --rebuild-link-id SPFILE,FIBRE
dl-rebuild-catalog-indices --survey 2DFGRS_DR3 --kind spectrum
dl-validate-catalog-spectra-link --survey 2DFGRS_DR3 --strict
```

#### 2dFGRS Slurm batch ingest (300 k files)

Use `scripts/slurm_ingest_2df_spectra.sh` for large-scale runs:

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

The script runs `dl-ingest-spectra-from-list` (sequential; add `--n-workers` in
the script or invoke the CLI directly for parallel decode on large 2dF lists),
with checkpoint + failure log,
- calls `dl-finalize-catalog` and `dl-validate-spectra-ingest` on completion.

**Restarting after preemption or timeout** — re-submit the same `sbatch`
command; the checkpoint file records completed paths and they are skipped.

**Stamp-only FITS (not spectra):** some paths under the FDR tree are
49×49 postage-stamp images with only a PRIMARY HDU (`SEQNUM`, `RA`, `DEC`)
and **no** `SPECTRUM` extension (e.g. `161216.fits`). Ingest correctly rejects
these with `2dF stamp-only FITS`. They are not 1-D spectra; remove them from
the file list or point ingest at the directory that holds the matching
`(3, n_pix)` spectrum files. To filter a list before Slurm:

```bash
python scripts/filter_2df_spectrum_list.py 2df_files.txt -o 2df_spectra_only.txt
```

**Reviewing / retrying failures:**

```bash
# List failed paths
python -c "
import json, sys
for line in open('ingest_state/2df/failures.jsonl'):
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

← [1-D readers reference](#1-d-spectrum-readers-reference) · [Catalog vs spectrum flags](#catalog-vs-spectrum-cli-flags) · [Validate linkage](#verify-catalog--spectra-linkage)

#### 6dFGS spectra ingest (all VR extensions)

6dFGS target FITS files are multi-extension products: stamp image HDUs followed
by repeating **V / R / VR** spectral blocks.  Ingest reads **every** combined
``SPECTRUM VR`` extension (not ``V`` or ``R`` alone), pairing each VR with its
immediately preceding V extension for the link key.  VR rows:
``[flux, variance, sky, wavelength?]``. If the 4th row (explicit wavelength)
exists it is preferred; otherwise wavelength is reconstructed from WCS header
keywords.  Some targets have multiple VR versions in one file (same
filename stem target, disambiguated by paired ``OBSID_V``/``OBSID_R``).

**Link key:** filename stem (target) plus ``OBSID_V`` from the paired V header
and ``OBSID_R`` from the paired R header → ``target|obsid_v|obsid_r`` (e.g.
``g2302140-251235|UK-SCHM.20011021.121407|UK-SCHM.20011021.104720``).  Ingest the catalog with a
**composite** source column so ``_source_id`` matches on both sides:

```bash
dl-ingest-catalog 6df_catalog.fits --survey SIXDF_DR3 \
  --link-id-col targetname,obsid_v,obsid_r --ra-col ra --dec-col dec
```

Spectrum ingest resolves the same composite label from FITS headers internally
(no sidecar lookup like SDSS spPlate). ``targetname`` must match the spectrum
filename stem.

To recompute catalog IDs for this key shape:

```bash
dl-repair-catalog-metadata --survey SIXDF_DR3 \
  --rebuild-link-id targetname,obsid_v,obsid_r
dl-rebuild-catalog-indices --survey SIXDF_DR3 --kind spectrum
```

Re-run failed spectrum paths from the checkpoint/failures log.  Zarr tiles already
written with old ``_source_id`` keys need re-ingest or separate cleanup.

Sky coordinates for HEALPix assignment come from each VR extension's ``OBSRA`` /
``OBSDEC`` keywords (degrees).  Sexagesimal ``OBJCTRA``/``OBJCTDEC`` on the
PRIMARY stamp are parsed as a fallback when ``OBSRA`` is absent.

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

Re-submit the same command to resume from checkpoint after timeout/preemption.

← [1-D readers reference](#1-d-spectrum-readers-reference) · [Catalog vs spectrum flags](#catalog-vs-spectrum-cli-flags) · [Validate linkage](#verify-catalog--spectra-linkage)

#### GAMA 1-D spectra ingest (stacked PRIMARY)

GAMA AAOMEGA-2dF spectra are stored as a **2-D PRIMARY image** ``(n_row, n_pix)``
with row labels in ``ROW1``…``ROW5`` (e.g. ``Spectrum``, ``Error``, sky rows).
Only the calibrated flux row and 1-σ error row are ingested (``ivar = 1/σ²``).

**Link key:** ``SPECID`` in the primary header (HDU 0), e.g. ``G23_Y7_015_265``.
Use the same column at catalog ingest so ``_source_id`` matches on both sides.

```bash
dl-ingest-catalog gama_targets.fits --survey GAMA_DR4 \
  --link-id-col SPECID --ra-col RA --dec-col DEC

dl-ingest-spectra G23_Y7_015_265.fit --survey GAMA_DR4 --fmt gama

# Auto-detect when ORIGIN=GAMA or ROW1=Spectrum (e.g. G23_* basenames)
dl-ingest-spectra G23_Y7_015_265.fit --survey GAMA_DR4 --link-id-col SPECID
```

#### OzDES spectra ingest (stacked only)

OzDES target FITS files store the **stacked** spectrum in the first three HDUs:
PRIMARY (flux), ``VARIANCE``, ``BADPIX`` (``0`` = good, ``1`` = bad).  Per-epoch
``SPECTRUM_*`` extensions are not ingested.

**Catalog linkage:** ingest the catalog with ``--link-id-col`` set to the column
that stores the spectrum **filename** (e.g. ``OzDES-DR2_00001.fits``). Spectrum
ingest derives ``_source_id`` from the FITS basename.  The ``SOURCE`` header
(e.g. ``04D1qt``) remains a science column in the catalog.

```bash
dl-ingest-catalog ozdes_catalog.fits --survey OZDES_DR2 \
  --link-id-col filename --ra-col RA --dec-col DEC

dl-ingest-spectra OzDES-DR2_00001.fits --survey OZDES_DR2 --fmt ozdes

# Auto-detect when the basename starts with OzDES and HDU layout matches
dl-ingest-spectra OzDES-DR2_00001.fits --survey OZDES_DR2
```

← [1-D readers reference](#1-d-spectrum-readers-reference) · [Catalog vs spectrum flags](#catalog-vs-spectrum-cli-flags)

#### VANDELS spectra ingest (stacked only)

VANDELS multi-extension FITS files store the stacked 1-D spectrum in PRIMARY with a
matching ``NOISE`` extension (1-σ noise estimate → IVAR). Per-epoch ``EXR2D`` / ``SKY`` /
``THUMB`` extensions are ignored.

**Catalog linkage:** ingest the catalog with ``--link-id-col`` set to the column that
stores the spectrum **filename** (e.g. ``sc_UDS313141_P3M1Q4_008_1.fits``). Spectrum
ingest derives the same ``source_id`` from ``normalize_object_id(path.name)``.

```bash
dl-ingest-catalog vandels_catalog.fits --survey VANDELS \
  --link-id-col <filename_column> --ra-col RA --dec-col DEC

dl-ingest-spectra sc_UDS313141_P3M1Q4_008_1.fits --survey VANDELS --fmt vandels

# Auto-detect works for sc_*.fits with PRIMARY + NOISE layout
dl-ingest-spectra sc_UDS313141_P3M1Q4_008_1.fits --survey VANDELS
```

← [1-D readers reference](#1-d-spectrum-readers-reference) · [Catalog vs spectrum flags](#catalog-vs-spectrum-cli-flags)

#### VIPERS spectra ingest

VIPERS 1-D spectra are stored as a row-per-pixel binary table with columns
``WAVES``, ``FLUXES``, ``NOISE``, and ``MASK``.  ``MASK`` values are stored as
ingested (no remapping).  Redshift is read from ``REDSHIFT``.

**Catalog linkage:** ingest the catalog with ``--link-id-col`` set to the column
that stores the spectrum **filename** (e.g. ``VIPERS_406064719.fits``).  Spectrum
ingest derives ``_source_id`` from the FITS basename.  The ``ID`` table header
remains a science column in the catalog.

```bash
dl-ingest-catalog vipers_catalog.fits --survey VIPERS \
  --link-id-col spectrum_filename --ra-col RA --dec-col DEC

dl-ingest-spectra VIPERS_406064719.fits --survey VIPERS --fmt vipers

# Auto-detect works for VIPERS_*.fits with the spectral table layout
dl-ingest-spectra VIPERS_406064719.fits --survey VIPERS
```

← [1-D readers reference](#1-d-spectrum-readers-reference) · [Catalog vs spectrum flags](#catalog-vs-spectrum-cli-flags)

#### VUDS spectra ingest

VUDS 1-D spectra are stored as a single PRIMARY image array with spectral WCS.
Object metadata uses ``LAM CESAM VO IDENT``, ``LAM CESAM VO ALPHA`` / ``DELTA``,
and ``LAM CESAM VO Z``.  No uncertainty or mask extensions are expected (IVAR=1,
mask=0).

**Catalog linkage:** ingest the catalog with ``--link-id-col`` set to the column
that stores the spectrum **filename** (e.g.
``sc_5101243705_F51P006_join_A_10_1_atm_clean.fits``).  ``LAM CESAM VO IDENT`` in
the FITS header is used for format detection only.

```bash
dl-ingest-catalog vuds_catalog.fits --survey VUDS \
  --link-id-col spectrum_filename --ra-col RA --dec-col DEC

dl-ingest-spectra sc_5101243705_F51P006_join_A_10_1_atm_clean.fits --survey VUDS \
  --fmt vuds

# Auto-detect works for sc_*.fits with LAM CESAM VO metadata
dl-ingest-spectra sc_5101243705_F51P006_join_A_10_1_atm_clean.fits --survey VUDS
```

← [1-D readers reference](#1-d-spectrum-readers-reference) · [Catalog vs spectrum flags](#catalog-vs-spectrum-cli-flags)

#### VVDS spectra ingest

VVDS 1-D spectra use a PRIMARY flux array (1-D or ``(1, n_pix)``) with spectral WCS.
Sky coordinates are read from ``RA`` / ``DEC`` when present, otherwise from
``ESO INS REF1 OBJ RA`` / ``ESO INS REF1 OBJ DEC`` (astropy stores hierarchical
keywords without the ``HIERARCH`` prefix).  Missing or invalid coordinates raise
an error (they are never defaulted to 0°, 0°).  No uncertainty or mask extensions
are expected (IVAR=1, mask=0).

**Catalog linkage:** ingest the catalog with ``--link-id-col`` set to the column
that stores the spectrum **filename** (e.g.
``sc_000030078_CDFS005_vmM1_red_30_1_atm_clean.fits``).  Files with VUDS
``LAM CESAM VO IDENT`` metadata are routed to the ``vuds`` reader instead.

```bash
dl-ingest-catalog vvds_catalog.fits --survey VVDS \
  --link-id-col spectrum_filename --ra-col RA --dec-col DEC

dl-ingest-spectra sc_000030078_CDFS005_vmM1_red_30_1_atm_clean.fits --survey VVDS \
  --fmt vvds

# Auto-detect works for sc_*.fits without VUDS metadata
dl-ingest-spectra sc_000030078_CDFS005_vmM1_red_30_1_atm_clean.fits --survey VVDS
```

← [1-D readers reference](#1-d-spectrum-readers-reference) · [Catalog vs spectrum flags](#catalog-vs-spectrum-cli-flags)

#### WiggleZ spectra ingest

WiggleZ 1-D FITS files use a 1-D flux array (PRIMARY / ``EXTNAME='spectrum'``) plus a
sibling ``VARIANCE`` extension.  Sky coordinates are in ``RA_OBJ`` / ``DEC_OBJ``.

**Catalog linkage:** ingest the catalog with ``--link-id-col`` set to the column
that stores the spectrum **filename** (e.g. ``wig225415.fits``).  Spectrum ingest
derives the same ``source_id`` from the file basename (``normalize_object_id`` of
``wig225415.fits``), so the stem alone (``wig225415``) will **not** match.

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

← [1-D readers reference](#1-d-spectrum-readers-reference) · [Catalog vs spectrum flags](#catalog-vs-spectrum-cli-flags)

#### Parallel batch ingest (many coadd files)

Export specObj rows with a constant ``survey`` column when merging multiple releases
into one lookup file.  spPlate wavelength grids (~3859 px, ``COEFF0``/``COEFF1``) differ
from per-object ``spec-*.fits`` coadds (~4628 px); do not expect pixel-identical spectra.

For survey-scale jobs (e.g. ~10 000 DESI coadd files) use the parallel
batch CLI.  It runs decode + ``coadd_cameras`` in N worker processes while
a single main-thread writer is the only process that touches the Zarr
store — that's the only multi-file pattern that is **safe** here, since
``LocalStore`` has no cross-process locks for shard writes.

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
    --n-workers 16
```

Key properties:

- **Resumable** — completed file paths are appended to
  `<output>/spectra/<survey>/.ingest_checkpoint.json.jsonl` (one path per
  line; legacy `.ingest_checkpoint.json` arrays are still read on resume).
  Restarts skip completed paths without rewriting a multi‑MB JSON file.
- **Error-isolated** — per-file failures are appended to a JSONL log
  (default `<output>/spectra/<survey>/.ingest_failures.jsonl`) and the
  run continues.  At 10k-file scale a small percentage of corrupt
  inputs is normal.
- **Vectorised HEALPix assignment** — one `assign_healpix` call per file
  instead of per record.
- **Automatic catalog patch** — after ingest the run patches
  `_spectrum_index` in the Parquet catalog (`--no-update-catalog` to
  skip; silently no-ops if no catalog exists yet for the survey).
  The ID column is read from `catalog_info.json` so DESI catalogs
  ingested with `--link-id-col TARGETID` are handled correctly.
- **`--n-workers` is required** — no implicit default; pick consciously
  (typical: `cpu_count - 1` to keep one core for the writer / OS).
- **Threads do not help here**: `read_spectra` is mostly Python under
  the GIL.  Stick to processes.

Rough timing on a single workstation for 10 000 DESI coadd files
(~500 spectra each, no resolution matrix):

| n_workers | Wall-clock estimate |
|---:|---|
|  1 |  6–11 h |
| 16 | 30–50 min |
| 32 | 20–35 min (SSD I/O may dominate beyond this) |

Disk footprint: ~120–170 GB compressed for ~5 M spectra at ~8 000 px.

**Large runs (10k+ coadds) and memory:** Logs go to
`<output>/spectra/<survey>/.ingest.log`, not the terminal.  If the shell or
IDE terminal dies with no Python traceback, check **systemd-oomd** (user-session
memory pressure) as well as the kernel OOM killer:
`journalctl -u systemd-oomd --since today` or `grep -i oom /var/log/syslog`.

Mitigations built into the batch command:

- Lower **`--n-workers`** (main lever for decoder process RAM).
- Lower **`--max-in-flight`** (default `n_workers + 2`; caps queued decode results).
- Keep default **`--max-open-tiles 64`** so the writer does not hold thousands of
  Zarr tiles open; use `0` only for small tests.
- Use **`--on-duplicate skip`** when resuming (writer caches IDs per open tile).

Run long jobs under `tmux`/`nohup` so oomd killing the terminal does not stop
the ingest.  Catalog patching scans Zarr tiles one at a time (bounded RAM); for a
manual rebuild after ingest use `dl-rebuild-catalog-indices`.



# Resolution matrix usage

#### Using the resolution matrix

When a tile was ingested with `--with-resolution`, each `Spectrum` object
returned by `SpectrumAccessor.get_spectrum()` has `.resolution` and
`.resolution_offsets` populated:

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

If you have the banded diagonals stored in a Zarr tile and only need scipy
(not the full `data-lake` package), you can rebuild `R` directly:

```python
from scipy.sparse import dia_matrix
import numpy as np

diags   = spec.resolution           # (n_diag, N_pix)
offsets = spec.resolution_offsets   # (n_diag,)
N       = spec.flux.shape[0]
R       = dia_matrix((diags, offsets), shape=(N, N))
```
