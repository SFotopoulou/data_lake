# Astronomy Data Lake

A local-first data lake for multi-survey astronomy catalogs, galaxy image cutouts, and 1-D spectra.

- **Wide catalogs** (>1 000 columns): stored as HATS-partitioned Parquet — no FITS column limit, column-projection so colleagues download only what they need.
- **Galaxy image cutouts**: stored as sharded Zarr v3 arrays — one tidy file per HEALPix tile, fast ML dataloading, lossless WCS round-trip to FITS.
- **1-D spectra** (SDSS/BOSS, DESI, generic): stored as sharded Zarr v3 stacks alongside cutouts — flux, IVAR, mask, shared or per-source wavelength, per-source scalar metadata.
- FITS is kept as the **ingest/export** format for observatory interoperability; it is not used as internal storage.

## Quick-start

Use [`uv`](https://docs.astral.sh/uv/) for all Python environments in this repo.
It installs from the pinned `uv.lock`, resolves `desispec` and its transitive
deps correctly, and keeps the env isolated from system Python.

```bash
# 1. Install uv (one-time, per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # add to your shell rc

# 2. Clone repo and create a Python 3.11 venv inside the project
git clone https://github.com/SFotopoulou/data_lake.git
cd data_lake
uv venv --python 3.11 .venv

# 3. Install from the lockfile (editable package + extras):
#    desi — DESI ingest (pulls in desispec)
#    dev  — pytest, Jupyter, matplotlib, napari, …
uv sync --extra desi --extra dev --extra fitsio

# 4. Activate the env (or use `uv run` / `.venv/bin/python` without activating)
source .venv/bin/activate
```

To add or bump dependencies, edit `pyproject.toml` and run `uv lock`, then
`uv sync` again.

**Jupyter / notebooks:** register the project kernel once:

```bash
uv run python -m ipykernel install --user --name=data-lake --display-name "Python (data-lake)"
```

Select **Python (data-lake)** in JupyterLab. All notebooks under `notebooks/`
assume this environment.

Smoke test before any large ingest:

```bash
uv run python -m pytest tests/test_desi_ingest.py -v       # unit tests
uv run python scripts/dry_run_desi_ingest.py               # end-to-end on 3 DESI files
```

The dry-run script ingests three small DESI coadd files in both
`--with-resolution` and plain modes, then reads them back to verify shape,
finiteness, and (for the resolution path) interior row sums ~1.0.

### Create a deployment

A deployment is *your* private instance of the data lake. Use `dl-init`
to scaffold one:

```bash
dl-init my_lake ~/projects \
    --description "Personal multi-survey lake" \
    --root /scratch/my_lake/data \
    --norder 5 \
    --ingest-token 'your-ingest-secret'
```

This creates `~/projects/my_lake/` with:

```
my_lake/
  lake_config.toml      # name, root, defaults — single source of truth
  README.md             # auto-generated, deployment-specific
  .gitignore            # excludes data/, logs/
  data/                 # actual tiles (lives at --root if you passed one)
    catalogs/ spectra/ cutouts/ shared/
  notebooks/  scripts/  # yours to fill in
```

Point every subsequent CLI at the deployment with a single env var:

```bash
export DATA_LAKE_CONFIG=~/projects/my_lake/lake_config.toml
```

You can either:

- `git init` the deployment to version-control your config, notebooks
  and scripts (but keep `data/` gitignored); or
- leave it un-tracked — it is just a working directory.

You can have multiple deployments side by side (e.g. `prod`, `staging`,
`personal`) and switch between them by exporting a different
`DATA_LAKE_CONFIG`.

### Roles: analysts vs ingest operators

| Role | Install | `DATA_LAKE_CONFIG` | `LAKE_INGEST_TOKEN` | Data path |
|------|---------|-------------------|---------------------|-----------|
| **Analyst** | `uv pip install data-lake` (or `uv sync` in a clone) | Shared deployment config | **Not used** | Read-only mount |
| **Ingest operator** | Same package | Same or writable deployment | **Required** for `dl-ingest-*` | Read-write |

Installing the package only provides scripts and the Python API. It does **not**
grant ingest rights. Anyone with shell access can still call ingest CLIs, but
production lakes should rely on **filesystem permissions** (analysts cannot write
Zarr tiles) as the real control.

### Ingest token (rudimentary operator authentication)

Every `dl-ingest-*` run **requires**:

1. A deployment config (`$DATA_LAKE_CONFIG` or `--config`).
2. A matching **ingest token** (`$LAKE_INGEST_TOKEN` or `--ingest-token`).

The deployment stores only a **SHA-256 hash** in `.ingest_token_hash` (mode
`0600`, gitignored) next to `lake_config.toml`. Plaintext tokens live in the
operator environment (Slurm secret, vault) — never in git.

**This is minimal security, not a strong boundary.** The token does not stop
someone who can (a) install this package, (b) obtain the token, and (c) write
the data directory from corrupting the lake. It is a lightweight “I am an ingest
operator” check. **Real protection** is correct Unix/NFS ACLs (read-only lake
for analysts) and future proper authentication.

**Create a deployment** (operators — token required at init):

```bash
dl-init mylake ~/projects --ingest-token 'your-secret' --root /scratch/mylake/data
export DATA_LAKE_CONFIG=~/projects/mylake/lake_config.toml
```

**Rotate token** on an existing deployment:

```bash
dl-set-ingest-token --ingest-token 'new-secret'
```

**Run ingest** (operators only):

```bash
export LAKE_INGEST_TOKEN='your-secret'   # prefer env over --ingest-token (ps visibility)
dl-ingest-spectra-batch --survey DESI_DR1 --file-list coadds.txt --n-workers 8
```

**Analysts** point at the shared config and use read APIs only — no token, no
`dl-init` on the production tree:

```bash
export DATA_LAKE_CONFIG=/path/to/mylake/lake_config.toml
# SpectrumAccessor, dl-extract-spectra-subset, notebooks, validators, …
```

Read-only tools never check the ingest token.

### Ingest a survey catalog

With a deployment config in place (`$DATA_LAKE_CONFIG` set), the
`OUTPUT_ROOT` argument is optional — it is filled in from the config:

```bash
dl-ingest-catalog survey_catalog.fits --survey des_dr2 --ra-col RA --dec-col DEC
# Also: .csv, .csv.gz, .tsv, .tsv.gz, .parquet, VOTable (in-memory path; --streaming is FITS-only).
# Delimited text: auto-detects comma vs tab vs semicolon (tab-in-.csv.gz works for GAIA-style exports).
```

Without a config you can still pass the path explicitly (legacy mode):

```bash
dl-ingest-catalog survey_catalog.fits /data/lake --survey des_dr2 --ra-col RA --dec-col DEC
```

Explicit CLI flags (`--norder`, etc.) override config defaults.

For very large FITS catalogs pass `--streaming`. The streaming path memory-maps
the FITS, sorts only RA/Dec/source_id columns, then writes one HEALPix
tile at a time via per-tile fancy indexing into the memmap:

```bash
dl-ingest-catalog zall-pix-iron.fits --survey desi_dr1 \
    --ra-col TARGET_RA --dec-col TARGET_DEC --source-id-col TARGETID \
    --streaming
```

Memory peak is bounded to ~one tile's worth of rows (tens of MB at
Norder=5) instead of the full table + sorted copy (~3× the raw size).
The on-disk Parquet output is identical to the in-memory path
(round-trip-tested), so consumers don't care which mode was used.
Trade-off: per-tile scattered I/O makes the streaming path ~1.5–2×
slower wall-clock; use only when memory is a constraint.

**Disk footprint:** Parquet is often **larger than a compressed FITS** file
because FITS may use internal compression, while we store a full typed,
queryable columnar layout (~12k tile files, ZSTD, per-column statistics).
To reduce size on re-ingest:

```bash
# Smaller tiles (ZSTD-9, no stats/dictionary, narrow strings per tile)
dl-ingest-catalog zall-pix-iron.fits --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --source-id-col TARGETID \
  --streaming --overwrite --compact

# Largest win: drop unused DESI columns (keep what you query/join on)
dl-ingest-catalog ... --columns TARGETID,TARGET_RA,TARGET_DEC,Z,MAG_G,MAG_R,MAG_Z,SPECTYPE
```

Expect **~2–4×** smaller than default ingest when combining `--compact` with a
sensible `--columns` list; exact ratio depends on which FITS columns you keep.

**Multiple FITS into one survey:** by default existing `Npix=*.parquet` tiles are
**skipped** (`--tile-mode skip`). To add rows from another file into the same
HEALPix pixel, use **append** (read–concat–write per tile):

```bash
dl-ingest-catalog-from-list desi_files.txt --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --source-id-col TARGETID \
  --streaming --tile-mode append
```

When appending with a native ID column (`TARGETID`), control duplicates with
`--on-duplicate-id skip|error|last` (default `skip`). Re-ingesting the **same**
FITS with ``append`` + ``skip`` is idempotent: rows already on disk (matched by
``source_id``) are dropped per tile; unchanged tiles are not rewritten. Use
``--tile-mode overwrite`` to rebuild a tile from one file only. ``--overwrite`` is
deprecated but still maps to ``--tile-mode overwrite``. After every ingest (and at
the end of ``dl-ingest-catalog-batch``), ``_metadata``, ``catalog_info.json``
(``total_rows``), and ``schema_manifest.json`` are refreshed from **all** tiles on
disk. If a batch job was killed before finalize, or the manifest is missing after a
checkpoint-only re-run, use ``dl-finalize-catalog --survey <name>``.

**Mixed numeric dtypes across files** (common in AllWISE/GALEX batches): the same
column may be ``E`` (float32) in one FITS and ``D`` (float64) in another. Ingest
promotes floats to **float64**, integers to **int64**, and inner elements of
``FixedSizeList`` columns likewise, before writing or appending tiles — so append
no longer fails on dtype mismatch and the first file no longer locks a narrower
Parquet type.  When rebuilding ``_metadata``, any tiles still on legacy dtypes
(e.g. float32 ``flux`` in one ``Npix=`` file and float64 in another) are rewritten
automatically before the aggregate footer is written.

#### Choosing HEALPix order (`--norder` / `hats_order`)

Catalogs, cutouts, and spectra for a survey share one **HEALPix nested** order
(`assign_healpix` in ingest). Each non-empty sky pixel becomes one on-disk shard
(`Norder=<N>/Dir=<D>/Npix=<P>.parquet` or `.zarr`). **Disk footprint and inode
count scale with the number of shard files**, not only with row count — sparse
all-sky catalogs at high order can be much larger than the source FITS.

**Mean tile area on the full sphere** (equal-area pixels; same convention as
`healpy.order2nside` / this repo’s `--norder`):

| `--norder` | `nside` | Max tiles (full sky) | Mean tile area |
|------------|---------|----------------------|----------------|
| 0 | 1 | 12 | 3438 deg² |
| 1 | 2 | 48 | 859 deg² |
| 2 | 4 | 192 | 215 deg² |
| 3 | 8 | 768 | 53.7 deg² |
| 4 | 16 | 3 072 | 13.4 deg² |
| 5 | 32 | 12 288 | **3.36 deg²** (default) |
| 6 | 64 | 49 152 | 0.84 deg² |
| 7 | 128 | 196 608 | 0.21 deg² |
| 8 | 256 | 786 432 | 0.052 deg² |
| 9 | 512 | 3 145 728 | 0.013 deg² |
| 10 | 1024 | 12 582 912 | 0.0033 deg² |

Only tiles that contain at least one source are written, but for a **sparse
all-sky** catalog the number of files still grows quickly with order (often
approaching one file per object at very high order).

**How to choose**

1. **Target rows per shard** — for TB-scale catalogs, aim for roughly **10⁴–10⁵
   rows per non-empty** `Npix=*.parquet` file. Too many tiny files → metadata and
   filesystem overhead (e.g. multi‑MB on-disk size from a 2 MB FITS when order is
   too high); too few huge tiles → slow append and heavy single-tile RAM.

2. **Survey density, not survey name** — use a **lower** order for sparse
   all-sky tables; keep **5** for dense survey footprints (DESI, deep drills)
   and Rubin/LSST **lsdb** interoperability.

3. **One order per survey** — set `--norder` (or `catalog_info.json`
   `hats_order`) per catalog ingest. Spectra/cutouts for that survey must use the
   **same** order. Different surveys in one lake **may** use different orders;
   tile-aligned cross-match in this repo expects the **same** order on both sides
   (otherwise join on sky position or via a master association table).

4. **Benchmark before TB ingests** — use **`dl-recommend-catalog-norder`** (quick
   FITS scan of RA/Dec only) or ingest a subset, then check tile counts and
   `du -sh catalogs/<survey>`.

| Catalog type | Typical `--norder` | Why |
|--------------|-------------------|-----|
| Sparse all-sky (10⁶–10⁷ rows, full sphere) | **3–4** | Fewer, larger Parquet tiles; smaller inode footprint |
| Dense survey footprint (DESI, deep fields) | **5** (default) | ~3.4 deg² tiles; matches common HATS/LSST practice |
| Very local, high density | **6–7** | Only if tiles still hold many rows per file |

Override the deployment default per run:

```bash
dl-ingest-catalog sparse_allsky.fits --survey my_sparse --norder 4 \
  --ra-col RA --dec-col DEC --source-id-col ID --streaming
```

**Pre-ingest norder scan** (reads FITS headers + a subsample of RA/Dec; no Parquet write):

```bash
# One file or directory of FITS
dl-recommend-catalog-norder /path/to/catalogs/*.fits \
  --ra-col TARGET_RA --dec-col TARGET_DEC

# File list (same paths as batch ingest)
dl-recommend-catalog-norder --file-list allwise_files.txt \
  --ra-col ra --dec-col dec --max-files 32 --sample-rows 500000
```

Prints a table of candidate orders with estimated **rows/tile**, **tile count**, and
**pixel_sky_frac**, and highlights a recommended `--norder` near 10⁴–10⁵ rows per
occupied pixel (default target 50 000). Re-run with more `--sample-rows` for large,
clustered footprints.

#### Object identifiers (`--source-id-col`)

The lake uses a single internal join column ``_source_id`` (int64) in Parquet
catalogs and Zarr ``_source_id/`` arrays. Survey-native columns (e.g. ultraVISTA
``SOURCE_ID``, DESI ``TARGETID``) are kept unchanged. Anything starting with ``_``
is lake-owned bookkeeping (like ``_healpix_norder*``, ``_cutout_index``,
``_spectrum_index``).

Catalog ingest records ``source_id_mode`` in ``catalog_info.json``; the join
column is always ``source_id_column: "_source_id"``. When you pass
``--source-id-col``, that native column is also stored as ``native_id_column``.

| Input type | Example | Parquet columns | ``source_id_mode`` |
|------------|---------|-----------------|---------------------|
| Integer column | DESI ``TARGETID`` | ``TARGETID`` (int64) + ``_source_id`` (same values) | ``column:TARGETID`` |
| Unsigned / uint64 column | SDSS ``objid`` (> ``2**63-1``) | Native cast + ``_source_id`` | ``column:objid`` |
| Decimal string column | ``"39627658462934656"`` in FITS ASCII | Parsed native + ``_source_id`` | ``column:TARGETID`` |
| Vector ID column | SDSS ``OBJID`` shape ``(5,)`` | **Error** — use scalar ``objid`` | — |
| Alphanumeric labels | ``J000000.00-314627.5`` in ``NAME`` | ``NAME`` kept; ``_source_id`` = stable hash | ``label:NAME`` |
| (none) | — | ``_source_id`` 0…N−1 only | ``sequential`` |

**Whitespace:** leading and trailing spaces are stripped before parsing or
hashing (common for fixed-width FITS strings). Internal spaces are preserved.

**Alphanumeric labels:** the human-readable name stays in your column (e.g.
``NAME``). ``_source_id`` holds the deterministic hash (BLAKE2b → 64-bit signed int).

```python
from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, normalize_object_id

label = "J000000.00-314627.5"
sid = normalize_object_id(label)   # same int64 as catalog _source_id / Zarr row
acc.get_spectrum(sid)
```

SQL on names: ``SELECT * FROM catalog WHERE NAME = 'J000000.00-314627.5'``.
Use the **same** spelling (after strip) in cutout/spectrum FITS headers via
``--source-id-col NAME`` so ingest hashes match the catalog.

If no ``--source-id-col`` is given, ingest tries common column names
(``TARGETID``, ``SOURCE_ID``, …) or generates sequential ``_source_id`` values.

**Upgrading existing lakes** (tiles still have legacy ``source_id``):

```bash
dl-repair-catalog-metadata /lake --survey MY_SURVEY --check-only
dl-repair-catalog-metadata /lake --survey MY_SURVEY --migrate-join-column
# spectra/cutouts Zarr tiles:
dl-repair-catalog-metadata /lake --survey MY_SURVEY --migrate-join-column --spectra --cutouts
```

**Reassign catalog link column** (recompute ``_source_id`` from another column without FITS re-ingest; catalog Parquet only):

```bash
dl-repair-catalog-metadata /lake --survey zCOSMOS_DR3 --rebuild-link-id filename
dl-rebuild-catalog-indices --survey zCOSMOS_DR3 --kind spectrum
dl-validate-catalog-spectra-link --survey zCOSMOS_DR3
```

Science columns (e.g. ``id``) are unchanged; ``_spectrum_index`` is reset and must be repatched.

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
  --ra-col TARGET_RA --dec-col TARGET_DEC --source-id-col TARGETID --streaming

# 2. One cutout FITS per object (paths in cutout_files.txt)
dl-ingest-cutouts-from-list cutout_files.txt --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --source-id-col TARGETID \
  --on-duplicate skip
```

Use the **same** ``--survey``, sky columns, and ``--source-id-col`` as the catalog
so ``update_index_column`` can patch ``_cutout_index``.

#### FITS header requirements (per cutout file)

| Purpose | CLI flag | Header keyword(s) | Notes |
|--------|----------|-------------------|--------|
| Object ID (join to catalog) | ``--source-id-col`` | e.g. ``TARGETID``, ``NAME`` | **Required** for production. Must match catalog ingest (int64 or hashed label). See [Object identifiers](#object-identifiers---source-id-col). If omitted, tries ``SOURCE_ID``, ``OBJ_ID``, ``TARGETID``, … then HDU index. |
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
  --ra-col TARGET_RA --dec-col TARGET_DEC --source-id-col TARGETID
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
  --source-id-col TARGETID --ra-col TARGET_RA --dec-col TARGET_DEC \\
  --band-names r,i,z
```

Band images must share a consistent astrometric grid (2-D WCS per FITS). Cutouts
use ``astropy.nddata.Cutout2D`` with ``mode='partial'`` (edge sources may include
``NaN`` fills).

### Ingest spectra

**DESI** ingest requires the `desispec` optional extra (uses `read_spectra` +
`coadd_cameras` for correct IVAR-weighted B/R/Z camera combination):

```bash
# Single file (debug / smoke-testing).
# With $DATA_LAKE_CONFIG set, OUTPUT_ROOT is taken from the config.
# DESI coadds: object ID comes from fibermap TARGETID (default); override with --source-id-col.
dl-ingest-spectra coadd-1-b0-0000p005-thru20210801.fits --survey desi_edr \
  --source-id-col TARGETID

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
  --source-id-col SPECOBJID
```

#### SDSS spPlate ingest (640 or 1000 fibers per file)

``spPlate-PLATE-MJD.fits`` holds plate-run spectra for all fibers (SDSS: 640 rows;
BOSS: 1000 plugmap rows, typically ~500 with non-zero flux).  There is **no
SPECOBJID** column in the FITS file.  Ingest maps each **FIBERID** to
``source_id`` via one of:

1. **Sidecar / catalog** — join on ``(survey,) PLATE, MJD, FIBERID``; ID from
   ``source_id``, ``specobjid``, ``TARGETID``, or ``--source-id-col`` (not
   photometric ``objid`` unless you set that column explicitly)
2. **Plate header** — synthesize CAS ``specObjID`` from plate/mjd/fiber
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

```bash
# Quick ingest without a specObj sidecar (DR8+/BOSS: primary header RUN2D only;
# VERS2D/VERSCOMB are pipeline versions, not used for specObjID)
dl-ingest-spectra data/spPlate-3523-55144.fits --survey boss_dr12 \
  --format sdss_spplate --specobj-lookup-from-plate --specobj-id-layout dr8plus

# SDSS-II / DR7 plates (low bits; no RUN2D) — force DR7 packing:
dl-ingest-spectra spPlate-287-52251.fits --survey sdss_dr7 \
  --format sdss_spplate --specobj-lookup-from-plate --specobj-id-layout dr7

# Sidecar Parquet/CSV: survey, PLATE, MJD, FIBERID, plus an ID column
dl-ingest-spectra spPlate-1960-53289.fits --survey sdss_dr17 \
  --format sdss_spplate \
  --specobj-lookup /path/to/specobj_lookup.parquet

# BOSS spPlates with an explicit specObj table
dl-ingest-spectra spPlate-5695-56191.fits --survey boss_dr12 \
  --format sdss_spplate \
  --specobj-lookup /path/to/boss_specobj_lookup.parquet

# Lake catalog: join on plate/mjd/fiber; ID from catalog source_id or specobjid
dl-ingest-spectra spPlate-1960-53289.fits --survey sdss_dr17 \
  --format sdss_spplate --specobj-lookup-from-catalog

# If the catalog used a non-default ID column at ingest time:
dl-ingest-spectra spPlate-1960-53289.fits --survey sdss_dr17 \
  --format sdss_spplate --specobj-lookup-from-catalog --source-id-col TARGETID
```

If catalog lookup finds **0 fibers**, run the debug helper before re-ingesting:

```bash
# Plate synthesis + lake catalog scan (typical BOSS / DR17 troubleshooting)
dl-debug-specobj-lookup data/spPlate-3523-55144.fits --survey SDSS_DR17 /path/to/lake

# Plate synthesis only (no catalog)
dl-debug-specobj-lookup spPlate-287-52251.fits --survey sdss_dr7 --no-catalog

# Sidecar + catalog
dl-debug-specobj-lookup spPlate-1960-53289.fits --survey sdss_dr17 /path/to/lake \
  --specobj-lookup /path/to/lookup.parquet
```

The tool reports **PLUGMAP column names** (flags ``OBJID`` as imaging ID, not
``specObjID``), plugmap fiber count, resolved catalog columns, row counts for
plate/mjd, sample ``fiber → specobjid`` pairs, and overlap with the plate file.
Exit code **1** when every mode maps zero fibers (same failure as ingest).

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
  --format sdss_spplate --specobj-lookup /path/to/lookup.parquet \
  --on-duplicate skip
```

#### 2dFGRS 1-D spectra ingest

2dFGRS 1-D FITS files contain one **SPECTRUM** HDU per observation.  HDU 0
carries object metadata (``SEQNUM`` = catalog ``serial``, ``NAME``, ``RA``,
``DEC``).  Each SPECTRUM extension holds a ``(3, 1024)`` image with rows
``[flux, variance, sky]`` plus per-observation headers (``SPFILE``, ``Z``,
``SNR``, ``OBSRA``, ``OBSDEC``).  Wavelength is reconstructed from
``CRVAL1 / CRPIX1 / CDELT1`` in each extension.

**Link key:** use ``SPFILE`` (unique per observation) for ``_source_id``, not
``serial``.  Many catalog rows share the same ``serial`` (multiple observations
of one target).  Keep ``serial`` as the science ID; ingest the catalog with
``--source-id-col SPFILE`` so ``_source_id = hash(SPFILE)`` on both sides.

```bash
# Catalog: serial kept; _source_id built from SPFILE
dl-ingest-catalog 2dfgrs_catalog.fits --survey 2DFGRS_DR3 \
  --source-id-col SPFILE --ra-col RA --dec-col DEC

# Single file (smoke test) — one Zarr row per SPECTRUM HDU
dl-ingest-spectra 389442.fits --survey 2DFGRS_DR3 \
  --fmt 2df --source-id-col SPFILE

# Auto-detection also works (SPECTRUM HDU + SEQNUM/BJSEL triggers 2df format)
dl-ingest-spectra 389442.fits --survey 2DFGRS_DR3 --source-id-col SPFILE

# File list (sequential, with checkpoint for restarts)
dl-ingest-spectra-from-list 2df_files.txt \
  --survey 2DFGRS_DR3 \
  --fmt 2df \
  --source-id-col SPFILE \
  --on-duplicate skip \
  --on-length-mismatch pad \
  --checkpoint /path/to/lake/ingest_state/2df/checkpoint.json \
  --failures-log /path/to/lake/ingest_state/2df/failures.jsonl
```

Multi-observation files (e.g. ``389442.fits`` with two SPECTRUM HDUs) auto-use
``wavelength_mode='per_source'`` when extensions have different WCS grids.

If the catalog was previously linked on ``serial``:

```bash
dl-repair-catalog-metadata /path/to/lake --survey 2DFGRS_DR3 --rebuild-link-id SPFILE
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

The script:
- runs `dl-ingest-spectra-from-list` with checkpoint + failure log,
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
dl-describe-lake --config "$DATA_LAKE_CONFIG" --survey 2DFGRS_DR3

# Spot-check one spectrum (use SPFILE label from catalog)
python - <<'PY'
from data_lake.io.spectra import SpectrumAccessor
from data_lake.ingest.fits_to_parquet import normalize_object_id
acc = SpectrumAccessor('/path/to/lake', '2DFGRS_DR3')
sp = acc.get_spectrum(normalize_object_id('sgp805_001203_2z.fits'))
print('flux shape:', sp.flux.shape, 'max_ivar:', sp.ivar.max())
PY

# Verify catalog linkage (SPFILE → _spectrum_index)
python - <<'PY'
import pyarrow.parquet as pq, pathlib
tiles = list(pathlib.Path('/path/to/lake/catalogs/2DFGRS_DR3').rglob('*.parquet'))
for t in tiles[:3]:
    tbl = pq.read_table(t, columns=['serial', 'SPFILE', '_spectrum_index'])
    unlinked = sum(1 for i in tbl['_spectrum_index'].to_pylist() if i < 0)
    print(t.name, 'rows:', len(tbl), 'unlinked:', unlinked)
PY
```

#### 6dFGS spectra ingest (VR extension only)

6dFGS target FITS files are multi-extension products. Ingest here reads only
the combined ``VR`` spectral extension (not ``V`` or ``R``), with rows:
``[flux, variance, sky, wavelength?]``. If the 4th row (explicit wavelength)
exists it is preferred; otherwise wavelength is reconstructed from WCS header
keywords.

Source IDs are derived from the file stem (for example
``g0001234-123456.fits`` → ``g0001234-123456``) and normalized with the same
stable hashing path used by catalog ingest for string IDs.

```bash
# Single-file smoke test
dl-ingest-spectra g0001234-123456.fits --survey SIXDF_DR3 \
  --fmt 6df --source-id-col targetname

# File-list ingest
dl-ingest-spectra-from-list 6df_files.txt \
  --survey SIXDF_DR3 \
  --fmt 6df \
  --source-id-col targetname \
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
export SOURCE_ID_COL=targetname
sbatch scripts/slurm_ingest_6df_spectra.sh
```

Re-submit the same command to resume from checkpoint after timeout/preemption.

#### OzDES spectra ingest (stacked only)

OzDES target FITS files store the **stacked** spectrum in the first three HDUs:
PRIMARY (flux), ``VARIANCE``, ``BADPIX`` (``0`` = good, ``1`` = bad).  Per-epoch
``SPECTRUM_*`` extensions are not ingested.

**Catalog linkage:** ingest the catalog with ``--source-id-col`` set to the column
that stores the spectrum **filename** (e.g. ``OzDES-DR2_00001.fits``). Spectrum
ingest derives ``_source_id`` from the FITS basename.  The ``SOURCE`` header
(e.g. ``04D1qt``) remains a science column in the catalog.

```bash
dl-ingest-catalog ozdes_catalog.fits --survey OZDES_DR2 \
  --source-id-col filename --ra-col RA --dec-col DEC

dl-ingest-spectra OzDES-DR2_00001.fits --survey OZDES_DR2 \
  --fmt ozdes --source-id-col filename

# Auto-detect when the basename starts with OzDES and HDU layout matches
dl-ingest-spectra OzDES-DR2_00001.fits --survey OZDES_DR2 --source-id-col filename
```

#### VANDELS spectra ingest (stacked only)

VANDELS multi-extension FITS files store the stacked 1-D spectrum in PRIMARY with a
matching ``NOISE`` extension (1-σ noise estimate → IVAR). Per-epoch ``EXR2D`` / ``SKY`` /
``THUMB`` extensions are ignored.

**Catalog linkage:** ingest the catalog with ``--source-id-col`` set to the column that
stores the spectrum **filename** (e.g. ``sc_UDS313141_P3M1Q4_008_1.fits``). Spectrum
ingest derives the same ``source_id`` from ``normalize_object_id(path.name)``.

```bash
dl-ingest-catalog vandels_catalog.fits --survey VANDELS \
  --source-id-col <filename_column> --ra-col RA --dec-col DEC

dl-ingest-spectra sc_UDS313141_P3M1Q4_008_1.fits --survey VANDELS --fmt vandels

# Auto-detect works for sc_*.fits with PRIMARY + NOISE layout
dl-ingest-spectra sc_UDS313141_P3M1Q4_008_1.fits --survey VANDELS
```

#### VIPERS spectra ingest

VIPERS 1-D spectra are stored as a row-per-pixel binary table with columns
``WAVES``, ``FLUXES``, ``NOISE``, and ``MASK``.  ``MASK`` values are stored as
ingested (no remapping).  Redshift is read from ``REDSHIFT``.

**Catalog linkage:** ingest the catalog with ``--source-id-col ID``; spectrum ingest
reads the same ``ID`` keyword from the table header (e.g. ``406064719``).

```bash
dl-ingest-catalog vipers_catalog.fits --survey VIPERS \
  --source-id-col ID --ra-col RA --dec-col DEC

dl-ingest-spectra VIPERS_406064719.fits --survey VIPERS \
  --fmt vipers --source-id-col ID

# Auto-detect works for VIPERS_*.fits with the spectral table layout
dl-ingest-spectra VIPERS_406064719.fits --survey VIPERS --source-id-col ID
```

#### VUDS spectra ingest

VUDS 1-D spectra are stored as a single PRIMARY image array with spectral WCS.
Object metadata uses ``LAM CESAM VO IDENT``, ``LAM CESAM VO ALPHA`` / ``DELTA``,
and ``LAM CESAM VO Z``.  No uncertainty or mask extensions are expected (IVAR=1,
mask=0).

**Catalog linkage:** ingest the catalog with ``--source-id-col ID``; spectrum
ingest reads ``LAM CESAM VO IDENT`` (``--source-id-col ID`` is accepted as an
alias).

```bash
dl-ingest-catalog vuds_catalog.fits --survey VUDS \
  --source-id-col ID --ra-col RA --dec-col DEC

dl-ingest-spectra sc_5101243705_F51P006_join_A_10_1_atm_clean.fits --survey VUDS \
  --fmt vuds --source-id-col ID

# Auto-detect works for sc_*.fits with LAM CESAM VO metadata
dl-ingest-spectra sc_5101243705_F51P006_join_A_10_1_atm_clean.fits --survey VUDS \
  --source-id-col ID
```

#### VVDS spectra ingest

VVDS 1-D spectra use a PRIMARY flux array (1-D or ``(1, n_pix)``) with spectral WCS.
Sky coordinates are in ``RA`` / ``DEC``.  No uncertainty or mask extensions are
expected (IVAR=1, mask=0).

**Catalog linkage:** ingest the catalog with ``--source-id-col ID``; spectrum ingest
derives ``source_id`` from the numeric segment in the filename prefix
``sc_<ID>_...`` (e.g. ``sc_000030078_...`` → ``30078``).  Files with VUDS
``LAM CESAM VO IDENT`` metadata are routed to the ``vuds`` reader instead.

```bash
dl-ingest-catalog vvds_catalog.fits --survey VVDS \
  --source-id-col ID --ra-col RA --dec-col DEC

dl-ingest-spectra sc_000030078_CDFS005_vmM1_red_30_1_atm_clean.fits --survey VVDS \
  --fmt vvds

# Auto-detect works for sc_*.fits without VUDS metadata
dl-ingest-spectra sc_000030078_CDFS005_vmM1_red_30_1_atm_clean.fits --survey VVDS
```

#### WiggleZ spectra ingest

WiggleZ 1-D FITS files use a 1-D flux array (PRIMARY / ``EXTNAME='spectrum'``) plus a
sibling ``VARIANCE`` extension.  Sky coordinates are in ``RA_OBJ`` / ``DEC_OBJ``.

**Catalog linkage:** ingest the catalog with ``--source-id-col`` set to the column
that stores the spectrum **filename** (e.g. ``wig225415.fits``).  Spectrum ingest
derives the same ``source_id`` from the file basename (``normalize_object_id`` of
``wig225415.fits``), so the stem alone (``wig225415``) will **not** match.

```bash
# Catalog (already ingested example)
dl-ingest-catalog wigglez_catalog.fits --survey WIGGLEZ \
  --source-id-col <filename_column> --ra-col RA --dec-col DEC

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
dl-ingest-spectra-batch \
    --survey desi_dr1 \
    --coadd-root /data/desi/coadds \
    --coadd-glob 'coadd-*.fits' \
    --n-workers 16

# Or by explicit file list (one path per line)
ls /data/desi/coadds/coadd-*.fits > coadds.txt
dl-ingest-spectra-batch \
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
  ingested with `--source-id-col TARGETID` are handled correctly.
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

#### Duplicate / resume flags by command

| Command | Flag | Values | Notes |
|---------|------|--------|--------|
| `dl-ingest-catalog` | `--on-duplicate-id` | `skip`, `error`, `last` | Only when `--tile-mode append` (Parquet rows) |
| `dl-ingest-catalog-from-list` | `--on-duplicate-id` | same | same |
| `dl-ingest-catalog-batch` | `--on-duplicate-id` | same | Parallel decode; default `--tile-mode append`; writes manifest at finalize |
| `dl-finalize-catalog` | — | — | Rebuild ``catalog_info.json``, ``_metadata``, ``schema_manifest.json`` from tiles |
| `dl-repair-catalog-metadata` | `--rebuild-link-id` | column name | Recompute catalog ``_source_id`` from column (catalog only); then run ``dl-rebuild-catalog-indices`` |
| `dl-repair-catalog-metadata` | — | — | Repair ``catalog_info.json``; ``--check-only``; ``--migrate-join-column`` renames legacy ``source_id`` → ``_source_id`` (``--spectra`` / ``--cutouts`` for Zarr) |
| `dl-ingest-cutouts` | `--on-duplicate` | `skip`, `error`, `append` | Default **`skip`**; per `source_id` in each `Npix=*.zarr` |
| `dl-ingest-cutouts-from-list` | `--on-duplicate` | same | same |
| `dl-ingest-spectra` | `--on-duplicate` | same | same |
| `dl-ingest-spectra-from-list` | `--on-duplicate` | same | Also ``--on-length-mismatch``, ``--wavelength-mode`` |
| `dl-ingest-spectra-batch` | `--on-duplicate` | same | DESI parallel batch; default **`skip`** |

Cutout/spectrum ingest defaults to **`--on-duplicate skip`** so file-list and batch re-runs
are idempotent. Use **`append`** only when you intentionally want duplicate Zarr rows.
Catalog append uses **`--on-duplicate-id skip`** (default) with **`--tile-mode append`**.

#### Parallel catalog batch (large file lists)

For surveys shipped as **many catalog files** (e.g. Gaia `GaiaSource_*.csv.gz`), use
parallel decode with a **single-thread Parquet writer** so overlapping HEALPix tiles
are merged safely:

```bash
dl-ingest-catalog-batch gaia_files.txt --survey GAIA_DR3_source \
  --ra-col ra --dec-col dec --source-id-col source_id --norder 5 \
  --tile-mode append --on-duplicate-id skip --n-workers 8
```

Same flags on `dl-ingest-catalog-from-list` when `--n-workers > 1` (default `1` =
sequential). **`--streaming` is not supported** on the parallel path.

Each batch exit (including when every file is already in the checkpoint) runs
**finalize**: ``catalog_info.json``, Parquet ``_metadata``, and
``schema_manifest.json``. If a Slurm job is killed mid-run, tiles may exist without
a manifest — run ``dl-finalize-catalog --survey <name>`` before
``dl-refresh-lake-registry``.

Peak RAM scales roughly as **`O(n_workers × largest catalog file)`** — each worker
still decodes a full file in memory, but tile tables are **spooled to temp Parquet
in the worker** (not pickled back to the parent). Lower `--n-workers` and use
`--columns` on wide surveys (ALLWISE). Default **`--max-in-flight`** is **`n_workers`**
(not `n_workers + 2` like spectra batch).

If the OS kills a worker (**OOM**), you may see `BrokenProcessPool`; the batch tool
**restarts the pool** and re-queues in-flight files. Persistent OOM → fewer workers
and a smaller `--columns` set.

#### Sequential file-list ingest (catalogs, cutouts, spectra)

```bash
find /data/cats -name '*.fits' > cat_files.txt
dl-ingest-catalog-from-list cat_files.txt --survey my_survey --ra-col RA --dec-col DEC

find /data/cutouts -name '*.fits' > cutout_files.txt
dl-ingest-cutouts-from-list cutout_files.txt --survey desi_dr1 \
  --ra-col TARGET_RA --dec-col TARGET_DEC --source-id-col TARGETID \
  --band-names r,i,z --on-duplicate skip

ls coadds.txt  # one DESI coadd path per line
dl-ingest-spectra-batch --survey desi_dr1 --file-list coadds.txt --n-workers 16 \
  --on-duplicate skip

# Generic / SDSS spectra (sequential, not parallel DESI batch):
dl-ingest-spectra-from-list spec_files.txt --survey sdss_dr17 \
  --source-id-col SPECOBJID --on-duplicate skip

# spPlate file lists (same flags as dl-ingest-spectra):
dl-ingest-spectra-from-list spPlate_files.txt --survey boss_dr12 \
  --format sdss_spplate --specobj-lookup /path/to/lookup.parquet --on-duplicate skip

SDSS spec lists: pixel lengths differ slightly; ingest auto-pads (or widens the
existing tile) when format is ``sdss_boss`` or ``sdss_spplate``.  Override with
``--on-length-mismatch pad`` (default; widens tiles for longer incoming spectra)
or ``truncate`` (same as ``dl-ingest-spectra``).

# Patch _cutout_index / _spectrum_index when --update-catalog (default).
```

Checkpoints default to ``catalogs/<survey>/.ingest_checkpoint.json`` or
``cutouts/<survey>/.ingest_checkpoint.json`` under the lake root.  Optional
``--failures-log`` writes JSONL per-file errors.

#### Validate Parquet / Zarr survey directories

```bash
dl-validate-catalog-ingest --survey des_dr2
dl-validate-cutout-ingest --survey des_dr2
dl-validate-spectra-ingest --survey desi_edr
```

These accept ``--file-list``, ``--checkpoint``, ``--inflight``, ``--max-tiles``, and ``--strict`` (same semantics as the spectrum validator).

#### Verify catalog ↔ spectra linkage

After spectrum ingest (with ``--update-catalog``, the default), each catalog row
should carry ``_source_id``, ``_healpix_norder{N}``, and ``_spectrum_index`` pointing
at the matching row in the paired ``Npix=*.zarr`` tile.  Cross-check with:

```bash
# 1. Zarr internal layout (flux/ivar/mask/_source_id shapes)
dl-validate-spectra-ingest --survey zCOSMOS_DR3

# 2. Catalog row index ↔ Zarr _source_id agreement (per HEALPix tile)
dl-validate-catalog-spectra-link --survey zCOSMOS_DR3

# Quick smoke: first tile only, sample 100 linked rows per tile
dl-validate-catalog-spectra-link --survey zCOSMOS_DR3 --max-tiles 1 --sample 100
```

If step 2 reports **unpatched catalog** (``_spectrum_index=-1`` but Zarr row exists)
or stale indices, rebuild from on-disk Zarr without re-ingesting FITS:

```bash
# --norder defaults to catalog_info.json hats_order (not always 5)
dl-rebuild-catalog-indices --survey zCOSMOS_DR3 --kind spectrum
dl-validate-catalog-spectra-link --survey zCOSMOS_DR3
```

Use ``--strict`` to treat orphan Zarr rows and unpatched catalog warnings as errors.
Override partitioning only when needed: ``--norder 1`` (must match catalog ``hats_order``).

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
survey = "desi_dr1"
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
    --survey desi_dr1 \
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
```

`--format` choices: `zarr` (default), `parquet`, `hdf5`, `fits`.  For FITS,
`--fits-layout` is `per-file` (default) or `catalog`.  All require
`wavelength_mode="shared"` in the source survey.

Output layout for **zarr**:

```
qso_subset.zarr/
  flux/        (N_written, N_pix) float32 sharded
  ivar/        (N_written, N_pix) float32 sharded
  mask/        (N_written, N_pix) uint8   sharded
  wavelength/  (N_pix,)           float64 shared grid
  _source_id/  (N_written,)       int64
  redshift/    (N_written,)       float32  (from catalog ``Z`` when catalog is used)
```

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
ingested with `--source-id-col TARGETID` (or any other native column) work
without extra configuration.

To backfill an existing lake where spectra or cutouts were ingested without
catalog patching (no FITS re-ingestion needed):

```bash
# Register console scripts after pulling (once per env):
uv sync --extra desi --extra dev

dl-rebuild-catalog-indices --survey desi_dr1 --kind spectrum
dl-rebuild-catalog-indices --survey desi_dr1 --kind cutout   # if cutouts exist

``--norder`` defaults to ``hats_order`` in ``catalogs/<survey>/catalog_info.json``.
Pass ``--norder`` only to override that metadata value.

# Without reinstalling, use the module directly:
uv run python -m data_lake.ingest.update_catalog_indices --survey desi_dr1 --kind spectrum
```

For manual / Python-API use:

```python
from data_lake.ingest.update_catalog_indices import update_index_column
update_index_column(lake_root="/data/lake", survey_name="sdss_dr17",
                    source_id_to_index=index_map, kind="spectrum")
```

### Pack a tile for sharing

```bash
dl-pack-tile /data/lake /data/share --norder 5 --npix 1234 --survey des_dr2 --manifest
# Exclude spectra if not needed:
dl-pack-tile /data/lake /data/share --norder 5 --npix 1234 --survey des_dr2 --no-spectra
```

## Repository layout

```
data_lake/
  ingest/
    fits_to_parquet.py        FITS/VOTable → HATS-partitioned Parquet
    fits_to_zarr.py           FITS cutouts → Zarr v3 sharded stacks
    fits_to_spectra_zarr.py   FITS 1-D spectra → Zarr v3 sharded stacks
    desi_parallel_ingest.py   Multi-process batch ingest of many DESI coadd files
    update_catalog_indices.py Patch _cutout_index / _spectrum_index in Parquet tiles; dl-rebuild-catalog-indices backfill CLI
  io/
    catalog.py           DuckDB-backed Parquet accessor
    cutouts.py           Zarr cutout accessor (O(1) source_id lookup)
    spectra.py           Zarr spectrum accessor (O(1) source_id lookup)
    crossmatch.py        Build & query precomputed cross-match catalogs
  ml/
    dataset.py           PyTorch Dataset / IterableDataset (cutouts)
    spectrum_dataset.py  PyTorch Dataset / IterableDataset (spectra) + transforms
  share/
    pack_tile.py         Per-tile .tar packaging + MANIFEST.json
  export/
    to_fits.py           Zarr cutout → standards-compliant FITS export
    to_spectrum_fits.py  Zarr spectrum → 1-D FITS + BINTABLE export
    spectra_subset.py    Curated source-id subset → single flat Zarr group
notebooks/
  01_duckdb_catalog_query.ipynb … 07_cutout_ingest.ipynb   # see “Example notebooks” below
examples/
  cross_survey_lsst_desi_euclid/   # synthetic lake + DuckDB join (master + modalities)
```

## Data layout on disk

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
        meta/       (N,) structured bytes (z, z_err, snr, exptime, R, instr)
      spectrum_info.json
```

Each catalog row carries:
- `_source_id` — stable int64 join key for Zarr/cross-match (sequential 0…N−1, copy of native int ID, or hash of a label column)
- native survey ID columns (e.g. `TARGETID`, `SOURCE_ID`) when ``source_id_mode`` is ``column:…`` or ``label:…``
- `_healpix_norder5` — HEALPix tile pixel (partitioning key)
- `_cutout_index` — position inside the tile's Zarr cutout array (O(1) lookup)
- `_spectrum_index` — position inside the tile's Zarr spectrum array (O(1) lookup; -1 = not ingested)

## Lake inventory and master association tables

The library does **not** emit one automatic “master file” that lists every survey
and object in the lake. **Discovery** is by convention: each modality keeps its
own metadata (`catalog_info.json`, `cutout_info.json`, `spectrum_info.json`,
Parquet `_metadata`, Zarr `_source_id` arrays, and optional ingest checkpoints).
For **multi-survey science** you maintain a separate **association table**
(usually columnar Parquet or CSV) that records how identifiers line up and,
when needed, how to open the right Zarr row.

### What goes in a master / association file

Shape it for how you query (DuckDB, Polars, ADQL). Typical columns:

| Column | Purpose |
|--------|--------|
| Primary `source_id` | int64 key for your “home” survey catalog row (or hash of a string label; see [Object identifiers](#object-identifiers---source-id-col)) |
| Partner IDs | e.g. `desi_targetid`, `euclid_source_id` — whatever the other survey stores |
| `sep_arcsec` | Sky separation from the matcher (optional but good for QA) |
| `match_rank` / `find` flag | If the matcher can return multiple neighbours, disambiguate |
| `healpix_npix`, `norder` | Same HEALPix **nested** index and order used for lake tiles (must match ingest `hats_order` / `--norder`) |
| Zarr row indices | After ingest, optional `desi_spectrum_row`, `euclid_cutout_row` — tile-local indices aligned with `_spectrum_index` / `_cutout_index` |

Tile paths under the lake follow `healpix_dir(norder, npix)` from
`data_lake.ingest.fits_to_parquet` (e.g. `Norder=5/Dir=10000/Npix=12345`).
Either store `npix` + `norder` and build paths in SQL/Python, or **join** the
master back to an ingested catalog on `source_id` and read `_healpix_norder5`
from Parquet.

Put the file wherever you prefer: many teams use
`<lake_root>/shared/associations/<name>.parquet` (easy to back up, not confused
with a formal `catalogs/<survey>/` ingest), or a dedicated tree under
`catalogs/` if you want the same validation tooling as other catalogs.

### Building an inventory of “what is in the lake”

Use the same tools as production queries:

1. **List surveys** — directories under `catalogs/`, `cutouts/`, `spectra/`
   (each name is the survey identifier you passed to ingest).
2. **Row counts / columns** — `duckdb` / `polars` over
   `read_parquet('.../catalogs/<survey>/**/*.parquet')`, or read each survey’s
   `catalog_info.json` (`total_rows`, `total_columns`, `hats_order`, …).
3. **Notebook** — `notebooks/05_ingestion_report.ipynb` walks a deployment tree
   and summarises what exists.

There is no requirement to materialise a single wide table of the whole lake;
often a **small association Parquet** plus **on-demand joins** to native
catalog tiles is enough.

### Catalog–catalog association (positional)

Catalog↔catalog matching is **sky-based** (STILTS, ``build_crossmatch``, etc.).
Catalog↔spectrum linkage is a **separate workflow**: the catalog row must carry
the native spectrum key (or ``_spectrum_index`` after ingest), not a positional
match to Zarr tiles.

**Export columns for matching** — ``dl-extract-catalog`` projects any columns
from raw catalog files (FITS, VOTable, Parquet, CSV, …) or from an ingested
lake catalog:

```bash
# Raw survey catalog → Parquet for STILTS
dl-extract-catalog survey_a.fits -o a_sky.parquet \
  -c TARGETID -c RA -c DEC --valid-sky-only

# Rename columns for STILTS (NAME:alias)
dl-extract-catalog survey_b.fits -o b_sky.csv --format csv \
  -c ID:id -c ra:RA -c dec:DEC

# Already-ingested lake catalog (streams; does not load full survey into RAM)
dl-extract-catalog --lake-root /data/lake --survey DESI_DR1 \
  -o desi_sky.parquet -c _source_id -c ra -c dec --engine tiles

# Same lake export to CSV or FITS (CSV streams tile-by-tile; FITS uses a temp Parquet pass)
dl-extract-catalog --lake-root /data/lake --survey zCOSMOS_DR3 \
  -o zCOSMOS_sky.csv -c _source_id -c ra -c dec_
dl-extract-catalog --lake-root /data/lake --survey zCOSMOS_DR3 \
  -o zCOSMOS_sky.fits -c _source_id -c ra -c dec_

# 100M+ rows: tiled export (parallel STILTS / bounded memory)
dl-extract-catalog --lake-root /data/lake --survey GAIA_DR3 \
  --output-dir /scratch/gaia_sky/ -c source_id -c ra -c dec --progress

# Large FITS before ingest
dl-extract-catalog huge_cat.fits -o sky.parquet --streaming \
  -c TARGETID -c RA -c DEC --valid-sky-only

# Many files
dl-extract-catalog --file-list catalog_paths.txt -o all_a.parquet \
  -c serial -c RA -c DEC --add-input-path
```

**Very large surveys** — avoid materialising hundreds of millions of rows in
one process. Prefer **in-lake cross-match** (``dl-crossmatch``) which streams
tile-by-tile like ingest. Use ``dl-extract-catalog`` only when an external tool
(STILTS) needs a portable extract.

```bash
# Positional catalog↔catalog match at lake scale (survey A defines partition)
dl-crossmatch SURVEY_A SURVEY_B /data/lake \
  --radius-arcsec 1.0 --n-workers 8 --progress

# Per-survey sky columns / Norder (defaults: each catalog_info.json)
dl-crossmatch SURVEY_A SURVEY_B /data/lake \
  --ra-col RA --dec-col DEC --norder 5 \
  --ra-col-b RAJ2000 --dec-col-b DEJ2000 --norder-b 6

# Output: catalogs/crossmatch/SURVEY_A_x_SURVEY_B/
# Query: CrossmatchAccessor or DuckDB over that tree
```

Lake exports read **one HEALPix tile at a time** (or use DuckDB
``COPY`` via ``--engine duckdb`` for a single Parquet file). Prefer
``--output-dir`` when you need a portable extract for external tools.

### Associations with STILTS

[STILTS](https://www.starlink.ac.uk/stilts/) is useful when you need
**explicit match semantics** (all neighbours in a radius, symmetric / mutual
best matches, proper motions, etc.) beyond ``dl-crossmatch`` (nearest neighbour
within a radius, survey-A-centric partitioning). At hundreds of millions of
rows, prefer ``dl-crossmatch``; use STILTS on smaller extracts or per-tile
exports from ``dl-extract-catalog --output-dir``.

**Suggested workflow:**

1. **Materialise inputs** — Use ``dl-extract-catalog`` (above) or DuckDB
   ``COPY (SELECT …)`` to write FITS/VOTable/Parquet with the ID and sky columns
   you need for matching. Keep **native IDs** consistent with how each catalog
   was (or will be) ingested.
2. **Run STILTS** — e.g. `tskymatch2` / `tmatch2` with your chosen `find=`
   policy, error circles, and output columns for both tables.
3. **Write the master** — Convert STILTS output to **Parquet** (columnar,
   typed). Add `healpix_npix` / `norder` if missing (recompute with the same
   `assign_healpix` / `hats_order` as the lake so tile paths stay consistent).
4. **Use in analysis** — DuckDB / Polars joins: filter the primary catalog,
   join to the master on `source_id`, optionally join to partner catalogs or
   open Zarr using `npix` + row indices. A minimal end-to-end pattern lives in
   `examples/cross_survey_lsst_desi_euclid/` (synthetic tiles + `query.sql`).
   Step-by-step examples: **`notebooks/11_duckdb_catalog_query.ipynb` §9**.

**After STILTS:** if you ingest new spectra or cutouts for matched IDs, call
`update_index_column` so `_spectrum_index` / `_cutout_index` on the **native**
survey catalog stay in sync; the master table can carry partner IDs and
separations while the lake catalog keeps machine indices for fast accessors.

### Schema registry (column discovery)

Each ingested catalog gets **`catalogs/<survey>/schema_manifest.json`** at finalize
(ingest or batch end): every column name, Arrow dtype, heuristic **role**
(`id`, `sky`, `redshift`, `photometry`, `healpix`, `index`, …), join columns, and
prefix **column_groups** (e.g. `MAG` → `MAG_G`, `MAG_R`).

```bash
dl-describe-survey DESI_DR1
dl-describe-survey EUCLID_DR1 --role photometry
dl-describe-survey DESI_DR1 --modality spectra
dl-describe-survey ALLWISE --rebuild   # refresh manifest from on-disk data
dl-describe-survey DESI_DR1 --json     # full manifest for tooling
```

Use this to pick columns before joining a **master association** table (see below).
Spectra/cutout manifests are written at ingest finalize; catalogs also get manifests from Parquet schema.
The master file should stay ID-centric; science columns come from per-survey catalogs.

### Lake registry and master metadata (P1)

**Lake index** — scan what is deployed:

```bash
dl-refresh-lake-registry              # write shared/registry/surveys.parquet
dl-describe-lake                      # print survey × modality summary
dl-describe-lake --refresh            # rebuild then print
```

For **spectra**, ``total_rows`` in the registry is the sum of ``source_id`` lengths
across all ``Npix=*.zarr`` tiles (same count as the ingestion report notebook’s
``n_spectra_in_zarr``). Catalog rows use ``catalog_info.json`` ``total_rows``.

**Master association** — map master ID columns to catalog join keys. Keep the
master Parquet thin; store mapping in a sidecar ``<master>.meta.json``:

```json
{
  "meta_version": "1",
  "master_parquet": "associations/master_desi_euclid.parquet",
  "primary_survey": "DESI_DR1",
  "partners": [
    {"survey": "DESI_DR1", "master_column": "desi_targetid", "catalog_id_column": "TARGETID"},
    {"survey": "EUCLID_DR1", "master_column": "euclid_source_id", "catalog_id_column": "SOURCE_ID"}
  ]
}
```

```bash
dl-describe-master associations/master.parquet
dl-describe-master associations/master.parquet --write-meta   # save guessed .meta.json
```

If ``.meta.json`` is missing, columns are matched heuristically against on-disk
``schema_manifest.json`` files (run ``dl-describe-survey <name> --rebuild`` first
for surveys without a manifest).

### Spectra, cutouts, and column overlays (P2)

**Spectra and cutout layers** get ``schema_manifest.json`` at ingest finalize
(from ``spectrum_info.json`` / ``cutout_info.json``): Zarr array names, dtypes,
roles (`flux`, `ivar`, `mask`, `wavelength`, `image`, `metadata`, …).

```bash
dl-describe-survey DESI_DR1 --modality catalog    # default
dl-describe-survey DESI_DR1 --modality spectra
dl-describe-survey LSST_DR1 --modality cutout
dl-describe-survey ALLWISE --rebuild
```

**Optional overlays** — analyst JSON under ``shared/registry/overlays/``
(see ``shared/registry/overlays/README.md``). Merged at describe time with
``unit``, ``description``, and homogenization hints (e.g. WISE Vega → AB offset).

**Discovery workflow** (master → columns → SQL):

1. ``dl-refresh-lake-registry`` then ``dl-describe-lake``
2. ``dl-describe-master associations/master.parquet`` (``--write-meta`` once)
3. ``dl-describe-survey <partner>`` for each catalog; ``--modality spectra`` if needed
4. DuckDB ``want → master → catalog`` (``notebooks/11_duckdb_catalog_query.ipynb`` §9–§10)

### SQL builder from master (P3)

After column discovery, generate join SQL from ``<master>.meta.json`` and your column picks:

```python
from data_lake.query_from_master import build_select_from_master, parse_column_picks

plan = build_select_from_master(
    lake_root,
    lake_root / "associations" / "master_desi_euclid.parquet",
    primary_survey="DESI_DR1",
    columns={
        "DESI_DR1": ["Z", "MAG_G", "MAG_R"],
        "EUCLID_DR1": ["SOURCE_ID"],
    },
)
print(plan.all_sql())  # CREATE VIEW … + SELECT want → master → catalogs
```

CLI (same logic):

```bash
dl-build-query-from-master associations/master.parquet /data/lake \
  --primary-survey DESI_DR1 \
  --column DESI_DR1:Z,MAG_G,MAG_R \
  --column EUCLID_DR1:SOURCE_ID
```

Register ``want`` in DuckDB (or ``MultiCatalogAccessor._con.register("want", df)``), run
``plan.view_ddls`` then ``plan.sql``.

### Fast retrieval with DuckDB (ID list → master → catalogs)

Cross-matching is **by sky position**; partner catalogs may use **different**
`--norder` values. Retrieval is keyed on **native object IDs** in the master
and in each `catalogs/<survey>/` tree — not on matching HEALPix orders between
surveys.

**Master file** — flat Parquet, e.g. `<lake_root>/associations/master_desi_euclid.parquet`:

| Column | Purpose |
|--------|---------|
| `desi_targetid` (example) | Primary key for your science sample |
| `euclid_source_id`, … | Partner survey IDs from the matcher |
| `sep_arcsec` | Match separation (QA) |
| `desi_norder`, `desi_npix` | Optional; **that** survey’s tile path for Zarr / single-tile reads |

**Register catalogs** (same glob as `CatalogAccessor`):

```sql
CREATE VIEW desi AS
SELECT * FROM parquet_scan('catalogs/DESI_DR1/Norder=5/**/*.parquet', hive_partitioning=false);

CREATE VIEW euclid AS
SELECT * FROM parquet_scan('catalogs/EUCLID_DR1/Norder=6/**/*.parquet', hive_partitioning=false);

CREATE VIEW master AS
SELECT * FROM read_parquet('associations/master_desi_euclid.parquet');
```

**ID list** — prefer a small table over a huge literal `IN (...)`:

```sql
CREATE TEMP TABLE want (id BIGINT);
-- INSERT from read_csv('my_ids.csv') or register from Python (see notebook §9)
```

**Join** (only listed columns are read from Parquet):

```sql
SELECT
    w.id,
    m.euclid_source_id,
    m.sep_arcsec,
    d.Z,
    d.MAG_G,
    d.MAG_R,
    e.SOURCE_ID
FROM want AS w
INNER JOIN master AS m ON m.desi_targetid = w.id
INNER JOIN desi AS d ON d.TARGETID = w.id
INNER JOIN euclid AS e ON e.SOURCE_ID = m.euclid_source_id;
```

Replace `TARGETID` / `SOURCE_ID` with the real ID columns (`CatalogAccessor(...).source_id_column`).

**Python** — single survey, batched `IN` queries:

```python
from data_lake.io.catalog import CatalogAccessor

cat = CatalogAccessor(lake_root, "DESI_DR1")
df = cat.get_sources_by_id(
    target_ids,
    columns=["TARGETID", "Z", "MAG_G", "MAG_R", "MAG_Z"],
)
```

**Multi-survey** — `MultiCatalogAccessor` + `want` + `master` (full example in
`notebooks/11_duckdb_catalog_query.ipynb` §9).

**Optional: one Parquet tile** when the master stores `desi_npix` at DESI’s order:

```sql
SELECT d.TARGETID, d.Z
FROM want w
JOIN master m ON m.desi_targetid = w.id
JOIN read_parquet(
  'catalogs/DESI_DR1/Norder=' || m.desi_norder::VARCHAR
  || '/Dir=' || ((m.desi_npix // 10000) * 10000)::VARCHAR
  || '/Npix=' || m.desi_npix::VARCHAR || '.parquet'
) AS d ON d.TARGETID = w.id;
```

**Avoid** scanning full survey globs when you only need columns for a fixed ID
list — filter `want` first, join `master`, then partner catalogs on IDs.

## Schema versioning policy

Parquet handles column add/drop natively.  The following rules apply:

| Change type | Policy |
|---|---|
| Add new column to existing survey | Add in-place with nullable default; regenerate `_metadata` |
| Remove column | Write tombstone `null` column in new files; drop from `_metadata` |
| Rename column | Add new column + deprecate old (keep both for one release cycle) |
| Breaking structural change | Bump `schema_version` in `catalog_info.json`; write new directory `<survey>_v2/` |

The `schema_version` field in `catalog_info.json` is a plain integer starting at `"1"`.

## Example notebooks

All notebooks expect the **uv** project environment from [Quick-start](#quick-start)
(`.venv` + `uv sync --extra desi --extra dev`, kernel **Python (data-lake)**).
Do not use a bare system Python or an ad-hoc `pip install` outside `uv`.

See `notebooks/` for worked examples:

1. **`01_catalog_ingest.ipynb`** — FITS → HEALPix Parquet ingest, validation, and `CatalogAccessor` queries (self-contained temp lake or your paths)
2. **`02_spectrum_workflow.ipynb`** — Ingest spectra, query, transform, subset export (catalog `Z`), ML loop, FITS export + round-trip
3. **`03_cutout_ingest.ipynb`** — FITS stamps → Zarr cutout stacks, validation, `CutoutAccessor`, optional `_cutout_index` catalog patch
4. **`04_ingestion_report.ipynb`** — Summarise what is on disk under a deployment (`lake_config.toml`)
5. **`11_duckdb_catalog_query.ipynb`** — SQL over Parquet catalogs; §9 master table + ID-list joins

Use **`dl-describe-survey <name>`** (or the manifest JSON) to choose columns before building joins.
6. **`12_visualization.ipynb`** — Matplotlib / Napari cutout visualization + DS9 FITS export
7. **`13_pytorch_training_loop.ipynb`** — PyTorch DataLoader over Zarr cutouts

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Catalog format | Parquet v2 + Zstd | No column limit; columnar projection; column stats for pushdown |
| Catalog partitioning | HEALPix HATS (`--norder`, default 5) | Default ~3.4 deg² tiles; see [Choosing HEALPix order](#choosing-healpix-order---norder--hats_order) |
| Cutout format | Zarr v3, sharded | Avoids file-per-cutout; sequential shard reads for ML |
| Cutout dtype | float32 | Full science precision; halve to float16 only for ML-only mirrors |
| Spectra format | Zarr v3, sharded (flux/ivar/mask) | Symmetric with cutout layer; shared wavelength saves ~30% space |
| Wavelength mode | shared (default) / per-source | Shared = one array per tile; per-source when grids differ across spectra |
| Spectrum mask | uint8 (default) / uint16 | 8 bits covers SDSS/DESI defaults; bump if >8 flag bits needed |
| Sharing unit | Per-tile .tar (catalog + cutouts + spectra) | Matches partition granularity; already compressed inside |
| ML dataloader | CutoutDataset / SpectrumDataset (map) or Tile* (iterable) | Map-style for random sampling; tile-iterable for full-epoch streaming |

## `lake_config.toml` reference

Generated by `dl-init`; edit by hand later if you need to.

```toml
schema_version = "1"                # data-lake config schema; bumped on breaking changes

[lake]
name        = "my_lake"             # short identifier (no spaces)
description = "Free-text"
root        = "/abs/path/to/data"   # where catalogs/spectra/cutouts live
created_utc = "2026-05-13T11:43:00Z"

[partitioning]
hats_order       = 5                # default HEALPix order; see README “Choosing HEALPix order” (~3.36 deg²/tile at 5)
chunks_per_shard = 512              # rows per Zarr shard

[defaults]
wavelength_mode  = "shared"         # "shared" | "per_source"
mask_dtype       = "uint8"          # "uint8" | "uint16"
with_resolution  = false            # default for DESI spectra ingest

[paths]                             # relative to lake.root
catalogs = "catalogs"
spectra  = "spectra"
cutouts  = "cutouts"
shared   = "shared"

[ingest]
num_workers = "auto"                # "auto" (= cpu_count) | <int>
log_level   = "INFO"
```

Discovery order for the config (highest priority first):

1. CLI: `dl-ingest-* --config /path/to/lake_config.toml`
2. Environment: `$DATA_LAKE_CONFIG=/path/to/lake_config.toml`
3. Walk up from the current working directory looking for `lake_config.toml`

Programmatic access:

```python
from data_lake.config import LakeConfig
cfg = LakeConfig.discover()                # explicit > env > cwd walk-up
print(cfg.lake.name, cfg.spectra_root)
```

## Dependencies

Core: `pyarrow`, `zarr>=3`, `numcodecs`, `duckdb`, `astropy`, `healpy`, `numpy`, `polars`, `torch`, `tqdm`, `click`

Catalog queries (`CatalogAccessor.query` and related helpers) return **Polars** DataFrames by default (`fmt="polars"`). Use `fmt="arrow"` or `fmt="astropy"` when you need those types instead.

Optional extras (included in `dev`): `napari`, `matplotlib`, `jupyterlab`, `ipykernel` —
install via `uv sync --extra dev` (or `uv sync --extra desi --extra dev` for full ingest + notebooks).
