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
uv sync --extra desi --extra dev

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
export DATA_LAKE_CONFIG=/shared/caspian/mylake/lake_config.toml
# SpectrumAccessor, dl-extract-spectra-subset, notebooks, validators, …
```

Read-only tools never check the ingest token.

### Ingest a survey catalog

With a deployment config in place (`$DATA_LAKE_CONFIG` set), the
`OUTPUT_ROOT` argument is optional — it is filled in from the config:

```bash
dl-ingest-catalog survey_catalog.fits --survey des_dr2 --ra-col RA --dec-col DEC
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
`--on-duplicate-id skip|error|last` (default `skip`). Use `--tile-mode overwrite`
to rebuild a tile from one file only. `--overwrite` is deprecated but still maps
to `--tile-mode overwrite`. After every ingest, `_metadata` and `catalog_info.json`
`total_rows` are refreshed from **all** tiles on disk.

### Ingest cutouts

Cutouts are stored as **Zarr v3** stacks (one group per HEALPix tile). Each ingested
FITS contributes one or more rows in the tile's ``source_id/``, ``images/``, and
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
| Object ID (join to catalog) | ``--source-id-col`` | e.g. ``TARGETID`` | **Required** for production; must equal catalog ID (int64). If omitted, tries ``SOURCE_ID``, ``OBJ_ID``, ``TARGETID``, … then HDU index (not suitable for catalog match). |
| Sky position (tile routing) | ``--ra-col`` / ``--dec-col`` | e.g. ``TARGET_RA``, ``TARGET_DEC`` | Degrees; fallbacks include ``RA_TARG``/``DEC_TARG``, ``CRVAL1``/``CRVAL2``. Should match catalog coordinates. |
| Astrometry (export) | — | Standard 2-D WCS | ``CTYPE*``, ``CRVAL*``, ``CRPIX*``, ``CD*_*`` (or CDELT/CROTA); stored in ``wcs/`` for FITS round-trip. |
| Image data | — | Primary or image HDU | 2-D ``(H,W)`` → one band; 3-D → set ``--band-axis``. Fixed ``(H,W)`` per survey tile after the first file. |

Optional: ``--band-names r,i,z``, ``--dtype float32``, ``--image-hdu N`` (select one
extension in multi-HDU files), ``--on-duplicate append|error|skip``.

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
dl-ingest-spectra spec-3586-55181-0001.fits --survey sdss_dr17
```

#### Parallel batch ingest (many coadd files)

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
| `dl-ingest-cutouts` | `--on-duplicate` | `append`, `error`, `skip` | Per `source_id` in each `Npix=*.zarr` |
| `dl-ingest-cutouts-from-list` | `--on-duplicate` | same | same |
| `dl-ingest-spectra` | `--on-duplicate` | same | same |
| `dl-ingest-spectra-from-list` | `--on-duplicate` | same | Sequential non-DESI / mixed FITS lists |
| `dl-ingest-spectra-batch` | `--on-duplicate` | same | DESI parallel batch (was missing before) |

For **resumable** file-list or batch re-runs, use **`--on-duplicate skip`** on cutout/spectrum
ingest (and **`--tile-mode skip`** on catalog). Default is **`append`**, which can add duplicate
Zarr rows if you re-ingest the same objects.

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
  --on-duplicate skip

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

# One FITS per spectrum (directory)
dl-extract-spectra-subset ... --format fits --fits-layout per-file \
    --output /scratch/qso_fits/

# One multi-row FITS catalog (BINTABLE: TARGETID, Z, FLUX, IVAR, MASK + WAVELENGTH HDU)
dl-extract-spectra-subset ... --format fits --fits-layout catalog \
    --output /scratch/qso_spectra.fits
```

`--format` choices: `zarr` (default), `parquet`, `fits`.  For FITS,
`--fits-layout` is `per-file` (default) or `catalog`.  All require
`wavelength_mode="shared"` in the source survey.

Output layout for **zarr**:

```
qso_subset.zarr/
  flux/        (N_written, N_pix) float32 sharded
  ivar/        (N_written, N_pix) float32 sharded
  mask/        (N_written, N_pix) uint8   sharded
  wavelength/  (N_pix,)           float64 shared grid
  source_id/   (N_written,)       int64
  redshift/    (N_written,)       float32  (from catalog ``Z`` when catalog is used)
```

Rows are written in HEALPix-tile-traversal order for fast contiguous
writes; the returned `id_to_row` mapping lets you reorder if needed.
Lookup uses the catalog's `_spectrum_index` column when available
(O(catalog SQL) batched), otherwise falls back to a vectorised tile
scan (one `source_id` array read per tile + `np.isin`).

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
        source_id/ (N,) int64
        wcs/      (N,) structured bytes
      cutout_info.json
  spectra/
    <survey>/
      Norder=5/Dir=0/Npix=0.zarr/   ← one Zarr group per tile
        flux/       (N, N_pix) float32, sharded
        ivar/       (N, N_pix) float32, sharded
        mask/       (N, N_pix) uint8,   sharded
        wavelength/ (N_pix,)   float64  (shared) or (N, N_pix) float32 (per-source)
        source_id/  (N,) int64
        meta/       (N,) structured bytes (z, z_err, snr, exptime, R, instr)
      spectrum_info.json
```

Each catalog row carries:
- `source_id` — stable integer ID
- `_healpix_norder5` — HEALPix tile pixel (partitioning key)
- `_cutout_index` — position inside the tile's Zarr cutout array (O(1) lookup)
- `_spectrum_index` — position inside the tile's Zarr spectrum array (O(1) lookup; -1 = not ingested)

## Lake inventory and master association tables

The library does **not** emit one automatic “master file” that lists every survey
and object in the lake. **Discovery** is by convention: each modality keeps its
own metadata (`catalog_info.json`, `cutout_info.json`, `spectrum_info.json`,
Parquet `_metadata`, Zarr `source_id` arrays, and optional ingest checkpoints).
For **multi-survey science** you maintain a separate **association table**
(usually columnar Parquet or CSV) that records how identifiers line up and,
when needed, how to open the right Zarr row.

### What goes in a master / association file

Shape it for how you query (DuckDB, Polars, ADQL). Typical columns:

| Column | Purpose |
|--------|--------|
| Primary `source_id` | Integer key for your “home” survey catalog row |
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

### Associations with STILTS

[STILTS](https://www.starlink.ac.uk/stilts/) is a strong choice when you need
**explicit match semantics** (all neighbours in a radius, symmetric / mutual
best matches, extra columns, proper motions, etc.) beyond the built-in
`build_crossmatch` helper (nearest neighbour within a radius, survey-A-centric
partitioning — see `data_lake/io/crossmatch.py`).

**Suggested workflow:**

1. **Materialise inputs** — Export the lake catalogs you need to FITS or
   VOTable (e.g. DuckDB `COPY (SELECT source_id, ra, dec, …) TO 'a.parquet'`
   then convert with Astropy / Polars, or write FITS directly). Keep
   **`source_id`** and sky columns consistent with the Parquet catalog.
2. **Run STILTS** — e.g. `tskymatch2` / `tmatch2` with your chosen `find=`
   policy, error circles, and output columns for both tables.
3. **Write the master** — Convert STILTS output to **Parquet** (columnar,
   typed). Add `healpix_npix` / `norder` if missing (recompute with the same
   `assign_healpix` / `hats_order` as the lake so tile paths stay consistent).
4. **Use in analysis** — DuckDB / Polars joins: filter the primary catalog,
   join to the master on `source_id`, optionally join to partner catalogs or
   open Zarr using `npix` + row indices. A minimal end-to-end pattern lives in
   `examples/cross_survey_lsst_desi_euclid/` (synthetic tiles + `query.sql`).

**After STILTS:** if you ingest new spectra or cutouts for matched IDs, call
`update_index_column` so `_spectrum_index` / `_cutout_index` on the **native**
survey catalog stay in sync; the master table can carry partner IDs and
separations while the lake catalog keeps machine indices for fast accessors.

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
5. **`11_duckdb_catalog_query.ipynb`** — SQL queries over multi-survey Parquet catalogs
6. **`12_visualization.ipynb`** — Matplotlib / Napari cutout visualization + DS9 FITS export
7. **`13_pytorch_training_loop.ipynb`** — PyTorch DataLoader over Zarr cutouts

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Catalog format | Parquet v2 + Zstd | No column limit; columnar projection; column stats for pushdown |
| Catalog partitioning | HEALPix Norder=5 HATS | ~3.7 deg² tiles; interop with Rubin/LSST tooling (lsdb) |
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
hats_order       = 5                # default HEALPix order (Norder=5 → ~3.7 deg² tiles)
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
