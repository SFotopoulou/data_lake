# Astronomy Data Lake

A local-first data lake for multi-survey astronomy catalogs, galaxy image cutouts, and 1-D spectra.

- **Wide catalogs** (>1 000 columns): stored as HATS-partitioned Parquet — no FITS column limit, column-projection so colleagues download only what they need.
- **Galaxy image cutouts**: stored as sharded Zarr v3 arrays — one tidy file per HEALPix tile, fast ML dataloading, lossless WCS round-trip to FITS.
- **1-D spectra** (SDSS/BOSS, DESI, generic): stored as sharded Zarr v3 stacks alongside cutouts — flux, IVAR, mask, shared or per-source wavelength, per-source scalar metadata.
- FITS is kept as the **ingest/export** format for observatory interoperability; it is not used as internal storage.

## Two layers: library vs. deployment

This repository is the **library**: a generic Python package (`data_lake`)
that you install once. Your actual data and configuration live in a
**deployment** directory — a thin instance that depends on the library:

```
github.com/SFotopoulou/data_lake       (this repo — the library, published)
        |
        | pip install -e ...
        v
~/projects/<your_lake_name>/           (the deployment, private)
    lake_config.toml                   (single source of truth)
    data/                              (actual Parquet / Zarr tiles)
    notebooks/  scripts/               (your own)
```

The library never knows the name of your lake or where its data lives;
all that is captured by one `lake_config.toml` in the deployment. The
`dl-init` CLI creates a deployment scaffold; the `dl-ingest-*` CLIs read
the config automatically (via `$DATA_LAKE_CONFIG` or `--config`).

This separation lets you publish a clean reusable library, while keeping
your private data, parameters, and notebooks isolated and free to
diverge.

## Quick-start

We recommend [`uv`](https://docs.astral.sh/uv/) for environment management - it is
fast, resolves the dependency graph correctly (including `desispec` and its
transitive deps), and keeps the env isolated from your system Python.

```bash
# 1. Install uv (one-time, per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # add to your shell rc

# 2. Create a Python 3.11 venv inside the project
cd data_lake
uv venv --python 3.11 .venv

# 3. Install the package editable, with whichever extras you need:
#    [desi] for DESI ingest (pulls in desispec)
#    [dev]  for tests + notebooks
uv pip install --python .venv/bin/python -e ".[desi,dev]"

# 4. Activate the env (or invoke the venv python directly)
source .venv/bin/activate
```

Smoke test before any large ingest:

```bash
.venv/bin/python -m pytest tests/test_desi_ingest.py -v       # unit tests
.venv/bin/python scripts/dry_run_desi_ingest.py               # end-to-end on 3 DESI files
```

The dry-run script ingests three small DESI coadd files in both
`--with-resolution` and plain modes, then reads them back to verify shape,
finiteness, and (for the resolution path) interior row sums ~1.0.

### Plain pip (alternative)

```bash
pip install -e ".[desi,dev]"
```

### Create a deployment

A deployment is *your* private instance of the data lake. Use `dl-init`
to scaffold one:

```bash
dl-init my_lake ~/projects \
    --description "Personal multi-survey lake" \
    --root /scratch/my_lake/data \
    --norder 5
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

### Ingest cutouts

```bash
dl-ingest-cutouts cutouts.fits --survey des_dr2 --ra-col RA --dec-col DEC
```

### Ingest spectra

**DESI** ingest requires the `desispec` optional extra (uses `read_spectra` +
`coadd_cameras` for correct IVAR-weighted B/R/Z camera combination):

```bash
pip install 'data-lake[desi]'

# With $DATA_LAKE_CONFIG set, OUTPUT_ROOT is taken from the config.
dl-ingest-spectra coadd-1-b0-0000p005-thru20210801.fits --survey desi_edr

# With resolution matrix (needed for redshift fitting / SPS / kinematic measurements)
# Storage cost: ~3× flux+ivar footprint (~170–200 GB per million coadded BRZ spectra)
dl-ingest-spectra coadd-1-b0-0000p005-thru20210801.fits \
    --survey desi_edr --with-resolution

# Without a config, pass OUTPUT_ROOT explicitly:
dl-ingest-spectra coadd-1-b0-0000p005-thru20210801.fits /data/lake --survey desi_edr

# SDSS/BOSS (no extra dependency needed)
dl-ingest-spectra spec-3586-55181-0001.fits --survey sdss_dr17
```

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

### Update catalog with spectrum index

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
    update_catalog_indices.py Patch _cutout_index / _spectrum_index in Parquet tiles
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
notebooks/
  01_duckdb_catalog_query.ipynb
  02_pytorch_training_loop.ipynb
  03_visualization.ipynb
  04_spectrum_workflow.ipynb
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

See `notebooks/` for worked examples:

1. **`01_duckdb_catalog_query.ipynb`** — SQL queries over multi-survey Parquet catalogs
2. **`02_pytorch_training_loop.ipynb`** — PyTorch DataLoader over Zarr cutouts
3. **`03_visualization.ipynb`** — Matplotlib / Napari cutout visualization + DS9 FITS export
4. **`04_spectrum_workflow.ipynb`** — Ingest spectra, query, transform, 1-D CNN training loop, FITS export + round-trip

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

Core: `pyarrow`, `zarr>=3`, `numcodecs`, `duckdb`, `astropy`, `healpy`, `numpy`, `pandas`, `polars`, `torch`, `tqdm`, `click`

Optional: `napari`, `matplotlib`, `jupyterlab` (install with `pip install -e ".[dev]"`)
