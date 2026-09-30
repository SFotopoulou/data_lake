# Quick-start: Day-1 journey

This guide takes you from a fresh clone to a working lake with your first catalog ingested and queried. Steps are sequential — each one builds on the last.

---

## Step 1 — Install

Use [`uv`](https://docs.astral.sh/uv/) for all Python environments.

```bash
# Install uv (one-time, per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # add to your shell rc

# Clone and create a Python 3.12 venv
git clone https://github.com/SFotopoulou/data_lake.git
cd data_lake
uv venv --python 3.12 .venv

# Install from the lockfile (editable, with extras):
#   dev      — pytest, Jupyter, matplotlib, napari, ...
#   fitsio   — fast FITS reading (recommended for large catalogs)
#   desi     — DESI coadd ingest (pulls in desispec); add when needed
uv sync --extra dev --extra fitsio

# Activate
source .venv/bin/activate
```

To add or bump dependencies, edit `pyproject.toml` and run `uv lock && uv sync`.

---

## Step 2 — Zero-data smoke test

Verify the installation without any external files:

```bash
# Unit tests (no external data required)
pytest

# Self-contained cross-survey demo (synthetic lake, runs in seconds)
python examples/cross_survey_lsst_desi_euclid/demo.py
```

Both should finish without errors. If they pass you are ready to create a real lake.

**Jupyter / notebooks:** register the project kernel once:

```bash
python -m ipykernel install --user --name=data-lake --display-name "Python (data-lake)"
```

Then open `notebooks/01_catalog_ingest.ipynb` for a guided walk-through of catalog ingest with synthetic data.

---

## Step 3 — Initialise your lake (`dl-init`)

A **deployment** is your private lake instance. `dl-init` scaffolds the directory layout and writes a `lake_config.toml`:

```bash
dl-init /data/lake \
    --description "My multi-survey lake" \
    --norder 5 \
    --ingest-token 'your-ingest-secret'
```

This creates:

```
/data/lake/
  lake_config.toml      # name, root, defaults — single source of truth
  .ingest_token_hash    # SHA-256 of the token (mode 0600, never commit)
  .gitignore
  catalogs/ spectra/ cutouts/ shared/
```

Point all subsequent `dl-*` commands at the deployment:

```bash
export DATA_LAKE_CONFIG=/data/lake/lake_config.toml
export LAKE_INGEST_TOKEN='your-ingest-secret'   # operators only
```

You can have multiple deployments (`prod`, `staging`, `personal`) and switch between them by exporting a different `DATA_LAKE_CONFIG`.

---

## Step 4 — First catalog ingest

**Required flags** for `dl-ingest-catalog`:

| Flag | Purpose |
|------|---------|
| `--survey` | Short name used for all lake paths (`catalogs/<SURVEY>/`) — case-sensitive |
| `--link-id-col` | Source integer ID column to use as `_source_id` (e.g. `TARGETID`, `SOURCE_ID`) |
| `--ra-col` | Right ascension column name (default `ra`) |
| `--dec-col` | Declination column name (default `dec`) |

```bash
dl-ingest-catalog survey.fits \
  --survey MY_SURVEY \
  --link-id-col TARGETID \
  --ra-col RA \
  --dec-col DEC
```

The `--link-id-col` value becomes the stable `_source_id` that links catalog rows to spectra and cutout tiles. Pick the column that is already used as a unique integer identifier in your survey (e.g. SDSS `SPECOBJID`, DESI `TARGETID`).

For large surveys or file lists, see [ingest/catalog.md](ingest/catalog.md) and [ingest/batch-and-checkpoints.md](ingest/batch-and-checkpoints.md).

---

## Step 5 — First query

After ingest, verify the lake and run a positional query:

```bash
# Inspect what is in the lake
dl-describe-lake --count-total

# Show columns for a survey
dl-describe-survey MY_SURVEY --modality catalog
```

Python one-liner (DuckDB-backed):

```python
from data_lake.io.catalog import CatalogAccessor

cat = CatalogAccessor("/data/lake", "MY_SURVEY")
df = cat.query("SELECT _source_id, ra, dec FROM catalog LIMIT 10")
print(df)
```

---

## Step 6 — Next steps

| You want to… | Go to… |
|-------------|--------|
| Ingest spectra (DESI, SDSS, generic FITS) | [ingest/spectra.md](ingest/spectra.md) |
| Ingest image cutouts | [ingest/cutouts.md](ingest/cutouts.md) |
| Add a second survey and crossmatch | [discovery/crossmatch.md](discovery/crossmatch.md) |
| Build a joined product catalog (≥2 surveys) | [discovery/workflow.md](discovery/workflow.md) · [notebook 14](../notebooks/14_discovery_workflow.ipynb) |
| Homogenize photometry to AB magnitudes | [homogenization.md](homogenization.md) |
| Export a source subset (spectra → Zarr/FITS) | [export-and-sharing.md](export-and-sharing.md) |
| Plot SED + spectrum for one source | [plotting.md](plotting.md) |
| All `dl-*` commands | [cli-reference.md](cli-reference.md) |
| Troubleshoot | [troubleshooting.md](troubleshooting.md) |

---

## Roles: analysts vs ingest operators

| Role | Install | `DATA_LAKE_CONFIG` | `LAKE_INGEST_TOKEN` | Data path |
|------|---------|-------------------|---------------------|-----------|
| **Analyst** | `uv sync --extra dev` (or `pip install data-lake`) | Shared deployment config | **Not used** | Read-only mount |
| **Ingest operator** | Same package | Same or writable deployment | **Required** for `dl-ingest-*` | Read-write |

Installing the package grants no ingest rights. Production lakes should rely on **filesystem permissions** (analysts cannot write Zarr tiles) as the primary control.

**Rotate the ingest token:**

```bash
dl-set-ingest-token --ingest-token 'new-secret'
```

---

## Dependencies

Core: `pyarrow`, `zarr>=3`, `numcodecs`, `duckdb`, `astropy`, `healpy`, `numpy`, `polars`, `torch`, `tqdm`, `click`.

Optional extras:

| Extra | When to add | How to install |
|-------|-------------|----------------|
| `dev` | Development, notebooks, matplotlib | `uv sync --extra dev` |
| `fitsio` | Fast FITS reading (large catalogs) | `uv sync --extra fitsio` |
| `desi` | DESI coadd ingest (adds `desispec`) | `uv sync --extra desi` |
| `mcp` | MCP server for AI agents | `uv sync --extra mcp` |
| `viz` | SED + spectrum plots (`dl-plot-source`) | `uv sync --extra viz` |

Catalog queries (`CatalogAccessor.query`) return **Polars** DataFrames by default (`fmt="polars"`). Use `fmt="arrow"` or `fmt="astropy"` when needed.
