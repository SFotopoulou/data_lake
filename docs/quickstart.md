# Quick-start

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

**Key environment variables:**

| Variable | Who sets it | Purpose |
|----------|-------------|---------|
| `DATA_LAKE_CONFIG` | Everyone | Path to `lake_config.toml`; read by all `dl-*` commands |
| `LAKE_INGEST_TOKEN` | Operators | Plaintext ingest token; never commit to git |

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
dl-ingest-spectra-batch-desi-coadds --survey DESI_DR1 --file-list coadds.txt --n-workers 8
```

**Analysts** point at the shared config and use read APIs only — no token, no
`dl-init` on the production tree:

```bash
export DATA_LAKE_CONFIG=/path/to/mylake/lake_config.toml
# SpectrumAccessor, dl-extract-spectra-subset, notebooks, validators, …
```

Read-only tools never check the ingest token.



## Dependencies


Core: `pyarrow`, `zarr>=3`, `numcodecs`, `duckdb`, `astropy`, `healpy`, `numpy`, `polars`, `torch`, `tqdm`, `click`

Catalog queries (`CatalogAccessor.query` and related helpers) return **Polars** DataFrames by default (`fmt="polars"`). Use `fmt="arrow"` or `fmt="astropy"` when you need those types instead.

Optional extras (included in `dev`): `napari`, `matplotlib`, `jupyterlab`, `ipykernel` —
install via `uv sync --extra dev` (or `uv sync --extra desi --extra dev` for full ingest + notebooks).
