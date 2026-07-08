# Contributing to `data_lake`

Thank you for considering a contribution. This project is small and
research-oriented; the guidelines below are deliberately lightweight.

**Documentation:** edit topic guides under `docs/` (not the root `README.md`,
which is a short navigation hub). Update `docs/cli-reference.md` when adding
or renaming `dl-*` entry points in `pyproject.toml`.

## Getting set up

The project is developed against Python 3.11 with [`uv`](https://docs.astral.sh/uv/)
as the recommended environment manager.

```bash
# One-time uv install
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Project venv
cd data_lake
uv venv --python 3.11 .venv
uv sync --extra desi --extra dev --extra fitsio
source .venv/bin/activate
```

Plain `pip install -e ".[desi,dev,fitsio]"` works too if you prefer.

## Before you open a PR

Run the unit tests and the end-to-end smoke test:

```bash
.venv/bin/python -m pytest tests/ -v
.venv/bin/python scripts/dry_run_desi_ingest.py
```

The dry-run script ingests three small DESI coadd files (with and without
`--with-resolution`) and verifies shape, finiteness, and resolution-matrix
row-sum invariants on a few sample sources. If you have changed anything in
`data_lake/ingest/` or `data_lake/io/`, please run it.

If you do not have local DESI coadd files, run the smoke script with `--help`
to see how to point it at your own files.

## Code style

- Python 3.10+ features (`X | None`, PEP 604 unions, walrus where helpful).
- Type hints on all public functions and dataclass fields.
- Docstrings use NumPy style on public functions.
- Inline comments are reserved for **why** a piece of code is the way it is.
  Avoid restating *what* the next line does.
- Imports grouped stdlib / third-party / first-party, sorted alphabetically
  inside each group.

The repo does not currently enforce a formatter, but `ruff` and `black` are
both fine choices if you want to format locally.

## Pull request expectations

- One topic per PR; prefer small, focused changes.
- Briefly describe the **why** in the PR body (the code already shows the
  **what**).
- Note any behavioural change to the on-disk layout, the Parquet schema, or
  the Zarr group attrs in the description. These are public contracts.
- New ingest readers (e.g., for a new survey) should follow the same shape
  used by `_read_sdss_boss`, `_read_desi_with_desispec`, `_read_generic_1d`
  in [`data_lake/ingest/fits_to_spectra_zarr.py`](data_lake/ingest/fits_to_spectra_zarr.py).
- If you add a new optional dependency, put it in a named extra under
  `[project.optional-dependencies]` in `pyproject.toml` rather than the core
  `dependencies` list.

## Adding a new survey reader

The pattern, in short:

1. Add a `_read_<survey>(hdul_or_path) -> tuple[list[SpectrumRecord], dict, ...]`
   function. Take a path if the reader manages I/O itself (as `desispec`
   does), otherwise an open `HDUList`.
2. Extend `_detect_format_from_path()` so the new survey can be auto-detected.
3. Add a dispatch branch in `ingest_spectra_from_fits()`.
4. Document the survey in the module docstring and the README.
5. Add a small test using `unittest.mock` or a synthetic FITS file in
   `tests/`.

## Reporting issues

When opening an issue, please include:

- The exact command you ran.
- The Python version and the versions of `data_lake`, `zarr`, `desispec`
  (if relevant), and `astropy`.
- A traceback if available.
- For data issues, a sample FITS file or a `fits-info` dump (`fitsinfo your.fits`).

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE) of this repository.
