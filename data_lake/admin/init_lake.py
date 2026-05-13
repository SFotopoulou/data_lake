"""
`dl-init` – scaffold a new data lake *deployment*.

A deployment is a directory that holds the actual data tiles plus a
`lake_config.toml` describing the lake.  It is **not** a fork of the
library; it is a thin instance that depends on the published
``data_lake`` package.

Example
-------
    dl-init caspian ~/projects \
        --root ~/projects/caspian/data \
        --norder 5 \
        --description "Personal multi-survey lake"

Creates::

    ~/projects/caspian/
        lake_config.toml
        README.md
        .gitignore
        data/
            catalogs/  spectra/  cutouts/  shared/
        notebooks/
        scripts/
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import click

from ..config import (
    SCHEMA_VERSION,
    LakeConfig,
    _Defaults,
    _Ingest,
    _Lake,
    _Partitioning,
    _Paths,
)

# ---------------------------------------------------------------------------
# Templates (kept tiny on purpose – users will customise)
# ---------------------------------------------------------------------------

_README_TEMPLATE = """\
# {name}

{description}

This directory is a **data lake deployment**: a thin instance of the
[`data_lake`](https://github.com/SFotopoulou/data_lake) library.
The library code lives elsewhere; this folder holds the *config*,
*data*, and *user-owned* scripts/notebooks for this particular lake.

## Layout

```
{name}/
  lake_config.toml      # single source of truth for this deployment
  README.md             # this file
  .gitignore            # excludes data/, logs/
  data/                 # actual tiles (gitignored)
    catalogs/  spectra/  cutouts/  shared/
  notebooks/            # your notebooks
  scripts/              # your scripts (cron jobs, batch wrappers)
```

## Quick start

```bash
# 1. Activate the data_lake venv (or any env with `data_lake` installed)
source /path/to/data_lake/.venv/bin/activate

# 2. Point all dl- CLIs at this deployment
export DATA_LAKE_CONFIG="$(pwd)/lake_config.toml"

# 3. Run an ingest – output_root, norder, etc. are read from the config
dl-ingest-spectra my_fits_dir/*.fits --fmt desi
```

## Updating the library

```bash
cd /path/to/data_lake && git pull
```

Library changes are picked up automatically because it is installed in
editable (`-e`) mode.
"""

_GITIGNORE_TEMPLATE = """\
# Data lake deployment: do not commit data tiles or logs
data/
logs/
*.zarr/
*.parquet
*.fits
*.fits.gz
__pycache__/
.ipynb_checkpoints/
.venv/
"""


# ---------------------------------------------------------------------------
# Core API (importable so the CLI is a thin wrapper)
# ---------------------------------------------------------------------------

def init_lake(
    name: str,
    parent: Path,
    *,
    root: Path | None = None,
    description: str = "",
    norder: int = 5,
    chunks_per_shard: int = 512,
    wavelength_mode: str = "shared",
    mask_dtype: str = "uint8",
    with_resolution: bool = False,
    num_workers: int | str = "auto",
    log_level: str = "INFO",
    force: bool = False,
) -> Path:
    """Bootstrap a new deployment.

    Parameters
    ----------
    name:
        Short identifier for the lake (e.g. ``"caspian"``).  Becomes both
        the deployment directory name and the ``lake.name`` field.
    parent:
        Directory in which to create ``<parent>/<name>/``.
    root:
        Where the actual data tiles will live.  Defaults to
        ``<parent>/<name>/data``.  If relative, resolved against the
        deployment directory.
    description, norder, ...:
        Values written to ``lake_config.toml``.
    force:
        If ``True``, overwrite an existing deployment directory.

    Returns
    -------
    Path
        The absolute path of the new deployment directory.
    """
    parent = Path(parent).expanduser().resolve()
    if not parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent}")

    deployment = parent / name
    if deployment.exists() and not force:
        raise FileExistsError(
            f"Deployment already exists: {deployment} (use force=True to overwrite)"
        )
    deployment.mkdir(parents=True, exist_ok=force)

    # Resolve data root
    if root is None:
        data_root = deployment / "data"
    else:
        root = Path(root).expanduser()
        data_root = root if root.is_absolute() else (deployment / root).resolve()

    # Build sub-trees
    for sub in (
        data_root / "catalogs",
        data_root / "spectra",
        data_root / "cutouts",
        data_root / "shared",
        deployment / "notebooks",
        deployment / "scripts",
    ):
        sub.mkdir(parents=True, exist_ok=True)

    # Build the config object and write it
    cfg = LakeConfig(
        lake=_Lake(
            name=name,
            root=data_root,
            description=description,
            created_utc=_dt.datetime.now(tz=_dt.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        partitioning=_Partitioning(hats_order=norder, chunks_per_shard=chunks_per_shard),
        defaults=_Defaults(
            wavelength_mode=wavelength_mode,
            mask_dtype=mask_dtype,
            with_resolution=with_resolution,
        ),
        paths=_Paths(),
        ingest=_Ingest(num_workers=num_workers, log_level=log_level),
        schema_version=SCHEMA_VERSION,
    )
    cfg.write(deployment / "lake_config.toml")

    (deployment / "README.md").write_text(
        _README_TEMPLATE.format(
            name=name,
            description=description or "Astronomy data lake deployment.",
        )
    )
    (deployment / ".gitignore").write_text(_GITIGNORE_TEMPLATE)

    return deployment


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------

@click.command("dl-init")
@click.argument("name")
@click.argument(
    "parent",
    type=click.Path(file_okay=False, path_type=Path),
    default=".",
)
@click.option(
    "--root", "root",
    type=click.Path(path_type=Path),
    default=None,
    help="Absolute path where data tiles will live. Default: <parent>/<name>/data",
)
@click.option(
    "--description", "-d",
    default="",
    help="Free-text description of this lake.",
)
@click.option("--norder", default=5, show_default=True, type=int,
              help="Default HEALPix order for HATS partitioning.")
@click.option("--chunks-per-shard", default=512, show_default=True, type=int,
              help="Rows per Zarr shard.")
@click.option("--wavelength-mode",
              type=click.Choice(["shared", "per_source"]),
              default="shared", show_default=True,
              help="Default spectra wavelength storage.")
@click.option("--mask-dtype",
              type=click.Choice(["uint8", "uint16"]),
              default="uint8", show_default=True)
@click.option("--with-resolution", is_flag=True,
              help="Default to storing DESI resolution matrices.")
@click.option("--num-workers", default="auto", show_default=True,
              help="Default ingest parallelism (an int or 'auto').")
@click.option("--log-level", default="INFO", show_default=True)
@click.option("--force", is_flag=True,
              help="Overwrite an existing deployment directory.")
def dl_init(
    name: str,
    parent: Path,
    root: Path | None,
    description: str,
    norder: int,
    chunks_per_shard: int,
    wavelength_mode: str,
    mask_dtype: str,
    with_resolution: bool,
    num_workers: str,
    log_level: str,
    force: bool,
) -> None:
    """Bootstrap a new data lake deployment at PARENT/NAME."""
    # 'num_workers' is a string from the CLI; coerce to int when possible.
    nw: int | str
    if num_workers.lower() == "auto":
        nw = "auto"
    else:
        try:
            nw = int(num_workers)
        except ValueError as exc:
            raise click.BadParameter(
                f"--num-workers must be 'auto' or an integer, got {num_workers!r}"
            ) from exc

    try:
        deployment = init_lake(
            name=name,
            parent=parent,
            root=root,
            description=description,
            norder=norder,
            chunks_per_shard=chunks_per_shard,
            wavelength_mode=wavelength_mode,
            mask_dtype=mask_dtype,
            with_resolution=with_resolution,
            num_workers=nw,
            log_level=log_level,
            force=force,
        )
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Created deployment: {deployment}")
    click.echo(f"Config:             {deployment / 'lake_config.toml'}")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  export DATA_LAKE_CONFIG={deployment / 'lake_config.toml'}")
    click.echo("  dl-ingest-spectra <fits files>  # output_root etc. come from the config")
