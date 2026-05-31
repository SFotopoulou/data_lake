"""
data_lake – astronomy data lake: wide catalogs (Parquet/HATS) + cutout stacks (Zarr v3).
"""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _version_from_pyproject() -> str | None:
    """Read version from the repo pyproject.toml (editable dev checkouts)."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.is_file():
        return None
    with open(pyproject, "rb") as fh:
        data = tomllib.load(fh)
    v = data.get("project", {}).get("version")
    return v if isinstance(v, str) else None


def _package_version() -> str:
    # Prefer pyproject.toml when present (editable install; egg-info can lag).
    from_pyproject = _version_from_pyproject()
    if from_pyproject is not None:
        return from_pyproject
    try:
        return version("data-lake")
    except PackageNotFoundError:
        return "unknown"


__version__ = _package_version()
