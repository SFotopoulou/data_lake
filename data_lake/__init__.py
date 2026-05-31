"""
data_lake – astronomy data lake: wide catalogs (Parquet/HATS) + cutout stacks (Zarr v3).
"""

from importlib.metadata import PackageNotFoundError, version


def _package_version() -> str:
    try:
        return version("data-lake")
    except PackageNotFoundError:
        return "unknown"


__version__ = _package_version()
