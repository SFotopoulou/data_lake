"""
SED assembly from a homogenized product catalog row.

The function dynamically discovers which ``phot_ab_*`` columns exist in the
product and intersects them with the BandpassRegistry to obtain effective
wavelengths and optional transmission curves.

Usage
-----
>>> from data_lake.io.catalog import CatalogAccessor
>>> from data_lake.homogenize.bandpass import BandpassRegistry
>>> from data_lake.plot.sed import assemble_sed
>>>
>>> product = CatalogAccessor("/shared/como", "EDFF_cone_joined")
>>> bp = BandpassRegistry(lake_root="/shared/como")
>>> sed = assemble_sed(product, source_id=123456789, bandpass_registry=bp)
>>> for pt in sed.points:
...     print(pt.band, pt.lambda_eff_um, pt.ab_mag)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from data_lake.homogenize.bandpass import BandpassRegistry
    from data_lake.io.catalog import CatalogAccessor

_PHOT_AB_RE = re.compile(r"^phot_ab_(\w+)$")


@dataclass
class SEDPoint:
    """Photometric data point for one band."""

    band: str
    lambda_eff_um: float
    ab_mag: float
    ab_mag_err: float | None
    survey_ref: str | None = None


@dataclass
class SEDPoints:
    """Collection of SED points for one source, sorted by wavelength."""

    source_id: int
    points: list[SEDPoint]

    @property
    def lambda_eff_um(self) -> np.ndarray:
        return np.array([p.lambda_eff_um for p in self.points])

    @property
    def ab_mag(self) -> np.ndarray:
        return np.array([p.ab_mag for p in self.points])

    @property
    def ab_mag_err(self) -> np.ndarray:
        return np.array(
            [p.ab_mag_err if p.ab_mag_err is not None else np.nan for p in self.points]
        )


def _discover_phot_ab_columns(columns: list[str]) -> dict[str, str | None]:
    """
    Return ``{band: err_column_or_None}`` for all ``phot_ab_*`` magnitude columns.

    Error columns (``phot_ab_*_err``) are not treated as independent bands;
    they are associated with their parent band column only.
    """
    bands: dict[str, str | None] = {}
    col_set = set(columns)
    for col in columns:
        m = _PHOT_AB_RE.match(col)
        if m and not col.endswith("_err"):
            err_col = col + "_err"
            bands[col] = err_col if err_col in col_set else None
    return bands


def assemble_sed(
    product_accessor: CatalogAccessor,
    source_id: int,
    bandpass_registry: BandpassRegistry,
) -> SEDPoints:
    """
    Build SED points for *source_id* from a homogenized product catalog.

    Parameters
    ----------
    product_accessor:
        ``CatalogAccessor`` opened on a product catalog that contains
        ``phot_ab_*`` columns (e.g. the output of ``dl-homogenize``).
    source_id:
        The lake ``_source_id`` integer that identifies the source.
    bandpass_registry:
        ``BandpassRegistry`` for resolving ``lambda_eff_um`` per band.

    Returns
    -------
    SEDPoints
        Sorted by effective wavelength.  An empty ``.points`` list means the
        source was found but no photometric bands with known wavelengths were
        available.

    Raises
    ------
    KeyError
        When *source_id* is not found in the product catalog.
    """
    all_cols = product_accessor.columns
    band_map = _discover_phot_ab_columns(all_cols)

    if not band_map:
        return SEDPoints(source_id=source_id, points=[])

    # Build the column list to fetch: all phot_ab_* + their error columns (deduplicated)
    fetch_cols: list[str] = []
    seen_fetch: set[str] = set()
    for band_col, err_col in band_map.items():
        if band_col not in seen_fetch:
            fetch_cols.append(band_col)
            seen_fetch.add(band_col)
        if err_col is not None and err_col not in seen_fetch:
            fetch_cols.append(err_col)
            seen_fetch.add(err_col)

    row_df = product_accessor.get_sources_by_id(
        [source_id], columns=fetch_cols, fmt="arrow"
    )

    if row_df.num_rows == 0:
        raise KeyError(
            f"source_id={source_id} not found in product "
            f"{product_accessor.survey_name!r}."
        )

    row = {col: row_df[col][0].as_py() for col in row_df.column_names}

    points: list[SEDPoint] = []
    for band_col, err_col in band_map.items():
        mag_val = row.get(band_col)
        if mag_val is None or (isinstance(mag_val, float) and np.isnan(mag_val)):
            continue

        lam = bandpass_registry.lambda_eff_um(band_col)
        if lam is None:
            continue

        err_val: float | None = None
        if err_col is not None:
            raw_err = row.get(err_col)
            if raw_err is not None and not (isinstance(raw_err, float) and np.isnan(raw_err)):
                err_val = float(raw_err)

        meta = bandpass_registry.band_meta(band_col) or {}
        points.append(
            SEDPoint(
                band=band_col,
                lambda_eff_um=lam,
                ab_mag=float(mag_val),
                ab_mag_err=err_val,
                survey_ref=meta.get("survey_ref"),
            )
        )

    points.sort(key=lambda p: p.lambda_eff_um)
    return SEDPoints(source_id=source_id, points=points)
