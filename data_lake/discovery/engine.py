"""Discovery engine: resolve a region to overlapping survey x modality tiles.

``resolve_region`` is registry/index-driven (no filesystem walk in the common
path): for each survey and modality it intersects the region (resolved to that
modality's ``hats_order``) with the cached populated-npix index.

Row counts:

- **rounded estimate (default)** — ``round(total_rows / n_tiles) * |overlap|``;
  zero tile reads.
- **exact (``count=True``)** — sum Parquet footer ``num_rows`` over the overlap
  tiles only (O(k) footer reads, never a row scan). Exact counts are currently
  catalog-only; spectra/cutout fall back to the estimate.
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow.parquet as pq

from data_lake.discovery import tile_index as ti
from data_lake.discovery.region import Region
from data_lake.ingest.fits_to_parquet import healpix_dir
from data_lake.schema_registry import (
    MODALITY_CATALOG,
    MODALITY_CUTOUT,
    MODALITY_SPECTRA,
)

log = logging.getLogger(__name__)

_MAX_COUNT_WORKERS = 64
_DEFAULT_MODALITIES = (MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT)


@dataclass
class DiscoveryRow:
    survey: str
    modality: str
    hats_order: int | None
    n_tiles_overlap: int
    est_rows: int
    exact_rows: int | None
    path: str


def round_count(n: float) -> str:
    """Human-friendly rounded count, e.g. ``~12k``, ``~750M`` (avoids implying exactness)."""
    n = float(n)
    if n < 1000:
        return f"~{int(round(n))}"
    for div, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            val = n / div
            # 2 significant figures for the rounded magnitude.
            if val >= 100:
                return f"~{int(round(val))}{suffix}"
            return f"~{val:.1f}{suffix}"
    return f"~{int(round(n))}"


def _read_total_rows(survey_root: Path, modality: str) -> int | None:
    info_name = {
        MODALITY_CATALOG: "catalog_info.json",
        MODALITY_SPECTRA: "spectrum_info.json",
        MODALITY_CUTOUT: "cutout_info.json",
    }[modality]
    info_path = survey_root / info_name
    if not info_path.is_file():
        return None
    try:
        with open(info_path) as fh:
            info = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    val = info.get("total_rows") or info.get("total_sources") or info.get("n_sources")
    return int(val) if val is not None else None


def _estimate_rows(total_rows: int | None, n_tiles_total: int, n_overlap: int) -> int:
    if not total_rows or n_tiles_total <= 0:
        return 0
    return int(round(total_rows / n_tiles_total * n_overlap))


def _exact_catalog_rows(
    survey_root: Path,
    hats_order: int,
    overlap_npix: Sequence[int],
    *,
    n_workers: int = 8,
) -> int:
    """Sum Parquet footer row counts over the overlap tiles (no row scan)."""
    def _rows(npix: int) -> int:
        tile = survey_root / healpix_dir(hats_order, npix) / f"Npix={npix}.parquet"
        if not tile.is_file():
            return 0
        try:
            return int(pq.read_metadata(str(tile)).num_rows)
        except Exception:
            return 0

    npix_list = list(overlap_npix)
    if not npix_list:
        return 0
    workers = max(1, min(n_workers, _MAX_COUNT_WORKERS, len(npix_list)))
    if workers == 1:
        return sum(_rows(p) for p in npix_list)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return int(sum(pool.map(_rows, npix_list)))


def resolve_region(
    lake_root: Path | str,
    region: Region,
    *,
    surveys: str | Iterable[str] = "all",
    modalities: Iterable[str] = (MODALITY_CATALOG,),
    count: bool = False,
    allow_scan: bool = True,
    n_workers: int = 8,
) -> list[DiscoveryRow]:
    """Return survey x modality rows overlapping *region*.

    Parameters
    ----------
    surveys:
        ``"all"`` (default) enumerates surveys per modality from the tile index /
        filesystem; otherwise an explicit iterable of survey names.
    modalities:
        Modalities to inspect (default: catalog only; opt-in spectra/cutout).
    count:
        When True, compute exact catalog counts via footer sums over overlap
        tiles. Spectra/cutout always use the rounded estimate.
    """
    lake_root = Path(lake_root)
    modalities = tuple(modalities)
    rows: list[DiscoveryRow] = []

    for modality in modalities:
        if surveys == "all":
            survey_names: list[str] = list(ti.iter_surveys_in_modality(lake_root, modality))
        else:
            survey_names = list(surveys)  # type: ignore[arg-type]

        for survey in survey_names:
            npix_set, hats_order = ti.survey_npix(
                lake_root, survey, modality, allow_scan=allow_scan
            )
            if hats_order is None or not npix_set:
                continue
            region_npix = region.to_npix(hats_order)
            overlap = sorted(region_npix & npix_set)
            if not overlap:
                continue

            survey_root = ti.survey_root_for(lake_root, survey, modality)
            total_rows = _read_total_rows(survey_root, modality)
            est = _estimate_rows(total_rows, len(npix_set), len(overlap))

            exact: int | None = None
            if count and modality == MODALITY_CATALOG:
                exact = _exact_catalog_rows(
                    survey_root, hats_order, overlap, n_workers=n_workers
                )

            rows.append(
                DiscoveryRow(
                    survey=survey,
                    modality=modality,
                    hats_order=hats_order,
                    n_tiles_overlap=len(overlap),
                    est_rows=est,
                    exact_rows=exact,
                    path=str(survey_root),
                )
            )
    return rows
