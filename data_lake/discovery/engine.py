"""Discovery engine: resolve a region to overlapping survey x modality tiles.

``resolve_region`` is registry/index-driven (no filesystem walk in the common
path): for each survey and modality it intersects the region (resolved to that
modality's ``hats_order``) with the cached populated-npix index.

Row counts:

- **rounded estimate (default)** — ``round(total_rows / n_tiles) * |overlap|``;
  zero tile reads when the modality info sidecar has a total. Spectra/cutout
  sidecars often omit totals (ingest never wrote them); then the survey-wide
  total is taken the same way as ``dl-describe-lake`` (sum of Zarr
  ``_source_id`` lengths).
- **exact (``count=True``)** — sum Parquet footer ``num_rows`` over overlap
  tiles for catalog/crossmatch, or sum Zarr ``_source_id`` lengths over
  overlap tiles for spectra/cutout (metadata only, never a full array read).
"""

from __future__ import annotations

import json
import logging
import re
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
    MODALITY_CROSSMATCH,
    MODALITY_CUTOUT,
    MODALITY_SPECTRA,
)

log = logging.getLogger(__name__)

_MAX_COUNT_WORKERS = 64
_DEFAULT_MODALITIES = (MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT)
_PARQUET_COUNT_MODALITIES = frozenset({MODALITY_CATALOG, MODALITY_CROSSMATCH})
_ZARR_COUNT_MODALITIES = frozenset({MODALITY_SPECTRA, MODALITY_CUTOUT})
_NPIX_PARQUET_RE = re.compile(r"Npix=(\d+)\.parquet$")


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


def _info_total_rows(info: dict, modality: str) -> int | None:
    """Pull a row/source total from a modality info sidecar, if present."""
    if modality == MODALITY_CROSSMATCH:
        val = info.get("total_rows") or info.get("n_match_rows")
    elif modality == MODALITY_SPECTRA:
        # Ingest historically wrote neither total_rows nor total_spectra;
        # registry fallback key is total_spectra.
        val = (
            info.get("total_rows")
            or info.get("total_spectra")
            or info.get("total_sources")
            or info.get("n_sources")
        )
    elif modality == MODALITY_CUTOUT:
        val = (
            info.get("total_rows")
            or info.get("total_cutouts")
            or info.get("total_sources")
            or info.get("n_sources")
        )
    else:
        val = info.get("total_rows") or info.get("total_sources") or info.get("n_sources")
    return int(val) if val is not None else None


def _read_total_rows(survey_root: Path, modality: str) -> int | None:
    info_name = {
        MODALITY_CATALOG: "catalog_info.json",
        MODALITY_SPECTRA: "spectrum_info.json",
        MODALITY_CUTOUT: "cutout_info.json",
        MODALITY_CROSSMATCH: "crossmatch_info.json",
    }[modality]
    info_path = survey_root / info_name
    info: dict | None = None
    if info_path.is_file():
        try:
            with open(info_path) as fh:
                info = json.load(fh)
        except (OSError, json.JSONDecodeError):
            info = None
    if info is not None:
        val = _info_total_rows(info, modality)
        if val is not None:
            return val
    # Spectra/cutout ingest often omits totals; match lake_registry and sum
    # Zarr ``_source_id`` lengths so estimates are not stuck at ~0.
    if modality in _ZARR_COUNT_MODALITIES:
        from data_lake.lake_registry import _count_zarr_sources

        return _count_zarr_sources(survey_root)
    return None


def _estimate_rows(total_rows: int | None, n_tiles_total: int, n_overlap: int) -> int:
    if not total_rows or n_tiles_total <= 0:
        return 0
    return int(round(total_rows / n_tiles_total * n_overlap))


def sum_parquet_rows_in_npix(
    survey_root: Path | str,
    hats_order: int,
    overlap_npix: Sequence[int],
    *,
    n_workers: int = 8,
) -> int:
    """Sum Parquet footer row counts over overlap tiles (no row scan).

    Shared by catalog and crossmatch (same ``healpix_dir`` + ``Npix=*.parquet``
    layout).
    """
    survey_root = Path(survey_root)

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


def sum_zarr_sources_in_npix(
    survey_root: Path | str,
    hats_order: int,
    overlap_npix: Sequence[int],
    *,
    n_workers: int = 8,
) -> int:
    """Sum Zarr ``_source_id`` lengths over overlap spectrum/cutout tiles.

    Uses the same per-tile metadata path as ``lake_registry._zarr_tile_n_sources``
    (shape from ``_source_id/zarr.json``, no array payload read).
    """
    from data_lake.lake_registry import _zarr_tile_n_sources

    survey_root = Path(survey_root)

    def _rows(npix: int) -> int:
        tile = survey_root / healpix_dir(hats_order, npix) / f"Npix={npix}.zarr"
        if not tile.is_dir():
            return 0
        n = _zarr_tile_n_sources(tile)
        return int(n) if n is not None else 0

    npix_list = list(overlap_npix)
    if not npix_list:
        return 0
    workers = max(1, min(n_workers, _MAX_COUNT_WORKERS, len(npix_list)))
    if workers == 1:
        return sum(_rows(p) for p in npix_list)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return int(sum(pool.map(_rows, npix_list)))


# Backward-compatible alias used by older call sites / tests.
_exact_catalog_rows = sum_parquet_rows_in_npix


def sum_crossmatch_rows_in_npix(
    catalog_root: Path | str,
    npix: set[int] | frozenset[int],
) -> tuple[int, int]:
    """Sum match rows / tiles whose ``Npix=`` is in *npix* (survey-A partition).

    Returns ``(total_rows, n_tiles)``. Prefer :func:`sum_parquet_rows_in_npix`
    when ``hats_order`` is known (direct path, no ``rglob``). This helper remains
    for callers that only have an npix set (e.g. lake registry area scoping).
    """
    catalog_root = Path(catalog_root)
    total = 0
    n_tiles = 0
    for path in catalog_root.rglob("Npix=*.parquet"):
        m = _NPIX_PARQUET_RE.search(path.name)
        if m is None:
            continue
        if int(m.group(1)) not in npix:
            continue
        total += int(pq.read_metadata(str(path)).num_rows)
        n_tiles += 1
    return total, n_tiles


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
        Modalities to inspect (default: catalog only; opt-in spectra/cutout/
        crossmatch).
    count:
        When True, compute exact counts over overlap tiles: Parquet footers for
        catalog/crossmatch, Zarr ``_source_id`` lengths for spectra/cutout.
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
            if count and modality in _PARQUET_COUNT_MODALITIES:
                exact = sum_parquet_rows_in_npix(
                    survey_root, hats_order, overlap, n_workers=n_workers
                )
            elif count and modality in _ZARR_COUNT_MODALITIES:
                exact = sum_zarr_sources_in_npix(
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
