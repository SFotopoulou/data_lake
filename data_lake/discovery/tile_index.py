"""Per-survey/per-modality HEALPix tile index.

Discovery must not walk the whole tree (``rglob`` over 12k-50k tiles per survey
per modality). This module persists a small JSON index per survey/modality at
``shared/registry/tile_index/<survey>.<modality>.json`` so that region overlap
becomes an in-memory set intersection.

The index records the modality's ``hats_order`` (needed to resolve a region to
this modality's pixel order) and the sorted populated ``npix`` list. When an
index is missing, callers may fall back to a one-off filesystem scan.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Iterator

from data_lake.schema_registry import (
    MODALITY_CATALOG,
    MODALITY_CUTOUT,
    MODALITY_SPECTRA,
)

log = logging.getLogger(__name__)

TILE_INDEX_DIRNAME = "tile_index"
_NPIX_RE = re.compile(r"Npix=(\d+)\.(?:parquet|zarr)$", re.IGNORECASE)


# modality -> (layer dir, info filename, tile suffix)
_MODALITY_CONFIG: dict[str, tuple[str, str, str]] = {
    MODALITY_CATALOG: ("catalogs", "catalog_info.json", "Npix=*.parquet"),
    MODALITY_SPECTRA: ("spectra", "spectrum_info.json", "Npix=*.zarr"),
    MODALITY_CUTOUT: ("cutouts", "cutout_info.json", "Npix=*.zarr"),
}


def modality_layer_dir(modality: str) -> str:
    try:
        return _MODALITY_CONFIG[modality][0]
    except KeyError:
        raise ValueError(f"unknown modality {modality!r}") from None


def survey_root_for(lake_root: Path | str, survey: str, modality: str) -> Path:
    return Path(lake_root) / modality_layer_dir(modality) / survey


def tile_index_dir(lake_root: Path | str) -> Path:
    return Path(lake_root) / "shared" / "registry" / TILE_INDEX_DIRNAME


def tile_index_path(lake_root: Path | str, survey: str, modality: str) -> Path:
    return tile_index_dir(lake_root) / f"{survey}.{modality}.json"


def _read_hats_order(survey_root: Path, modality: str) -> int | None:
    info_name = _MODALITY_CONFIG[modality][1]
    info_path = survey_root / info_name
    if not info_path.is_file():
        return None
    try:
        with open(info_path) as fh:
            return int(json.load(fh).get("hats_order"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _scan_npix(survey_root: Path, modality: str) -> list[int]:
    suffix = _MODALITY_CONFIG[modality][2]
    npix: set[int] = set()
    if not survey_root.is_dir():
        return []
    for path in survey_root.rglob(suffix):
        m = _NPIX_RE.search(path.name)
        if m:
            npix.add(int(m.group(1)))
    return sorted(npix)


def build_tile_index(lake_root: Path | str, survey: str, modality: str) -> dict:
    """Scan the filesystem once and return an index dict (does not write)."""
    survey_root = survey_root_for(lake_root, survey, modality)
    npix = _scan_npix(survey_root, modality)
    return {
        "survey": survey,
        "modality": modality,
        "hats_order": _read_hats_order(survey_root, modality),
        "n_tiles": len(npix),
        "npix": npix,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_tile_index(lake_root: Path | str, survey: str, modality: str) -> dict:
    """Build and persist the tile index; returns the index dict."""
    index = build_tile_index(lake_root, survey, modality)
    path = tile_index_path(lake_root, survey, modality)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(index, fh)
    return index


def load_tile_index(lake_root: Path | str, survey: str, modality: str) -> dict | None:
    path = tile_index_path(lake_root, survey, modality)
    if not path.is_file():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def survey_npix(
    lake_root: Path | str,
    survey: str,
    modality: str,
    *,
    allow_scan: bool = True,
) -> tuple[set[int], int | None]:
    """Return ``(npix_set, hats_order)`` for a survey/modality.

    Prefers the persisted tile index; falls back to a one-off scan when the
    index is absent and ``allow_scan`` is True.
    """
    index = load_tile_index(lake_root, survey, modality)
    if index is not None:
        return {int(p) for p in index.get("npix", [])}, index.get("hats_order")
    if not allow_scan:
        return set(), None
    survey_root = survey_root_for(lake_root, survey, modality)
    return set(_scan_npix(survey_root, modality)), _read_hats_order(survey_root, modality)


def iter_surveys_in_modality(lake_root: Path | str, modality: str) -> Iterator[str]:
    """Yield survey names present in a modality layer (directories with tiles/info)."""
    layer = Path(lake_root) / modality_layer_dir(modality)
    if not layer.is_dir():
        return
    info_name = _MODALITY_CONFIG[modality][1]
    for p in sorted(layer.iterdir()):
        if not p.is_dir() or p.name == "crossmatch":
            continue
        if (p / info_name).is_file() or any(p.glob("Norder=*")):
            yield p.name


def refresh_tile_indices(
    lake_root: Path | str,
    modalities: tuple[str, ...] = (MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT),
) -> dict[str, int]:
    """Rebuild tile indices for every survey in each modality.

    Returns a ``{modality: n_surveys_indexed}`` summary.
    """
    summary: dict[str, int] = {}
    for modality in modalities:
        count = 0
        for survey in iter_surveys_in_modality(lake_root, modality):
            write_tile_index(lake_root, survey, modality)
            count += 1
        summary[modality] = count
    return summary
