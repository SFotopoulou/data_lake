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
    CATALOGS_LAYER,
    MODALITY_CATALOG,
    MODALITY_CROSSMATCH,
    MODALITY_CUTOUT,
    MODALITY_SPECTRA,
    PRODUCTS_LAYER,
    resolve_catalog_root,
)

log = logging.getLogger(__name__)

TILE_INDEX_DIRNAME = "tile_index"
_NPIX_RE = re.compile(r"Npix=(\d+)\.(?:parquet|zarr)$", re.IGNORECASE)


# modality -> (layer dir, info filename, tile suffix)
_MODALITY_CONFIG: dict[str, tuple[str, str, str]] = {
    MODALITY_CATALOG: ("catalogs", "catalog_info.json", "Npix=*.parquet"),
    MODALITY_SPECTRA: ("spectra", "spectrum_info.json", "Npix=*.zarr"),
    MODALITY_CUTOUT: ("cutouts", "cutout_info.json", "Npix=*.zarr"),
    MODALITY_CROSSMATCH: ("crossmatch", "crossmatch_info.json", "Npix=*.parquet"),
}


def modality_layer_dir(modality: str) -> str:
    try:
        return _MODALITY_CONFIG[modality][0]
    except KeyError:
        raise ValueError(f"unknown modality {modality!r}") from None


def survey_root_for(lake_root: Path | str, survey: str, modality: str) -> Path:
    if modality == MODALITY_CATALOG:
        return resolve_catalog_root(lake_root, survey)
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
    """Yield survey/product names present in a modality layer.

    For ``MODALITY_CATALOG`` both ``catalogs/`` and ``products/`` are scanned
    so that derived products are discovered alongside ingested surveys.

    For ``MODALITY_CROSSMATCH`` yields tree names under ``crossmatch/`` that have
    ``crossmatch_info.json`` or any ``Npix=*.parquet`` (names containing ``_x_``
    are valid and are not skipped).
    """
    root = Path(lake_root)
    info_name = _MODALITY_CONFIG[modality][1]

    def _iter_layer(layer: Path) -> Iterator[str]:
        if not layer.is_dir():
            return
        for p in sorted(layer.iterdir()):
            if not p.is_dir() or p.name == "crossmatch":
                continue
            if (p / info_name).is_file() or any(p.glob("Norder=*")):
                yield p.name

    def _iter_crossmatch_trees(layer: Path) -> Iterator[str]:
        if not layer.is_dir():
            return
        for p in sorted(layer.iterdir()):
            if not p.is_dir():
                continue
            # Do not skip names containing ``_x_`` — those are normal XM tree ids.
            if (
                (p / info_name).is_file()
                or any(p.glob("Norder=*"))
                or any(p.rglob("Npix=*.parquet"))
            ):
                yield p.name

    if modality == MODALITY_CATALOG:
        seen: set[str] = set()
        for name in _iter_layer(root / CATALOGS_LAYER):
            seen.add(name)
            yield name
        for name in _iter_layer(root / PRODUCTS_LAYER):
            if name not in seen:
                yield name
    elif modality == MODALITY_CROSSMATCH:
        yield from _iter_crossmatch_trees(root / modality_layer_dir(modality))
    else:
        yield from _iter_layer(root / modality_layer_dir(modality))


def refresh_tile_indices(
    lake_root: Path | str,
    modalities: tuple[str, ...] = (
        MODALITY_CATALOG,
        MODALITY_SPECTRA,
        MODALITY_CUTOUT,
        MODALITY_CROSSMATCH,
    ),
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
