"""Export sky regions and survey footprints as IVOA Multi-Order Coverage maps."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from data_lake.discovery.region import Region, rescale_npix_nested
from data_lake.discovery.tile_index import survey_npix
from data_lake.schema_registry import MODALITY_CATALOG

log = logging.getLogger(__name__)

MocFormat = Literal["fits", "json", "ascii"]


@dataclass(frozen=True)
class MocExportResult:
    output: Path
    format: MocFormat
    moc_order: int
    n_cells: int
    max_order: int


def _require_mocpy():
    try:
        from mocpy import MOC  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "MOC export requires the optional 'mocpy' dependency. "
            "Install with: uv sync --extra moc"
        ) from exc
    return MOC


def npix_set_to_moc(npix: Iterable[int], moc_order: int):
    """Build a compressed spatial MOC from NESTED HEALPix cells at *moc_order*."""
    if moc_order < 0:
        raise ValueError("moc_order must be >= 0")
    MOC = _require_mocpy()
    cells = sorted({int(p) for p in npix})
    if not cells:
        return MOC.from_string(f"{moc_order}/")
    ipix = np.array(cells, dtype=np.uint64)
    depth = np.uint8(moc_order)
    return MOC.from_healpix_cells(ipix, depth=depth, max_depth=moc_order)


def region_to_moc(region: Region, moc_order: int):
    """Resolve a :class:`Region` to a MOC at the requested HEALPix order."""
    npix = region.to_npix(moc_order)
    return npix_set_to_moc(npix, moc_order)


def survey_footprint_npix(
    lake_root: Path | str,
    survey: str,
    modality: str = MODALITY_CATALOG,
    *,
    region: Region | None = None,
    moc_order: int,
    allow_scan: bool = True,
) -> set[int]:
    """Populated survey tiles, optionally clipped to *region*, at *moc_order*."""
    npix_native, hats_order = survey_npix(
        lake_root, survey, modality, allow_scan=allow_scan,
    )
    if not npix_native:
        raise FileNotFoundError(
            f"No tiles found for {survey!r} modality {modality!r}"
        )
    if hats_order is None:
        raise ValueError(
            f"Cannot determine hats_order for {survey!r} {modality!r}; "
            "ingest metadata or refresh the tile index."
        )
    if region is not None:
        region_npix = region.to_npix(hats_order)
        npix_native &= region_npix
    if moc_order == hats_order:
        return npix_native
    return rescale_npix_nested(npix_native, hats_order, moc_order)


def write_moc(
    moc,
    output: Path | str,
    *,
    fmt: MocFormat = "fits",
    overwrite: bool = False,
    fits_keywords: dict[str, Any] | None = None,
) -> Path:
    """Serialize a MOC to disk (IVOA FITS by default)."""
    path = Path(output)
    if fmt not in ("fits", "json", "ascii"):
        raise ValueError(f"unsupported MOC format {fmt!r}")
    moc.save(str(path), format=fmt, overwrite=overwrite, fits_keywords=fits_keywords)
    return path


def export_region_moc(
    region: Region,
    output: Path | str,
    *,
    moc_order: int,
    fmt: MocFormat = "fits",
    overwrite: bool = False,
    provenance: dict[str, Any] | None = None,
) -> MocExportResult:
    """Export any region selector as a MOC file."""
    moc = region_to_moc(region, moc_order)
    keywords = _fits_keywords(provenance, moc_order=moc_order)
    out = write_moc(moc, output, fmt=fmt, overwrite=overwrite, fits_keywords=keywords)
    n_cells = len(region.to_npix(moc_order))
    return MocExportResult(
        output=out,
        format=fmt,
        moc_order=moc_order,
        n_cells=n_cells,
        max_order=int(moc.max_order),
    )


def export_survey_moc(
    lake_root: Path | str,
    survey: str,
    output: Path | str,
    *,
    modality: str = MODALITY_CATALOG,
    moc_order: int,
    region: Region | None = None,
    fmt: MocFormat = "fits",
    overwrite: bool = False,
) -> MocExportResult:
    """Export populated survey tiles (optionally region-clipped) as a MOC."""
    npix = survey_footprint_npix(
        lake_root, survey, modality, region=region, moc_order=moc_order,
    )
    moc = npix_set_to_moc(npix, moc_order)
    provenance = {
        "survey": survey,
        "modality": modality,
        "source": "data_lake_tile_index",
    }
    keywords = _fits_keywords(provenance, moc_order=moc_order)
    out = write_moc(moc, output, fmt=fmt, overwrite=overwrite, fits_keywords=keywords)
    return MocExportResult(
        output=out,
        format=fmt,
        moc_order=moc_order,
        n_cells=len(npix),
        max_order=int(moc.max_order),
    )


def _fits_keywords(
    provenance: dict[str, Any] | None,
    *,
    moc_order: int,
) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "ORIGIN": "data-lake",
        "MOCORD": moc_order,
    }
    if provenance:
        for key, value in provenance.items():
            if value is not None:
                kw[key.upper()[:8]] = str(value)[:68]
    return kw
