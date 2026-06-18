"""Flat ``areas/<area_id>.json`` definitions.

An *area* is a metadata-only, self-contained spatial selection plus optional
crossmatch and gather plans. It is **not** tile data: the file fully defines the
region and what to do with it, across all modalities.

Schema (all blocks except ``region`` are optional)::

    {
      "area_id": "Wide_Field_47",
      "region": { "type": "cone", "ra_deg": 150.1, "dec_deg": 2.2,
                  "radius_arcsec": 600 },
      "discover": { "surveys": "all",
                    "modalities": ["catalog", "spectra", "cutout"] },
      "crossmatch_plan": { "base_catalog": "EUCLID",
                           "partners": [ {"survey": "DESI_DR1",
                                          "radius_arcsec": 1.0,
                                          "modalities": ["catalog", "spectra"]} ],
                           "reuse_existing": true },
      "gather": { "base": "EUCLID",
                  "select": {"type": "region", "from_area": "Wide_Field_47"},
                  "columns": {"EUCLID": ["ra", "dec"], "DESI_DR1": ["z"]},
                  "include_sep": true,
                  "multiplicity": "nearest",
                  "materialize_as": "EUCLID_wide47_joined" }
    }

Areas are lake-level (they span surveys), so they live under ``areas/`` and are
surfaced in their own block by ``dl-describe-lake`` rather than as fake
survey rows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from data_lake.discovery.region import Region
from data_lake.schema_registry import (
    MODALITY_CATALOG,
    MODALITY_CUTOUT,
    MODALITY_SPECTRA,
)

log = logging.getLogger(__name__)

AREA_SCHEMA_VERSION = "1"
_VALID_MODALITIES = {MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT}
_VALID_MULTIPLICITY = {"nearest", "all"}


def areas_dir(lake_root: Path | str) -> Path:
    return Path(lake_root) / "areas"


def area_path(lake_root: Path | str, area_id: str) -> Path:
    return areas_dir(lake_root) / f"{normalize_area_id(area_id)}.json"


def normalize_area_id(area_id: str) -> str:
    """Strip optional ``.area`` / ``.json`` suffixes from a CLI area id."""
    aid = area_id.strip()
    for suffix in (".json", ".area"):
        if aid.endswith(suffix):
            return aid[: -len(suffix)]
    return aid


def resolve_area_path(lake_root: Path | str, area_id: str) -> Path | None:
    """Return the first existing area JSON path for *area_id*, or ``None``."""
    raw = area_id.strip()
    normalized = normalize_area_id(raw)
    d = areas_dir(lake_root)
    seen: set[Path] = set()
    for stem in (normalized, raw):
        candidate = d / f"{stem}.json"
        if candidate not in seen:
            seen.add(candidate)
            if candidate.is_file():
                return candidate
    return None


@dataclass
class Area:
    """Parsed view over an ``areas/<area_id>.json`` definition."""

    area_id: str
    data: dict[str, Any]
    path: Path | None = None

    @property
    def region(self) -> Region:
        return Region.from_dict(self.data["region"])

    @property
    def discover_surveys(self) -> str | list[str]:
        return self.data.get("discover", {}).get("surveys", "all")

    @property
    def discover_modalities(self) -> list[str]:
        return list(
            self.data.get("discover", {}).get(
                "modalities", [MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT]
            )
        )

    @property
    def crossmatch_plan(self) -> dict[str, Any] | None:
        return self.data.get("crossmatch_plan")

    @property
    def gather(self) -> dict[str, Any] | None:
        return self.data.get("gather")

    @property
    def homogenize(self) -> dict[str, Any] | None:
        return self.data.get("homogenize")

    def to_dict(self) -> dict[str, Any]:
        return self.data


def make_area(area_id: str, region: Region, **blocks: Any) -> Area:
    """Build an :class:`Area` from a region and optional blocks."""
    data: dict[str, Any] = {
        "area_id": area_id,
        "schema_version": AREA_SCHEMA_VERSION,
        "region": region.to_dict(),
    }
    for key, value in blocks.items():
        if value is not None:
            data[key] = value
    return Area(area_id=area_id, data=data)


def load_area(lake_root: Path | str, area_id: str) -> Area:
    normalized = normalize_area_id(area_id)
    path = resolve_area_path(lake_root, area_id)
    if path is None:
        raise FileNotFoundError(
            f"area not found: {area_path(lake_root, normalized)} "
            f"(also tried {areas_dir(lake_root) / f'{area_id.strip()}.json'})"
        )
    with open(path) as fh:
        data = json.load(fh)
    return Area(area_id=data.get("area_id", normalized), data=data, path=path)


def resolve_region_ref(
    lake_root: Path | str,
    region_spec: dict[str, Any] | None,
    *,
    fallback: Region,
) -> Region:
    """Resolve an area ``region`` block or ``{"from_area": id}`` reference."""
    if region_spec is None:
        return fallback
    ref = region_spec.get("from_area")
    if ref is not None:
        return load_area(lake_root, str(ref)).region
    return Region.from_dict(region_spec)


def save_area(lake_root: Path | str, area: Area, *, overwrite: bool = False) -> Path:
    errors = validate_area(area.data)
    fatal = [e for e in errors if e.startswith("ERROR")]
    if fatal:
        raise ValueError("invalid area: " + "; ".join(fatal))
    path = area_path(lake_root, area.area_id)
    if path.exists() and not overwrite:
        raise FileExistsError(f"area already exists: {path} (use overwrite=True)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(area.data, fh, indent=2)
    return path


def list_areas(lake_root: Path | str) -> list[str]:
    d = areas_dir(lake_root)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def iter_areas(lake_root: Path | str) -> Iterator[Area]:
    for area_id in list_areas(lake_root):
        try:
            yield load_area(lake_root, area_id)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            log.warning("Skipping unreadable area %s: %s", area_id, exc)


def validate_area(data: dict[str, Any]) -> list[str]:
    """Validate an area definition.

    Returns a list of messages prefixed ``ERROR``/``WARN``. ``ERROR`` entries
    mean the area is unusable; ``WARN`` entries are advisory (e.g. unknown
    modality names).
    """
    msgs: list[str] = []
    if not data.get("area_id"):
        msgs.append("ERROR: missing 'area_id'")
    region = data.get("region")
    if not isinstance(region, dict):
        msgs.append("ERROR: missing or invalid 'region'")
    else:
        try:
            Region.from_dict(region)
        except (ValueError, KeyError, TypeError) as exc:
            msgs.append(f"ERROR: invalid region: {exc}")

    discover = data.get("discover")
    if discover is not None:
        for m in discover.get("modalities", []):
            if m not in _VALID_MODALITIES:
                msgs.append(f"WARN: unknown discover modality {m!r}")

    plan = data.get("crossmatch_plan")
    if plan is not None:
        if not plan.get("base_catalog"):
            msgs.append("ERROR: crossmatch_plan missing 'base_catalog'")
        for p in plan.get("partners", []):
            if not p.get("survey"):
                msgs.append("ERROR: crossmatch_plan partner missing 'survey'")
            if p.get("radius_arcsec") is None:
                msgs.append(
                    f"ERROR: crossmatch_plan partner {p.get('survey')!r} missing 'radius_arcsec'"
                )

    gather = data.get("gather")
    if gather is not None:
        if not gather.get("base"):
            msgs.append("ERROR: gather missing 'base'")
        mult = gather.get("multiplicity", "nearest")
        if mult not in _VALID_MULTIPLICITY:
            msgs.append(f"ERROR: gather multiplicity must be one of {_VALID_MULTIPLICITY}")
        cols = gather.get("columns")
        if cols is not None and not isinstance(cols, dict):
            msgs.append("ERROR: gather 'columns' must be a mapping survey -> [columns]")

    hom = data.get("homogenize")
    if hom is not None:
        has_survey = bool(hom.get("survey"))
        has_product = bool(hom.get("from_product"))
        if has_survey and has_product:
            msgs.append("ERROR: homogenize: use survey+region OR from_product, not both")
        elif not has_survey and not has_product:
            msgs.append("ERROR: homogenize requires 'survey' or 'from_product'")
        if not hom.get("transform"):
            msgs.append("ERROR: homogenize missing 'transform'")
        if not hom.get("materialize_as"):
            msgs.append("ERROR: homogenize missing 'materialize_as'")
        if has_survey and not hom.get("region") and not has_product:
            msgs.append(
                "WARN: homogenize with survey but no inline region; "
                "use CLI selectors or region.from_area"
            )
    return msgs


def area_is_valid(data: dict[str, Any]) -> bool:
    return not any(m.startswith("ERROR") for m in validate_area(data))
