"""Helpers for building and updating area plan blocks (crossmatch, gather, homogenize)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_lake.discovery.areas import Area, load_area, save_area, validate_area


def parse_partner_spec(value: str) -> dict[str, Any]:
    """Parse a ``--partner`` value into a crossmatch_plan partner dict.

    Sky (positional)::

        SURVEY:RADIUS_ARCSEC
        e.g. DESI_DR1:1.0

    Column equality::

        SURVEY:col:COL_A:COL_B
        e.g. DESI_DR1:col:TARGETID:TARGETID

    Returns a dict suitable for ``crossmatch_plan.partners[]``.
    """
    raw = value.strip()
    if not raw:
        raise ValueError("empty partner spec")

    parts = raw.split(":")
    if len(parts) >= 4 and parts[1].lower() == "col":
        survey = parts[0].strip()
        match_col_a = parts[2].strip()
        match_col_b = ":".join(parts[3:]).strip()
        if not survey:
            raise ValueError(f"empty survey in column partner spec {value!r}")
        if not match_col_a or not match_col_b:
            raise ValueError(
                f"column partner must be SURVEY:col:COL_A:COL_B, got {value!r}"
            )
        from data_lake.io.crossmatch import validate_match_column

        validate_match_column(match_col_a, "--match-col-a")
        validate_match_column(match_col_b, "--match-col-b")
        return {
            "survey": survey,
            "match_mode": "column",
            "match_col_a": match_col_a,
            "match_col_b": match_col_b,
        }

    if ":" not in raw:
        raise ValueError(
            f"partner must be SURVEY:RADIUS_ARCSEC or "
            f"SURVEY:col:COL_A:COL_B, got {value!r}"
        )
    # Incomplete column form, e.g. SURVEY:col (missing columns)
    if len(parts) >= 2 and parts[1].lower() == "col":
        raise ValueError(
            f"column partner must be SURVEY:col:COL_A:COL_B, got {value!r}"
        )

    survey, radius_s = raw.rsplit(":", 1)
    survey = survey.strip()
    if not survey:
        raise ValueError(f"empty survey in partner spec {value!r}")
    try:
        radius = float(radius_s.strip())
    except ValueError as exc:
        raise ValueError(
            f"invalid radius in partner spec {value!r} "
            f"(expected SURVEY:RADIUS_ARCSEC or SURVEY:col:COL_A:COL_B)"
        ) from exc
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    return {
        "survey": survey,
        "match_mode": "sky",
        "radius_arcsec": radius,
    }


def partner_match_mode(partner: dict[str, Any]) -> str:
    """Return ``\"sky\"`` or ``\"column\"`` for a plan partner entry."""
    mode = partner.get("match_mode")
    if mode in ("sky", "column"):
        return mode
    if partner.get("match_col_a") or partner.get("match_col_b"):
        return "column"
    return "sky"


def build_crossmatch_plan(
    base_catalog: str,
    partners: list[dict[str, Any]],
    *,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    if not base_catalog.strip():
        raise ValueError("base_catalog is required")
    if not partners:
        raise ValueError(
            "at least one --partner SURVEY:RADIUS_ARCSEC or "
            "SURVEY:col:COL_A:COL_B is required"
        )
    normalised: list[dict[str, Any]] = []
    for p in partners:
        mode = partner_match_mode(p)
        survey = str(p.get("survey", "")).strip()
        if not survey:
            raise ValueError("partner missing 'survey'")
        if mode == "column":
            for key in ("match_col_a", "match_col_b"):
                if not p.get(key):
                    raise ValueError(f"column partner {survey!r} missing {key!r}")
            from data_lake.io.crossmatch import validate_match_column

            validate_match_column(str(p["match_col_a"]), "--match-col-a")
            validate_match_column(str(p["match_col_b"]), "--match-col-b")
            normalised.append({
                "survey": survey,
                "match_mode": "column",
                "match_col_a": str(p["match_col_a"]),
                "match_col_b": str(p["match_col_b"]),
            })
        else:
            radius = p.get("radius_arcsec")
            if radius is None:
                raise ValueError(f"sky partner {survey!r} missing 'radius_arcsec'")
            radius_f = float(radius)
            if radius_f <= 0:
                raise ValueError(f"radius must be positive, got {radius_f}")
            normalised.append({
                "survey": survey,
                "match_mode": "sky",
                "radius_arcsec": radius_f,
            })
    return {
        "base_catalog": base_catalog.strip(),
        "partners": normalised,
        "reuse_existing": reuse_existing,
    }


def build_gather_block(
    base: str,
    columns: dict[str, list[str]],
    materialize_as: str,
    *,
    multiplicity: str = "nearest",
    include_sep: bool = True,
    keep_all: bool = True,
    where_joined: str | None = None,
) -> dict[str, Any]:
    if not base.strip():
        raise ValueError("gather base is required")
    if not columns:
        raise ValueError("gather columns mapping is required")
    if not materialize_as.strip():
        raise ValueError("materialize_as is required")
    block: dict[str, Any] = {
        "base": base.strip(),
        "columns": columns,
        "multiplicity": multiplicity,
        "include_sep": include_sep,
        "keep_all": keep_all,
        "materialize_as": materialize_as.strip(),
    }
    if where_joined:
        block["where_joined"] = where_joined
    return block


def build_homogenize_block(
    *,
    survey: str | None = None,
    from_product: str | None = None,
    transform: str,
    materialize_as: str,
    area_id: str | None = None,
) -> dict[str, Any]:
    has_survey = bool(survey and survey.strip())
    has_product = bool(from_product and from_product.strip())
    if has_survey == has_product:
        raise ValueError("provide exactly one of survey or from_product")
    block: dict[str, Any] = {
        "transform": transform.strip(),
        "materialize_as": materialize_as.strip(),
    }
    if has_product:
        block["from_product"] = from_product.strip()  # type: ignore[union-attr]
        if area_id:
            block["region"] = {"from_area": area_id}
    else:
        block["survey"] = survey.strip()  # type: ignore[union-attr]
        if area_id:
            block["region"] = {"from_area": area_id}
    return block


def parse_columns_json(raw: str) -> dict[str, list[str]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid --columns JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("--columns must be a JSON object {survey: [col, ...]}")
    out: dict[str, list[str]] = {}
    for survey, cols in data.items():
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError(
                f"columns for {survey!r} must be a list of column name strings"
            )
        out[str(survey)] = list(cols)
    return out


def load_plan_fragment(path: Path | str) -> dict[str, Any]:
    """Load a full area file or a partial JSON with plan blocks only."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("area JSON must be an object")
    return data


def merge_plan_blocks(area: Area, fragment: dict[str, Any]) -> None:
    """Merge optional blocks from *fragment* into *area* (in place)."""
    for key in ("discover", "crossmatch_plan", "gather", "homogenize"):
        if key in fragment and fragment[key] is not None:
            area.data[key] = fragment[key]
    if fragment.get("area_id") and not area.data.get("area_id"):
        area.data["area_id"] = fragment["area_id"]
    if fragment.get("region") and not area.data.get("region"):
        area.data["region"] = fragment["region"]


def update_area(
    lake_root: Path | str,
    area_id: str,
    *,
    crossmatch_plan: dict[str, Any] | None = None,
    gather: dict[str, Any] | None = None,
    homogenize: dict[str, Any] | None = None,
    discover: dict[str, Any] | None = None,
) -> Path:
    """Load an area, replace plan blocks, validate, and save."""
    area = load_area(lake_root, area_id)
    if discover is not None:
        area.data["discover"] = discover
    if crossmatch_plan is not None:
        area.data["crossmatch_plan"] = crossmatch_plan
    if gather is not None:
        area.data["gather"] = gather
    if homogenize is not None:
        area.data["homogenize"] = homogenize

    fatal = [m for m in validate_area(area.data) if m.startswith("ERROR")]
    if fatal:
        raise ValueError("; ".join(fatal))

    return save_area(lake_root, area, overwrite=True)
