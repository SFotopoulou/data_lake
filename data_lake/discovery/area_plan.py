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


# ---------------------------------------------------------------------------
# Suggest draft plans (match_mode heuristics; for hand-edit, not auto-run)
# ---------------------------------------------------------------------------

# Prefer these when names match exactly on both surveys (order = priority).
_ID_LIKE_COLUMNS: tuple[str, ...] = (
    "TARGETID",
    "TARGET_ID",
    "SOURCE_ID",
    "source_id",
    "OBJECT_ID",
    "OBJID",
    "OBJ_ID",
    "SPECOBJID",
    "specObjID",
    "designation",
    "DESIGNATION",
    "LEGACY_TARGETID",
    "LSID",
)

_LAKE_BOOKKEEPING_PREFIX = "_"
_DEFAULT_SUGGEST_SKY_RADIUS_ARCSEC = 1.0


def strip_crossmatch_plan_annotations(plan: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *plan* without ``_suggestions`` / other ``_`` draft keys.

    Executable partners are passed through :func:`build_crossmatch_plan` so
    unknown partner fields are dropped.
    """
    base = str(plan.get("base_catalog") or "").strip()
    partners_in = plan.get("partners") or []
    if not isinstance(partners_in, list):
        raise ValueError("crossmatch_plan 'partners' must be a list")
    clean_partners: list[dict[str, Any]] = []
    for p in partners_in:
        if not isinstance(p, dict):
            raise ValueError("crossmatch_plan partner entries must be objects")
        clean_partners.append({k: v for k, v in p.items() if not str(k).startswith("_")})
    reuse = plan.get("reuse_existing", True)
    return build_crossmatch_plan(base, clean_partners, reuse_existing=bool(reuse))


def _column_names_from_manifest(manifest: dict[str, Any]) -> list[str]:
    cols = manifest.get("columns") or []
    names: list[str] = []
    for c in cols:
        if isinstance(c, dict) and c.get("name"):
            names.append(str(c["name"]))
        elif isinstance(c, str):
            names.append(c)
    return names


def _native_id_column(manifest: dict[str, Any], columns: list[str]) -> str | None:
    """Best-effort science ID column for a survey (never ``_source_id``)."""
    link_mode = str(manifest.get("link_id_mode") or "")
    if link_mode.startswith("column:") or link_mode.startswith("label:"):
        native = link_mode.split(":", 1)[1].strip()
        if native and native in columns and not native.startswith(_LAKE_BOOKKEEPING_PREFIX):
            return native
    for name in _ID_LIKE_COLUMNS:
        if name in columns:
            return name
    return None


def _casefold_map(columns: list[str]) -> dict[str, str]:
    """Map casefold(name) → first original name."""
    out: dict[str, str] = {}
    for name in columns:
        key = name.casefold()
        if key not in out:
            out[key] = name
    return out


def suggest_partner_match(
    base_columns: list[str],
    partner_columns: list[str],
    *,
    partner_survey: str,
    base_manifest: dict[str, Any] | None = None,
    partner_manifest: dict[str, Any] | None = None,
    default_sky_radius_arcsec: float = _DEFAULT_SUGGEST_SKY_RADIUS_ARCSEC,
) -> tuple[dict[str, Any], str]:
    """Suggest one partner entry and a human reason string.

    Returns ``(partner_dict, reason)`` where *partner_dict* is executable
    (no ``_`` keys).
    """
    base_set = {c for c in base_columns if not c.startswith(_LAKE_BOOKKEEPING_PREFIX)}
    partner_set = {
        c for c in partner_columns if not c.startswith(_LAKE_BOOKKEEPING_PREFIX)
    }
    base_cf = _casefold_map(sorted(base_set))
    partner_cf = _casefold_map(sorted(partner_set))

    # 1) Shared ID-like column names (exact, then case-insensitive).
    for name in _ID_LIKE_COLUMNS:
        if name in base_set and name in partner_set:
            return (
                {
                    "survey": partner_survey,
                    "match_mode": "column",
                    "match_col_a": name,
                    "match_col_b": name,
                },
                f"column: shared ID-like column {name!r}",
            )
    for name in _ID_LIKE_COLUMNS:
        key = name.casefold()
        if key in base_cf and key in partner_cf:
            ca, cb = base_cf[key], partner_cf[key]
            return (
                {
                    "survey": partner_survey,
                    "match_mode": "column",
                    "match_col_a": ca,
                    "match_col_b": cb,
                },
                f"column: shared ID-like column {ca!r}/{cb!r} (case-insensitive)",
            )

    partner_native = None
    if partner_manifest is not None:
        partner_native = _native_id_column(partner_manifest, list(partner_set))
    if partner_native is None:
        partner_native = _native_id_column({}, list(partner_set))
    # Euclid-style GAIA_ID stores the numeric Gaia source_id, not designation.
    if "gaia" in partner_survey.casefold() and "source_id" in partner_set:
        partner_native = "source_id"

    # 2) Foreign-key style on base → partner native id (e.g. GAIA_ID → source_id).
    survey_token = partner_survey.split("_")[0].casefold()
    fk_candidates: list[str] = []
    if survey_token:
        fk_candidates.extend(
            [
                f"{survey_token.upper()}_ID",
                f"{survey_token.upper()}ID",
                f"{survey_token}_id",
                f"{survey_token}id",
            ]
        )
    # Common Euclid→Gaia spelling
    if "gaia" in partner_survey.casefold():
        fk_candidates.extend(["GAIA_ID", "Gaia_ID", "gaia_id", "GAIA_SOURCE_ID"])

    for fk in fk_candidates:
        if fk in base_set and partner_native:
            return (
                {
                    "survey": partner_survey,
                    "match_mode": "column",
                    "match_col_a": fk,
                    "match_col_b": partner_native,
                },
                f"column: base.{fk} → partner.{partner_native}",
            )
        fk_cf = fk.casefold()
        if fk_cf in base_cf and partner_native:
            return (
                {
                    "survey": partner_survey,
                    "match_mode": "column",
                    "match_col_a": base_cf[fk_cf],
                    "match_col_b": partner_native,
                },
                f"column: base.{base_cf[fk_cf]} → partner.{partner_native}",
            )

    # 3) Base column ending _ID whose stem matches partner survey token.
    if partner_native and survey_token:
        for col in sorted(base_set):
            if not col.upper().endswith("_ID") and not col.upper().endswith("ID"):
                continue
            stem = col
            if stem.upper().endswith("_ID"):
                stem = stem[:-3]
            elif stem.upper().endswith("ID"):
                stem = stem[:-2]
            if stem.casefold() == survey_token or survey_token in stem.casefold():
                return (
                    {
                        "survey": partner_survey,
                        "match_mode": "column",
                        "match_col_a": col,
                        "match_col_b": partner_native,
                    },
                    f"column: base.{col} → partner.{partner_native} (name heuristic)",
                )

    _ = base_manifest  # reserved for future native-id hints
    radius = float(default_sky_radius_arcsec)
    return (
        {
            "survey": partner_survey,
            "match_mode": "sky",
            "radius_arcsec": radius,
        },
        f"sky: no shared ID column found; default radius {radius} arcsec — edit as needed",
    )


def suggest_crossmatch_plan(
    lake_root: Path | str,
    base_catalog: str,
    partner_surveys: list[str],
    *,
    reuse_existing: bool = True,
    default_sky_radius_arcsec: float = _DEFAULT_SUGGEST_SKY_RADIUS_ARCSEC,
) -> dict[str, Any]:
    """Build a draft ``crossmatch_plan`` from catalog schema manifests.

    Adds a top-level ``_suggestions`` map (partner → reason) for hand editing.
    Strip annotations with :func:`strip_crossmatch_plan_annotations` before run.
    """
    from data_lake.schema_registry import load_schema_manifest, resolve_catalog_root

    lake_root = Path(lake_root)
    base = base_catalog.strip()
    if not base:
        raise ValueError("base_catalog is required")
    partners = [p.strip() for p in partner_surveys if p and p.strip()]
    if not partners:
        raise ValueError("at least one partner survey is required")
    if base in partners:
        raise ValueError(f"base catalog {base!r} cannot also be a partner")

    base_root = resolve_catalog_root(lake_root, base)
    if not base_root.is_dir():
        raise FileNotFoundError(f"base catalog not found: {base_root}")
    base_manifest = load_schema_manifest(base_root)
    base_columns = _column_names_from_manifest(base_manifest)

    suggested_partners: list[dict[str, Any]] = []
    reasons: dict[str, str] = {}
    for survey_b in partners:
        partner_root = resolve_catalog_root(lake_root, survey_b)
        if not partner_root.is_dir():
            raise FileNotFoundError(f"partner catalog not found: {partner_root}")
        partner_manifest = load_schema_manifest(partner_root)
        partner_columns = _column_names_from_manifest(partner_manifest)
        entry, reason = suggest_partner_match(
            base_columns,
            partner_columns,
            partner_survey=survey_b,
            base_manifest=base_manifest,
            partner_manifest=partner_manifest,
            default_sky_radius_arcsec=default_sky_radius_arcsec,
        )
        suggested_partners.append(entry)
        reasons[survey_b] = reason

    plan = build_crossmatch_plan(
        base, suggested_partners, reuse_existing=reuse_existing
    )
    plan["_suggestions"] = reasons
    return plan


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
