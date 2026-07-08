"""Per-survey homogenization recipes spanning catalog, spectra, and cutout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_lake.homogenize.registry import load_transform
from data_lake.homogenize.transforms import TransformRule
from data_lake.schema_registry import MODALITY_CATALOG, MODALITY_CUTOUT, MODALITY_SPECTRA

_PKG_SURVEYS = Path(__file__).resolve().parent / "surveys"

_DEFAULT_TRANSFORMS = {
    MODALITY_CATALOG: "phot_ab_v1",
    MODALITY_SPECTRA: "spec_observed_v1",
    MODALITY_CUTOUT: "cutout_njy_v1",
}


def surveys_homogenize_dir(lake_root: Path | str) -> Path:
    return Path(lake_root) / "shared" / "registry" / "homogenize"


def survey_homogenize_path(lake_root: Path | str | None, survey: str) -> Path | None:
    """Resolve per-survey homogenize JSON: lake override, then bundled default."""
    if lake_root is not None:
        lake_path = surveys_homogenize_dir(lake_root) / f"{survey}.json"
        if lake_path.is_file():
            return lake_path
    bundled = _PKG_SURVEYS / f"{survey}.json"
    if bundled.is_file():
        return bundled
    return None


def load_survey_homogenize(
    lake_root: Path | str | None,
    survey: str,
) -> dict[str, Any] | None:
    path = survey_homogenize_path(lake_root, survey)
    if path is None:
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("survey") and data["survey"] != survey:
        raise ValueError(
            f"Survey homogenize file {path} has survey={data['survey']!r}, expected {survey!r}"
        )
    return data


def _modality_block(doc: dict[str, Any], modality: str) -> dict[str, Any] | None:
    block = doc.get(modality)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(f"survey homogenize {modality!r} block must be an object")
    return block


def _rules_from_block(
    survey: str,
    block: dict[str, Any],
    *,
    transform_id: str | None = None,
) -> list[dict[str, Any]]:
    """Extract raw rule dicts from a modality block (flat or keyed by transform_id)."""
    if transform_id and transform_id in block:
        nested = block[transform_id]
        if not isinstance(nested, dict):
            raise ValueError(f"transform block {transform_id!r} must be an object")
        return list(nested.get("rules") or [])
    if "rules" in block:
        return list(block.get("rules") or [])
    return []


def catalog_rules_for_survey(
    lake_root: Path | str | None,
    survey: str,
    transform_id: str,
) -> list[TransformRule] | None:
    """Return catalog rules from the per-survey file, or None to use the transform pack."""
    doc = load_survey_homogenize(lake_root, survey)
    if doc is None:
        return None
    block = _modality_block(doc, MODALITY_CATALOG)
    if block is None:
        return None
    if transform_id in block:
        nested = block[transform_id]
        if not isinstance(nested, dict):
            raise ValueError(f"catalog.{transform_id} must be an object")
        raw = list(nested.get("rules") or [])
    elif "rules" in block:
        raw = list(block.get("rules") or [])
    else:
        return None
    return [
        TransformRule.from_dict({**rule, "survey": survey})
        for rule in raw
    ]


def zarr_rule_for_survey(
    lake_root: Path | str | None,
    survey: str,
    modality: str,
    transform_id: str,
) -> dict[str, Any] | None:
    """Return spectra/cutout transform spec from the per-survey file."""
    doc = load_survey_homogenize(lake_root, survey)
    if doc is None:
        return None
    block = _modality_block(doc, modality)
    if block is None:
        return None
    if transform_id in block:
        spec = dict(block[transform_id])
    else:
        spec = dict(block)
    spec.setdefault("survey", survey)
    return spec


class SurveyHomogenizeNotFound(LookupError):
    """Raised when no per-survey homogenize recipe exists for a transform."""


def resolve_catalog_rules(
    lake_root: Path | str | None,
    survey: str,
    transform_id: str,
) -> list[TransformRule]:
    """Return catalog rules from the per-survey homogenize file only."""
    rules = catalog_rules_for_survey(lake_root, survey, transform_id)
    if rules is not None:
        return rules
    path = survey_homogenize_path(lake_root, survey)
    if path is None:
        raise SurveyHomogenizeNotFound(
            f"No homogenize recipe for survey {survey!r} "
            f"(add homogenize/{survey}.json with catalog.{transform_id}.rules)"
        )
    raise SurveyHomogenizeNotFound(
        f"Survey {survey!r} homogenize file {path} has no catalog.{transform_id} rules"
    )


def default_transform_id(modality: str) -> str:
    try:
        return _DEFAULT_TRANSFORMS[modality]
    except KeyError as exc:
        raise ValueError(f"unknown modality {modality!r}") from exc


def validate_survey_homogenize(data: dict[str, Any]) -> list[str]:
    """Validate a per-survey homogenize document."""
    msgs: list[str] = []
    if not data.get("survey"):
        msgs.append("ERROR: missing survey")
    status = str(data.get("recipe_status") or "")
    if status == "skeleton":
        msgs.append(
            "WARN: recipe_status is skeleton — fill calibration before production homogenize"
        )
    for modality in (MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT):
        block = data.get(modality)
        if block is None:
            continue
        if not isinstance(block, dict):
            msgs.append(f"ERROR: {modality} block must be an object")
            continue
        rules = _rules_from_block(str(data.get("survey", "")), block)
        if modality == MODALITY_CATALOG:
            for i, rule in enumerate(rules):
                if not rule.get("source_column"):
                    msgs.append(f"ERROR: catalog rule[{i}] missing source_column")
                if not rule.get("target_column"):
                    msgs.append(f"ERROR: catalog rule[{i}] missing target_column")
                t = (rule.get("transform") or {}).get("type")
                if t not in ("mag_offset", "scale", "identity", "null_if_sentinel", "flux_to_ab"):
                    msgs.append(f"ERROR: catalog rule[{i}] unknown transform {t!r}")
                if t == "null_if_sentinel":
                    vals = (rule.get("transform") or {}).get("values")
                    if vals is not None and not isinstance(vals, list):
                        msgs.append(f"ERROR: catalog rule[{i}] null_if_sentinel.values must be a list")
        elif modality == MODALITY_SPECTRA:
            if block.get("flux_calibration"):
                cal = block["flux_calibration"]
                if not isinstance(cal, dict) or cal.get("flux_scale") is None:
                    msgs.append("ERROR: spectra.flux_calibration missing flux_scale")
                continue
            if not block.get("flux_array") and not any(
                isinstance(v, dict) and v.get("flux_array")
                for v in block.values()
                if isinstance(v, dict)
            ):
                msgs.append(f"WARN: {modality} block missing flux_array")
        elif modality == MODALITY_CUTOUT:
            has_image = block.get("image_array") or any(
                isinstance(v, dict) and v.get("image_array")
                for v in block.values()
                if isinstance(v, dict)
            )
            if not has_image:
                msgs.append(f"WARN: {modality} block missing image_array")
    return msgs
