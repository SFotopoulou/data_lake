"""Report catalog surveys that need phot_ab_v1 homogenization recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_lake.homogenize.survey_registry import (
    SurveyHomogenizeNotFound,
    load_survey_homogenize,
    resolve_catalog_rules,
    survey_homogenize_path,
)
from data_lake.schema_registry import (
    MODALITY_CATALOG,
    ROLE_PHOTOMETRY,
    get_survey_manifest,
)

TRANSFORM_ID = "phot_ab_v1"

# Ingested catalog names treated as near- / mid-infrared for homogenization planning.
NEAR_MID_IR_SURVEY_NAMES: frozenset[str] = frozenset({
    "2MASS_PSC",
    "2MASS_XSC",
    "ALLWISE",
    "UNWISE_W1",
    "UNWISE_W2",
    "VHS_DR3",
    "VIDEO_DR5",
    "VIKING_DR4",
    "ultraVISTA_DR6",
    "ASSEF18_C75",
    "ASSEF18_C75_extended",
    "ASSEF18_R90",
    "ASSEF18_R90_extended",
})


@dataclass
class SurveyAbStatus:
    survey: str
    category: str
    has_photometry: bool
    recipe_path: str | None
    recipe_status: str | None
    n_rules: int
    note: str | None = None


@dataclass
class PhotAbCoverageReport:
    near_mid_ir: list[SurveyAbStatus] = field(default_factory=list)
    other_needing_ab: list[SurveyAbStatus] = field(default_factory=list)
    ready: list[str] = field(default_factory=list)
    skeleton: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def warnings(self) -> list[str]:
        msgs: list[str] = []
        if self.missing:
            msgs.append(
                "Missing phot_ab_v1 recipe: "
                + ", ".join(sorted(self.missing))
            )
        if self.skeleton:
            msgs.append(
                "Skeleton recipes (fill calibration): "
                + ", ".join(sorted(self.skeleton))
            )
        if self.other_needing_ab:
            names = [s.survey for s in self.other_needing_ab]
            msgs.append(
                "Other ingested catalogs with photometry but no phot_ab_v1 recipe: "
                + ", ".join(sorted(names))
            )
        return msgs


def _ingested_catalog_surveys(lake_root: Path) -> list[str]:
    from data_lake.lake_registry import (
        filter_lake_registry_table,
        filter_registry_by_kind,
        load_lake_registry,
        registry_path,
        refresh_lake_registry,
    )

    if not registry_path(lake_root).is_file():
        refresh_lake_registry(lake_root)
    table = filter_lake_registry_table(load_lake_registry(lake_root), MODALITY_CATALOG)
    table = filter_registry_by_kind(table, "ingested")
    return sorted(
        row["survey"]
        for row in table.to_pylist()
        if row.get("survey")
    )


def _has_photometry_columns(lake_root: Path, survey: str) -> bool:
    try:
        manifest = get_survey_manifest(lake_root, survey, MODALITY_CATALOG, apply_overlay=True)
    except (FileNotFoundError, OSError, ValueError):
        return False
    for col in manifest.get("columns") or []:
        if col.get("role") == ROLE_PHOTOMETRY:
            return True
    return False


def _survey_ab_status(
    lake_root: Path | None,
    survey: str,
    *,
    category: str,
) -> SurveyAbStatus:
    path = survey_homogenize_path(lake_root, survey)
    doc = load_survey_homogenize(lake_root, survey)
    status = (doc or {}).get("recipe_status")
    try:
        rules = resolve_catalog_rules(lake_root, survey, TRANSFORM_ID)
        n_rules = len(rules)
    except SurveyHomogenizeNotFound:
        rules = []
        n_rules = 0
    has_photo = (
        _has_photometry_columns(lake_root, survey)
        if lake_root is not None
        else n_rules > 0
    )
    note = None
    if survey in {"UNWISE_W1", "UNWISE_W2"} and status == "skeleton":
        note = "native flux columns; needs flux_to_ab recipe"
    return SurveyAbStatus(
        survey=survey,
        category=category,
        has_photometry=has_photo,
        recipe_path=str(path) if path else None,
        recipe_status=status,
        n_rules=n_rules,
        note=note,
    )


def phot_ab_coverage_report(lake_root: Path | str | None) -> PhotAbCoverageReport:
    """Summarize phot_ab_v1 recipe coverage for ingested catalogs on a lake."""
    rep = PhotAbCoverageReport()
    if lake_root is None:
        return rep

    root = Path(lake_root)
    surveys = _ingested_catalog_surveys(root)
    for survey in surveys:
        is_ir = survey in NEAR_MID_IR_SURVEY_NAMES
        cat = "near_mid_ir" if is_ir else "other"
        st = _survey_ab_status(root, survey, category=cat)
        if not st.has_photometry and st.n_rules == 0:
            continue
        if is_ir:
            rep.near_mid_ir.append(st)
        elif st.n_rules == 0:
            rep.other_needing_ab.append(st)
        if st.n_rules == 0:
            rep.missing.append(survey)
        elif st.recipe_status == "skeleton":
            rep.skeleton.append(survey)
        else:
            rep.ready.append(survey)
    return rep


def phot_ab_coverage_dict(lake_root: Path | str | None) -> dict[str, Any]:
    """JSON-serializable coverage summary."""
    rep = phot_ab_coverage_report(lake_root)

    def _row(st: SurveyAbStatus) -> dict[str, Any]:
        return {
            "survey": st.survey,
            "category": st.category,
            "recipe_path": st.recipe_path,
            "recipe_status": st.recipe_status,
            "n_rules": st.n_rules,
            "note": st.note,
        }

    return {
        "transform_id": TRANSFORM_ID,
        "near_mid_ir": [_row(s) for s in rep.near_mid_ir],
        "other_needing_ab": [_row(s) for s in rep.other_needing_ab],
        "ready": sorted(rep.ready),
        "skeleton": sorted(rep.skeleton),
        "missing": sorted(rep.missing),
        "warnings": rep.warnings(),
    }
