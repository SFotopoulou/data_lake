"""Homogenization rule resolution for gathered wide product catalogs."""

from __future__ import annotations

from typing import Any, Sequence

from data_lake.homogenize.survey_registry import SurveyHomogenizeNotFound, resolve_catalog_rules
from data_lake.homogenize.transforms import RuleResolution, TransformRule


def product_column_name(base_catalog: str, survey: str, source_column: str) -> str:
    """Map a native survey column to its name in a gather product."""
    if survey == base_catalog:
        return source_column
    if source_column.startswith(f"{survey}_"):
        return source_column
    return f"{survey}_{source_column}"


def product_surveys(provenance: dict[str, Any]) -> frozenset[str]:
    """Return survey names contributing columns to a gather product."""
    base = provenance.get("base_catalog")
    if not base:
        return frozenset()
    surveys: set[str] = {str(base)}
    for partner in provenance.get("partners") or []:
        if partner.get("survey"):
            surveys.add(str(partner["survey"]))
    return frozenset(surveys)


def resolve_rules_for_product(
    lake_root,
    transform: dict[str, Any],
    provenance: dict[str, Any],
    available_columns: set[str],
    *,
    columns: Sequence[str] | None = None,
) -> RuleResolution:
    """Intersect transform rules with columns present in a gather product."""
    base = provenance.get("base_catalog")
    if not base:
        raise ValueError(
            "product provenance missing base_catalog; "
            "only gather products support --from-product homogenization"
        )
    contributors = product_surveys(provenance)
    res = RuleResolution()
    want_native: set[str] | None = set(columns) if columns else None
    transform_id = str(transform.get("transform_id", "phot_ab_v1"))

    for survey in sorted(contributors):
        try:
            survey_rules = resolve_catalog_rules(lake_root, survey, transform_id)
        except SurveyHomogenizeNotFound:
            res.skipped_survey.append(
                {"survey": survey, "reason": f"no homogenize/{survey}.json recipe"}
            )
            continue
        for rule in survey_rules:
            prod_col = product_column_name(base, survey, rule.source_column)
            if prod_col not in available_columns:
                res.skipped_missing.append(
                    {
                        "survey": survey,
                        "source_column": rule.source_column,
                        "product_column": prod_col,
                        "reason": "column not in product",
                    }
                )
                continue
            if want_native is not None and rule.source_column not in want_native:
                continue
            res.applied.append(
                TransformRule(
                    survey=rule.survey,
                    source_column=prod_col,
                    target_column=rule.target_column,
                    transform_steps=rule.transform_steps,
                    uncertainty_column=(
                        product_column_name(base, survey, rule.uncertainty_column)
                        if rule.uncertainty_column
                        else None
                    ),
                    target_uncertainty_column=rule.target_uncertainty_column,
                    uncertainty_transform_steps=rule.uncertainty_transform_steps,
                    native_system=rule.native_system,
                )
            )
    return res
