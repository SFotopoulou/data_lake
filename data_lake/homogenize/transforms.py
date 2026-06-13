"""Apply homogenization rules to catalog columns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from data_lake.schema_registry import ROLE_PHOTOMETRY, get_survey_manifest

_MAG_SENTINELS = (-9999.0, 9999.0, -999.0, 999.0)


@dataclass(frozen=True)
class TransformRule:
    survey: str
    source_column: str
    target_column: str
    transform: dict[str, Any]
    uncertainty_column: str | None = None
    target_uncertainty_column: str | None = None
    native_system: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransformRule:
        return cls(
            survey=str(data["survey"]),
            source_column=str(data["source_column"]),
            target_column=str(data["target_column"]),
            transform=dict(data["transform"]),
            uncertainty_column=data.get("uncertainty_column"),
            target_uncertainty_column=data.get("target_uncertainty_column"),
            native_system=data.get("native_system"),
        )


@dataclass
class RuleResolution:
    applied: list[TransformRule] = field(default_factory=list)
    skipped_missing: list[dict[str, str]] = field(default_factory=list)
    skipped_survey: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_applied": len(self.applied),
            "applied": [
                {
                    "source": r.source_column,
                    "target": r.target_column,
                    "type": r.transform.get("type"),
                }
                for r in self.applied
            ],
            "skipped_missing": self.skipped_missing,
            "skipped_survey": self.skipped_survey,
        }


def _rules_for_survey(transform: dict[str, Any], survey: str) -> list[TransformRule]:
    out: list[TransformRule] = []
    for raw in transform.get("rules") or []:
        if raw.get("survey") != survey:
            continue
        out.append(TransformRule.from_dict(raw))
    return out


def _manifest_column_names(manifest: dict[str, Any]) -> set[str]:
    return {c["name"] for c in manifest.get("columns") or []}


def _photometry_columns(manifest: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for col in manifest.get("columns") or []:
        if col.get("role") == ROLE_PHOTOMETRY:
            names.add(col["name"])
    return names


def resolve_applicable_rules(
    transform: dict[str, Any],
    survey: str,
    available_columns: set[str],
    *,
    columns: Sequence[str] | None = None,
) -> RuleResolution:
    """Intersect transform rules with columns present on disk."""
    res = RuleResolution()
    survey_rules = _rules_for_survey(transform, survey)
    if not survey_rules:
        res.skipped_survey.append(
            {"survey": survey, "reason": f"no rules in {transform.get('transform_id')}"}
        )
        return res

    want_sources: set[str] | None = None
    if columns:
        want_sources = set(columns)

    for rule in survey_rules:
        if rule.source_column not in available_columns:
            res.skipped_missing.append(
                {
                    "survey": survey,
                    "source_column": rule.source_column,
                    "reason": "column not in manifest/tiles",
                }
            )
            continue
        if want_sources is not None and rule.source_column not in want_sources:
            continue
        res.applied.append(rule)
    return res


def resolve_rules_from_manifest(
    lake_root,
    transform: dict[str, Any],
    survey: str,
    *,
    columns: Sequence[str] | None = None,
) -> RuleResolution:
    from data_lake.homogenize.survey_registry import resolve_catalog_rules

    manifest = get_survey_manifest(lake_root, survey, "catalog", apply_overlay=True)
    available = _manifest_column_names(manifest)

    survey_rules = resolve_catalog_rules(
        lake_root, survey, str(transform.get("transform_id", "")),
    )

    if columns is None:
        photo = _photometry_columns(manifest)
        rule_sources = {r.source_column for r in survey_rules}
        columns = sorted(photo & rule_sources) if photo else None

    res = RuleResolution()
    want_sources: set[str] | None = set(columns) if columns else None
    for rule in survey_rules:
        if rule.source_column not in available:
            res.skipped_missing.append(
                {
                    "survey": survey,
                    "source_column": rule.source_column,
                    "reason": "column not in manifest/tiles",
                }
            )
            continue
        if want_sources is not None and rule.source_column not in want_sources:
            continue
        res.applied.append(rule)

    if not survey_rules:
        res.skipped_survey.append(
            {
                "survey": survey,
                "reason": f"no rules in survey homogenize or {transform.get('transform_id')}",
            }
        )
    return res


def _mag_expr(col: str):
    import polars as pl

    expr = pl.col(col).cast(pl.Float64)
    for sentinel in _MAG_SENTINELS:
        expr = pl.when(expr == sentinel).then(None).otherwise(expr)
    return pl.when(expr.is_nan()).then(None).otherwise(expr)


def apply_rules_to_frame(df, rules: Sequence[TransformRule]):
    """Return a new Polars frame with homogenized columns added."""
    import polars as pl

    if not isinstance(df, pl.DataFrame):
        df = pl.from_arrow(df)

    out = df
    lineage: list[dict[str, Any]] = []

    for rule in rules:
        if rule.source_column not in out.columns:
            continue
        src = _mag_expr(rule.source_column)
        ttype = rule.transform.get("type")
        if ttype == "mag_offset":
            tgt = src + float(rule.transform["delta"])
        elif ttype == "scale":
            tgt = src * float(rule.transform["factor"])
        elif ttype == "identity":
            tgt = src
        else:
            raise ValueError(f"Unsupported catalog transform type {ttype!r}")

        out = out.with_columns(tgt.alias(rule.target_column))
        lineage.append(
            {
                "target": rule.target_column,
                "source": rule.source_column,
                "survey": rule.survey,
                "transform": rule.transform,
            }
        )

        u_src = rule.uncertainty_column
        u_tgt = rule.target_uncertainty_column
        if u_src and u_tgt and u_src in out.columns:
            u_expr = pl.col(u_src).cast(pl.Float64)
            if ttype == "scale":
                u_expr = u_expr * float(rule.transform["factor"])
            out = out.with_columns(u_expr.alias(u_tgt))
            lineage[-1]["uncertainty_target"] = u_tgt

    return out, lineage


def build_homogenized_view_sql(
    survey: str,
    transform: dict[str, Any],
    *,
    catalog_view: str = "catalog",
) -> str:
    """Generate DuckDB SQL expressions for query-time homogenization (no materialize)."""
    lines: list[str] = []
    for rule in _rules_for_survey(transform, survey):
        src = rule.source_column
        tgt = rule.target_column
        ttype = rule.transform.get("type")
        if ttype == "mag_offset":
            expr = f'("{src}" + {float(rule.transform["delta"])}) AS "{tgt}"'
        elif ttype == "scale":
            expr = f'("{src}" * {float(rule.transform["factor"])}) AS "{tgt}"'
        elif ttype == "identity":
            expr = f'"{src}" AS "{tgt}"'
        else:
            continue
        lines.append(expr)
    if not lines:
        return f"SELECT * FROM {catalog_view}"
    extra = ",\n  ".join(lines)
    return f"SELECT *,\n  {extra}\nFROM {catalog_view}"
