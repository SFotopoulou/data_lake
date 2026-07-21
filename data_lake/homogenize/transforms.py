"""Apply homogenization rules to catalog columns."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from data_lake.schema_registry import ROLE_PHOTOMETRY, get_survey_manifest

log = logging.getLogger(__name__)

_MAG_SENTINELS = (-9999.0, 9999.0, -999.0, 999.0)
_FLUX_TO_AB_K = 1.0857362047461345  # 2.5 / ln(10)


# Catalog transform types shared by value and uncertainty rules.
CATALOG_TRANSFORM_TYPES = frozenset({
    "mag_offset",
    "scale",
    "identity",
    "null_if_sentinel",
    "flux_to_ab",
})


@dataclass(frozen=True)
class TransformRule:
    survey: str
    source_column: str
    target_column: str
    transform: dict[str, Any]
    uncertainty_column: str | None = None
    target_uncertainty_column: str | None = None
    uncertainty_transform: dict[str, Any] | None = None
    native_system: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransformRule:
        u_xf = data.get("uncertainty_transform")
        return cls(
            survey=str(data["survey"]),
            source_column=str(data["source_column"]),
            target_column=str(data["target_column"]),
            transform=dict(data["transform"]),
            uncertainty_column=data.get("uncertainty_column"),
            target_uncertainty_column=data.get("target_uncertainty_column"),
            uncertainty_transform=dict(u_xf) if isinstance(u_xf, dict) else None,
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


def _mag_expr(col: str, extra_sentinels: Sequence[float] = ()):
    import polars as pl

    expr = pl.col(col).cast(pl.Float64)
    for sentinel in (*_MAG_SENTINELS, *extra_sentinels):
        expr = pl.when(expr == sentinel).then(None).otherwise(expr)
    return pl.when(expr.is_nan()).then(None).otherwise(expr)


def _flux_expr(col: str):
    import polars as pl

    expr = pl.col(col).cast(pl.Float64)
    for sentinel in _MAG_SENTINELS:
        expr = pl.when(expr == sentinel).then(None).otherwise(expr)
    return (
        pl.when(expr.is_nan() | (expr <= 0))
        .then(None)
        .otherwise(expr)
    )


def _apply_value_transform(
    *,
    source_column: str,
    transform: dict[str, Any],
):
    """Build a Polars expression for a catalog value transform."""
    import polars as pl

    ttype = transform.get("type")
    extra: list[float] = [float(v) for v in transform.get("values") or []]
    src = _mag_expr(source_column, extra)
    if ttype == "mag_offset":
        return src + float(transform["delta"])
    if ttype == "scale":
        return src * float(transform["factor"])
    if ttype == "identity":
        return src
    if ttype == "null_if_sentinel":
        return src
    if ttype == "flux_to_ab":
        flux = _flux_expr(source_column)
        zp = float(transform["zp"])
        return pl.when(flux.is_not_null()).then(-2.5 * flux.log10() + zp).otherwise(None)
    raise ValueError(f"Unsupported catalog transform type {ttype!r}")


def _apply_uncertainty_transform(
    *,
    uncertainty_column: str,
    source_column: str,
    target_column: str,
    transform: dict[str, Any],
):
    """Build a Polars expression for an explicit uncertainty transform.

    Supported types match value transforms.  ``flux_to_ab`` means flux→mag
    error propagation using *source_column* as the flux and *uncertainty_column*
    as ``dflux`` (``zp`` is unused for the error).
    """
    import polars as pl

    ttype = transform.get("type")
    if ttype == "scale":
        return pl.col(uncertainty_column).cast(pl.Float64) * float(transform["factor"])
    if ttype == "mag_offset":
        # Additive offset on an uncertainty is unusual but allowed for parity.
        return pl.col(uncertainty_column).cast(pl.Float64) + float(transform["delta"])
    if ttype in ("identity", "null_if_sentinel"):
        return (
            pl.when(pl.col(target_column).is_null())
            .then(None)
            .otherwise(pl.col(uncertainty_column).cast(pl.Float64))
        )
    if ttype == "flux_to_ab":
        flux = _flux_expr(source_column)
        dflux = pl.col(uncertainty_column).cast(pl.Float64)
        return (
            pl.when(flux.is_not_null())
            .then(_FLUX_TO_AB_K * dflux / flux)
            .otherwise(None)
        )
    raise ValueError(f"Unsupported uncertainty transform type {ttype!r}")


def _default_uncertainty_expr(
    *,
    uncertainty_column: str,
    source_column: str,
    target_column: str,
    value_transform: dict[str, Any],
):
    """Auto-propagate uncertainty from the value transform type (legacy default)."""
    import polars as pl

    ttype = value_transform.get("type")
    if ttype == "scale":
        return pl.col(uncertainty_column).cast(pl.Float64) * float(
            value_transform["factor"]
        )
    if ttype == "flux_to_ab":
        flux = _flux_expr(source_column)
        dflux = pl.col(uncertainty_column).cast(pl.Float64)
        return (
            pl.when(flux.is_not_null())
            .then(_FLUX_TO_AB_K * dflux / flux)
            .otherwise(None)
        )
    # mag_offset, identity, null_if_sentinel: copy, null when target nulled
    return (
        pl.when(pl.col(target_column).is_null())
        .then(None)
        .otherwise(pl.col(uncertainty_column).cast(pl.Float64))
    )


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
        src_dtype = out.schema[rule.source_column]
        if not (src_dtype.is_numeric() or src_dtype == pl.Null):
            log.warning(
                "Skipping homogenize rule %r → %r: source column dtype %s is not "
                "numeric (product column name collision with a non-photometric field).",
                rule.source_column,
                rule.target_column,
                src_dtype,
            )
            continue
        tgt = _apply_value_transform(
            source_column=rule.source_column,
            transform=rule.transform,
        )
        out = out.with_columns(tgt.alias(rule.target_column))
        entry: dict[str, Any] = {
            "target": rule.target_column,
            "source": rule.source_column,
            "survey": rule.survey,
            "transform": rule.transform,
        }

        u_src = rule.uncertainty_column
        u_tgt = rule.target_uncertainty_column
        if u_src and u_tgt and u_src in out.columns:
            if rule.uncertainty_transform is not None:
                u_expr = _apply_uncertainty_transform(
                    uncertainty_column=u_src,
                    source_column=rule.source_column,
                    target_column=rule.target_column,
                    transform=rule.uncertainty_transform,
                )
                entry["uncertainty_transform"] = rule.uncertainty_transform
            else:
                u_expr = _default_uncertainty_expr(
                    uncertainty_column=u_src,
                    source_column=rule.source_column,
                    target_column=rule.target_column,
                    value_transform=rule.transform,
                )
            out = out.with_columns(u_expr.alias(u_tgt))
            entry["uncertainty_target"] = u_tgt

        lineage.append(entry)

    return out, lineage


def build_homogenized_view_sql(
    survey: str,
    transform: dict[str, Any],
    *,
    lake_root=None,
    catalog_view: str = "catalog",
) -> str:
    """Generate DuckDB SQL expressions for query-time homogenization (no materialize)."""
    from data_lake.homogenize.survey_registry import resolve_catalog_rules

    transform_id = str(transform.get("transform_id", "phot_ab_v1"))
    rules = resolve_catalog_rules(lake_root, survey, transform_id)
    lines: list[str] = []
    for rule in rules:
        src = rule.source_column
        tgt = rule.target_column
        ttype = rule.transform.get("type")
        if ttype == "mag_offset":
            expr = f'("{src}" + {float(rule.transform["delta"])}) AS "{tgt}"'
        elif ttype == "scale":
            expr = f'("{src}" * {float(rule.transform["factor"])}) AS "{tgt}"'
        elif ttype == "identity":
            expr = f'"{src}" AS "{tgt}"'
        elif ttype == "null_if_sentinel":
            extra_vals = [float(v) for v in rule.transform.get("values") or []]
            all_sentinels = [*_MAG_SENTINELS, *extra_vals]
            not_null = " AND ".join(f'"{src}" <> {v}' for v in all_sentinels)
            expr = (
                f'(CASE WHEN "{src}" IS NOT NULL AND ({not_null}) '
                f'THEN "{src}" ELSE NULL END) AS "{tgt}"'
            )
        elif ttype == "flux_to_ab":
            zp = float(rule.transform["zp"])
            expr = (
                f'(CASE WHEN "{src}" > 0 THEN -2.5 * log10("{src}") + {zp} '
                f'ELSE NULL END) AS "{tgt}"'
            )
        else:
            continue
        lines.append(expr)
    if not lines:
        return f"SELECT * FROM {catalog_view}"
    extra = ",\n  ".join(lines)
    return f"SELECT *,\n  {extra}\nFROM {catalog_view}"
