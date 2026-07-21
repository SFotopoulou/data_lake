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
    "inverse",
})


@dataclass(frozen=True)
class TransformRule:
    survey: str
    source_column: str
    target_column: str
    transform_steps: tuple[dict[str, Any], ...]
    uncertainty_column: str | None = None
    target_uncertainty_column: str | None = None
    uncertainty_transform_steps: tuple[dict[str, Any], ...] | None = None
    native_system: str | None = None

    @property
    def transform(self) -> dict[str, Any]:
        """First (or only) value-transform step — backward-compatible accessor."""
        return self.transform_steps[0]

    @property
    def uncertainty_transform(self) -> dict[str, Any] | None:
        """First uncertainty step, or None when unset."""
        if not self.uncertainty_transform_steps:
            return None
        return self.uncertainty_transform_steps[0]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransformRule:
        return cls(
            survey=str(data["survey"]),
            source_column=str(data["source_column"]),
            target_column=str(data["target_column"]),
            transform_steps=tuple(normalize_transform_steps(data["transform"])),
            uncertainty_column=data.get("uncertainty_column"),
            target_uncertainty_column=data.get("target_uncertainty_column"),
            uncertainty_transform_steps=(
                tuple(normalize_transform_steps(data["uncertainty_transform"]))
                if data.get("uncertainty_transform") is not None
                else None
            ),
            native_system=data.get("native_system"),
        )


def normalize_transform_steps(raw: Any) -> list[dict[str, Any]]:
    """Accept a single transform object or a non-empty list of steps."""
    if isinstance(raw, dict):
        return [dict(raw)]
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ValueError("transform chain must be a non-empty list")
        steps: list[dict[str, Any]] = []
        for i, step in enumerate(raw):
            if not isinstance(step, dict):
                raise ValueError(f"transform step[{i}] must be an object, got {type(step).__name__}")
            steps.append(dict(step))
        return steps
    raise ValueError(
        f"transform must be an object or list of objects, got {type(raw).__name__}"
    )


def validate_transform_steps(steps: Sequence[dict[str, Any]], *, label: str) -> list[str]:
    """Return ERROR messages for one transform chain (value or uncertainty)."""
    msgs: list[str] = []
    if not steps:
        msgs.append(f"ERROR: {label} chain is empty")
        return msgs
    for i, step in enumerate(steps):
        t = step.get("type")
        prefix = f"ERROR: {label}" if len(steps) == 1 else f"ERROR: {label} step[{i}]"
        if t not in CATALOG_TRANSFORM_TYPES:
            msgs.append(f"{prefix} unknown transform {t!r}")
            continue
        if t == "scale" and step.get("factor") is None:
            msgs.append(f"{prefix} scale needs factor")
        if t == "mag_offset" and step.get("delta") is None:
            msgs.append(f"{prefix} mag_offset needs delta")
        if t == "flux_to_ab" and step.get("zp") is None:
            msgs.append(f"{prefix} flux_to_ab needs zp")
        if t == "null_if_sentinel":
            vals = step.get("values")
            if vals is not None and not isinstance(vals, list):
                msgs.append(f"{prefix} null_if_sentinel.values must be a list")
    return msgs


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
                    "types": [s.get("type") for s in r.transform_steps],
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
    """Build a Polars expression for a single catalog value transform on a column."""
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
    if ttype == "inverse":
        return (
            pl.when(src.is_not_null() & (src != 0))
            .then(1.0 / src)
            .otherwise(None)
        )
    raise ValueError(f"Unsupported catalog transform type {ttype!r}")


def _apply_value_step_on_expr(src, transform: dict[str, Any]):
    """Apply one value-transform step to an existing Polars expression."""
    import polars as pl

    ttype = transform.get("type")
    if ttype == "mag_offset":
        return src + float(transform["delta"])
    if ttype == "scale":
        return src * float(transform["factor"])
    if ttype == "identity":
        return src
    if ttype == "null_if_sentinel":
        expr = src
        for sentinel in [float(v) for v in transform.get("values") or []]:
            expr = pl.when(expr == sentinel).then(None).otherwise(expr)
        return pl.when(expr.is_nan()).then(None).otherwise(expr)
    if ttype == "flux_to_ab":
        zp = float(transform["zp"])
        flux = (
            pl.when(src.is_nan() | (src <= 0))
            .then(None)
            .otherwise(src)
        )
        return pl.when(flux.is_not_null()).then(-2.5 * flux.log10() + zp).otherwise(None)
    if ttype == "inverse":
        return (
            pl.when(src.is_not_null() & (src != 0))
            .then(1.0 / src)
            .otherwise(None)
        )
    raise ValueError(f"Unsupported catalog transform type {ttype!r}")


def _apply_value_transform_chain(
    *,
    source_column: str,
    steps: Sequence[dict[str, Any]],
):
    """Apply one or more value-transform steps left-to-right."""
    if not steps:
        raise ValueError("value transform chain is empty")
    expr = _apply_value_transform(source_column=source_column, transform=steps[0])
    for step in steps[1:]:
        expr = _apply_value_step_on_expr(expr, step)
    return expr


def _apply_uncertainty_transform(
    *,
    uncertainty_column: str,
    source_column: str,
    target_column: str,
    transform: dict[str, Any],
    err_expr=None,
    flux_expr=None,
):
    """Build a Polars expression for an explicit uncertainty transform step.

    Supported types match value transforms.  ``flux_to_ab`` means flux→mag
    error propagation using *flux_expr* (or *source_column*) as the flux and
    *err_expr* / *uncertainty_column* as ``dflux`` (``zp`` is unused for the error).
    """
    import polars as pl

    ttype = transform.get("type")
    err = err_expr if err_expr is not None else pl.col(uncertainty_column).cast(pl.Float64)
    if ttype == "scale":
        return err * float(transform["factor"])
    if ttype == "mag_offset":
        return err + float(transform["delta"])
    if ttype in ("identity", "null_if_sentinel"):
        return (
            pl.when(pl.col(target_column).is_null())
            .then(None)
            .otherwise(err)
        )
    if ttype == "flux_to_ab":
        flux = flux_expr if flux_expr is not None else _flux_expr(source_column)
        return (
            pl.when(flux.is_not_null())
            .then(_FLUX_TO_AB_K * err / flux)
            .otherwise(None)
        )
    if ttype == "inverse":
        # Reciprocal of the uncertainty column (or current err expr in a chain).
        return (
            pl.when(err.is_not_null() & (err != 0))
            .then(1.0 / err)
            .otherwise(None)
        )
    raise ValueError(f"Unsupported uncertainty transform type {ttype!r}")


def _apply_uncertainty_transform_chain(
    *,
    uncertainty_column: str,
    source_column: str,
    target_column: str,
    steps: Sequence[dict[str, Any]],
    value_steps: Sequence[dict[str, Any]],
):
    """Apply uncertainty steps; track intermediate value for ``flux_to_ab`` / ``inverse``."""
    import polars as pl

    err = pl.col(uncertainty_column).cast(pl.Float64)
    # Intermediate value for inverse; flux track for flux_to_ab.
    value = _mag_expr(source_column)
    flux = _flux_expr(source_column)
    for step in value_steps:
        ttype = step.get("type")
        if ttype == "scale":
            factor = float(step["factor"])
            value = value * factor
            flux = flux * factor
        elif ttype == "mag_offset":
            value = value + float(step["delta"])
        elif ttype == "inverse":
            value = (
                pl.when(value.is_not_null() & (value != 0))
                .then(1.0 / value)
                .otherwise(None)
            )
        elif ttype == "flux_to_ab":
            break

    for step in steps:
        ttype = step.get("type")
        err = _apply_uncertainty_transform(
            uncertainty_column=uncertainty_column,
            source_column=source_column,
            target_column=target_column,
            transform=step,
            err_expr=err,
            flux_expr=flux if ttype == "flux_to_ab" else value,
        )
        if ttype == "scale":
            factor = float(step["factor"])
            value = value * factor
            flux = flux * factor
        elif ttype == "inverse":
            value = (
                pl.when(value.is_not_null() & (value != 0))
                .then(1.0 / value)
                .otherwise(None)
            )
    return err


def _default_uncertainty_expr(
    *,
    uncertainty_column: str,
    source_column: str,
    target_column: str,
    value_transform: dict[str, Any],
):
    """Auto-propagate uncertainty from a single value transform type (legacy).

    Note: value ``mag_offset`` does **not** add ``delta`` to the error — that
    only happens for an explicit ``uncertainty_transform`` of type ``mag_offset``.
    """
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
    # mag_offset, identity, null_if_sentinel, inverse: copy, null when target nulled
    # (inverse does not auto-propagate; use uncertainty_transform: {type: inverse}
    #  to take the reciprocal of the uncertainty column itself.)
    return (
        pl.when(pl.col(target_column).is_null())
        .then(None)
        .otherwise(pl.col(uncertainty_column).cast(pl.Float64))
    )


def _default_uncertainty_from_value_chain(
    *,
    uncertainty_column: str,
    source_column: str,
    target_column: str,
    value_steps: Sequence[dict[str, Any]],
):
    """Compose auto uncertainty propagation across a value-transform chain.

    ``scale`` multiplies the error; ``flux_to_ab`` converts using the flux after
    any preceding scales; ``inverse`` does **not** auto-propagate (error is
    copied; set ``uncertainty_transform: {type: inverse}`` to take ``1/err``);
    ``mag_offset`` / ``identity`` / ``null_if_sentinel`` leave the error unchanged
    (nulled when the value target is null).
    """
    import polars as pl

    err = pl.col(uncertainty_column).cast(pl.Float64)
    value = _mag_expr(source_column)
    flux = _flux_expr(source_column)
    saw_flux_to_ab = False
    for step in value_steps:
        ttype = step.get("type")
        if ttype == "scale":
            factor = float(step["factor"])
            err = err * factor
            value = value * factor
            if not saw_flux_to_ab:
                flux = flux * factor
        elif ttype == "mag_offset":
            value = value + float(step["delta"])
        elif ttype == "inverse":
            # No auto uncertainty change for inverse — only track value for later steps.
            value = (
                pl.when(value.is_not_null() & (value != 0))
                .then(1.0 / value)
                .otherwise(None)
            )
        elif ttype == "flux_to_ab":
            err = (
                pl.when(flux.is_not_null())
                .then(_FLUX_TO_AB_K * err / flux)
                .otherwise(None)
            )
            saw_flux_to_ab = True
        elif ttype in ("identity", "null_if_sentinel"):
            continue
        else:
            raise ValueError(f"Unsupported catalog transform type {ttype!r}")
    return (
        pl.when(pl.col(target_column).is_null())
        .then(None)
        .otherwise(err)
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
        tgt = _apply_value_transform_chain(
            source_column=rule.source_column,
            steps=rule.transform_steps,
        )
        out = out.with_columns(tgt.alias(rule.target_column))
        entry: dict[str, Any] = {
            "target": rule.target_column,
            "source": rule.source_column,
            "survey": rule.survey,
            "transform": (
                list(rule.transform_steps)
                if len(rule.transform_steps) > 1
                else rule.transform
            ),
        }

        u_src = rule.uncertainty_column
        u_tgt = rule.target_uncertainty_column
        if u_src and u_tgt and u_src in out.columns:
            if rule.uncertainty_transform_steps is not None:
                u_expr = _apply_uncertainty_transform_chain(
                    uncertainty_column=u_src,
                    source_column=rule.source_column,
                    target_column=rule.target_column,
                    steps=rule.uncertainty_transform_steps,
                    value_steps=rule.transform_steps,
                )
                entry["uncertainty_transform"] = (
                    list(rule.uncertainty_transform_steps)
                    if len(rule.uncertainty_transform_steps) > 1
                    else rule.uncertainty_transform
                )
            elif len(rule.transform_steps) == 1:
                u_expr = _default_uncertainty_expr(
                    uncertainty_column=u_src,
                    source_column=rule.source_column,
                    target_column=rule.target_column,
                    value_transform=rule.transform,
                )
            else:
                u_expr = _default_uncertainty_from_value_chain(
                    uncertainty_column=u_src,
                    source_column=rule.source_column,
                    target_column=rule.target_column,
                    value_steps=rule.transform_steps,
                )
            out = out.with_columns(u_expr.alias(u_tgt))
            entry["uncertainty_target"] = u_tgt

        lineage.append(entry)

    return out, lineage


def _sql_expr_for_value_steps(src: str, steps: Sequence[dict[str, Any]]) -> str:
    """Nest DuckDB SQL for a value-transform chain starting from column *src*."""
    expr = f'"{src}"'
    for i, step in enumerate(steps):
        ttype = step.get("type")
        if ttype == "mag_offset":
            expr = f"({expr} + {float(step['delta'])})"
        elif ttype == "scale":
            expr = f"({expr} * {float(step['factor'])})"
        elif ttype == "identity":
            continue
        elif ttype == "null_if_sentinel":
            extra_vals = [float(v) for v in step.get("values") or []]
            all_sentinels = [*_MAG_SENTINELS, *extra_vals] if i == 0 else extra_vals
            if all_sentinels:
                not_null = " AND ".join(f"{expr} <> {v}" for v in all_sentinels)
                expr = f"(CASE WHEN {expr} IS NOT NULL AND ({not_null}) THEN {expr} ELSE NULL END)"
        elif ttype == "flux_to_ab":
            zp = float(step["zp"])
            expr = (
                f"(CASE WHEN {expr} > 0 THEN -2.5 * log10({expr}) + {zp} ELSE NULL END)"
            )
        elif ttype == "inverse":
            expr = (
                f"(CASE WHEN {expr} IS NOT NULL AND {expr} <> 0 "
                f"THEN 1.0 / ({expr}) ELSE NULL END)"
            )
        else:
            raise ValueError(f"Unsupported catalog transform type {ttype!r}")
    return expr


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
        try:
            inner = _sql_expr_for_value_steps(rule.source_column, rule.transform_steps)
        except ValueError:
            continue
        lines.append(f'{inner} AS "{rule.target_column}"')
    if not lines:
        return f"SELECT * FROM {catalog_view}"
    extra = ",\n  ".join(lines)
    return f"SELECT *,\n  {extra}\nFROM {catalog_view}"
