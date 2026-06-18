"""Validation for homogenization transform registry and products."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq

from data_lake.homogenize.registry import (
    load_transform,
    transforms_dir,
    validate_transform_schema,
)
from data_lake.homogenize.transforms import TransformRule, apply_rules_to_frame
from data_lake.schema_registry import PRODUCT_SUBTYPE_HOMOGENIZED

_PKG_GOLDEN = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "homogenization"
)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def ok(self, *, strict: bool) -> bool:
        if self.errors:
            return False
        if strict and self.warnings:
            return False
        return True


def _bundled_transform_ids() -> list[str]:
    root = Path(__file__).resolve().parent / "transforms"
    return sorted(p.stem for p in root.glob("*.json"))


def _bundled_survey_ids() -> list[str]:
    root = Path(__file__).resolve().parent / "surveys"
    return sorted(p.stem for p in root.glob("*.json"))


def validate_survey_homogenize_registry(lake_root: Path | str | None) -> ValidationReport:
    """Lint bundled and lake-local per-survey homogenize files."""
    from data_lake.homogenize.survey_registry import (
        load_survey_homogenize,
        validate_survey_homogenize,
    )

    rep = ValidationReport()
    seen: set[str] = set()
    for sid in _bundled_survey_ids():
        seen.add(sid)
        try:
            data = load_survey_homogenize(None, sid)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            rep.errors.append(f"survey homogenize {sid!r}: {exc}")
            continue
        if data is None:
            continue
        for msg in validate_survey_homogenize(data):
            if msg.startswith("ERROR"):
                rep.errors.append(f"survey homogenize {sid!r}: {msg}")
            else:
                rep.warnings.append(f"survey homogenize {sid!r}: {msg}")

    if lake_root is not None:
        lake_dir = Path(lake_root) / "shared" / "registry" / "homogenize"
        if lake_dir.is_dir():
            for path in sorted(lake_dir.glob("*.json")):
                sid = path.stem
                try:
                    data = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    rep.errors.append(f"survey homogenize {sid!r}: {exc}")
                    continue
                for msg in validate_survey_homogenize(data):
                    if msg.startswith("ERROR"):
                        rep.errors.append(f"survey homogenize {sid!r}: {msg}")
                    else:
                        rep.warnings.append(f"survey homogenize {sid!r}: {msg}")
                if sid in seen:
                    rep.warnings.append(
                        f"survey homogenize {sid!r}: lake override shadows bundled default"
                    )
    return rep


def validate_transform_registry(lake_root: Path | str | None) -> ValidationReport:
    """Lint bundled and lake-local transform packs."""
    rep = ValidationReport()
    seen: set[str] = set()
    for tid in _bundled_transform_ids():
        seen.add(tid)
        try:
            data = load_transform(lake_root, tid)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            rep.errors.append(f"transform {tid!r}: {exc}")
            continue
        for msg in validate_transform_schema(data):
            if msg.startswith("ERROR"):
                rep.errors.append(f"transform {tid!r}: {msg}")
            else:
                rep.warnings.append(f"transform {tid!r}: {msg}")

    if lake_root is not None:
        lake_dir = transforms_dir(lake_root)
        if lake_dir.is_dir():
            for path in sorted(lake_dir.glob("*.json")):
                tid = path.stem
                if tid in seen:
                    rep.warnings.append(
                        f"transform {tid!r}: lake override shadows bundled default"
                    )
                try:
                    data = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    rep.errors.append(f"transform {tid!r}: {exc}")
                    continue
                for msg in validate_transform_schema(data):
                    if msg.startswith("ERROR"):
                        rep.errors.append(f"transform {tid!r}: {msg}")
                    else:
                        rep.warnings.append(f"transform {tid!r}: {msg}")
    return rep


def _golden_path(name: str) -> Path:
    path = _PKG_GOLDEN / name
    if not path.is_file():
        raise FileNotFoundError(f"golden fixture not found: {path}")
    return path


def validate_golden_transform(
    transform_id: str,
    *,
    fixture_name: str | None = None,
    lake_root: Path | str | None = None,
) -> ValidationReport:
    """Spot-check transform rules against golden input/output pairs."""
    from data_lake.homogenize.survey_registry import resolve_catalog_rules

    rep = ValidationReport()
    fixture = _golden_path(fixture_name or f"{transform_id}_golden.json")
    spec = json.loads(fixture.read_text())
    if spec.get("transform_id") != transform_id:
        rep.errors.append(
            f"fixture transform_id {spec.get('transform_id')!r} != {transform_id!r}"
        )
        return rep

    import polars as pl

    for case in spec.get("cases") or []:
        survey = case["survey"]
        src = case["source_column"]
        tgt = case["target_column"]
        try:
            survey_rules = resolve_catalog_rules(lake_root, survey, transform_id)
        except Exception as exc:
            rep.errors.append(f"golden case {survey}.{src}: {exc}")
            continue
        rules = [r for r in survey_rules if r.source_column == src and r.target_column == tgt]
        if not rules:
            rep.errors.append(f"golden case missing rule for {survey}.{src}")
            continue
        raw = {
            "survey": rules[0].survey,
            "source_column": rules[0].source_column,
            "target_column": rules[0].target_column,
            "transform": dict(rules[0].transform),
            "uncertainty_column": rules[0].uncertainty_column,
            "target_uncertainty_column": rules[0].target_uncertainty_column,
        }
        if case.get("transform_override"):
            raw["transform"] = case["transform_override"]
        rule = TransformRule.from_dict(raw)
        df = pl.DataFrame({src: [float(case["input_value"])]})
        out, _ = apply_rules_to_frame(df, [rule])
        got = out[tgt][0]
        expected = float(case["expected_output"])
        tol = float(case.get("tolerance", 1e-6))
        if abs(got - expected) > tol:
            rep.errors.append(
                f"{case.get('description', tgt)}: expected {expected}, got {got} "
                f"(tol {tol})"
            )
    return rep


def validate_homogenized_catalog_product(
    lake_root: Path | str,
    product: str,
    *,
    transform_id: str | None = None,
    max_tiles: int = 3,
) -> ValidationReport:
    """Validate a homogenized catalog product's metadata and sample values."""
    rep = ValidationReport()
    root = Path(lake_root) / "catalogs" / product
    info_path = root / "catalog_info.json"
    if not info_path.is_file():
        rep.errors.append(f"missing catalog_info.json for product {product!r}")
        return rep

    info = json.loads(info_path.read_text())
    if info.get("product_subtype") != PRODUCT_SUBTYPE_HOMOGENIZED:
        rep.errors.append(
            f"product {product!r} product_subtype={info.get('product_subtype')!r} "
            f"(expected {PRODUCT_SUBTYPE_HOMOGENIZED!r})"
        )
    prov = info.get("provenance") or {}
    tid = prov.get("transform_id")
    if transform_id and tid != transform_id:
        rep.errors.append(
            f"product transform_id {tid!r} != expected {transform_id!r}"
        )
    if not tid:
        rep.warnings.append(f"product {product!r} missing provenance.transform_id")

    resolution = prov.get("resolution") or {}
    if resolution.get("n_applied", 0) == 0:
        rep.errors.append(f"product {product!r} has zero applied transform rules")

    tiles = sorted(root.rglob("Npix=*.parquet"))[: max(0, max_tiles)]
    if not tiles:
        rep.warnings.append(f"product {product!r} has no Parquet tiles to sample")
        return rep

    applied = resolution.get("applied") or []
    for tile in tiles:
        schema = pq.read_schema(str(tile))
        for rule in applied:
            tgt = rule.get("target")
            if tgt and tgt not in schema.names:
                rep.errors.append(f"{tile}: missing homogenized column {tgt!r}")
    return rep


def run_validation(
    lake_root: Path | str | None,
    *,
    transform_ids: Sequence[str] | None = None,
    golden: bool = False,
    products: Sequence[str] = (),
    strict: bool = False,
    ab_coverage: bool = False,
) -> ValidationReport:
    """Run registry lint, optional golden checks, and product validation."""
    rep = ValidationReport()
    reg = validate_transform_registry(lake_root)
    rep.errors.extend(reg.errors)
    rep.warnings.extend(reg.warnings)
    sreg = validate_survey_homogenize_registry(lake_root)
    rep.errors.extend(sreg.errors)
    rep.warnings.extend(sreg.warnings)

    tids = list(transform_ids) if transform_ids else _bundled_transform_ids()
    if golden:
        for tid in tids:
            try:
                gold = validate_golden_transform(tid, lake_root=lake_root)
            except FileNotFoundError as exc:
                rep.warnings.append(str(exc))
                continue
            rep.errors.extend(gold.errors)
            rep.warnings.extend(gold.warnings)

    if ab_coverage and lake_root is not None:
        from data_lake.homogenize.phot_ab_coverage import phot_ab_coverage_report

        cov = phot_ab_coverage_report(lake_root)
        rep.warnings.extend(cov.warnings())

    if lake_root is not None:
        for product in products:
            prod = validate_homogenized_catalog_product(
                lake_root, product, transform_id=transform_ids[0] if transform_ids else None,
            )
            rep.errors.extend(prod.errors)
            rep.warnings.extend(prod.warnings)

    return rep
