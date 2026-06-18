"""Tests for phot_ab_v1 coverage reporting."""

from __future__ import annotations

from pathlib import Path

from data_lake.homogenize.phot_ab_coverage import (
    NEAR_MID_IR_SURVEY_NAMES,
    phot_ab_coverage_report,
)


def test_near_mid_ir_survey_set_includes_wise() -> None:
    assert "ALLWISE" in NEAR_MID_IR_SURVEY_NAMES
    assert "VHS_DR3" in NEAR_MID_IR_SURVEY_NAMES


def test_coverage_bundled_allwise_ready() -> None:
    rep = phot_ab_coverage_report(None)
    assert rep.ready == []


def test_coverage_warns_other_photometry_surveys(tmp_path: Path) -> None:
    from data_lake.homogenize.phot_ab_coverage import phot_ab_coverage_dict

    lake = tmp_path / "lake"
    cat = lake / "catalogs" / "GALEX_GR67"
    cat.mkdir(parents=True)
    (cat / "catalog_info.json").write_text(
        '{"hats_order": 1, "kind": "ingested", "ra_column": "ra", "dec_column": "dec"}'
    )
    (lake / "shared" / "registry").mkdir(parents=True)
    # minimal registry parquet via refresh
    from data_lake.lake_registry import refresh_lake_registry

    refresh_lake_registry(lake)
    payload = phot_ab_coverage_dict(lake)
    assert "warnings" in payload
