"""Unit tests for crossmatch plan suggest heuristics."""

from __future__ import annotations

import json
from pathlib import Path

from data_lake.discovery.area_plan import (
    strip_crossmatch_plan_annotations,
    suggest_crossmatch_plan,
    suggest_partner_match,
)


class TestSuggestPartnerMatch:
    def test_shared_targetid(self) -> None:
        partner, reason = suggest_partner_match(
            ["TARGETID", "RA", "DEC"],
            ["TARGETID", "Z"],
            partner_survey="DESI_DR1",
        )
        assert partner["match_mode"] == "column"
        assert partner["match_col_a"] == "TARGETID"
        assert partner["match_col_b"] == "TARGETID"
        assert "TARGETID" in reason

    def test_gaia_fk(self) -> None:
        partner, reason = suggest_partner_match(
            ["OBJECT_ID", "GAIA_ID", "right_ascension"],
            ["source_id", "ra", "dec", "designation"],
            partner_survey="GAIA_DR3_source",
            partner_manifest={"link_id_mode": "label:designation"},
        )
        assert partner["match_mode"] == "column"
        assert partner["match_col_a"] == "GAIA_ID"
        assert partner["match_col_b"] == "source_id"
        assert "GAIA_ID" in reason

    def test_sky_fallback(self) -> None:
        partner, reason = suggest_partner_match(
            ["OBJECT_ID", "FLUX"],
            ["w1mpro", "ra", "dec"],
            partner_survey="ALLWISE",
            default_sky_radius_arcsec=2.5,
        )
        assert partner == {
            "survey": "ALLWISE",
            "match_mode": "sky",
            "radius_arcsec": 2.5,
        }
        assert "sky" in reason.casefold()

    def test_never_joins_source_id_hash(self) -> None:
        partner, _reason = suggest_partner_match(
            ["_source_id", "RA"],
            ["_source_id", "ra"],
            partner_survey="OTHER",
        )
        assert partner["match_mode"] == "sky"
        assert partner.get("match_col_a") != "_source_id"


class TestStripAnnotations:
    def test_strips_suggestions(self) -> None:
        draft = {
            "base_catalog": "EUCLID_Q1_MER",
            "partners": [
                {
                    "survey": "GAIA_DR3_source",
                    "match_mode": "column",
                    "match_col_a": "GAIA_ID",
                    "match_col_b": "source_id",
                },
                {
                    "survey": "ALLWISE",
                    "match_mode": "sky",
                    "radius_arcsec": 1.0,
                },
            ],
            "reuse_existing": True,
            "_suggestions": {"ALLWISE": "sky: …"},
        }
        clean = strip_crossmatch_plan_annotations(draft)
        assert "_suggestions" not in clean
        assert clean["base_catalog"] == "EUCLID_Q1_MER"
        assert len(clean["partners"]) == 2


def _write_manifest(catalog_root: Path, survey: str, columns: list[str], **extra) -> None:
    catalog_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": "1",
        "survey": survey,
        "modality": "catalog",
        "columns": [{"name": c, "dtype": "string", "nullable": True, "role": "science"}
                    for c in columns],
        **extra,
    }
    (catalog_root / "schema_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class TestSuggestCrossmatchPlanLake:
    def test_end_to_end_tmp_lake(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_manifest(
            lake / "catalogs" / "EUCLID_Q1_MER",
            "EUCLID_Q1_MER",
            ["OBJECT_ID", "GAIA_ID", "right_ascension", "declination"],
            link_id_mode="column:OBJECT_ID",
        )
        _write_manifest(
            lake / "catalogs" / "GAIA_DR3_source",
            "GAIA_DR3_source",
            ["source_id", "designation", "ra", "dec"],
            link_id_mode="label:designation",
        )
        _write_manifest(
            lake / "catalogs" / "ALLWISE",
            "ALLWISE",
            ["designation", "ra", "dec", "w1mpro"],
            link_id_mode="label:designation",
        )
        plan = suggest_crossmatch_plan(
            lake,
            "EUCLID_Q1_MER",
            ["GAIA_DR3_source", "ALLWISE"],
        )
        assert plan["base_catalog"] == "EUCLID_Q1_MER"
        by_name = {p["survey"]: p for p in plan["partners"]}
        assert by_name["GAIA_DR3_source"]["match_mode"] == "column"
        assert by_name["GAIA_DR3_source"]["match_col_a"] == "GAIA_ID"
        assert by_name["GAIA_DR3_source"]["match_col_b"] == "source_id"
        assert by_name["ALLWISE"]["match_mode"] == "sky"
        assert "GAIA_DR3_source" in plan["_suggestions"]
        clean = strip_crossmatch_plan_annotations(plan)
        assert "_suggestions" not in clean
