"""Tests for dl-area CLI and area plan helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from data_lake.discovery.area_cli import cli
from data_lake.discovery.area_plan import (
    build_crossmatch_plan,
    build_gather_block,
    parse_partner_spec,
)
from data_lake.discovery.areas import load_area, make_area, save_area
from data_lake.discovery.region import Region


@pytest.fixture
def lake(tmp_path: Path) -> Path:
    lake = tmp_path / "lake"
    area = make_area("MyCone", Region.cone(150.1, 2.2, 600.0))
    save_area(lake, area)
    return lake


class TestAreaPlanHelpers:
    def test_parse_partner(self) -> None:
        assert parse_partner_spec("DESI_DR1:1.0") == {
            "survey": "DESI_DR1",
            "match_mode": "sky",
            "radius_arcsec": 1.0,
        }

    def test_parse_column_partner(self) -> None:
        assert parse_partner_spec("DESI_DR1:col:desi_tid:TARGETID:TARGETID") == {
            "survey": "DESI_DR1",
            "match_mode": "column",
            "match_id": "desi_tid",
            "match_col_a": "TARGETID",
            "match_col_b": "TARGETID",
        }

    def test_parse_partner_rejects_bad(self) -> None:
        with pytest.raises(ValueError, match="SURVEY:RADIUS|SURVEY:col"):
            parse_partner_spec("DESI_DR1")

    def test_build_crossmatch_plan(self) -> None:
        plan = build_crossmatch_plan(
            "EUCLID",
            [
                {"survey": "DESI_DR1", "match_mode": "sky", "radius_arcsec": 1.0},
                {"survey": "ALLWISE", "match_mode": "sky", "radius_arcsec": 2.0},
            ],
        )
        assert plan["base_catalog"] == "EUCLID"
        assert len(plan["partners"]) == 2

    def test_build_crossmatch_plan_mixed(self) -> None:
        plan = build_crossmatch_plan(
            "EUCLID",
            [
                parse_partner_spec("ALLWISE:2.0"),
                parse_partner_spec("DESI_DR1:col:desi_tid:TARGETID:TARGETID"),
            ],
        )
        assert plan["partners"][0]["match_mode"] == "sky"
        assert plan["partners"][1]["match_mode"] == "column"
        assert plan["partners"][1]["match_id"] == "desi_tid"


class TestAreaCli:
    def test_list_and_show(self, lake: Path) -> None:
        runner = CliRunner()
        r = runner.invoke(cli, [str(lake), "list"])
        assert r.exit_code == 0
        assert "MyCone" in r.output

        r2 = runner.invoke(cli, [str(lake), "show", "MyCone"])
        assert r2.exit_code == 0
        assert "cone" in r2.output

    def test_set_crossmatch_column_partner(self, lake: Path) -> None:
        runner = CliRunner()
        r = runner.invoke(
            cli,
            [
                str(lake), "set-crossmatch", "MyCone",
                "--base", "EUCLID_DR1",
                "--partner", "ALLWISE:2.0",
                "--partner", "DESI_DR1:col:desi_tid:TARGETID:TARGETID",
            ],
        )
        assert r.exit_code == 0, r.output
        area = load_area(lake, "MyCone")
        partners = area.crossmatch_plan["partners"]
        assert partners[0]["match_mode"] == "sky"
        assert partners[1]["match_mode"] == "column"
        assert partners[1]["match_id"] == "desi_tid"

    def test_set_crossmatch_and_gather(self, lake: Path) -> None:
        runner = CliRunner()
        r1 = runner.invoke(
            cli,
            [
                str(lake), "set-crossmatch", "MyCone",
                "--base", "EUCLID_DR1",
                "--partner", "DESI_DR1:1.0",
                "--partner", "ALLWISE:2.0",
            ],
        )
        assert r1.exit_code == 0, r1.output

        cols = json.dumps({
            "EUCLID_DR1": ["ra", "dec"],
            "DESI_DR1": ["z"],
            "ALLWISE": ["w1mpro"],
        })
        r2 = runner.invoke(
            cli,
            [
                str(lake), "set-gather", "MyCone",
                "--base", "EUCLID_DR1",
                "--columns", cols,
                "--materialize-as", "euclid_native_v1",
            ],
        )
        assert r2.exit_code == 0, r2.output

        area = load_area(lake, "MyCone")
        assert area.crossmatch_plan["base_catalog"] == "EUCLID_DR1"
        assert area.gather["materialize_as"] == "euclid_native_v1"
        assert "DESI_DR1" in area.gather["columns"]

    def test_set_homogenize(self, lake: Path) -> None:
        runner = CliRunner()
        gather = build_gather_block(
            "EUCLID_DR1", {"EUCLID_DR1": ["ra"]}, "native_v1",
        )
        from data_lake.discovery.area_plan import update_area
        update_area(lake, "MyCone", gather=gather)

        r = runner.invoke(
            cli,
            [
                str(lake), "set-homogenize", "MyCone",
                "--from-product", "native_v1",
                "--transform", "phot_ab_v1",
                "--materialize-as", "ab_v1",
            ],
        )
        assert r.exit_code == 0, r.output
        area = load_area(lake, "MyCone")
        assert area.homogenize["from_product"] == "native_v1"
        assert area.homogenize["region"] == {"from_area": "MyCone"}

    def test_import_merge(self, lake: Path, tmp_path: Path) -> None:
        fragment = {
            "crossmatch_plan": build_crossmatch_plan(
                "ALLWISE", [parse_partner_spec("GAIA_DR3_source:0.5")],
            ),
            "notes": "imported",
        }
        frag_path = tmp_path / "frag.json"
        frag_path.write_text(json.dumps(fragment))

        runner = CliRunner()
        r = runner.invoke(
            cli, [str(lake), "import", "MyCone", "--from-file", str(frag_path)],
        )
        assert r.exit_code == 0, r.output
        area = load_area(lake, "MyCone")
        assert area.crossmatch_plan["base_catalog"] == "ALLWISE"

    def test_validate_errors(self, lake: Path) -> None:
        path = lake / "areas" / "Bad.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "area_id": "Bad",
            "schema_version": "1",
            "region": Region.cone(1.0, 2.0, 10.0).to_dict(),
            "gather": {"base": "X", "multiplicity": "weird"},
        }))
        runner = CliRunner()
        r = runner.invoke(cli, [str(lake), "validate", "Bad"])
        assert r.exit_code != 0
        assert "multiplicity" in r.output
