"""Tests for flat area definitions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_lake.discovery.areas import (
    Area,
    area_is_valid,
    list_areas,
    load_area,
    make_area,
    save_area,
    validate_area,
)
from data_lake.discovery.region import Region


class TestAreaRoundtrip:
    def test_make_save_load(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        area = make_area(
            "Wide_Field_47",
            Region.cone(150.1, 2.2, 600.0),
            discover={"surveys": "all", "modalities": ["catalog", "spectra"]},
        )
        path = save_area(lake, area)
        assert path.is_file()
        assert list_areas(lake) == ["Wide_Field_47"]

        loaded = load_area(lake, "Wide_Field_47")
        assert loaded.region.type == "cone"
        assert loaded.discover_modalities == ["catalog", "spectra"]

    def test_no_overwrite_by_default(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        area = make_area("A", Region.cone(1.0, 2.0, 10.0))
        save_area(lake, area)
        with pytest.raises(FileExistsError):
            save_area(lake, area)
        save_area(lake, area, overwrite=True)  # ok


class TestAreaValidation:
    def test_valid_cone(self) -> None:
        area = make_area("A", Region.cone(1.0, 2.0, 10.0))
        assert area_is_valid(area.data)
        assert validate_area(area.data) == []

    def test_missing_region(self) -> None:
        msgs = validate_area({"area_id": "A"})
        assert any(m.startswith("ERROR") for m in msgs)

    def test_unknown_modality_is_warning(self) -> None:
        area = make_area(
            "A", Region.cone(1.0, 2.0, 10.0),
            discover={"modalities": ["catalog", "bogus"]},
        )
        msgs = validate_area(area.data)
        assert any("WARN" in m and "bogus" in m for m in msgs)
        assert area_is_valid(area.data)  # warnings don't invalidate

    def test_crossmatch_plan_requires_radius(self) -> None:
        area = make_area(
            "A", Region.cone(1.0, 2.0, 10.0),
            crossmatch_plan={"base_catalog": "EUCLID",
                             "partners": [{"survey": "DESI_DR1"}]},
        )
        msgs = validate_area(area.data)
        assert any("radius_arcsec" in m for m in msgs)

    def test_gather_multiplicity_validated(self) -> None:
        area = make_area(
            "A", Region.cone(1.0, 2.0, 10.0),
            gather={"base": "EUCLID", "multiplicity": "weird"},
        )
        msgs = validate_area(area.data)
        assert any("multiplicity" in m for m in msgs)

    def test_homogenize_block_valid(self) -> None:
        area = make_area(
            "A", Region.cone(1.0, 2.0, 10.0),
            homogenize={
                "survey": "ALLWISE",
                "transform": "phot_ab_v1",
                "materialize_as": "ALLWISE_ab",
            },
        )
        assert area_is_valid(area.data)

    def test_homogenize_requires_transform(self) -> None:
        area = make_area(
            "A", Region.cone(1.0, 2.0, 10.0),
            homogenize={"survey": "ALLWISE", "materialize_as": "x"},
        )
        msgs = validate_area(area.data)
        assert any("transform" in m for m in msgs)

    def test_homogenize_survey_and_product_mutually_exclusive(self) -> None:
        area = make_area(
            "A", Region.cone(1.0, 2.0, 10.0),
            homogenize={
                "survey": "ALLWISE",
                "from_product": "joined",
                "transform": "phot_ab_v1",
                "materialize_as": "x",
            },
        )
        msgs = validate_area(area.data)
        assert any("survey+region OR from_product" in m for m in msgs)
