"""Tests for sdss_specobj_lookup (survey-scoped plate/mjd/fiber → SPECOBJID)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_lake.ingest.fits_to_parquet import normalize_object_id
from data_lake.ingest.sdss_specobj_lookup import (
    build_fiber_to_specobjid_map,
    sdss_specobjid_from_plate_fiber,
)


def _write_lookup(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


class TestSpecobjIdEncoding:
    def test_dr8_reference_value(self) -> None:
        sid = sdss_specobjid_from_plate_fiber(4055, 408, 55359, "v5_7_0")
        assert sid == normalize_object_id(4565636362342690816)


class TestSpecobjLookupSurveyScope:
    def test_filters_by_survey_in_sidecar(self, tmp_path: Path) -> None:
        lookup = tmp_path / "lookup.parquet"
        _write_lookup(
            lookup,
            [
                {
                    "survey": "sdss_dr17",
                    "PLATE": 100,
                    "MJD": 50000,
                    "FIBERID": 1,
                    "SPECOBJID": 1001,
                },
                {
                    "survey": "boss_dr12",
                    "PLATE": 100,
                    "MJD": 50000,
                    "FIBERID": 1,
                    "SPECOBJID": 2001,
                },
                {
                    "survey": "sdss_dr17",
                    "PLATE": 100,
                    "MJD": 50000,
                    "FIBERID": 2,
                    "SPECOBJID": 1002,
                },
            ],
        )
        sdss = build_fiber_to_specobjid_map(
            "sdss_dr17", 100, 50000, lookup_path=lookup,
        )
        boss = build_fiber_to_specobjid_map(
            "boss_dr12", 100, 50000, lookup_path=lookup,
        )
        assert sdss == {
            1: normalize_object_id(1001),
            2: normalize_object_id(1002),
        }
        assert boss == {1: normalize_object_id(2001)}

    def test_requires_survey_column_or_override(self, tmp_path: Path) -> None:
        lookup = tmp_path / "no_survey.parquet"
        _write_lookup(
            lookup,
            [{"PLATE": 1, "MJD": 2, "FIBERID": 3, "SPECOBJID": 99}],
        )
        with pytest.raises(ValueError, match="no survey column"):
            build_fiber_to_specobjid_map("sdss_dr17", 1, 2, lookup_path=lookup)

        m = build_fiber_to_specobjid_map(
            "sdss_dr17",
            1,
            2,
            lookup_path=lookup,
            lookup_survey="sdss_dr17",
        )
        assert m == {3: normalize_object_id(99)}

    def test_rejects_both_lookup_sources(self) -> None:
        with pytest.raises(ValueError, match="only one"):
            build_fiber_to_specobjid_map(
                "sdss_dr17",
                1,
                2,
                lookup_path="a.parquet",
                catalog_root="/tmp/lake",
            )


class TestCatalogLookup:
    def test_catalog_requires_specobjid_not_objid(self, tmp_path: Path) -> None:
        """Photometric ``objid`` must not be used as the spPlate join key."""
        lake = tmp_path / "lake"
        cat_dir = lake / "catalogs" / "sdss_spec"
        tile_dir = cat_dir / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "plate": pa.array([3523], type=pa.int32()),
                "mjd": pa.array([55144], type=pa.int64()),
                "fiber": pa.array([501], type=pa.int16()),
                "objid": pa.array([9001], type=pa.int64()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        with pytest.raises(ValueError, match="SPECOBJID"):
            build_fiber_to_specobjid_map(
                "sdss_spec",
                3523,
                55144,
                catalog_root=lake,
            )

    def test_catalog_with_specobjid_column(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "sdss_spec" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "plate": pa.array([3523, 3523], type=pa.int32()),
                "mjd": pa.array([55144.0, 55144.0], type=pa.float64()),
                "fiber": pa.array([501, 502], type=pa.int16()),
                "specobjid": pa.array([9001, 9002], type=pa.int64()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        m = build_fiber_to_specobjid_map(
            "sdss_spec",
            3523,
            55144,
            catalog_root=lake,
        )
        assert m == {501: normalize_object_id(9001), 502: normalize_object_id(9002)}
