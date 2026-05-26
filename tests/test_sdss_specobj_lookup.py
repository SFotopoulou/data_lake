"""Tests for sdss_specobj_lookup (survey-scoped plate/mjd/fiber → SPECOBJID)."""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_lake.ingest.fits_to_parquet import normalize_object_id
from data_lake.ingest.sdss_specobj_lookup import (
    build_fiber_to_specobjid_map,
    encode_sdss_run2d,
    run2d_from_spplate_header,
    sdss_specobjid_dr7_from_plate_fiber,
    sdss_specobjid_dr8plus_from_plate_fiber,
    sdss_specobjid_from_plate_fiber,
)

_CAS_EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "sdss-specobjid.txt"


def _load_cas_specobjid_examples(path: Path = _CAS_EXAMPLES) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(
            r'(\w+)\s+"?\s*(\d+)\s*"\s+(\S+)\s+(""|"[^"]*"|\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
            line,
        )
        assert m, f"unparseable CAS example line: {line!r}"
        survey, sid, run2d, _, plate, _tile, mjd, fiber = m.groups()
        rows.append({
            "survey": survey,
            "specobjid": int(sid),
            "run2d": run2d.strip('"'),
            "plate": int(plate),
            "mjd": int(mjd),
            "fiber": int(fiber),
        })
    return rows


def _write_lookup(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


class TestSpecobjIdEncoding:
    def test_dr8_reference_value(self) -> None:
        sid = sdss_specobjid_from_plate_fiber(4055, 408, 55359, "v5_7_0")
        assert sid == normalize_object_id(4565636362342690816)

    def test_dr7_and_dr8plus_differ(self) -> None:
        dr7 = sdss_specobjid_dr7_from_plate_fiber(287, 320, 52251)
        dr8 = sdss_specobjid_dr8plus_from_plate_fiber(287, 320, 52251, 26)
        assert dr7 != dr8

    def test_dr7_roundtrip_bits(self) -> None:
        plate, fiber, mjd = 287, 320, 52251
        raw = int(sdss_specobjid_dr7_from_plate_fiber(plate, fiber, mjd))
        assert (raw & 0xFFFF) == plate
        assert ((raw >> 16) & 0xFFFF) == mjd
        assert ((raw >> 32) & 0x3FF) == fiber

    def test_encode_run2d_integer_string(self) -> None:
        assert encode_sdss_run2d("26") == 26
        assert encode_sdss_run2d("v5_13_2") == 1302

    def test_run2d_header_ignores_vers2d(self) -> None:
        phdr = {"RUN2D": 26, "VERS2D": "v5_13_2", "VERSCOMB": "v5_13_2"}
        assert run2d_from_spplate_header(phdr) == 26
        sid_run2d = int(sdss_specobjid_dr8plus_from_plate_fiber(266, 15, 51602, 26))
        sid_vers2d = int(sdss_specobjid_dr8plus_from_plate_fiber(266, 15, 51602, "v5_13_2"))
        assert sid_run2d == 299493525265868800
        assert sid_vers2d != sid_run2d

    @pytest.mark.skipif(not _CAS_EXAMPLES.is_file(), reason="CAS fixture missing")
    def test_cas_examples_match_dr8plus_layout(self) -> None:
        """Real SkyServer rows in data/sdss-specobjid.txt (SDSS + eBOSS)."""
        for row in _load_cas_specobjid_examples():
            got = int(
                sdss_specobjid_dr8plus_from_plate_fiber(
                    row["plate"], row["fiber"], row["mjd"], row["run2d"],
                )
            )
            dr7 = int(
                sdss_specobjid_dr7_from_plate_fiber(
                    row["plate"], row["fiber"], row["mjd"],
                )
            )
            assert got == row["specobjid"], (
                f"{row['survey']} plate={row['plate']} mjd={row['mjd']} "
                f"fiber={row['fiber']} run2d={row['run2d']!r}"
            )
            assert dr7 != row["specobjid"]


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
    def test_catalog_join_plate_mjd_fiber_without_specobjid(self, tmp_path: Path) -> None:
        """Join on plate/mjd/fiber; ID from ``source_id`` (no specobjid column)."""
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "sdss_spec" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "plate": pa.array([3523], type=pa.int32()),
                "mjd": pa.array([55144], type=pa.int64()),
                "fiber": pa.array([501], type=pa.int16()),
                "source_id": pa.array([4242], type=pa.int64()),
                "objid": pa.array([9001], type=pa.int64()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        m = build_fiber_to_specobjid_map(
            "sdss_spec",
            3523,
            55144,
            catalog_root=lake,
        )
        assert m == {501: normalize_object_id(4242)}

    def test_catalog_photo_objid_only_requires_explicit_col(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "photo" / "Norder=5" / "Dir=0"
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
        with pytest.raises((ValueError, KeyError)):
            build_fiber_to_specobjid_map(
                "photo",
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
