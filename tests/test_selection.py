"""Tests for the base-source selection abstraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.discovery.region import Region
from data_lake.discovery.selection import (
    read_ids_file,
    selection_from_ids,
    selection_from_region,
    selection_from_where,
)
from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, assign_healpix, healpix_dir


def _write_base(lake: Path, survey: str, norder: int) -> dict[int, int]:
    """Write a base catalog with a few sources across two tiles. Returns {sid: npix}."""
    pts = [
        (1, 120.0, 45.0, 5.0),
        (2, 120.001, 45.001, 9.0),
        (3, 10.0, -30.0, 1.0),
    ]
    by_tile: dict[int, list[tuple]] = {}
    sid_npix: dict[int, int] = {}
    for sid, ra, dec, mag in pts:
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        by_tile.setdefault(npix, []).append((sid, ra, dec, mag))
        sid_npix[sid] = npix

    hp_col = f"_healpix_norder{norder}"
    for npix, rows in by_tile.items():
        tile_dir = lake / "catalogs" / survey / healpix_dir(norder, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({
                LAKE_JOIN_ID_COLUMN: pa.array([r[0] for r in rows], type=pa.int64()),
                "ra": pa.array([r[1] for r in rows], type=pa.float64()),
                "dec": pa.array([r[2] for r in rows], type=pa.float64()),
                "mag": pa.array([r[3] for r in rows], type=pa.float64()),
                hp_col: pa.array([npix] * len(rows), type=pa.int64()),
            }),
            tile_dir / f"Npix={npix}.parquet",
        )
    (lake / "catalogs" / survey / "catalog_info.json").write_text(
        json.dumps({
            "hats_order": norder,
            "ra_column": "ra",
            "dec_column": "dec",
            "link_id_mode": "sequential",
            "link_id_column": LAKE_JOIN_ID_COLUMN,
            "total_rows": len(pts),
        })
    )
    return sid_npix


class TestSelectionFromRegion:
    def test_tile_granular(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        sid_npix = _write_base(lake, "BASE", 5)
        region = Region.cone(120.0, 45.0, radius_arcsec=30.0)
        sel = selection_from_region(lake, "BASE", region)
        assert sel.is_tile_granular
        assert sel.source_ids is None
        assert sid_npix[1] in sel.npix


class TestSelectionFromIds:
    def test_lookup_npix(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        sid_npix = _write_base(lake, "BASE", 5)
        sel = selection_from_ids(lake, "BASE", [1, 3])
        assert sorted(sel.source_ids) == [1, 3]
        assert sel.npix == {sid_npix[1], sid_npix[3]}


class TestSelectionFromWhere:
    def test_predicate_base_only(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        sid_npix = _write_base(lake, "BASE", 5)
        sel = selection_from_where(lake, "BASE", "mag > 4.0")
        assert sorted(sel.source_ids) == [1, 2]
        assert sel.npix == {sid_npix[1]}  # sources 1,2 share a tile

    def test_where_prefix_stripped(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_base(lake, "BASE", 5)
        sel = selection_from_where(lake, "BASE", "WHERE mag < 2.0")
        assert sel.source_ids == [3]


class TestReadIdsFile:
    def test_parquet(self, tmp_path: Path) -> None:
        p = tmp_path / "ids.parquet"
        pq.write_table(pa.table({"source_id": pa.array([10, 20, 30], type=pa.int64())}), p)
        assert read_ids_file(p) == [10, 20, 30]

    def test_csv_explicit_col(self, tmp_path: Path) -> None:
        p = tmp_path / "ids.csv"
        p.write_text("name,sid\na,1\nb,2\n")
        assert read_ids_file(p, id_col="sid") == [1, 2]
