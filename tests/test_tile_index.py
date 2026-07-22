"""Tests for the per-survey tile index."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.discovery import tile_index as ti
from data_lake.ingest.fits_to_parquet import healpix_dir
from data_lake.schema_registry import MODALITY_CROSSMATCH


def _write_catalog_tile(lake: Path, survey: str, norder: int, npix: int) -> None:
    tile_dir = lake / "catalogs" / survey / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), tile_dir / f"Npix={npix}.parquet")


def _write_info(lake: Path, survey: str, norder: int) -> None:
    (lake / "catalogs" / survey / "catalog_info.json").write_text(
        json.dumps({"hats_order": norder})
    )


def _write_crossmatch_tree(lake: Path, tree: str, norder: int, npix: int) -> None:
    tile_dir = lake / "crossmatch" / tree / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), tile_dir / f"Npix={npix}.parquet")
    (lake / "crossmatch" / tree / "crossmatch_info.json").write_text(
        json.dumps({"hats_order": norder, "total_rows": 1})
    )


class TestTileIndex:
    def test_build_and_load(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        for npix in (10, 20, 30):
            _write_catalog_tile(lake, "ALLWISE", norder=5, npix=npix)
        _write_info(lake, "ALLWISE", 5)

        index = ti.write_tile_index(lake, "ALLWISE", "catalog")
        assert index["npix"] == [10, 20, 30]
        assert index["hats_order"] == 5
        assert index["n_tiles"] == 3

        loaded = ti.load_tile_index(lake, "ALLWISE", "catalog")
        assert loaded == index

    def test_survey_npix_uses_index(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        for npix in (1, 2):
            _write_catalog_tile(lake, "S", norder=4, npix=npix)
        _write_info(lake, "S", 4)
        ti.write_tile_index(lake, "S", "catalog")

        npix, order = ti.survey_npix(lake, "S", "catalog", allow_scan=False)
        assert npix == {1, 2}
        assert order == 4

    def test_survey_npix_scan_fallback(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_catalog_tile(lake, "S", norder=4, npix=7)
        _write_info(lake, "S", 4)
        # No index written -> scan fallback.
        npix, order = ti.survey_npix(lake, "S", "catalog", allow_scan=True)
        assert npix == {7}
        assert order == 4
        # Disallowing scan returns empty.
        npix2, _ = ti.survey_npix(lake, "S", "catalog", allow_scan=False)
        assert npix2 == set()

    def test_refresh_all(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_catalog_tile(lake, "A", norder=5, npix=1)
        _write_info(lake, "A", 5)
        _write_catalog_tile(lake, "B", norder=5, npix=2)
        _write_info(lake, "B", 5)
        summary = ti.refresh_tile_indices(lake)
        assert summary["catalog"] == 2
        assert ti.load_tile_index(lake, "A", "catalog")["npix"] == [1]
        assert ti.load_tile_index(lake, "B", "catalog")["npix"] == [2]

    def test_crossmatch_iter_and_refresh(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        tree = "A_x_B__r1.0"
        _write_crossmatch_tree(lake, tree, norder=5, npix=42)

        names = list(ti.iter_surveys_in_modality(lake, MODALITY_CROSSMATCH))
        assert tree in names

        summary = ti.refresh_tile_indices(lake)
        assert summary.get(MODALITY_CROSSMATCH) == 1
        index_path = ti.tile_index_path(lake, tree, MODALITY_CROSSMATCH)
        assert index_path.name == f"{tree}.crossmatch.json"
        assert index_path.is_file()
        loaded = ti.load_tile_index(lake, tree, MODALITY_CROSSMATCH)
        assert loaded is not None
        assert loaded["npix"] == [42]
        assert loaded["hats_order"] == 5
