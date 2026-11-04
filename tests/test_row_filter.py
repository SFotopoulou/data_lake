"""Tests for extract --where / --limit helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_lake.export.row_filter import (
    CUTOUT_INDEX_COLUMN,
    SPECTRUM_INDEX_COLUMN,
    load_extract_source_ids,
    source_ids_from_catalog,
    validate_predicate,
)
from data_lake.io.catalog import CatalogAccessor


SURVEY = "FILT"


def _write_filter_catalog(lake: Path) -> Path:
    tile_dir = lake / "catalogs" / SURVEY / "Norder=5" / "Dir=0"
    tile_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({
            "_source_id": pa.array([1, 2, 3, 4], type=pa.int64()),
            "SPECTYPE": pa.array(["STAR", "GALAXY", "STAR", "STAR"], type=pa.string()),
            "_spectrum_index": pa.array([0, 1, -1, 2], type=pa.int64()),
            "_cutout_index": pa.array([0, -1, 1, 2], type=pa.int64()),
            "ra": pa.array([10.0, 11.0, 12.0, 13.0], type=pa.float64()),
            "dec": pa.array([0.0, 0.1, 0.2, 0.3], type=pa.float64()),
        }),
        tile_dir / "Npix=1.parquet",
    )
    (lake / "catalogs" / SURVEY / "catalog_info.json").write_text(
        '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
        '"link_id_mode": "sequential", "link_id_column": "_source_id", '
        '"total_rows": 4}'
    )
    return lake


class TestValidatePredicate:
    def test_strips_and_accepts(self) -> None:
        assert validate_predicate("  SPECTYPE = 'STAR'  ") == "SPECTYPE = 'STAR'"

    def test_strips_leading_where(self) -> None:
        assert validate_predicate("WHERE SPECTYPE = 'STAR'") == "SPECTYPE = 'STAR'"

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_predicate("   ")

    def test_rejects_semicolon(self) -> None:
        with pytest.raises(ValueError, match="single boolean"):
            validate_predicate("SPECTYPE = 'STAR'; DROP TABLE catalog")

    def test_rejects_comments(self) -> None:
        with pytest.raises(ValueError, match="comments"):
            validate_predicate("SPECTYPE = 'STAR' -- oops")
        with pytest.raises(ValueError, match="comments"):
            validate_predicate("SPECTYPE = 'STAR' /* x */")


class TestSourceIdsFromCatalog:
    def test_filters_index_and_limit(self, tmp_path: Path) -> None:
        lake = _write_filter_catalog(tmp_path / "lake")
        cat = CatalogAccessor(lake, SURVEY)
        ids = source_ids_from_catalog(
            cat,
            where="SPECTYPE = 'STAR'",
            limit=2,
            index_column=SPECTRUM_INDEX_COLUMN,
        )
        # STAR rows: 1 (idx 0), 3 (idx -1 skipped), 4 (idx 2) → 1, 4
        assert list(ids) == [1, 4]

    def test_cutout_index_skips_negative(self, tmp_path: Path) -> None:
        lake = _write_filter_catalog(tmp_path / "lake")
        cat = CatalogAccessor(lake, SURVEY)
        ids = source_ids_from_catalog(
            cat,
            where="SPECTYPE = 'STAR'",
            index_column=CUTOUT_INDEX_COLUMN,
        )
        # STAR with _cutout_index >= 0: 1 (0), 3 (1), 4 (2)
        assert list(ids) == [1, 3, 4]

    def test_missing_index_column_is_skipped(self, tmp_path: Path) -> None:
        lake = _write_filter_catalog(tmp_path / "lake")
        cat = CatalogAccessor(lake, SURVEY)
        ids = source_ids_from_catalog(
            cat,
            where="SPECTYPE = 'STAR'",
            index_column="_not_a_column",
        )
        assert list(ids) == [1, 3, 4]


class TestLoadExtractSourceIds:
    def test_rejects_both_and_neither(self, tmp_path: Path) -> None:
        lake = _write_filter_catalog(tmp_path / "lake")
        cat = CatalogAccessor(lake, SURVEY)
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("1\n")

        def _read(path, col):
            return np.loadtxt(str(path), dtype=np.int64, ndmin=1)

        with pytest.raises(ValueError, match="not both"):
            load_extract_source_ids(
                catalog_accessor=cat,
                target_list=ids_file,
                target_id_col="_source_id",
                where_sql="SPECTYPE = 'STAR'",
                row_limit=None,
                index_column=SPECTRUM_INDEX_COLUMN,
                read_target_ids=_read,
            )
        with pytest.raises(ValueError, match="Pass --target-list or --where"):
            load_extract_source_ids(
                catalog_accessor=cat,
                target_list=None,
                target_id_col="_source_id",
                where_sql=None,
                row_limit=None,
                index_column=SPECTRUM_INDEX_COLUMN,
                read_target_ids=_read,
            )

    def test_truncates_target_list(self, tmp_path: Path) -> None:
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("10\n20\n30\n")

        def _read(path, col):
            return np.loadtxt(str(path), dtype=np.int64, ndmin=1)

        ids = load_extract_source_ids(
            catalog_accessor=None,
            target_list=ids_file,
            target_id_col="_source_id",
            where_sql=None,
            row_limit=2,
            index_column=None,
            read_target_ids=_read,
        )
        assert list(ids) == [10, 20]
