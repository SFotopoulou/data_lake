"""Tests for reconcile_catalog_column_names and the repair CLI flag."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_lake.ingest.fits_to_parquet import (
    LAKE_JOIN_ID_COLUMN,
    assign_healpix,
    healpix_dir,
)


def _write_padded_tile(catalog_root: Path, norder: int, npix: int) -> Path:
    """Write one Npix=*.parquet tile with padded column names."""
    tile_dir = catalog_root / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_path = tile_dir / f"Npix={npix}.parquet"
    hp_col = f"_healpix_norder{norder}"
    table = pa.table({
        LAKE_JOIN_ID_COLUMN: pa.array([1, 2], type=pa.int64()),
        " ra": pa.array([120.0, 121.0], type=pa.float64()),
        " dec": pa.array([45.0, 45.5], type=pa.float64()),
        hp_col: pa.array([npix, npix], type=pa.int64()),
        "_cutout_index": pa.array([-1, -1], type=pa.int64()),
        "_spectrum_index": pa.array([-1, -1], type=pa.int64()),
    })
    pq.write_table(table, tile_path)
    return tile_path


class TestReconcileCatalogColumnNames:
    def test_rewrites_padded_tiles(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import reconcile_catalog_column_names

        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        catalog_root = tmp_path / "catalogs" / "ALLWISE"
        tile_path = _write_padded_tile(catalog_root, norder, npix)

        assert pq.read_schema(str(tile_path)).names[1] == " ra"

        n = reconcile_catalog_column_names(catalog_root)
        assert n == 1
        names = pq.read_schema(str(tile_path)).names
        assert "ra" in names
        assert " ra" not in names
        assert "dec" in names
        assert " dec" not in names

    def test_check_only_does_not_rewrite(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import reconcile_catalog_column_names

        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        catalog_root = tmp_path / "catalogs" / "ALLWISE"
        tile_path = _write_padded_tile(catalog_root, norder, npix)

        n = reconcile_catalog_column_names(catalog_root, check_only=True)
        assert n == 1
        # File must NOT have been rewritten
        assert " ra" in pq.read_schema(str(tile_path)).names

    def test_no_op_on_clean_names(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import reconcile_catalog_column_names

        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        catalog_root = tmp_path / "catalogs" / "DESI"
        tile_dir = catalog_root / healpix_dir(norder, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path = tile_dir / f"Npix={npix}.parquet"
        clean = pa.table({"_source_id": pa.array([1], type=pa.int64()),
                          "ra": pa.array([120.0]), "dec": pa.array([45.0])})
        pq.write_table(clean, tile_path)

        n = reconcile_catalog_column_names(catalog_root)
        assert n == 0

    def test_has_padded_column_names(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import has_padded_column_names

        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        catalog_root = tmp_path / "catalogs" / "ALLWISE"
        _write_padded_tile(catalog_root, norder, npix)

        padded = has_padded_column_names(catalog_root)
        assert " ra" in padded
        assert " dec" in padded


class TestRepairCatalogMetadataNormalizeFlag:
    def test_normalize_column_names_flag(self, tmp_path: Path) -> None:
        from data_lake.ingest.repair_catalog_metadata import repair_catalog_metadata

        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        catalog_root = tmp_path / "catalogs" / "ALLWISE"
        _write_padded_tile(catalog_root, norder, npix)
        catalog_root.joinpath("catalog_info.json").write_text(json.dumps({
            "hats_order": norder,
            "ra_column": "ra",
            "dec_column": "dec",
            "link_id_mode": "sequential",
            "link_id_column": LAKE_JOIN_ID_COLUMN,
            "total_rows": 2,
            "total_columns": 6,
        }))

        result = repair_catalog_metadata(
            catalog_root, "ALLWISE",
            normalize_column_names=True,
        )
        assert result.ok
        assert result.tiles_column_renamed == 1
        names = pq.read_schema(str(
            catalog_root / healpix_dir(norder, npix) / f"Npix={npix}.parquet"
        )).names
        assert "ra" in names
        assert " ra" not in names
