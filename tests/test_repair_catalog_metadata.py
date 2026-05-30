"""Tests for dl-repair-catalog-metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import healpix_dir
from data_lake.ingest.repair_catalog_metadata import repair_catalog_metadata


def _write_id_only_tile(
    lake: Path,
    survey: str,
    *,
    norder: int,
    npix: int,
    ids: list[int],
) -> None:
    tile_dir = lake / "catalogs" / survey / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    hp_col = f"_healpix_norder{norder}"
    pq.write_table(
        pa.table({
            "id": pa.array(ids, type=pa.int64()),
            "ra": pa.array([10.0] * len(ids), type=pa.float64()),
            "dec": pa.array([0.0] * len(ids), type=pa.float64()),
            hp_col: pa.array([npix] * len(ids), type=pa.int64()),
            "_cutout_index": pa.array([-1] * len(ids), type=pa.int64()),
            "_spectrum_index": pa.array([-1] * len(ids), type=pa.int64()),
        }),
        tile_dir / f"Npix={npix}.parquet",
    )
    info = {
        "hats_order": norder,
        "ra_column": "ra",
        "dec_column": "dec",
        "source_id_mode": "sequential",
        "total_rows": len(ids),
    }
    (lake / "catalogs" / survey / "catalog_info.json").write_text(json.dumps(info))


class TestRepairCatalogMetadata:
    def test_fixes_source_id_column_from_tiles(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_id_only_tile(lake, "SURVEY_X", norder=5, npix=42, ids=[1, 2, 3])

        catalog_root = lake / "catalogs" / "SURVEY_X"
        res = repair_catalog_metadata(
            catalog_root, "SURVEY_X", migrate_join_column=True,
        )
        assert res.ok
        from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN

        assert res.source_id_column_after == LAKE_JOIN_ID_COLUMN
        assert res.source_id_mode_after == "column:id"

        info = json.loads((catalog_root / "catalog_info.json").read_text())
        assert info["source_id_column"] == LAKE_JOIN_ID_COLUMN
        assert info["source_id_mode"] == "column:id"
        assert info.get("native_id_column") == "id"
        assert (catalog_root / "_metadata").is_file()
        assert (catalog_root / "schema_manifest.json").is_file()

    def test_missing_catalog(self, tmp_path: Path) -> None:
        from data_lake.ingest.repair_catalog_metadata import repair_catalogs_under_lake

        results = repair_catalogs_under_lake(tmp_path / "lake", ["NOPE"])
        assert len(results) == 1
        assert not results[0].ok
