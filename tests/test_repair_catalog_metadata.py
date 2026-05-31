"""Tests for dl-repair-catalog-metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, healpix_dir
from data_lake.ingest.repair_catalog_metadata import repair_catalog_metadata


def _write_catalog_tile(
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
            LAKE_JOIN_ID_COLUMN: pa.array(ids, type=pa.int64()),
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
        "link_id_mode": "column:id",
        "link_id_column": LAKE_JOIN_ID_COLUMN,
        "native_id_column": "id",
        "total_rows": len(ids),
    }
    (lake / "catalogs" / survey / "catalog_info.json").write_text(json.dumps(info))


class TestRepairCatalogMetadata:
    def test_refreshes_metadata_from_tiles(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_catalog_tile(lake, "SURVEY_X", norder=5, npix=42, ids=[1, 2, 3])

        catalog_root = lake / "catalogs" / "SURVEY_X"
        (catalog_root / "catalog_info.json").write_text(
            json.dumps({"hats_order": 5, "link_id_column": "id", "link_id_mode": "sequential"})
        )
        res = repair_catalog_metadata(catalog_root, "SURVEY_X")
        assert res.ok
        assert res.link_id_column_after == LAKE_JOIN_ID_COLUMN
        assert res.link_id_mode_after == "column:id"

        info = json.loads((catalog_root / "catalog_info.json").read_text())
        assert info["link_id_column"] == LAKE_JOIN_ID_COLUMN
        assert info["link_id_mode"] == "column:id"
        assert info.get("native_id_column") == "id"
        assert (catalog_root / "_metadata").is_file()
        assert (catalog_root / "schema_manifest.json").is_file()

    def test_missing_catalog(self, tmp_path: Path) -> None:
        from data_lake.ingest.repair_catalog_metadata import repair_catalogs_under_lake

        results = repair_catalogs_under_lake(tmp_path / "lake", ["NOPE"])
        assert len(results) == 1
        assert not results[0].ok


class TestRebuildLinkId:
    def test_rebuild_link_id_filename(self, tmp_path: Path) -> None:
        """Recompute _source_id from filename; science id and indices updated."""
        from data_lake.ingest.fits_to_parquet import stable_object_id_from_string

        lake = tmp_path / "lake"
        norder = 5
        npix = 42
        tile_dir = lake / "catalogs" / "ZCOS" / healpix_dir(norder, npix)
        tile_dir.mkdir(parents=True)
        hp_col = f"_healpix_norder{norder}"
        filenames = ["foo.fits", "bar.fits"]
        expected = [stable_object_id_from_string(n) for n in filenames]
        pq.write_table(
            pa.table({
                "id": pa.array([1, 2], type=pa.int64()),
                "filename": pa.array(filenames, type=pa.string()),
                "ra": pa.array([10.0, 11.0], type=pa.float64()),
                "dec": pa.array([0.0, 0.1], type=pa.float64()),
                hp_col: pa.array([npix, npix], type=pa.int64()),
                LAKE_JOIN_ID_COLUMN: pa.array([1, 2], type=pa.int64()),
                "_cutout_index": pa.array([0, 1], type=pa.int64()),
                "_spectrum_index": pa.array([5, 10], type=pa.int64()),
            }),
            tile_dir / f"Npix={npix}.parquet",
        )
        (lake / "catalogs" / "ZCOS" / "catalog_info.json").write_text(
            json.dumps({
                "hats_order": norder,
                "ra_column": "ra",
                "dec_column": "dec",
                "link_id_mode": "column:id",
                "link_id_column": LAKE_JOIN_ID_COLUMN,
                "native_id_column": "id",
            })
        )

        catalog_root = lake / "catalogs" / "ZCOS"
        res = repair_catalog_metadata(
            catalog_root, "ZCOS", rebuild_link_id="filename",
        )
        assert res.ok, res.error
        assert res.parquet_tiles_rebuilt == 1
        assert res.link_id_mode_after == "label:filename"

        tbl = pq.ParquetFile(tile_dir / f"Npix={npix}.parquet").read()
        assert tbl.column("id").to_pylist() == [1, 2]
        assert tbl.column(LAKE_JOIN_ID_COLUMN).to_pylist() == expected
        assert tbl.column("_spectrum_index").to_pylist() == [-1, -1]
        assert tbl.column("_cutout_index").to_pylist() == [-1, -1]

        info = json.loads((catalog_root / "catalog_info.json").read_text())
        assert info["link_id_mode"] == "label:filename"
        assert info.get("native_id_column") == "filename"

    def test_rebuild_link_id_missing_column(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_catalog_tile(lake, "Y", norder=5, npix=1, ids=[1])

        res = repair_catalog_metadata(
            lake / "catalogs" / "Y",
            "Y",
            rebuild_link_id="filename",
        )
        assert not res.ok
        assert res.error is not None
        assert "filename" in res.error
