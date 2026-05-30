"""
Tests for update_catalog_indices: resolve_source_id_column, update_index_column,
and build_index_map_from_zarr.

Key cases covered:
  - catalog with TARGETID (DESI-style) – the main production bug
  - catalog with sequential source_id (regression guard)
  - update_index_column uses the correct column from catalog_info.json
  - zero-match warning path (wrong survey name)
  - build_index_map_from_zarr reconstructs the mapping from Zarr on disk
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.table import Table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_mini_catalog(
    catalog_root: Path,
    survey_name: str,
    id_col: str,
    ids: list[int],
    ra: list[float],
    dec: list[float],
    norder: int = 5,
    source_id_mode: str | None = None,
) -> None:
    """Write a minimal HEALPix-partitioned Parquet catalog for testing."""
    from data_lake.ingest.fits_to_parquet import (
        LAKE_JOIN_ID_COLUMN,
        assign_healpix,
        healpix_dir,
        _ZSTD_LEVEL,
    )

    ids_arr = np.array(ids, dtype=np.int64)
    ra_arr = np.array(ra, dtype=np.float64)
    dec_arr = np.array(dec, dtype=np.float64)
    pix_arr = assign_healpix(ra_arr, dec_arr, norder)
    hp_col = f"_healpix_norder{norder}"

    cols: dict = {
        LAKE_JOIN_ID_COLUMN: pa.array(ids_arr, type=pa.int64()),
        "ra": pa.array(ra_arr, type=pa.float64()),
        "dec": pa.array(dec_arr, type=pa.float64()),
        hp_col: pa.array(pix_arr, type=pa.int64()),
        "_cutout_index": pa.array(np.full(len(ids), -1, dtype=np.int64), type=pa.int64()),
        "_spectrum_index": pa.array(np.full(len(ids), -1, dtype=np.int64), type=pa.int64()),
    }
    if id_col != LAKE_JOIN_ID_COLUMN:
        cols[id_col] = pa.array(ids_arr, type=pa.int64())
    table = pa.table(cols)

    survey_dir = catalog_root / "catalogs" / survey_name
    unique_pixels = np.unique(pix_arr)
    for npix in unique_pixels.tolist():
        mask = pix_arr == npix
        tile_table = table.filter(pa.array(mask))
        tile_dir = survey_dir / healpix_dir(norder, int(npix))
        tile_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            tile_table,
            str(tile_dir / f"Npix={int(npix)}.parquet"),
            compression="zstd",
            compression_level=_ZSTD_LEVEL,
        )

    sid_mode = source_id_mode or (
        f"column:{id_col}" if id_col != LAKE_JOIN_ID_COLUMN else "sequential"
    )
    info = {
        "catalog_name": survey_name,
        "hats_order": norder,
        "ra_column": "ra",
        "dec_column": "dec",
        "source_id_mode": sid_mode,
        "source_id_column": LAKE_JOIN_ID_COLUMN,
    }
    if id_col != LAKE_JOIN_ID_COLUMN:
        info["native_id_column"] = id_col
    with open(survey_dir / "catalog_info.json", "w") as fh:
        json.dump(info, fh)


def _write_mini_spectra_zarr(
    lake_root: Path,
    survey_name: str,
    ids: list[int],
    ra: list[float],
    dec: list[float],
    norder: int = 5,
) -> dict[int, int]:
    """Write synthetic Zarr spectrum tiles and return the expected index_map."""
    import zarr
    from data_lake.ingest.fits_to_parquet import assign_healpix, healpix_dir
    from data_lake.ingest.zarr_ids import create_zarr_join_array, zarr_join_array

    ids_arr = np.array(ids, dtype=np.int64)
    ra_arr = np.array(ra, dtype=np.float64)
    dec_arr = np.array(dec, dtype=np.float64)
    pix_arr = assign_healpix(ra_arr, dec_arr, norder)

    survey_root = lake_root / "spectra" / survey_name
    index_map: dict[int, int] = {}
    tile_buffers: dict[int, list[int]] = {}
    for sid, npix in zip(ids_arr.tolist(), pix_arr.tolist()):
        tile_buffers.setdefault(npix, []).append(sid)

    for npix, tile_ids in tile_buffers.items():
        tile_dir = survey_root / healpix_dir(norder, int(npix))
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path = tile_dir / f"Npix={int(npix)}.zarr"
        store = zarr.storage.LocalStore(str(tile_path))
        root = zarr.open_group(store=store, mode="w", zarr_format=3)
        create_zarr_join_array(root, shape=(0,), chunks=(4096,), dtype=np.int64, fill_value=-1)
        zarr_join_array(root).append(np.array(tile_ids, dtype=np.int64))
        for local_i, sid in enumerate(tile_ids):
            index_map[sid] = local_i

    return index_map


# ---------------------------------------------------------------------------
# resolve_source_id_column
# ---------------------------------------------------------------------------

class TestNormalizeObjectId:
    def test_int_and_numpy_int64(self) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id

        assert normalize_object_id(39627658462934656) == 39627658462934656
        assert normalize_object_id(np.int64(39627658462934656)) == 39627658462934656

    def test_string_digits(self) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id

        assert normalize_object_id("39627658462934656") == 39627658462934656

    def test_alphanumeric_label_hashed(self) -> None:
        from data_lake.ingest.fits_to_parquet import (
            normalize_object_id,
            stable_object_id_from_string,
        )

        label = "J000000.00-314627.5"
        h = stable_object_id_from_string(label)
        assert normalize_object_id(label) == h
        assert normalize_object_id(label) == normalize_object_id(label)

    def test_float_rejected(self) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id

        with pytest.raises(ValueError, match="floating-point"):
            normalize_object_id(39627658462934656.0)

    def test_none_rejected(self) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id

        with pytest.raises(ValueError, match="None"):
            normalize_object_id(None)

    def test_sdss_uint64_range_objid(self) -> None:
        from data_lake.ingest.fits_to_parquet import (
            cast_object_id_column_to_int64,
            normalize_object_id,
            storage_int64_from_integer,
        )

        v = 9223372435012999168
        expected = int(np.int64(np.uint64(v)))
        assert storage_int64_from_integer(v) == expected
        assert normalize_object_id(str(v)) == expected
        assert normalize_object_id(np.uint64(v)) == expected
        col = pa.array([v], type=pa.uint64())
        assert cast_object_id_column_to_int64(col).to_pylist() == [expected]


class TestResolveSourceIdColumn:
    def test_targetid_catalog(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, resolve_source_id_column

        info = {"source_id_mode": "column:TARGETID", "source_id_column": LAKE_JOIN_ID_COLUMN}
        (tmp_path / "catalog_info.json").write_text(json.dumps(info))
        assert resolve_source_id_column(tmp_path) == LAKE_JOIN_ID_COLUMN

    def test_sequential_catalog(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, resolve_source_id_column

        info = {"source_id_mode": "sequential", "source_id_column": LAKE_JOIN_ID_COLUMN}
        (tmp_path / "catalog_info.json").write_text(json.dumps(info))
        assert resolve_source_id_column(tmp_path) == LAKE_JOIN_ID_COLUMN

    def test_missing_info_file_defaults_to_source_id(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, resolve_source_id_column

        assert resolve_source_id_column(tmp_path) == LAKE_JOIN_ID_COLUMN

    def test_arbitrary_column_name(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, resolve_source_id_column

        info = {
            "source_id_mode": "column:OBJ_ID",
            "source_id_column": LAKE_JOIN_ID_COLUMN,
            "native_id_column": "OBJ_ID",
        }
        (tmp_path / "catalog_info.json").write_text(json.dumps(info))
        assert resolve_source_id_column(tmp_path) == LAKE_JOIN_ID_COLUMN

    def test_schema_fallback_targetid_when_info_says_sequential(self, tmp_path: Path) -> None:
        """Tiles with TARGETID + _source_id resolve to the lake join column."""
        from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, resolve_source_id_column

        info = {"source_id_mode": "sequential"}
        (tmp_path / "catalog_info.json").write_text(json.dumps(info))
        schema_names = [
            "TARGETID", LAKE_JOIN_ID_COLUMN, "ra", "dec",
            "_healpix_norder5", "_spectrum_index",
        ]
        assert resolve_source_id_column(tmp_path, schema_names=schema_names) == LAKE_JOIN_ID_COLUMN

    def test_schema_fallback_id_when_info_says_sequential(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, resolve_source_id_column

        info = {"source_id_mode": "sequential", "source_id_column": "source_id"}
        (tmp_path / "catalog_info.json").write_text(json.dumps(info))
        schema_names = ["id", LAKE_JOIN_ID_COLUMN, "ALPHA_J2000", "DELTA_J2000", "_healpix_norder5"]
        assert resolve_source_id_column(tmp_path, schema_names=schema_names) == LAKE_JOIN_ID_COLUMN

    def test_recorded_source_id_column_in_info(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, resolve_source_id_column

        info = {
            "source_id_mode": "column:id",
            "source_id_column": LAKE_JOIN_ID_COLUMN,
            "native_id_column": "id",
        }
        (tmp_path / "catalog_info.json").write_text(json.dumps(info))
        schema_names = ["id", LAKE_JOIN_ID_COLUMN, "ra", "dec"]
        assert resolve_source_id_column(tmp_path, schema_names=schema_names) == LAKE_JOIN_ID_COLUMN


# ---------------------------------------------------------------------------
# update_index_column – TARGETID catalog (the production bug)
# ---------------------------------------------------------------------------

class TestUpdateIndexColumnTargetId:
    def test_patches_spectrum_index_via_targetid(self, tmp_path: Path) -> None:
        """update_index_column must update _spectrum_index when ID col is TARGETID."""
        from data_lake.ingest.update_catalog_indices import update_index_column

        ids = [1_000_001, 1_000_002, 1_000_003]
        ra = [10.0, 20.0, 30.0]
        dec = [5.0, -5.0, 15.0]
        _write_mini_catalog(tmp_path, "desi_test", "TARGETID", ids, ra, dec)

        index_map = {ids[0]: 0, ids[1]: 1, ids[2]: 2}
        n_modified = update_index_column(
            lake_root=tmp_path,
            survey_name="desi_test",
            source_id_to_index=index_map,
            kind="spectrum",
        )
        assert n_modified > 0

        # Verify the written values (read each file directly to avoid Hive partition discovery)
        tiles = list((tmp_path / "catalogs" / "desi_test").rglob("Npix=*.parquet"))
        merged = pa.concat_tables([pq.ParquetFile(str(t)).read() for t in tiles])
        targetids = merged.column("TARGETID").to_pylist()
        spec_idx = merged.column("_spectrum_index").to_pylist()
        for tid, expected_idx in index_map.items():
            row_i = targetids.index(tid)
            assert spec_idx[row_i] == expected_idx, (
                f"TARGETID={tid}: expected _spectrum_index={expected_idx}, "
                f"got {spec_idx[row_i]}"
            )

    def test_patches_cutout_index_via_targetid(self, tmp_path: Path) -> None:
        from data_lake.ingest.update_catalog_indices import update_index_column

        ids = [2_000_001, 2_000_002]
        ra = [45.0, 90.0]
        dec = [10.0, -10.0]
        _write_mini_catalog(tmp_path, "survey_cut", "TARGETID", ids, ra, dec)

        index_map = {ids[0]: 7, ids[1]: 3}
        n_modified = update_index_column(
            lake_root=tmp_path,
            survey_name="survey_cut",
            source_id_to_index=index_map,
            kind="cutout",
        )
        assert n_modified > 0

        tiles = list((tmp_path / "catalogs" / "survey_cut").rglob("Npix=*.parquet"))
        merged = pa.concat_tables([pq.ParquetFile(str(t)).read() for t in tiles])
        targetids = merged.column("TARGETID").to_pylist()
        cut_idx = merged.column("_cutout_index").to_pylist()
        for tid, expected_idx in index_map.items():
            row_i = targetids.index(tid)
            assert cut_idx[row_i] == expected_idx


# ---------------------------------------------------------------------------
# update_index_column – sequential source_id catalog (regression guard)
# ---------------------------------------------------------------------------

class TestUpdateIndexColumnSourceId:
    def test_patches_via_source_id_column(self, tmp_path: Path) -> None:
        from data_lake.ingest.update_catalog_indices import update_index_column

        ids = [0, 1, 2, 3]
        ra = [5.0, 15.0, 25.0, 35.0]
        dec = [0.0, 0.0, 0.0, 0.0]
        _write_mini_catalog(
            tmp_path, "seq_survey", "source_id", ids, ra, dec,
            source_id_mode="sequential",
        )

        index_map = {0: 10, 2: 20}
        n_modified = update_index_column(
            lake_root=tmp_path,
            survey_name="seq_survey",
            source_id_to_index=index_map,
            kind="spectrum",
        )
        assert n_modified > 0

        tiles = list((tmp_path / "catalogs" / "seq_survey").rglob("Npix=*.parquet"))
        merged = pa.concat_tables([pq.ParquetFile(str(t)).read() for t in tiles])
        sids = merged.column("source_id").to_pylist()
        spec_idx = merged.column("_spectrum_index").to_pylist()
        assert spec_idx[sids.index(0)] == 10
        assert spec_idx[sids.index(2)] == 20
        assert spec_idx[sids.index(1)] == -1
        assert spec_idx[sids.index(3)] == -1

    def test_no_match_returns_zero_and_warns(self, tmp_path: Path, caplog) -> None:
        """When no tiles match, n_modified==0 and a warning is emitted."""
        import logging
        from data_lake.ingest.update_catalog_indices import update_index_column

        ids = [0, 1]
        ra = [5.0, 15.0]
        dec = [0.0, 0.0]
        _write_mini_catalog(
            tmp_path, "seq_survey", "source_id", ids, ra, dec,
            source_id_mode="sequential",
        )

        # IDs that don't exist in the catalog
        index_map = {9999: 0, 8888: 1}
        with caplog.at_level(logging.WARNING):
            n_modified = update_index_column(
                lake_root=tmp_path,
                survey_name="seq_survey",
                source_id_to_index=index_map,
                kind="spectrum",
            )
        assert n_modified == 0
        assert any("No catalog tiles were modified" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# build_index_map_from_zarr + update roundtrip
# ---------------------------------------------------------------------------

class TestUpdateIndexFromZarrTiles:
    def test_patches_catalog_without_full_index_map(self, tmp_path: Path) -> None:
        """Per-tile Zarr scan patches catalog without a global source_id dict."""
        from data_lake.ingest.update_catalog_indices import (
            update_index_column_from_zarr_tiles,
        )

        ids = [5_000_001, 5_000_002, 5_000_003]
        ra = [50.0, 60.0, 70.0]
        dec = [20.0, -20.0, 5.0]
        _write_mini_catalog(tmp_path, "desi_tile_patch", "TARGETID", ids, ra, dec)
        _write_mini_spectra_zarr(tmp_path, "desi_tile_patch", ids, ra, dec)

        n_modified = update_index_column_from_zarr_tiles(
            lake_root=tmp_path,
            survey_name="desi_tile_patch",
            kind="spectrum",
        )
        assert n_modified > 0

        tiles = list((tmp_path / "catalogs" / "desi_tile_patch").rglob("Npix=*.parquet"))
        merged = pa.concat_tables([pq.ParquetFile(str(t)).read() for t in tiles])
        spec_idx = merged.column("_spectrum_index").to_pylist()
        assert all(v >= 0 for v in spec_idx)


class TestBuildIndexMapFromZarr:
    def test_roundtrip_via_zarr_scan(self, tmp_path: Path) -> None:
        """Simulate a lake where spectra exist but catalog was never patched.

        build_index_map_from_zarr must reconstruct the index_map; then
        update_index_column must correctly patch TARGETID catalog tiles.
        """
        from data_lake.ingest.update_catalog_indices import (
            build_index_map_from_zarr,
            update_index_column,
        )

        ids = [5_000_001, 5_000_002, 5_000_003]
        ra = [50.0, 60.0, 70.0]
        dec = [20.0, -20.0, 5.0]

        # 1. Write catalog (TARGETID)
        _write_mini_catalog(tmp_path, "desi_backfill", "TARGETID", ids, ra, dec)

        # 2. Write Zarr tiles (simulating a prior ingest run with no catalog patch)
        written_map = _write_mini_spectra_zarr(tmp_path, "desi_backfill", ids, ra, dec)

        # 3. Reconstruct map from Zarr
        recovered_map = build_index_map_from_zarr(tmp_path, "desi_backfill", kind="spectrum")
        assert set(recovered_map.keys()) == set(written_map.keys())
        for sid in written_map:
            assert recovered_map[sid] == written_map[sid]

        # 4. Patch catalog using the recovered map
        n_modified = update_index_column(
            lake_root=tmp_path,
            survey_name="desi_backfill",
            source_id_to_index=recovered_map,
            kind="spectrum",
        )
        assert n_modified > 0

        # 5. Verify all rows are patched
        tiles = list((tmp_path / "catalogs" / "desi_backfill").rglob("Npix=*.parquet"))
        merged = pa.concat_tables([pq.ParquetFile(str(t)).read() for t in tiles])
        spec_idx = merged.column("_spectrum_index").to_pylist()
        assert all(v >= 0 for v in spec_idx), f"Some _spectrum_index still -1: {spec_idx}"
