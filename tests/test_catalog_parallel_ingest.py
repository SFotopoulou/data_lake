"""Tests for parallel catalog file-list ingest."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from data_lake.ingest.catalog_parallel_ingest import (
    CatalogDecodeConfig,
    ingest_catalogs_parallel,
)
import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import (
    catalog_table_to_tile_batches,
    decode_catalog_file_to_batches,
)
from data_lake.schema_registry import MANIFEST_FILENAME, load_catalog_schema_manifest


def _make_same_pixel_table(targetids: list[int], *, ra: float = 120.0, dec: float = 45.0) -> Table:
    n = len(targetids)
    return Table({
        "TARGETID": np.array(targetids, dtype=np.int64),
        "TARGET_RA": np.full(n, ra, dtype=np.float64),
        "TARGET_DEC": np.full(n, dec, dtype=np.float64),
        "Z": np.arange(n, dtype=np.float64),
    })


def _write_table_as_fits(tbl: Table, path: Path) -> None:
    tbl.write(str(path), format="fits", overwrite=True)


def _read_merged_catalog(lake_root: Path, survey: str) -> tuple[list[Path], pa.Table]:
    root = lake_root / "catalogs" / survey
    tiles = sorted(root.rglob("Npix=*.parquet"))
    merged = pa.concat_tables([pq.ParquetFile(str(p)).read() for p in tiles])
    return tiles, merged


def _thread_executor(n_workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=n_workers)


class TestDecodeCatalogFileToBatches:
    def test_partitions_by_healpix(self, tmp_path: Path) -> None:
        fits_path = tmp_path / "one.fits"
        _write_table_as_fits(_make_same_pixel_table([1, 2, 3, 4, 5]), fits_path)
        batches, sid_mode, n_rows = decode_catalog_file_to_batches(
            fits_path,
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            link_id_col="TARGETID",
        )
        assert n_rows == 5
        assert sid_mode == "column:TARGETID"
        assert len(batches) == 1
        npix, tbl = batches[0]
        assert tbl.num_rows == 5
        assert int(npix) == npix

        # Round-trip through helper used by ingest
        hp_col = "_healpix_norder5"
        prepared = tbl  # already has healpix col from decode
        assert hp_col in prepared.schema.names
        repart = catalog_table_to_tile_batches(prepared, 5)
        assert len(repart) == 1
        assert repart[0][1].num_rows == 5


class TestParallelCatalogIngest:
    def test_two_files_append_same_tile(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        fits_a = tmp_path / "a.fits"
        fits_b = tmp_path / "b.fits"
        _write_table_as_fits(_make_same_pixel_table([1, 2, 3]), fits_a)
        _write_table_as_fits(_make_same_pixel_table([4, 5]), fits_b)

        result = ingest_catalogs_parallel(
            [fits_a, fits_b],
            output_root=lake,
            survey_name="par_tile",
            n_workers=2,
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            link_id_col="TARGETID",
            tile_mode="append",
            show_progress=False,
            executor_factory=_thread_executor,
        )
        assert result["n_files_succeeded"] == 2
        assert result["n_files_failed"] == 0
        assert result["n_rows"] == 5

        _, merged = _read_merged_catalog(lake, "par_tile")
        assert merged.num_rows == 5
        assert set(np.asarray(merged.column("TARGETID")).tolist()) == {1, 2, 3, 4, 5}

        manifest_path = lake / "catalogs" / "par_tile" / MANIFEST_FILENAME
        assert manifest_path.is_file()
        manifest = load_catalog_schema_manifest(lake / "catalogs" / "par_tile")
        assert manifest["survey"] == "par_tile"
        assert manifest["total_rows"] == 5

    def test_checkpoint_skips_completed(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        fits_a = tmp_path / "a.fits"
        _write_table_as_fits(_make_same_pixel_table([1]), fits_a)
        ckpt = tmp_path / "ckpt.json"

        ingest_catalogs_parallel(
            [fits_a],
            output_root=lake,
            survey_name="ck",
            n_workers=1,
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            link_id_col="TARGETID",
            tile_mode="append",
            checkpoint_path=ckpt,
            show_progress=False,
            executor_factory=_thread_executor,
        )
        result = ingest_catalogs_parallel(
            [fits_a],
            output_root=lake,
            survey_name="ck",
            n_workers=1,
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            link_id_col="TARGETID",
            tile_mode="append",
            checkpoint_path=ckpt,
            show_progress=False,
            executor_factory=_thread_executor,
        )
        assert result["n_files_processed"] == 0
        assert result["n_files_skipped"] == 1

    def test_checkpoint_skip_still_writes_manifest(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        fits_a = tmp_path / "a.fits"
        _write_table_as_fits(_make_same_pixel_table([1]), fits_a)
        ckpt = tmp_path / "ckpt.json"
        survey_root = lake / "catalogs" / "ck"

        ingest_catalogs_parallel(
            [fits_a],
            output_root=lake,
            survey_name="ck",
            n_workers=1,
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            link_id_col="TARGETID",
            tile_mode="append",
            checkpoint_path=ckpt,
            show_progress=False,
            executor_factory=_thread_executor,
        )
        manifest_path = survey_root / MANIFEST_FILENAME
        assert manifest_path.is_file()
        manifest_path.unlink()

        ingest_catalogs_parallel(
            [fits_a],
            output_root=lake,
            survey_name="ck",
            n_workers=1,
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            link_id_col="TARGETID",
            tile_mode="append",
            checkpoint_path=ckpt,
            show_progress=False,
            executor_factory=_thread_executor,
        )
        assert manifest_path.is_file()
        manifest = load_catalog_schema_manifest(survey_root)
        assert manifest["total_rows"] == 1

    def test_failure_logged(self, tmp_path: Path) -> None:
        from data_lake.ingest.catalog_parallel_ingest import CatalogWorkerResult

        lake = tmp_path / "lake"
        bad = tmp_path / "missing.fits"
        fail_log = tmp_path / "fail.jsonl"

        def _boom(path_str: str, _cfg: CatalogDecodeConfig) -> CatalogWorkerResult:
            return CatalogWorkerResult(path=path_str, ok=False, error="synthetic fail")

        result = ingest_catalogs_parallel(
            [bad],
            output_root=lake,
            survey_name="fail",
            n_workers=1,
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            failures_log=fail_log,
            show_progress=False,
            decoder=_boom,
            executor_factory=_thread_executor,
        )
        assert result["n_files_failed"] == 1
        lines = fail_log.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["error"] == "synthetic fail"
