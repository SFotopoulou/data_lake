"""Tests for per-process DuckDB thread / memory limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_lake.io import duckdb_limits as dl
from data_lake.io.duckdb_limits import (
    MIB,
    MIN_MEMORY_LIMIT_BYTES,
    DuckDBLimits,
    apply_limits,
    connect,
    current_limits,
    per_worker_limits,
    process_limits,
    set_process_limits,
)
from tests.test_crossmatch import _write_catalog_tile

GIB = 1024 * MIB


def _settings(con) -> tuple[int, str]:
    threads, mem = con.execute(
        "SELECT current_setting('threads'), current_setting('memory_limit')"
    ).fetchone()
    return int(threads), str(mem)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(dl.THREADS_ENV, raising=False)
    monkeypatch.delenv(dl.MEMORY_LIMIT_ENV, raising=False)
    set_process_limits(None)
    yield
    set_process_limits(None)


@pytest.fixture
def node_16cpu_64gib(monkeypatch):
    monkeypatch.setattr(dl, "available_cpus", lambda: 16)
    monkeypatch.setattr(dl, "available_memory_bytes", lambda: 64 * GIB)


class TestPerWorkerLimits:
    def test_splits_cpus_and_memory(self, node_16cpu_64gib) -> None:
        limits = per_worker_limits(4, memory_fraction=0.25)
        assert limits.threads == 4
        assert limits.memory_limit_bytes == 4 * GIB

    def test_more_workers_than_cpus_keeps_one_thread(self, node_16cpu_64gib) -> None:
        assert per_worker_limits(64).threads == 1

    def test_memory_floor(self, node_16cpu_64gib) -> None:
        limits = per_worker_limits(64, memory_budget_bytes=GIB, memory_fraction=0.01)
        assert limits.memory_limit_bytes == MIN_MEMORY_LIMIT_BYTES

    def test_explicit_budget_wins_over_detected(self, node_16cpu_64gib) -> None:
        limits = per_worker_limits(2, memory_budget_bytes=8 * GIB, memory_fraction=0.5)
        assert limits.memory_limit_bytes == 2 * GIB

    def test_env_overrides_computed(self, node_16cpu_64gib, monkeypatch) -> None:
        monkeypatch.setenv(dl.THREADS_ENV, "3")
        monkeypatch.setenv(dl.MEMORY_LIMIT_ENV, "700")
        limits = per_worker_limits(4)
        assert limits == DuckDBLimits(threads=3, memory_limit_bytes=700 * MIB)

    def test_bad_fraction(self, node_16cpu_64gib) -> None:
        with pytest.raises(ValueError, match="memory_fraction"):
            per_worker_limits(2, memory_fraction=0)


class TestDetection:
    def test_slurm_mem_per_node(self, monkeypatch) -> None:
        monkeypatch.setenv("SLURM_MEM_PER_NODE", "32768")
        assert dl._slurm_memory_limit() == 32 * GIB

    def test_slurm_mem_per_cpu(self, monkeypatch) -> None:
        monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
        monkeypatch.setenv("SLURM_MEM_PER_CPU", "4096")
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
        assert dl._slurm_memory_limit() == 32 * GIB

    def test_detected_memory_is_positive(self) -> None:
        assert dl.available_memory_bytes() > 0
        assert dl.available_cpus() >= 1


class TestExtractDuckDBLimits:
    def test_caps_threads_on_large_node(self, node_16cpu_64gib) -> None:
        limits = dl.extract_duckdb_limits(n_workers=1)
        assert limits.threads == 4  # EXTRACT_DUCKDB_THREADS default
        assert limits.memory_limit_bytes > 0

    def test_splits_across_workers_then_caps(self, node_16cpu_64gib) -> None:
        # 16 CPUs / 8 workers = 2 threads each, under the extract cap.
        assert dl.extract_duckdb_limits(n_workers=8).threads == 2

    def test_env_overrides_extract_cap(self, node_16cpu_64gib, monkeypatch) -> None:
        monkeypatch.setenv(dl.THREADS_ENV, "7")
        assert dl.extract_duckdb_limits(n_workers=1).threads == 7

    def test_process_limits_preferred(self, node_16cpu_64gib) -> None:
        with process_limits(DuckDBLimits(1, 256 * MIB)):
            assert dl.extract_duckdb_limits(n_workers=1) == DuckDBLimits(1, 256 * MIB)


class TestConnections:
    def test_apply_limits(self) -> None:
        import duckdb

        con = apply_limits(duckdb.connect(":memory:"), DuckDBLimits(2, 512 * MIB))
        assert _settings(con) == (2, "512.0 MiB")

    def test_process_limits_scope(self) -> None:
        with process_limits(DuckDBLimits(3, 300 * MIB)):
            assert _settings(connect())[0] == 3
        assert current_limits() == DuckDBLimits()

    def test_env_used_without_process_limits(self, monkeypatch) -> None:
        monkeypatch.setenv(dl.THREADS_ENV, "2")
        assert _settings(connect())[0] == 2

    def test_catalog_accessor_uses_process_limits(self, tmp_path: Path) -> None:
        from data_lake.io.catalog import CatalogAccessor

        lake = tmp_path / "lake"
        _write_catalog_tile(lake, "A", norder=5, npix=100, source_ids=[1], ra=[30.0], dec=[5.0])
        with process_limits(DuckDBLimits(2, 512 * MIB)):
            with CatalogAccessor(lake, "A") as acc:
                assert _settings(acc._con) == (2, "512.0 MiB")


def _metadata_cache_on(con) -> bool:
    value = con.execute("SELECT current_setting('parquet_metadata_cache')").fetchone()[0]
    return str(value).lower() == "true"


class TestParquetMetadataCache:
    def test_connect_default_off(self) -> None:
        assert not _metadata_cache_on(connect())
        assert _metadata_cache_on(connect(parquet_metadata_cache=True))

    @pytest.mark.parametrize(
        ("env", "expected"), [(None, True), ("0", False), ("off", False), ("1", True)],
    )
    def test_accessor_default_and_env(self, tmp_path: Path, monkeypatch, env, expected) -> None:
        from data_lake.io.catalog import CatalogAccessor

        if env is None:
            monkeypatch.delenv("DATA_LAKE_PARQUET_METADATA_CACHE", raising=False)
        else:
            monkeypatch.setenv("DATA_LAKE_PARQUET_METADATA_CACHE", env)
        lake = tmp_path / "lake"
        _write_catalog_tile(lake, "A", norder=5, npix=100, source_ids=[1], ra=[30.0], dec=[5.0])
        with CatalogAccessor(lake, "A") as acc:
            assert _metadata_cache_on(acc._con) is expected
        with CatalogAccessor(lake, "A", metadata_cache=False) as acc:
            assert not _metadata_cache_on(acc._con)


class TestCrossmatchWorkers:
    def test_worker_installs_config_limits(self, tmp_path: Path) -> None:
        from data_lake.io.crossmatch import CrossmatchTileConfig, _crossmatch_tile_worker

        lake = tmp_path / "lake"
        _write_catalog_tile(lake, "A", norder=5, npix=100, source_ids=[1], ra=[30.0], dec=[5.0])
        _write_catalog_tile(lake, "B", norder=5, npix=100, source_ids=[2], ra=[30.0], dec=[5.0])
        limits = DuckDBLimits(1, 256 * MIB)
        res = _crossmatch_tile_worker(
            CrossmatchTileConfig(
                lake_root=str(lake), survey_a="A", survey_b="B", npix_a_list=(100,),
                norder_a=5, norder_b=5, ra_col_a="ra", dec_col_a="dec",
                ra_col_b="ra", dec_col_b="dec", radius_arcsec=1.0,
                out_root=str(tmp_path / "xm"), duckdb_limits=limits,
            )
        )
        assert res.error is None
        assert current_limits() == limits

    def test_parallel_build_with_budget(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import assign_healpix
        from data_lake.io.crossmatch import build_crossmatch
        import numpy as np

        lake = tmp_path / "lake"
        ra, dec = 30.0, 5.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), 5)[0])
        _write_catalog_tile(lake, "A", norder=5, npix=npix, source_ids=[1], ra=[ra], dec=[dec])
        _write_catalog_tile(
            lake, "B", norder=5, npix=npix, source_ids=[2], ra=[ra + 1e-5], dec=[dec],
        )
        result = build_crossmatch(
            lake, "A", "B", radius_arcsec=1.0, n_workers=2,
            memory_budget_bytes=2 * GIB,
        )
        assert result.n_match_rows == 1
