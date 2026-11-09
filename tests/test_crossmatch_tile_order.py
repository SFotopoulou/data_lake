"""Unit tests for crossmatch pending sort + mega-strip partitioning."""

from __future__ import annotations

import numpy as np
import pytest

from data_lake.io.crossmatch import (
    estimate_crossmatch_tile_costs,
    nested_parent_order,
    partition_pending_among_workers,
    partition_pending_by_cost,
    sort_pending_by_nested_parent,
)


class TestNestedParentSort:
    def test_parent_order_default(self):
        assert nested_parent_order(5) == 3
        assert nested_parent_order(2) == 0
        assert nested_parent_order(0) == 0

    def test_clusters_sparse_tiles_by_parent(self):
        # At norder=5, parent_order=3 → shift 4 bits (2*2).
        # Pixels under parent 0: 0..15; under parent 1: 16..31; etc.
        # Interleave distant Morton indices so a plain sorted() would keep them
        # interleaved; parent sort should group by parent first.
        pending = [0, 16, 1, 17, 2, 18]
        out = sort_pending_by_nested_parent(pending, norder=5)
        # All parent-0 tiles before parent-1
        assert out == [0, 1, 2, 16, 17, 18]

    def test_plain_ascending_when_parent_equals_norder(self):
        pending = [5, 1, 3]
        assert sort_pending_by_nested_parent(
            pending, norder=5, parent_order=5
        ) == [1, 3, 5]


class TestPartitionPending:
    def test_empty(self):
        assert partition_pending_among_workers([], 4) == []

    def test_at_most_n_workers_strips(self):
        pending = list(range(100))
        strips = partition_pending_among_workers(pending, 8)
        assert len(strips) == 8
        assert sum(len(s) for s in strips) == 100
        # Contiguous: concat restores order
        assert [p for s in strips for p in s] == pending
        # Mega-strips are roughly equal
        lengths = [len(s) for s in strips]
        assert max(lengths) - min(lengths) <= 1

    def test_fewer_tiles_than_workers(self):
        strips = partition_pending_among_workers([10, 20, 30], 8)
        assert len(strips) == 3
        assert all(len(s) == 1 for s in strips)

    def test_single_worker_one_strip(self):
        pending = [1, 2, 3, 4]
        strips = partition_pending_among_workers(pending, 1)
        assert strips == [(1, 2, 3, 4)]


class TestPartitionByCost:
    def test_contiguous_and_complete(self):
        pending = list(range(100))
        strips = partition_pending_by_cost(pending, [1.0] * 100, 8)
        assert len(strips) == 8
        assert [p for s in strips for p in s] == pending

    def test_dense_region_gets_short_strips(self):
        # First 10 tiles are 10x denser than the remaining 90.
        pending = list(range(100))
        costs = [10.0] * 10 + [1.0] * 90
        strips = partition_pending_by_cost(pending, costs, 4)
        cost_of = dict(zip(pending, costs))
        strip_costs = [sum(cost_of[p] for p in s) for s in strips]
        assert len(strips[0]) < len(strips[-1])
        assert max(strip_costs) / (sum(strip_costs) / len(strip_costs)) < 1.25
        assert [p for s in strips for p in s] == pending

    def test_single_heavy_tile_does_not_lose_tiles(self):
        pending = [1, 2, 3, 4, 5]
        strips = partition_pending_by_cost(pending, [0, 100, 0, 0, 1], 4)
        assert [p for s in strips for p in s] == pending
        assert 1 <= len(strips) <= 4
        assert all(strips)

    def test_more_strips_than_tiles(self):
        strips = partition_pending_by_cost([7, 8], [1.0, 1.0], 16)
        assert strips == [(7,), (8,)]

    def test_zero_cost_falls_back_to_equal_counts(self):
        strips = partition_pending_by_cost(list(range(8)), [0.0] * 8, 4)
        assert [len(s) for s in strips] == [2, 2, 2, 2]

    def test_validation(self):
        with pytest.raises(ValueError):
            partition_pending_by_cost([1], [1.0], 0)
        with pytest.raises(ValueError):
            partition_pending_by_cost([1, 2], [1.0], 2)


class TestBalancedParallelCrossmatch:
    def _lake(self, tmp_path):
        from data_lake.ingest.fits_to_parquet import assign_healpix
        from tests.test_crossmatch import _write_catalog_tile

        lake = tmp_path / "lake"
        rng = np.random.default_rng(0)
        npix_rows: dict[int, list[tuple[float, float]]] = {}
        # A dense patch plus a sparse spread, so strip costs differ.
        ra = np.concatenate([rng.uniform(30, 31, 400), rng.uniform(30, 60, 100)])
        dec = np.concatenate([rng.uniform(5, 6, 400), rng.uniform(5, 30, 100)])
        npix = assign_healpix(ra, dec, 5)
        for r, d, p in zip(ra, dec, npix):
            npix_rows.setdefault(int(p), []).append((float(r), float(d)))
        sid = 1
        for survey, offset in (("A", 0.0), ("B", 1e-5)):
            for p, rows in npix_rows.items():
                ids = list(range(sid, sid + len(rows)))
                sid += len(rows)
                _write_catalog_tile(
                    lake, survey, norder=5, npix=p, source_ids=ids,
                    ra=[r + offset for r, _ in rows], dec=[d for _, d in rows],
                )
        return lake, len(ra)

    def test_costs_follow_density(self, tmp_path):
        lake, _ = self._lake(tmp_path)
        from data_lake.io.crossmatch import iter_populated_tile_npixels
        from data_lake.schema_registry import resolve_catalog_root

        pending = iter_populated_tile_npixels(resolve_catalog_root(lake, "A"), norder=5)
        costs = estimate_crossmatch_tile_costs(
            lake, "A", "B", pending, norder_a=5, norder_b=5,
            radius_rad=np.radians(1 / 3600),
        )
        assert len(costs) == len(pending)
        assert max(costs) > 3 * min(costs) > 0

    def test_balanced_parallel_matches_sequential(self, tmp_path):
        from data_lake.io.crossmatch import build_crossmatch, load_crossmatch_table

        lake, n_rows = self._lake(tmp_path)
        seq = build_crossmatch(lake, "A", "B", radius_arcsec=1.0, n_workers=1)
        par = build_crossmatch(
            lake, "A", "B", radius_arcsec=1.0, n_workers=2, strips_per_worker=4,
            overwrite=True,
        )
        assert seq.n_match_rows == par.n_match_rows == n_rows
        tbl = load_crossmatch_table(par.output_root)
        assert tbl.num_rows == n_rows

    def test_state_reused_only_inside_pool_workers(self, tmp_path):
        import data_lake.io.crossmatch as xm

        lake, _ = self._lake(tmp_path)
        cfg = xm.CrossmatchTileConfig(
            lake_root=str(lake), survey_a="A", survey_b="B", npix_a_list=(),
            norder_a=5, norder_b=5, ra_col_a="ra", dec_col_a="dec",
            ra_col_b="ra", dec_col_b="dec", radius_arcsec=1.0,
            out_root=str(tmp_path / "xm"),
        )
        assert xm._crossmatch_tile_worker(cfg).error is None
        assert xm._worker_state is None

        xm._REUSE_WORKER_STATE = True
        try:
            assert xm._crossmatch_tile_worker(cfg).error is None
            first = xm._worker_state
            assert first is not None
            assert xm._crossmatch_tile_worker(cfg).error is None
            assert xm._worker_state is first
        finally:
            xm._close_worker_state()
            xm._REUSE_WORKER_STATE = False
