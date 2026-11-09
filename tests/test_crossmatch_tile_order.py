"""Unit tests for crossmatch pending sort + mega-strip partitioning."""

from __future__ import annotations

from data_lake.io.crossmatch import (
    nested_parent_order,
    partition_pending_among_workers,
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
