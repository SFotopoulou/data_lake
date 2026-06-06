"""Tests for data_lake.io.fits_read."""

from __future__ import annotations

import numpy as np

from data_lake.io.fits_read import (
    FitsReadPolicy,
    contiguous_runs,
    materialize_fits_columns,
    materialize_fits_rows,
    resolve_memmap,
)


def test_resolve_memmap_auto_small(tmp_path) -> None:
    p = tmp_path / "small.fits"
    p.write_bytes(b"x" * 1024)
    policy = FitsReadPolicy(memmap="auto", small_file_bytes=8192)
    assert resolve_memmap(p, policy) is False


def test_resolve_memmap_auto_large(tmp_path) -> None:
    p = tmp_path / "large.fits"
    p.write_bytes(b"x" * 20_000)
    policy = FitsReadPolicy(memmap="auto", small_file_bytes=8192)
    assert resolve_memmap(p, policy) is True


def test_resolve_memmap_forced_on(tmp_path) -> None:
    p = tmp_path / "tiny.fits"
    p.write_bytes(b"x")
    assert resolve_memmap(p, FitsReadPolicy(memmap="on")) is True


def test_contiguous_runs() -> None:
    assert contiguous_runs(np.array([1, 2, 3, 7, 8, 12])) == [(1, 4), (7, 9), (12, 13)]
    assert contiguous_runs(np.array([], dtype=np.int64)) == []


def test_materialize_fits_rows_sequential_equivalent() -> None:
    dtype = np.dtype([("a", "i4"), ("b", "f8")])
    data = np.array([(i, float(i)) for i in range(10)], dtype=dtype)
    row_idx = np.array([3, 1, 2, 7, 9, 0])
    got = materialize_fits_rows(data, row_idx)
    want = np.asarray(data[row_idx])
    assert np.array_equal(got["a"], want["a"])
    assert np.allclose(got["b"], want["b"])


def test_materialize_fits_columns_subset() -> None:
    dtype = np.dtype([("a", "i4"), ("b", "f8"), ("c", "i4")])
    data = np.array([(i, float(i), i * 10) for i in range(10)], dtype=dtype)
    row_idx = np.array([1, 3, 5])
    got = materialize_fits_columns(data, row_idx, ["a", "c"])
    assert got.dtype.names == ("a", "c")
    assert list(got["a"]) == [1, 3, 5]
    assert list(got["c"]) == [10, 30, 50]


def test_materialize_fits_rows_empty() -> None:
    dtype = np.dtype([("x", "i4")])
    data = np.array([], dtype=dtype)
    got = materialize_fits_rows(data, np.array([], dtype=np.int64))
    assert got.size == 0
