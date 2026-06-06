"""
Batch reads from Zarr arrays using contiguous index runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from data_lake.io.fits_read import contiguous_runs


def _normalize_row_slab(slab: np.ndarray) -> np.ndarray:
    """Ensure a row slab is ``(n_rows, *row_shape)``."""
    arr = np.asarray(slab)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr.reshape(arr.shape[0], arr.shape[2])
    return arr


def read_zarr_rows(
    zarr_array,
    indices: Sequence[int] | np.ndarray,
    *,
    stack: Callable[[list], np.ndarray] | None = None,
) -> np.ndarray:
    """Read rows from a Zarr array, slice-reading contiguous index runs.

    Parameters
    ----------
    zarr_array:
        Zarr array with first dimension = row index.
    indices:
        Row indices to read (any order; output matches input order).
    stack:
        Optional stack function (default ``np.stack``). Pass ``None`` for 1-D rows
        to use ``np.concatenate``.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return np.asarray(zarr_array[:0])

    order = np.argsort(indices, kind="stable")
    sorted_idx = indices[order]

    parts: list[np.ndarray] = []
    for run_start, run_end in contiguous_runs(sorted_idx):
        if run_end - run_start == 1:
            slab = _normalize_row_slab(zarr_array[int(run_start)])
        else:
            slab = _normalize_row_slab(zarr_array[run_start:run_end])
        parts.append(slab)

    if not parts:
        return np.asarray(zarr_array[:0])

    if stack is None:
        if parts[0].ndim == 1:
            stacked = np.concatenate(parts, axis=0)
        else:
            stacked = np.vstack(parts)
    else:
        stacked = stack(parts)

    unshuffle = np.empty_like(order)
    unshuffle[order] = np.arange(len(order))
    return stacked[unshuffle]
