"""Tests for Zarr duplicate-ID policy helper."""

from __future__ import annotations

import numpy as np
import pytest

from data_lake.ingest.duplicate_policy import zarr_row_keep_mask


def test_skip_existing_ids() -> None:
    sids = np.array([1, 2, 3], dtype=np.int64)
    keep = zarr_row_keep_mask(sids, {2}, "skip")
    np.testing.assert_array_equal(keep, [True, False, True])


def test_error_on_existing() -> None:
    with pytest.raises(ValueError, match="already exists"):
        zarr_row_keep_mask(np.array([5], dtype=np.int64), {5}, "error")


def test_within_batch_duplicate_raises() -> None:
    with pytest.raises(ValueError, match="within a single ingest batch"):
        zarr_row_keep_mask(np.array([1, 1], dtype=np.int64), set(), "skip")
