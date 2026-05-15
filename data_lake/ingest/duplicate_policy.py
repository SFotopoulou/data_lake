"""Shared duplicate-ID policies for Zarr append ingest (cutouts and spectra)."""

from __future__ import annotations

from typing import Literal

import numpy as np

ZarrDuplicateMode = Literal["append", "error", "skip"]


def zarr_row_keep_mask(
    source_ids: np.ndarray,
    existing_source_ids: set[int] | np.ndarray,
    on_duplicate: ZarrDuplicateMode,
) -> np.ndarray:
    """Return a boolean mask of rows to append for one tile batch.

    Raises ``ValueError`` on duplicate IDs within *source_ids* or when
    ``on_duplicate='error'`` and an ID is already in *existing_source_ids*.
    """
    sids = np.asarray(source_ids, dtype=np.int64)
    n = int(sids.size)
    if n == 0:
        return np.zeros(0, dtype=bool)

    seen_in_batch: set[int] = set()
    for sid in sids.tolist():
        sid_int = int(sid)
        if sid_int in seen_in_batch:
            raise ValueError(
                f"Duplicate source_id {sid_int} within a single ingest batch for one tile"
            )
        seen_in_batch.add(sid_int)

    if on_duplicate == "append":
        return np.ones(n, dtype=bool)

    if isinstance(existing_source_ids, np.ndarray):
        existing_arr = np.asarray(existing_source_ids, dtype=np.int64)
        if existing_arr.size == 0:
            return np.ones(n, dtype=bool)
        dup = np.isin(sids, existing_arr, assume_unique=False)
        if on_duplicate == "error" and dup.any():
            first = int(sids[np.argmax(dup)])
            raise ValueError(
                f"source_id {first} already exists in this tile's Zarr; "
                f"use on_duplicate='skip' or 'append'."
            )
        return ~dup if on_duplicate == "skip" else np.ones(n, dtype=bool)

    keep = np.ones(n, dtype=bool)
    for i, sid in enumerate(sids.tolist()):
        sid_int = int(sid)
        if sid_int in existing_source_ids:
            if on_duplicate == "error":
                raise ValueError(
                    f"source_id {sid_int} already exists in this tile's Zarr; "
                    f"use on_duplicate='skip' or 'append'."
                )
            keep[i] = False
    return keep
