"""Zarr array names for the lake-internal object join key."""

from __future__ import annotations

import zarr

from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, LEGACY_JOIN_ID_COLUMN


def zarr_join_array(root: zarr.Group) -> zarr.Array:
    """Return the per-tile int64 join array (current or legacy name)."""
    if LAKE_JOIN_ID_COLUMN in root:
        return root[LAKE_JOIN_ID_COLUMN]
    if LEGACY_JOIN_ID_COLUMN in root:
        return root[LEGACY_JOIN_ID_COLUMN]
    raise KeyError(
        f"Zarr tile has no join array {LAKE_JOIN_ID_COLUMN!r} or "
        f"{LEGACY_JOIN_ID_COLUMN!r}; keys: {list(root.keys())}"
    )


def create_zarr_join_array(root: zarr.Group, **kwargs) -> zarr.Array:
    """Create the per-tile join array using the canonical name."""
    return root.create_array(LAKE_JOIN_ID_COLUMN, **kwargs)
