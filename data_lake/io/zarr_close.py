"""Shared helper for closing Zarr LocalStore file handles (best-effort)."""

from __future__ import annotations

from typing import Any


def close_zarr_group(root: Any) -> None:
    """Release Zarr file handles for a group (best-effort, idempotent).

    Calls ``store.close()`` on the underlying store object if that method exists.
    Safe to call with ``None`` or any non-Zarr object.
    """
    if root is None:
        return
    store = getattr(root, "store", None)
    if store is None:
        return
    close = getattr(store, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
