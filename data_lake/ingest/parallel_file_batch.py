"""Batch helpers for parallel FITS file-list ingest."""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


def decode_path_batch(
    decoder: Callable[..., T],
    paths: list[str],
    *args,
    **kwargs,
) -> list[T]:
    """Run *decoder* once per path in *paths* (same worker process)."""
    return [decoder(p, *args, **kwargs) for p in paths]
