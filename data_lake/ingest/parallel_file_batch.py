"""Batch helpers for parallel FITS file-list ingest."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def decode_path_batch(
    decoder: Callable[..., T],
    paths: list[str],
    *args,
    **kwargs,
) -> list[T]:
    """Run *decoder* once per path in *paths* (same worker process)."""
    return [decoder(p, *args, **kwargs) for p in paths]


def partition_paths_by_directory(
    paths: Sequence[str] | list[str],
    n_workers: int,
) -> list[list[str]]:
    """Group sorted paths by parent directory, then assign groups to workers.

    Within each directory, path order is preserved.  Worker chunks are
    balanced by file count (not byte size).
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    sorted_paths = sorted(paths)
    if not sorted_paths:
        return []
    by_dir: dict[str, list[str]] = defaultdict(list)
    for p in sorted_paths:
        by_dir[str(Path(p).parent)].append(p)
    dir_groups = [by_dir[k] for k in sorted(by_dir)]
    if n_workers == 1 or len(sorted_paths) <= n_workers:
        return dir_groups
    # Flatten dir groups round-robin into n_workers queues
    buckets: list[list[str]] = [[] for _ in range(n_workers)]
    for i, group in enumerate(dir_groups):
        buckets[i % n_workers].extend(group)
    return [p for d in sorted(by_dir) for p in by_dir[d]]


def paths_grouped_by_directory(paths: Sequence[str]) -> list[str]:
    """Return *paths* sorted by parent directory then filename."""
    by_dir: dict[str, list[str]] = defaultdict(list)
    for p in sorted(paths):
        by_dir[str(Path(p).parent)].append(p)
    return [p for d in sorted(by_dir) for p in by_dir[d]]
