"""
Unified FITS open policy for ingest: adaptive memmap and sequential row reads.

Local FITS files are not copied into Astropy's download cache; ``memmap=True``
maps the file in place and the OS page cache retains recently read pages.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np

MemmapMode = Literal["auto", "on", "off"]

DEFAULT_SMALL_FILE_BYTES = 8 * 1024 * 1024
DEFAULT_PARALLEL_CATALOG_MAX_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class FitsReadPolicy:
    """Controls ``astropy.io.fits.open`` behaviour for data_lake ingest."""

    memmap: MemmapMode = "auto"
    small_file_bytes: int = DEFAULT_SMALL_FILE_BYTES
    parallel_catalog_max_bytes: int = DEFAULT_PARALLEL_CATALOG_MAX_BYTES
    lazy_load_hdus: bool = False
    ignore_missing_simple: bool = True

    @classmethod
    def for_sniff(cls) -> FitsReadPolicy:
        """Header-only open for format detection (no data read)."""
        return cls(memmap="off", lazy_load_hdus=True)

    @classmethod
    def from_env(cls) -> FitsReadPolicy:
        memmap_raw = os.environ.get("DATA_LAKE_FITS_MEMMAP", "auto").strip().lower()
        if memmap_raw not in ("auto", "on", "off"):
            memmap_raw = "auto"
        memmap: MemmapMode = memmap_raw  # type: ignore[assignment]
        small = int(os.environ.get("DATA_LAKE_FITS_SMALL_BYTES", DEFAULT_SMALL_FILE_BYTES))
        parallel_max = int(
            os.environ.get(
                "DATA_LAKE_PARALLEL_CATALOG_MAX_BYTES",
                DEFAULT_PARALLEL_CATALOG_MAX_BYTES,
            )
        )
        return cls(
            memmap=memmap,
            small_file_bytes=small,
            parallel_catalog_max_bytes=parallel_max,
        )


def parse_memmap_mode(value: str | None) -> MemmapMode:
    if value is None:
        return "auto"
    v = value.strip().lower()
    if v in ("auto", "on", "off"):
        return v  # type: ignore[return-value]
    raise ValueError(f"fits memmap mode must be auto, on, or off; got {value!r}")


def default_fits_read_policy(memmap: str | None = None) -> FitsReadPolicy:
    """Build a read policy from env defaults, optionally overriding memmap mode."""
    env = FitsReadPolicy.from_env()
    if memmap is None:
        return env
    return FitsReadPolicy(
        memmap=parse_memmap_mode(memmap),
        small_file_bytes=env.small_file_bytes,
        parallel_catalog_max_bytes=env.parallel_catalog_max_bytes,
    )


def resolve_memmap(path: Path | str, policy: FitsReadPolicy) -> bool:
    """Return the boolean ``memmap=`` argument for ``fits.open``."""
    if policy.memmap == "on":
        return True
    if policy.memmap == "off":
        return False
    try:
        size = Path(path).stat().st_size
    except OSError:
        return True
    return size >= policy.small_file_bytes


def file_size_bytes(path: Path | str) -> int:
    return Path(path).stat().st_size


def check_parallel_catalog_file_size(path: Path | str, policy: FitsReadPolicy) -> None:
    """Reject files too large for parallel whole-file catalog decode."""
    try:
        size = file_size_bytes(path)
    except OSError as exc:
        raise ValueError(f"Cannot stat catalog FITS {path!r}: {exc}") from exc
    if size > policy.parallel_catalog_max_bytes:
        mb = policy.parallel_catalog_max_bytes / (1024 * 1024)
        raise ValueError(
            f"Catalog file {Path(path).name!r} is {size / (1024 * 1024):.1f} MiB; "
            f"parallel ingest loads each file fully in RAM (limit {mb:.0f} MiB). "
            "Use sequential dl-ingest-catalog --streaming for large FITS BINTABLEs."
        )


@contextmanager
def open_fits(path: Path | str, policy: FitsReadPolicy | None = None):
    """Open a local FITS file with the ingest read policy."""
    hdul = open_fits_raw(path, policy)
    try:
        yield hdul
    finally:
        hdul.close()


def open_fits_raw(path: Path | str, policy: FitsReadPolicy | None = None):
    """Open FITS and return an HDUList (caller must ``close()``)."""
    from astropy.io import fits

    resolved = policy or FitsReadPolicy.from_env()
    kwargs: dict = {
        "memmap": resolve_memmap(path, resolved),
        "ignore_missing_simple": resolved.ignore_missing_simple,
    }
    if resolved.lazy_load_hdus:
        kwargs["lazy_load_hdus"] = True
    return fits.open(str(path), **kwargs)


def contiguous_runs(sorted_indices: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` intervals where *sorted_indices* increase by 1."""
    idx = np.asarray(sorted_indices, dtype=np.int64)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for val in idx[1:]:
        v = int(val)
        if v == prev + 1:
            prev = v
            continue
        runs.append((start, prev + 1))
        start = prev = v
    runs.append((start, prev + 1))
    return runs


def materialize_fits_rows(data, row_indices: np.ndarray) -> np.ndarray:
    """Materialize BINTABLE rows using sequential file-order reads when possible."""
    row_indices = np.asarray(row_indices, dtype=np.int64)
    n = int(row_indices.size)
    if n == 0:
        return np.asarray(data[:0])
    order = np.argsort(row_indices, kind="stable")
    sorted_rows = row_indices[order]
    out = np.empty(n, dtype=data.dtype)
    pos = 0
    for run_start, run_end in contiguous_runs(sorted_rows):
        run_len = run_end - run_start
        slab = np.asarray(data[run_start:run_end])
        out[order[pos : pos + run_len]] = slab
        pos += run_len
    return out


def iter_path_batches(paths: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    """Yield consecutive path batches of at most *batch_size*."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    batch: list[str] = []
    for p in paths:
        batch.append(p)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
