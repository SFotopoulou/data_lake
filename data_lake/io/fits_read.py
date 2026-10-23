"""
Unified FITS open policy for ingest: adaptive memmap and sequential row reads.

Local FITS files are not copied into Astropy's download cache; ``memmap=True``
maps the file in place and the OS page cache retains recently read pages.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np

MemmapMode = Literal["auto", "on", "off"]

FITS_HDU_ENV = "DATA_LAKE_FITS_HDU"
HDU_ALL = "all"

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
    #: Catalog table HDU selector (see :func:`parse_hdu_spec`); None = first table HDU.
    table_hdu: str | None = None

    @classmethod
    def for_sniff(cls, table_hdu: str | None = None) -> FitsReadPolicy:
        """Header-only open for format detection (no data read)."""
        return cls(memmap="off", lazy_load_hdus=True, table_hdu=parse_hdu_spec(table_hdu))

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
            table_hdu=parse_hdu_spec(os.environ.get(FITS_HDU_ENV)),
        )


def parse_memmap_mode(value: str | None) -> MemmapMode:
    if value is None:
        return "auto"
    v = value.strip().lower()
    if v in ("auto", "on", "off"):
        return v  # type: ignore[return-value]
    raise ValueError(f"fits memmap mode must be auto, on, or off; got {value!r}")


def parse_hdu_spec(value: str | int | None) -> str | None:
    """Normalise a ``--hdu`` value.

    Accepted forms: ``None`` / ``""`` (first table HDU), ``"all"`` (every
    non-empty table HDU), an integer index (``"2"``), an EXTNAME (``"OBJECTS"``),
    or ``"EXTNAME,EXTVER"`` (``"SCI,2"``).  Names are matched case-insensitively.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() == HDU_ALL:
        return HDU_ALL
    if text.lstrip("-").isdigit():
        if int(text) < 0:
            raise ValueError(f"--hdu index must be >= 0; got {value!r}")
        return str(int(text))
    name, sep, ver = text.partition(",")
    name = name.strip()
    if not name:
        raise ValueError(f"--hdu name is empty in {value!r}")
    if sep:
        ver = ver.strip()
        if not ver.isdigit():
            raise ValueError(f"--hdu EXTVER must be an integer in {value!r}")
        return f"{name},{int(ver)}"
    return name


def _describe_hdus(hdul) -> str:
    parts = []
    for i, hdu in enumerate(hdul):
        name = (hdu.name or "").strip() or "-"
        ver = getattr(hdu, "ver", 1)
        label = f"{name},{ver}" if ver and ver != 1 else name
        parts.append(f"{i}:{label}({type(hdu).__name__})")
    return ", ".join(parts)


def _hdu_rows(hdu) -> int:
    return int(hdu.header.get("NAXIS2", 0) or 0)


def resolve_table_hdus(hdul, spec: str | None = None, *, source: str = "") -> list[int]:
    """Return the table HDU indices selected by *spec* (see :func:`parse_hdu_spec`).

    ``None`` → ``[first table HDU]``; ``"all"`` → every table HDU with rows (the
    first table HDU when all are empty).  Raises ``ValueError`` listing the
    file's HDUs when the selection does not exist or is not a table.
    """
    from astropy.io.fits.hdu.table import _TableLikeHDU

    spec = parse_hdu_spec(spec)
    where = f" in {source}" if source else ""
    tables = [i for i, hdu in enumerate(hdul) if isinstance(hdu, _TableLikeHDU)]
    if not tables:
        raise ValueError(f"No BINTABLE / table HDU found{where}. HDUs: {_describe_hdus(hdul)}")
    if spec is None:
        return [tables[0]]
    if spec == HDU_ALL:
        non_empty = [i for i in tables if _hdu_rows(hdul[i]) > 0]
        return non_empty or [tables[0]]

    if spec.isdigit():
        idx = int(spec)
        if idx >= len(hdul):
            raise ValueError(
                f"--hdu {idx} out of range{where} ({len(hdul)} HDUs: {_describe_hdus(hdul)})"
            )
    else:
        name, _, ver = spec.partition(",")
        want_ver = int(ver) if ver else None
        matches = [
            i for i, hdu in enumerate(hdul)
            if (hdu.name or "").strip().upper() == name.upper()
            and (want_ver is None or int(getattr(hdu, "ver", 1) or 1) == want_ver)
        ]
        if not matches:
            raise ValueError(f"--hdu {spec!r} not found{where}. HDUs: {_describe_hdus(hdul)}")
        if len(matches) > 1:
            raise ValueError(
                f"--hdu {spec!r} is ambiguous{where} (HDUs {matches}); "
                "pass NAME,EXTVER or an index. HDUs: " + _describe_hdus(hdul)
            )
        idx = matches[0]
    if idx not in tables:
        raise ValueError(
            f"--hdu {spec!r}{where} selects HDU {idx} ({type(hdul[idx]).__name__}), "
            f"not a table. HDUs: {_describe_hdus(hdul)}"
        )
    return [idx]


def default_fits_read_policy(
    memmap: str | None = None,
    table_hdu: str | None = None,
) -> FitsReadPolicy:
    """Build a read policy from env defaults, optionally overriding memmap / HDU."""
    policy = FitsReadPolicy.from_env()
    if memmap is not None:
        policy = replace(policy, memmap=parse_memmap_mode(memmap))
    if table_hdu is not None:
        policy = replace(policy, table_hdu=parse_hdu_spec(table_hdu))
    return policy


def catalog_fits_read_policy(memmap: str | None, table_hdu: str | None) -> FitsReadPolicy:
    """Read policy for catalog workers: *table_hdu* is authoritative (None = first table)."""
    return replace(default_fits_read_policy(memmap), table_hdu=parse_hdu_spec(table_hdu))


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


def materialize_fits_columns(
    data,
    row_indices: np.ndarray,
    column_names: Sequence[str],
) -> np.ndarray:
    """Materialize selected BINTABLE columns for *row_indices*."""
    row_indices = np.asarray(row_indices, dtype=np.int64)
    names = list(column_names)
    if not names:
        return materialize_fits_rows(data, row_indices)
    missing = [c for c in names if c not in data.dtype.names]
    if missing:
        raise KeyError(f"FITS columns not found: {missing}")
    full = materialize_fits_rows(data, row_indices)
    if len(names) == len(full.dtype.names):
        return full
    return full[names]


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
