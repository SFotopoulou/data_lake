"""Per-process DuckDB thread and memory limits.

A DuckDB connection defaults to every core on the node and 80 % of physical
RAM.  Parallel workers each open their own connections, so ``N`` workers
otherwise run ``N × cores`` scan threads and each assumes it owns most of the
machine.  Under a Slurm or cgroup memory limit that is a common cause of a
worker being killed (``BrokenProcessPool``).

Resolution order for a new connection:

1. limits installed for this process with :func:`set_process_limits`
   (parallel workers) or :func:`process_limits` (sequential runs);
2. ``$DATA_LAKE_DUCKDB_THREADS`` / ``$DATA_LAKE_DUCKDB_MEMORY_LIMIT_MIB``;
3. DuckDB defaults.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

THREADS_ENV = "DATA_LAKE_DUCKDB_THREADS"
MEMORY_LIMIT_ENV = "DATA_LAKE_DUCKDB_MEMORY_LIMIT_MIB"

MIB = 1024 * 1024
MIN_MEMORY_LIMIT_BYTES = 256 * MIB


@dataclass(frozen=True)
class DuckDBLimits:
    """``threads`` / ``memory_limit_bytes`` of 0 leave the DuckDB default."""

    threads: int = 0
    memory_limit_bytes: int = 0

    def describe(self) -> str:
        threads = str(self.threads) if self.threads > 0 else "default"
        mem = (
            f"{self.memory_limit_bytes / MIB:.0f} MiB"
            if self.memory_limit_bytes > 0
            else "default"
        )
        return f"threads={threads}, memory_limit={mem}"


_process_limits: DuckDBLimits | None = None


def available_cpus() -> int:
    """CPUs this process may run on (respects Slurm / taskset affinity)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _read_limit(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if not text or text == "max":
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    # cgroup v1 reports "unlimited" as a huge page-aligned number.
    return value if 0 < value < (1 << 62) else None


def _cgroup_memory_limits() -> list[int]:
    limits: list[int] = []
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return limits
    root = Path("/sys/fs/cgroup")
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, rel = parts
        rel = rel.lstrip("/")
        if controllers == "":
            base, filename = root, "memory.max"
        elif "memory" in controllers.split(","):
            base, filename = root / "memory", "memory.limit_in_bytes"
        else:
            continue
        # A job's limit may sit on any ancestor cgroup; the tightest one applies.
        node = base / rel if rel else base
        while True:
            value = _read_limit(node / filename)
            if value is not None:
                limits.append(value)
            if node == base:
                break
            node = node.parent
    return limits


def _slurm_memory_limit() -> int | None:
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node and per_node.isdigit():
        return int(per_node) * MIB
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus = os.environ.get("SLURM_CPUS_ON_NODE")
    if per_cpu and per_cpu.isdigit() and cpus and cpus.isdigit():
        return int(per_cpu) * int(cpus) * MIB
    return None


def available_memory_bytes() -> int:
    """Smallest of physical RAM, cgroup memory limits, and the Slurm allocation."""
    candidates = _cgroup_memory_limits()
    slurm = _slurm_memory_limit()
    if slurm is not None:
        candidates.append(slurm)
    try:
        candidates.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    return min(candidates) if candidates else 0


def _env_int(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        log.warning("Ignoring non-integer $%s=%r", name, raw)
        return 0


def env_limits() -> DuckDBLimits:
    return DuckDBLimits(
        threads=_env_int(THREADS_ENV),
        memory_limit_bytes=_env_int(MEMORY_LIMIT_ENV) * MIB,
    )


def per_worker_limits(
    n_workers: int,
    *,
    memory_budget_bytes: int | None = None,
    memory_fraction: float | None = None,
) -> DuckDBLimits:
    """Split the CPUs and memory budget of this job across *n_workers* processes.

    Each worker gets ``available_cpus() // n_workers`` DuckDB threads and
    ``memory_fraction`` of its share of the budget (the rest is left for
    Polars frames, partner caches, and the matcher).  ``memory_budget_bytes``
    defaults to :func:`available_memory_bytes`.  Explicit environment overrides
    win over the computed values.
    """
    from data_lake.knobs import DUCKDB_WORKER_MEMORY_FRACTION

    n = max(1, int(n_workers))
    fraction = DUCKDB_WORKER_MEMORY_FRACTION if memory_fraction is None else memory_fraction
    if not 0 < fraction <= 1:
        raise ValueError(f"memory_fraction must be in (0, 1]; got {fraction}")
    budget = memory_budget_bytes if memory_budget_bytes else available_memory_bytes()

    threads = max(1, available_cpus() // n)
    memory = max(MIN_MEMORY_LIMIT_BYTES, int(budget / n * fraction)) if budget > 0 else 0

    env = env_limits()
    return DuckDBLimits(
        threads=env.threads or threads,
        memory_limit_bytes=env.memory_limit_bytes or memory,
    )


def set_process_limits(limits: DuckDBLimits | None) -> None:
    """Install *limits* for connections opened later in this process."""
    global _process_limits
    _process_limits = limits


@contextmanager
def process_limits(limits: DuckDBLimits | None) -> Iterator[None]:
    """Temporarily install *limits* for connections opened inside the block."""
    global _process_limits
    previous = _process_limits
    _process_limits = limits
    try:
        yield
    finally:
        _process_limits = previous


def current_limits() -> DuckDBLimits:
    return _process_limits if _process_limits is not None else env_limits()


def apply_limits(con, limits: DuckDBLimits | None = None):
    """Apply *limits* (default :func:`current_limits`) to an open connection."""
    limits = limits if limits is not None else current_limits()
    if limits.threads > 0:
        con.execute(f"SET threads = {int(limits.threads)}")
    if limits.memory_limit_bytes > 0:
        con.execute(f"SET memory_limit = '{max(1, limits.memory_limit_bytes // MIB)}MiB'")
    return con


def parquet_metadata_cache_default() -> bool:
    """``DATA_LAKE_PARQUET_METADATA_CACHE`` when set, else the knob default."""
    raw = os.environ.get("DATA_LAKE_PARQUET_METADATA_CACHE", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    from data_lake.knobs import CATALOG_PARQUET_METADATA_CACHE

    return CATALOG_PARQUET_METADATA_CACHE


def connect(
    database: str = ":memory:",
    limits: DuckDBLimits | None = None,
    *,
    parquet_metadata_cache: bool = False,
):
    """``duckdb.connect`` with the process limits applied.

    ``parquet_metadata_cache`` keeps parsed Parquet footers across queries on
    this connection, so repeated queries over the same files skip re-parsing.
    """
    import duckdb

    con = apply_limits(duckdb.connect(database=database), limits)
    if parquet_metadata_cache:
        con.execute("SET parquet_metadata_cache = true")
    return con
