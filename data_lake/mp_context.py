"""Safe multiprocessing contexts for ``ProcessPoolExecutor``.

Forking a parent that already runs DuckDB / Arrow / Polars / BLAS thread pools
can deadlock child processes on locks held by those threads.  Prefer
``forkserver`` (or ``spawn`` when forkserver is unavailable) so workers start
from a clean process.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def safe_mp_context(*, preload: Sequence[str] | None = None) -> Any:
    """Return a multiprocessing context that never forks the current process.

    Parameters
    ----------
    preload:
        Module names passed to ``set_forkserver_preload`` when the start method
        is ``forkserver``.  Ignored under ``spawn``.
    """
    import multiprocessing as mp

    if "forkserver" in mp.get_all_start_methods():
        ctx = mp.get_context("forkserver")
        if preload:
            ctx.set_forkserver_preload(list(preload))
        return ctx
    return mp.get_context("spawn")
