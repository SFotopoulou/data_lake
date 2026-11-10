"""Tests for data_lake.mp_context.safe_mp_context."""

from __future__ import annotations

import multiprocessing as mp

from data_lake.mp_context import safe_mp_context


def test_safe_mp_context_never_uses_fork() -> None:
    ctx = safe_mp_context()
    assert ctx.get_start_method() in {"forkserver", "spawn"}
    assert ctx.get_start_method() != "fork"


def test_safe_mp_context_preload_on_forkserver() -> None:
    if "forkserver" not in mp.get_all_start_methods():
        return
    ctx = safe_mp_context(preload=["data_lake.mp_context"])
    assert ctx.get_start_method() == "forkserver"
