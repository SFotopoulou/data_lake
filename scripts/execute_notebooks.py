#!/usr/bin/env python
"""Execute notebooks against the synthetic demo lake and store their outputs.

    uv run --extra dev python scripts/execute_notebooks.py                 # all, in order
    uv run --extra dev python scripts/execute_notebooks.py notebooks/13_*.ipynb --fresh
    uv run --extra dev python scripts/execute_notebooks.py --demo-dir .scratch_nb/a --no-write

Notebooks run one at a time (they share one lake) with ``DATA_LAKE_CONFIG``
unset, so each falls back to ``ensure_demo_lake()`` at ``--demo-dir``.
Absolute paths in outputs are rewritten relative to the notebook folder, so
committed outputs do not leak local paths.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import click
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOB = "notebooks/[0-9]*.ipynb"
MAX_OUTPUT_KIB = 1024


def _scrub(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _scrub_outputs(nb, replacements: list[tuple[str, str]]) -> None:
    for cell in nb.cells:
        cell.get("metadata", {}).pop("execution", None)
        for out in cell.get("outputs", []):
            if "text" in out:
                out["text"] = _scrub(out["text"], replacements)
            for key, val in (out.get("data") or {}).items():
                if key.startswith("text/") and isinstance(val, str):
                    out["data"][key] = _scrub(val, replacements)
            if "traceback" in out:
                out["traceback"] = [_scrub(t, replacements) for t in out["traceback"]]


def _run_one(path: Path, *, kernel: str, timeout: int, write: bool) -> tuple[bool, float, int]:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, kernel_name=kernel, timeout=timeout,
        resources={"metadata": {"path": str(path.parent)}},
    )
    t0 = time.perf_counter()
    ok = True
    try:
        client.execute()
    except CellExecutionError as exc:
        ok = False
        print(f"  FAILED: {str(exc).splitlines()[0] if str(exc) else exc}", file=sys.stderr)
        print(str(exc)[-3000:], file=sys.stderr)
    elapsed = time.perf_counter() - t0
    home = str(Path.home())
    _scrub_outputs(nb, [
        (str(path.parent) + "/", ""),
        (str(path.parent), "."),
        (str(REPO_ROOT) + "/", "../"),
        (str(REPO_ROOT), ".."),
        (home, "~"),
    ])
    text = nbformat.writes(nb)
    if write and ok:
        path.write_text(text, encoding="utf-8")
    return ok, elapsed, len(text.encode()) // 1024


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=__doc__,
)
@click.argument(
    "notebooks",
    nargs=-1,
    type=click.Path(path_type=Path),
)
@click.option(
    "--demo-dir",
    type=click.Path(path_type=Path),
    default=str(REPO_ROOT / "notebooks" / "_demo_lake"),
    show_default=True,
    help="Demo deployment used by the notebooks.",
)
@click.option("--fresh", is_flag=True, help="Delete --demo-dir first so it is rebuilt.")
@click.option("--kernel", default="python3", show_default=True, help="Jupyter kernel name.")
@click.option("--timeout", type=int, default=900, show_default=True, help="Per-cell timeout in seconds.")
@click.option("--no-write", is_flag=True, help="Execute only; leave notebook files unchanged.")
def cli(
    notebooks: tuple[Path, ...],
    demo_dir: Path,
    fresh: bool,
    kernel: str,
    timeout: int,
    no_write: bool,
) -> int:
    paths = [p.resolve() for p in notebooks] or sorted(REPO_ROOT.glob(DEFAULT_GLOB))
    demo_dir = demo_dir.resolve()
    if fresh and demo_dir.exists():
        shutil.rmtree(demo_dir)

    os.environ.pop("DATA_LAKE_CONFIG", None)
    os.environ["DATA_LAKE_DEMO_DIR"] = str(demo_dir)
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")

    failed = []
    for path in paths:
        print(f"{path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path} …", flush=True)
        ok, elapsed, kib = _run_one(path, kernel=kernel, timeout=timeout, write=not no_write)
        flag = "" if kib <= MAX_OUTPUT_KIB else f"  (over {MAX_OUTPUT_KIB} KiB)"
        print(f"  {'ok' if ok else 'FAILED'} in {elapsed:.1f} s, {kib} KiB{flag}", flush=True)
        if not ok:
            failed.append(path.name)
    if failed:
        print(f"{len(failed)} notebook(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return cli.main(args=argv, prog_name="execute_notebooks", standalone_mode=False)  # type: ignore[return-value]


if __name__ == "__main__":
    sys.exit(main())
