"""Guardrails: CLI table and README hub stay in sync with pyproject and docs/."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from data_lake.doc_index import docs_root, list_doc_paths


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _pyproject_scripts() -> set[str]:
    data = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    return {
        name
        for name in data.get("project", {}).get("scripts", {})
        if name.startswith("dl-")
    }


def test_every_cli_script_in_cli_reference() -> None:
    cli_ref = (docs_root() / "cli-reference.md").read_text(encoding="utf-8")
    missing = sorted(cmd for cmd in _pyproject_scripts() if cmd not in cli_ref)
    assert not missing, f"Add to docs/cli-reference.md: {missing}"


def test_every_doc_linked_from_readme() -> None:
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    unlinked: list[str] = []
    for rel in list_doc_paths():
        # README links use docs/<path> or (docs/<path>)
        if rel == "mcp.md":
            needle = "docs/mcp.md"
        else:
            needle = f"docs/{rel}"
        if needle not in readme and f"]({needle})" not in readme:
            # Also accept anchor-only links like docs/quickstart.md#foo
            if f"docs/{rel}" not in readme:
                unlinked.append(rel)
    assert not unlinked, f"Link from README Documentation section: {unlinked}"
