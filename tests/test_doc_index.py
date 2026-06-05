"""Tests for data_lake.doc_index."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_lake.doc_index import (
    clear_doc_index_cache,
    docs_root,
    get_section,
    list_cli_commands,
    list_doc_paths,
    resource_uri_for_path,
    search_docs,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_doc_index_cache()
    yield
    clear_doc_index_cache()


def test_docs_root_exists() -> None:
    assert (docs_root() / "quickstart.md").is_file()


def test_list_doc_paths_includes_cli_reference() -> None:
    paths = list_doc_paths()
    assert "cli-reference.md" in paths
    assert "ingest/spectra.md" in paths


def test_search_docs_finds_cli_reference() -> None:
    hits = search_docs("dl-describe-lake", limit=3)
    assert hits
    assert any("cli-reference" in h["path"] for h in hits)


def test_get_section_by_path() -> None:
    text = get_section("cli-reference.md")
    assert text is not None
    assert "dl-ingest-catalog" in text


def test_get_section_by_docs_uri() -> None:
    text = get_section("docs://ingest/logging")
    assert text is not None
    assert "quiet" in text.lower() or "heartbeat" in text.lower()


def test_list_cli_commands_matches_pyproject() -> None:
    import tomllib

    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    expected = {k for k in data["project"]["scripts"] if k.startswith("dl-")}
    got = {c["name"] for c in list_cli_commands()}
    assert expected == got


def test_resource_uri_for_path() -> None:
    assert resource_uri_for_path("ingest/spectra.md") == "docs://ingest/spectra"
