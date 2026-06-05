"""
Documentation index for human and MCP consumers.

Chunks markdown under ``docs/`` by heading, supports keyword search, and lists
``dl-*`` CLI entry points from ``pyproject.toml``.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class DocChunk:
    """One section of a documentation file."""

    id: str
    path: str
    heading: str
    level: int
    text: str


def _repo_root() -> Path:
    """Repository root (parent of the ``data_lake`` package directory)."""
    return Path(__file__).resolve().parent.parent


def docs_root() -> Path:
    """Absolute path to the ``docs/`` directory."""
    return _repo_root() / "docs"


def _pyproject_path() -> Path:
    return _repo_root() / "pyproject.toml"


def list_doc_paths() -> list[str]:
    """Relative paths of all ``*.md`` files under ``docs/``, sorted."""
    root = docs_root()
    if not root.is_dir():
        return []
    return sorted(
        str(p.relative_to(root)).replace("\\", "/")
        for p in root.rglob("*.md")
    )


def _slugify(path: str, heading: str) -> str:
    base = path.replace("/", "-").replace(".md", "")
    h = re.sub(r"[^\w\s-]", "", heading.lower())
    h = re.sub(r"[-\s]+", "-", h).strip("-")
    return f"{base}#{h}" if h else base


def _iter_chunks_for_file(rel_path: str) -> Iterator[DocChunk]:
    full = docs_root() / rel_path
    if not full.is_file():
        return
    raw = full.read_text(encoding="utf-8")
    matches = list(_HEADING_RE.finditer(raw))
    if not matches:
        yield DocChunk(
            id=rel_path.replace(".md", "").replace("/", "/"),
            path=rel_path,
            heading="",
            level=0,
            text=raw.strip(),
        )
        return

    # Preamble before first heading
    first = matches[0]
    preamble = raw[: first.start()].strip()
    if preamble:
        yield DocChunk(
            id=rel_path.replace(".md", ""),
            path=rel_path,
            heading="(intro)",
            level=1,
            text=preamble,
        )

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        text = f"{'#' * level} {heading}\n\n{body}".strip() if body else f"{'#' * level} {heading}"
        yield DocChunk(
            id=_slugify(rel_path, heading),
            path=rel_path,
            heading=heading,
            level=level,
            text=text,
        )


@lru_cache(maxsize=1)
def _all_chunks() -> tuple[DocChunk, ...]:
    chunks: list[DocChunk] = []
    for rel in list_doc_paths():
        chunks.extend(_iter_chunks_for_file(rel))
    return tuple(chunks)


def clear_doc_index_cache() -> None:
    """Clear cached chunks (for tests)."""
    _all_chunks.cache_clear()


def get_section(path_or_id: str) -> str | None:
    """Return chunk text by relative path, ``docs://`` URI, chunk id, or heading."""
    key = path_or_id.strip()
    if key.startswith("docs://"):
        key = key[len("docs://") :]

    root = docs_root()
    if "#" not in key:
        rel = key if key.endswith(".md") else f"{key}.md"
        full = root / rel
        if full.is_file():
            return full.read_text(encoding="utf-8")

    for chunk in _all_chunks():
        if chunk.id == key or chunk.path == key or chunk.heading.lower() == key.lower():
            return chunk.text

    # Partial path match
    for chunk in _all_chunks():
        if key in chunk.path or key in chunk.id:
            return chunk.text
    return None


def _score_chunk(chunk: DocChunk, terms: list[str]) -> int:
    score = 0
    heading_l = chunk.heading.lower()
    path_l = chunk.path.lower()
    text_l = chunk.text.lower()
    for term in terms:
        if term in heading_l:
            score += 10
        if term in path_l:
            score += 5
        if term in text_l:
            score += text_l.count(term)
    return score


def search_docs(query: str, *, limit: int = 5) -> list[dict[str, str]]:
    """Search documentation chunks by keyword (heading-weighted)."""
    terms = [t.lower() for t in re.split(r"\W+", query.strip()) if len(t) >= 2]
    if not terms:
        return []

    scored: list[tuple[int, DocChunk]] = []
    for chunk in _all_chunks():
        s = _score_chunk(chunk, terms)
        if s > 0:
            scored.append((s, chunk))
    scored.sort(key=lambda x: (-x[0], x[1].path, x[1].heading))

    out: list[dict[str, str]] = []
    for s, chunk in scored[: max(1, limit)]:
        preview = chunk.text[:400].replace("\n", " ")
        if len(chunk.text) > 400:
            preview += "…"
        out.append({
            "id": chunk.id,
            "path": chunk.path,
            "heading": chunk.heading,
            "score": str(s),
            "preview": preview,
        })
    return out


def list_cli_commands() -> list[dict[str, str]]:
    """Return ``dl-*`` script names and module entry points from pyproject.toml."""
    data = tomllib.loads(_pyproject_path().read_text(encoding="utf-8"))
    scripts: dict[str, str] = data.get("project", {}).get("scripts", {})
    return [
        {"name": name, "entry_point": target}
        for name, target in sorted(scripts.items())
        if name.startswith("dl-")
    ]


def resource_uri_for_path(rel_path: str) -> str:
    """Map ``ingest/spectra.md`` → ``docs://ingest/spectra``."""
    p = rel_path.replace("\\", "/")
    if p.endswith(".md"):
        p = p[:-3]
    return f"docs://{p}"
