"""Shared checkpoint / file-list validation for ingest runs (catalog, cutouts, spectra)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _canonical(p: Path | str) -> str:
    return str(Path(p).expanduser().resolve())


def paths_from_file_list_file(file_list_path: Path) -> list[Path]:
    """Resolve paths from a text file (one path per line); relative paths use list parent."""
    fl = file_list_path.resolve()
    base = fl.parent
    paths: list[Path] = []
    for line in fl.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        paths.append(p if p.is_absolute() else (base / p))
    return paths


def validate_ingest_sidecars(
    survey_root: Path,
    rep: Any,
    *,
    checkpoint_path: Path | None,
    inflight_path: Path | None,
    file_list: Path | None,
) -> None:
    """
    Validate optional ``.ingest_checkpoint.json`` / ``.ingest_inflight.json``.

    ``rep`` must be duck-typed like ``ValidationReport`` (``.errors`` / ``.warnings`` lists).
    """
    ck = checkpoint_path or (survey_root / ".ingest_checkpoint.json")
    if ck.exists():
        try:
            data = json.loads(ck.read_text())
        except Exception as exc:
            rep.errors.append(f"Invalid checkpoint JSON {ck}: {exc}")
            return
        done: set[str] = set()
        for p in data.get("completed", []):
            if not p or not isinstance(p, str):
                continue
            try:
                done.add(_canonical(p))
            except OSError:
                done.add(p)
        for p in sorted(done):
            if not Path(p).is_file():
                rep.warnings.append(f"Checkpoint lists missing file: {p}")
        if file_list is not None:
            listed = [_canonical(x) for x in paths_from_file_list_file(file_list)]
            missing_ck = [p for p in listed if p not in done]
            for p in missing_ck:
                rep.warnings.append(f"File-list entry not in checkpoint (not ingested?): {p}")
    elif file_list is not None:
        rep.warnings.append(f"No checkpoint at {ck}; cannot compare --file-list")

    infl = inflight_path or (survey_root / ".ingest_inflight.json")
    if infl.exists():
        try:
            infl_data = json.loads(infl.read_text())
        except Exception as exc:
            rep.errors.append(f"Invalid inflight JSON {infl}: {exc}")
            return
        commit = infl_data.get("commit")
        if commit:
            rep.warnings.append(
                f"Non-null inflight commit in {infl} (possible mid-file crash): {commit!r}"
            )
