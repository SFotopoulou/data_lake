"""Load homogenization transform packs from the lake or bundled defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PKG_TRANSFORMS = Path(__file__).resolve().parent / "transforms"


def transforms_dir(lake_root: Path | str) -> Path:
    return Path(lake_root) / "shared" / "registry" / "transforms"


def transform_path(lake_root: Path | str | None, transform_id: str) -> Path:
    """Resolve transform JSON: lake override first, then bundled package default."""
    if lake_root is not None:
        lake_path = transforms_dir(lake_root) / f"{transform_id}.json"
        if lake_path.is_file():
            return lake_path
    bundled = _PKG_TRANSFORMS / f"{transform_id}.json"
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(
        f"Transform {transform_id!r} not found under "
        f"{transforms_dir(lake_root or '/lake')} or {_PKG_TRANSFORMS}"
    )


def load_transform(lake_root: Path | str | None, transform_id: str) -> dict[str, Any]:
    path = transform_path(lake_root, transform_id)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("transform_id") != transform_id:
        raise ValueError(
            f"Transform file {path} has transform_id={data.get('transform_id')!r}, "
            f"expected {transform_id!r}"
        )
    return data


def load_bandpass_metadata() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "bandpass.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def validate_transform_schema(data: dict[str, Any]) -> list[str]:
    """Return ERROR/WARN messages for a transform pack."""
    msgs: list[str] = []
    if not data.get("transform_id"):
        msgs.append("ERROR: missing transform_id")
    if not data.get("rules"):
        msgs.append("ERROR: missing rules")
    for i, rule in enumerate(data.get("rules") or []):
        if not rule.get("survey"):
            msgs.append(f"ERROR: rule[{i}] missing survey")
        modality = data.get("modality", "catalog")
        if modality == "catalog":
            if not rule.get("source_column"):
                msgs.append(f"ERROR: rule[{i}] missing source_column")
            if not rule.get("target_column"):
                msgs.append(f"ERROR: rule[{i}] missing target_column")
        elif modality == "spectra":
            if not rule.get("flux_array"):
                msgs.append(f"ERROR: rule[{i}] missing flux_array")
        elif modality == "cutout":
            if not rule.get("image_array"):
                msgs.append(f"ERROR: rule[{i}] missing image_array")
        t = (rule.get("transform") or {}).get("type")
        if t not in ("mag_offset", "scale", "identity", "null_if_sentinel", "flux_scale"):
            msgs.append(f"ERROR: rule[{i}] unknown transform type {t!r}")
    return msgs
