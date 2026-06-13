"""Shared helpers for MCP servers (docs and lake explorer)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_lake_root(lake_root: str | None) -> Path:
    if lake_root:
        return Path(lake_root).expanduser().resolve()
    from data_lake.config import LakeConfig, LakeConfigNotFound

    try:
        cfg = LakeConfig.discover(None)
    except LakeConfigNotFound as exc:
        raise ValueError(
            "No lake root: pass lake_root or set DATA_LAKE_CONFIG / lake_config.toml."
        ) from exc
    return cfg.lake.root


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)
