"""Tests for optional ingest guardrails (lake_config.toml + LAKE_INGEST_TOKEN)."""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from data_lake.cli_utils import (
    hash_ingest_token,
    require_ingest_permission,
    resolve_expected_ingest_hash,
    write_ingest_token_hash,
)
from data_lake.config import LakeConfig, _Guardrails, _Lake, _Paths


def _cfg(tmp_path: Path, *, require: bool, token: str = "secret") -> LakeConfig:
    return LakeConfig(
        lake=_Lake(name="t", root=tmp_path / "data"),
        paths=_Paths(),
        guardrails=_Guardrails(
            require_ingest_token=require,
            ingest_token_hash=hash_ingest_token(token) if require else "",
        ),
    )


def test_guardrails_off_by_default(tmp_path: Path) -> None:
    require_ingest_permission(_cfg(tmp_path, require=False), None)


def test_guardrails_disabled_with_cfg(tmp_path: Path) -> None:
    require_ingest_permission(None, None)


def test_guardrails_requires_token(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, require=True)
    with pytest.raises(click.ClickException, match="requires an ingest token"):
        require_ingest_permission(cfg, None)


def test_guardrails_rejects_wrong_token(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, require=True, token="correct")
    with pytest.raises(click.ClickException, match="Invalid ingest token"):
        require_ingest_permission(cfg, "wrong")


def test_guardrails_accepts_token(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, require=True, token="correct")
    require_ingest_permission(cfg, "correct")


def test_guardrails_sidecar_hash(tmp_path: Path) -> None:
    dep = tmp_path / "mylake"
    dep.mkdir()
    toml = dep / "lake_config.toml"
    toml.write_text(
        f"""
schema_version = "1"
[lake]
name = "prod"
root = "{(dep / 'data').as_posix()}"
[guardrails]
require_ingest_token = true
"""
    )
    write_ingest_token_hash(dep, "lake-secret")
    cfg = LakeConfig.load(toml)
    assert resolve_expected_ingest_hash(cfg) == hash_ingest_token("lake-secret")
    require_ingest_permission(cfg, "lake-secret")


def test_guardrails_load_from_toml(tmp_path: Path) -> None:
    toml = tmp_path / "lake_config.toml"
    h = hash_ingest_token("lake-secret")
    toml.write_text(
        f"""
schema_version = "1"
[lake]
name = "prod"
root = "{(tmp_path / 'data').as_posix()}"
[guardrails]
require_ingest_token = true
ingest_token_hash = "{h}"
"""
    )
    cfg = LakeConfig.load(toml)
    require_ingest_permission(cfg, "lake-secret")
