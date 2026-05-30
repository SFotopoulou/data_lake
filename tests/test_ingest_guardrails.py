"""Tests for mandatory ingest token (deployment hash + LAKE_INGEST_TOKEN)."""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from data_lake.cli_utils import (
    INGEST_TOKEN_ENV,
    hash_ingest_token,
    require_ingest_permission,
    write_ingest_token_hash,
)
from data_lake.config import LakeConfig, _Guardrails, _Lake, _Paths


def _cfg(tmp_path: Path, token: str = "secret") -> LakeConfig:
    dep = tmp_path / "dep"
    dep.mkdir()
    cfg_path = dep / "lake_config.toml"
    cfg_path.write_text(
        f"""
schema_version = "1"
[lake]
name = "t"
root = "{(dep / 'data').as_posix()}"
[guardrails]
ingest_token_file = ".ingest_token_hash"
"""
    )
    write_ingest_token_hash(dep, token)
    return LakeConfig.load(cfg_path)


def test_ingest_requires_deployment_config(tmp_path: Path) -> None:
    with pytest.raises(click.ClickException, match="deployment config"):
        require_ingest_permission(None, "any")


def test_ingest_requires_hash_on_disk(tmp_path: Path) -> None:
    dep = tmp_path / "bare"
    dep.mkdir()
    cfg_path = dep / "lake_config.toml"
    cfg_path.write_text(
        f"""
schema_version = "1"
[lake]
name = "t"
root = "{(dep / 'data').as_posix()}"
"""
    )
    cfg = LakeConfig.load(cfg_path)
    with pytest.raises(click.ClickException, match="No ingest token hash"):
        require_ingest_permission(cfg, "secret")


def test_ingest_requires_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.delenv(INGEST_TOKEN_ENV, raising=False)
    with pytest.raises(click.ClickException, match="requires an ingest token"):
        require_ingest_permission(cfg, None)


def test_ingest_rejects_wrong_token(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, token="correct")
    with pytest.raises(click.ClickException, match="Invalid ingest token"):
        require_ingest_permission(cfg, "wrong")


def test_ingest_accepts_token(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, token="correct")
    require_ingest_permission(cfg, "correct")


def test_legacy_guardrails_keys_ignored_in_toml(tmp_path: Path) -> None:
    """Old require_ingest_token / ingest_token_hash keys do not break load."""
    dep = tmp_path / "legacy"
    dep.mkdir()
    cfg_path = dep / "lake_config.toml"
    cfg_path.write_text(
        f"""
schema_version = "1"
[lake]
name = "t"
root = "{(dep / 'data').as_posix()}"
[guardrails]
require_ingest_token = false
ingest_token_hash = "deadbeef"
ingest_token_file = ".ingest_token_hash"
"""
    )
    write_ingest_token_hash(dep, "lake-secret")
    cfg = LakeConfig.load(cfg_path)
    assert cfg.guardrails == _Guardrails(ingest_token_file=".ingest_token_hash")
    require_ingest_permission(cfg, "lake-secret")
