"""Tests for ``dl-set-ingest-token``."""

from __future__ import annotations

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from data_lake.admin.set_ingest_token import dl_set_ingest_token, set_ingest_token
from data_lake.cli_utils import hash_ingest_token, require_ingest_permission
from data_lake.config import LakeConfig


def _write_minimal_config(tmp_path: Path) -> Path:
    dep = tmp_path / "existing"
    dep.mkdir()
    cfg_path = dep / "lake_config.toml"
    cfg_path.write_text(
        f"""
schema_version = "1"
[lake]
name = "existing"
root = "{(dep / 'data').as_posix()}"
[guardrails]
require_ingest_token = false
"""
    )
    return cfg_path


def test_set_ingest_token_enables_guardrails(tmp_path: Path) -> None:
    cfg_path = _write_minimal_config(tmp_path)
    deployment = set_ingest_token("new-secret", config_path=cfg_path)

    sidecar = deployment / ".ingest_token_hash"
    assert sidecar.is_file()
    assert sidecar.read_text().strip() == hash_ingest_token("new-secret")

    cfg = LakeConfig.load(cfg_path)
    assert cfg.guardrails.require_ingest_token is True
    assert cfg.guardrails.ingest_token_hash == ""
    require_ingest_permission(cfg, "new-secret")


def test_set_ingest_token_rotates_hash(tmp_path: Path) -> None:
    cfg_path = _write_minimal_config(tmp_path)
    set_ingest_token("old", config_path=cfg_path)
    set_ingest_token("new", config_path=cfg_path)
    cfg = LakeConfig.load(cfg_path)
    require_ingest_permission(cfg, "new")
    with pytest.raises(click.ClickException, match="Invalid ingest token"):
        require_ingest_permission(cfg, "old")


def test_set_ingest_token_updates_gitignore(tmp_path: Path) -> None:
    cfg_path = _write_minimal_config(tmp_path)
    dep = cfg_path.parent
    (dep / ".gitignore").write_text("data/\n")
    set_ingest_token("x", config_path=cfg_path)
    assert ".ingest_token_hash" in (dep / ".gitignore").read_text()


def test_set_ingest_token_rejects_empty(tmp_path: Path) -> None:
    cfg_path = _write_minimal_config(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        set_ingest_token("   ", config_path=cfg_path)


def test_cli_set_ingest_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = _write_minimal_config(tmp_path)
    monkeypatch.setenv("DATA_LAKE_CONFIG", str(cfg_path))
    runner = CliRunner()
    result = runner.invoke(dl_set_ingest_token, ["--ingest-token", "cli-secret"])
    assert result.exit_code == 0, result.output
    cfg = LakeConfig.load(cfg_path)
    require_ingest_permission(cfg, "cli-secret")
