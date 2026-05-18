"""Tests for ``data_lake.admin.init_lake``."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from data_lake.admin.init_lake import dl_init, init_lake
from data_lake.cli_utils import hash_ingest_token, require_ingest_permission
from data_lake.config import LakeConfig

_TEST_TOKEN = "test-ingest-token"


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------

def test_init_lake_creates_expected_tree(tmp_path: Path) -> None:
    deployment = init_lake(
        name="test_lake",
        parent=tmp_path,
        description="A test deployment",
        ingest_token=_TEST_TOKEN,
    )

    assert deployment == tmp_path / "test_lake"
    assert (deployment / "lake_config.toml").is_file()
    assert (deployment / ".ingest_token_hash").is_file()
    assert (deployment / "README.md").is_file()
    assert (deployment / ".gitignore").is_file()
    for sub in ("data/catalogs", "data/spectra", "data/cutouts",
                "data/shared", "notebooks", "scripts"):
        assert (deployment / sub).is_dir(), f"missing {sub}"


def test_init_lake_writes_valid_loadable_config(tmp_path: Path) -> None:
    deployment = init_lake(
        name="lake_alpha",
        parent=tmp_path,
        description="hello",
        norder=7,
        with_resolution=True,
        ingest_token=_TEST_TOKEN,
    )
    cfg = LakeConfig.load(deployment / "lake_config.toml")

    assert cfg.lake.name == "lake_alpha"
    assert cfg.lake.description == "hello"
    assert cfg.lake.root == (deployment / "data").resolve()
    assert cfg.partitioning.hats_order == 7
    assert cfg.defaults.with_resolution is True
    require_ingest_permission(cfg, _TEST_TOKEN)


def test_init_lake_external_root(tmp_path: Path) -> None:
    external = tmp_path / "external_data"
    deployment = init_lake(
        name="ext",
        parent=tmp_path,
        root=external,
        ingest_token=_TEST_TOKEN,
    )
    assert external.is_dir()
    assert (external / "spectra").is_dir()

    cfg = LakeConfig.load(deployment / "lake_config.toml")
    assert cfg.lake.root == external.resolve()


def test_init_lake_refuses_existing_dir(tmp_path: Path) -> None:
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        init_lake(name="exists", parent=tmp_path, ingest_token=_TEST_TOKEN)


def test_init_lake_force_overwrites(tmp_path: Path) -> None:
    (tmp_path / "force_me").mkdir()
    deployment = init_lake(
        name="force_me", parent=tmp_path, force=True, ingest_token=_TEST_TOKEN,
    )
    assert (deployment / "lake_config.toml").is_file()


def test_init_lake_missing_parent_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        init_lake(name="x", parent=tmp_path / "does_not_exist", ingest_token=_TEST_TOKEN)


def test_init_lake_rejects_empty_token(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        init_lake(name="x", parent=tmp_path, ingest_token="   ")


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def test_cli_creates_deployment(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        dl_init,
        ["my_lake", str(tmp_path), "--description", "from CLI", "--norder", "6",
         "--ingest-token", _TEST_TOKEN],
    )
    assert result.exit_code == 0, result.output
    assert "Created deployment" in result.output

    cfg = LakeConfig.load(tmp_path / "my_lake" / "lake_config.toml")
    assert cfg.lake.name == "my_lake"
    assert cfg.partitioning.hats_order == 6


def test_cli_requires_ingest_token(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(dl_init, ["x", str(tmp_path)])
    assert result.exit_code != 0


def test_cli_with_resolution_flag(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        dl_init,
        ["resolved_lake", str(tmp_path), "--with-resolution", "--ingest-token", _TEST_TOKEN],
    )
    assert result.exit_code == 0, result.output
    cfg = LakeConfig.load(tmp_path / "resolved_lake" / "lake_config.toml")
    assert cfg.defaults.with_resolution is True


def test_cli_num_workers_int(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        dl_init,
        ["w_lake", str(tmp_path), "--num-workers", "8", "--ingest-token", _TEST_TOKEN],
    )
    assert result.exit_code == 0, result.output
    cfg = LakeConfig.load(tmp_path / "w_lake" / "lake_config.toml")
    assert cfg.ingest.num_workers == 8


def test_cli_num_workers_invalid(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        dl_init,
        ["w_lake", str(tmp_path), "--num-workers", "many", "--ingest-token", _TEST_TOKEN],
    )
    assert result.exit_code != 0
    assert "must be 'auto' or an integer" in result.output


def test_cli_refuses_existing_without_force(tmp_path: Path) -> None:
    (tmp_path / "dup").mkdir()
    runner = CliRunner()
    result = runner.invoke(
        dl_init, ["dup", str(tmp_path), "--ingest-token", _TEST_TOKEN],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output
