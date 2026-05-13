"""Tests for ``data_lake.config.LakeConfig``."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from data_lake.config import (
    CONFIG_FILENAME,
    ENV_VAR,
    SCHEMA_VERSION,
    LakeConfig,
    LakeConfigInvalid,
    LakeConfigNotFound,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_minimal_config(path: Path, root: str = "/tmp/lake_data") -> None:
    path.write_text(textwrap.dedent(f"""
        schema_version = "{SCHEMA_VERSION}"

        [lake]
        name = "test_lake"
        root = "{root}"
        description = "minimal"
    """).strip() + "\n")


# ---------------------------------------------------------------------------
# load(): explicit path
# ---------------------------------------------------------------------------

def test_load_minimal_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / CONFIG_FILENAME
    _write_minimal_config(cfg_path, root=str(tmp_path / "data"))

    cfg = LakeConfig.load(cfg_path)

    assert cfg.lake.name == "test_lake"
    assert cfg.lake.root == (tmp_path / "data").resolve()
    assert cfg.partitioning.hats_order == 5
    assert cfg.defaults.wavelength_mode == "shared"
    assert cfg.defaults.with_resolution is False
    assert cfg.paths.spectra == "spectra"
    assert cfg.ingest.num_workers == "auto"
    assert cfg.source_path == cfg_path.resolve()


def test_load_relative_root_resolved_against_config_dir(tmp_path: Path) -> None:
    """root = './data' must resolve against the config file's directory."""
    cfg_path = tmp_path / CONFIG_FILENAME
    _write_minimal_config(cfg_path, root="./data")

    cfg = LakeConfig.load(cfg_path)
    assert cfg.lake.root == (tmp_path / "data").resolve()


def test_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(LakeConfigNotFound):
        LakeConfig.load(tmp_path / "does_not_exist.toml")


def test_load_invalid_toml_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / CONFIG_FILENAME
    cfg_path.write_text("this is = not = valid = toml [\n")
    with pytest.raises(LakeConfigInvalid):
        LakeConfig.load(cfg_path)


def test_load_missing_required_section_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / CONFIG_FILENAME
    cfg_path.write_text(f'schema_version = "{SCHEMA_VERSION}"\n')
    with pytest.raises(LakeConfigInvalid, match=r"\[lake\]"):
        LakeConfig.load(cfg_path)


def test_load_missing_required_key_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / CONFIG_FILENAME
    cfg_path.write_text(textwrap.dedent(f"""
        schema_version = "{SCHEMA_VERSION}"
        [lake]
        name = "lake_without_root"
    """).strip() + "\n")
    with pytest.raises(LakeConfigInvalid, match=r"lake\.root"):
        LakeConfig.load(cfg_path)


def test_load_unknown_schema_version_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / CONFIG_FILENAME
    cfg_path.write_text(textwrap.dedent("""
        schema_version = "99"
        [lake]
        name = "x"
        root = "/tmp/x"
    """).strip() + "\n")
    with pytest.raises(LakeConfigInvalid, match=r"schema_version"):
        LakeConfig.load(cfg_path)


# ---------------------------------------------------------------------------
# discover(): explicit > env > cwd walk-up
# ---------------------------------------------------------------------------

def test_discover_explicit_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "explicit" / CONFIG_FILENAME
    env_cfg  = tmp_path / "env"      / CONFIG_FILENAME
    explicit.parent.mkdir()
    env_cfg.parent.mkdir()
    _write_minimal_config(explicit, root=str(tmp_path / "explicit_data"))
    _write_minimal_config(env_cfg,  root=str(tmp_path / "env_data"))

    monkeypatch.setenv(ENV_VAR, str(env_cfg))
    cfg = LakeConfig.discover(explicit)
    assert cfg.source_path == explicit.resolve()


def test_discover_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_cfg = tmp_path / "env" / CONFIG_FILENAME
    env_cfg.parent.mkdir()
    _write_minimal_config(env_cfg, root=str(tmp_path / "env_data"))

    monkeypatch.setenv(ENV_VAR, str(env_cfg))
    # cwd is somewhere with no config
    monkeypatch.chdir(tmp_path)
    cfg = LakeConfig.discover()
    assert cfg.source_path == env_cfg.resolve()


def test_discover_cwd_walk_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / CONFIG_FILENAME
    _write_minimal_config(cfg_path, root=str(tmp_path / "data"))

    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)

    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.chdir(deep)
    cfg = LakeConfig.discover()
    assert cfg.source_path == cfg_path.resolve()


def test_discover_raises_when_nothing_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(LakeConfigNotFound):
        LakeConfig.discover()


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def test_subdir_properties(tmp_path: Path) -> None:
    cfg_path = tmp_path / CONFIG_FILENAME
    _write_minimal_config(cfg_path, root=str(tmp_path / "data"))
    cfg = LakeConfig.load(cfg_path)

    assert cfg.catalogs_root == (tmp_path / "data" / "catalogs").resolve()
    assert cfg.spectra_root  == (tmp_path / "data" / "spectra").resolve()
    assert cfg.cutouts_root  == (tmp_path / "data" / "cutouts").resolve()
    assert cfg.shared_root   == (tmp_path / "data" / "shared").resolve()


def test_resolved_num_workers_auto(tmp_path: Path) -> None:
    cfg_path = tmp_path / CONFIG_FILENAME
    _write_minimal_config(cfg_path, root=str(tmp_path / "data"))
    cfg = LakeConfig.load(cfg_path)

    n = cfg.resolved_num_workers
    assert isinstance(n, int) and n >= 1
    assert n == (os.cpu_count() or 1)


def test_resolved_num_workers_explicit(tmp_path: Path) -> None:
    cfg_path = tmp_path / CONFIG_FILENAME
    cfg_path.write_text(textwrap.dedent(f"""
        schema_version = "{SCHEMA_VERSION}"
        [lake]
        name = "x"
        root = "{tmp_path / 'data'}"
        [ingest]
        num_workers = 4
    """).strip() + "\n")
    cfg = LakeConfig.load(cfg_path)
    assert cfg.resolved_num_workers == 4


# ---------------------------------------------------------------------------
# Round-trip: write -> load
# ---------------------------------------------------------------------------

def test_write_then_load_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / CONFIG_FILENAME
    _write_minimal_config(cfg_path, root=str(tmp_path / "data"))
    cfg = LakeConfig.load(cfg_path)

    out_path = tmp_path / "copy.toml"
    cfg.write(out_path)

    cfg2 = LakeConfig.load(out_path)
    assert cfg2.lake.name == cfg.lake.name
    assert cfg2.lake.root == cfg.lake.root
    assert cfg2.partitioning.hats_order == cfg.partitioning.hats_order
    assert cfg2.defaults.wavelength_mode == cfg.defaults.wavelength_mode
    assert cfg2.paths.spectra == cfg.paths.spectra
    assert cfg2.ingest.num_workers == cfg.ingest.num_workers
