"""
LakeConfig - single source of truth for a deployment.

A *deployment* (your private data lake instance, e.g. ``mylake``) is a
directory containing a ``lake_config.toml`` file plus user-managed
notebooks, scripts, and data subdirectories.  The library itself is
deployment-agnostic; everything specific to a particular lake lives in
the config file.

Schema (lake_config.toml)
-------------------------
    schema_version = "1"

    [lake]
    name        = "mylake"                 # short identifier
    description = "Personal multi-survey"  # free text
    root        = "/path/to/lake/data"     # where tiles live
    created_utc = "2026-05-13T11:43:00Z"   # ISO-8601

    [partitioning]
    hats_order       = 5                   # default HEALPix order
    chunks_per_shard = 512                 # Zarr shard size in rows

    [defaults]
    wavelength_mode  = "shared"            # "shared" | "per_source"
    mask_dtype       = "uint8"             # "uint8" | "uint16"
    with_resolution  = false               # default for DESI spectra ingest

    [paths]
    catalogs = "catalogs"                  # subdir under root
    spectra  = "spectra"
    cutouts  = "cutouts"
    shared   = "shared"

    [ingest]
    num_workers = "auto"                   # "auto" | <int>
    log_level   = "INFO"                   # standard logging level

    [guardrails]
    require_ingest_token = false           # when true, dl-ingest-* needs $LAKE_INGEST_TOKEN
    ingest_token_hash  = ""                # SHA-256 hex (optional; prefer .ingest_token_hash)
    ingest_token_file  = ".ingest_token_hash"  # sidecar next to lake_config.toml (gitignored)

Discovery order
---------------
``LakeConfig.discover(explicit_path=None)`` looks for a config in:

    1. ``explicit_path``                            (highest priority)
    2. ``$DATA_LAKE_CONFIG`` environment variable
    3. ``./lake_config.toml`` walking up from CWD   (lowest priority)

Discovery raises :class:`LakeConfigNotFound` if no config is found.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "lake_config.toml"
ENV_VAR = "DATA_LAKE_CONFIG"
SCHEMA_VERSION = "1"


class LakeConfigError(Exception):
    """Base class for lake-config errors."""


class LakeConfigNotFound(LakeConfigError, FileNotFoundError):
    """Raised when no lake_config.toml could be located."""


class LakeConfigInvalid(LakeConfigError, ValueError):
    """Raised when a found lake_config.toml is malformed."""


# ---------------------------------------------------------------------------
# Default values (used when a section/key is absent)
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, dict[str, Any]] = {
    "partitioning": {
        "hats_order": 5,
        "chunks_per_shard": 512,
    },
    "defaults": {
        "wavelength_mode": "shared",
        "mask_dtype": "uint8",
        "with_resolution": False,
    },
    "paths": {
        "catalogs": "catalogs",
        "spectra":  "spectra",
        "cutouts":  "cutouts",
        "shared":   "shared",
    },
    "ingest": {
        "num_workers": "auto",
        "log_level":   "INFO",
    },
    "guardrails": {
        "require_ingest_token": False,
        "ingest_token_hash": "",
        "ingest_token_file": ".ingest_token_hash",
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Lake:
    name: str
    root: Path
    description: str = ""
    created_utc: str = ""


@dataclass(slots=True)
class _Partitioning:
    hats_order: int = 5
    chunks_per_shard: int = 512


@dataclass(slots=True)
class _Defaults:
    wavelength_mode: str = "shared"
    mask_dtype: str = "uint8"
    with_resolution: bool = False


@dataclass(slots=True)
class _Paths:
    catalogs: str = "catalogs"
    spectra:  str = "spectra"
    cutouts:  str = "cutouts"
    shared:   str = "shared"


@dataclass(slots=True)
class _Ingest:
    num_workers: int | str = "auto"
    log_level: str = "INFO"


@dataclass(slots=True)
class _Guardrails:
    require_ingest_token: bool = False
    ingest_token_hash: str = ""
    ingest_token_file: str = ".ingest_token_hash"


@dataclass(slots=True)
class LakeConfig:
    """In-memory representation of a deployment's ``lake_config.toml``."""

    lake: _Lake
    partitioning: _Partitioning = field(default_factory=_Partitioning)
    defaults: _Defaults = field(default_factory=_Defaults)
    paths: _Paths = field(default_factory=_Paths)
    ingest: _Ingest = field(default_factory=_Ingest)
    guardrails: _Guardrails = field(default_factory=_Guardrails)
    schema_version: str = SCHEMA_VERSION
    source_path: Path | None = None

    # ------------------------------------------------------------------
    # Convenience accessors (absolute paths under the lake root)
    # ------------------------------------------------------------------

    @property
    def catalogs_root(self) -> Path:
        return self.lake.root / self.paths.catalogs

    @property
    def spectra_root(self) -> Path:
        return self.lake.root / self.paths.spectra

    @property
    def cutouts_root(self) -> Path:
        return self.lake.root / self.paths.cutouts

    @property
    def shared_root(self) -> Path:
        return self.lake.root / self.paths.shared

    @property
    def resolved_num_workers(self) -> int:
        """Resolve ``num_workers`` to a concrete int (cpu_count for 'auto')."""
        if self.ingest.num_workers == "auto":
            return os.cpu_count() or 1
        return int(self.ingest.num_workers)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "LakeConfig":
        """Load a ``LakeConfig`` from an explicit path."""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise LakeConfigNotFound(f"Lake config not found: {path}")
        try:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise LakeConfigInvalid(f"Could not parse {path}: {exc}") from exc

        cfg = cls._from_dict(data, source_path=path)
        return cfg

    @classmethod
    def discover(cls, explicit_path: str | Path | None = None) -> "LakeConfig":
        """Locate a lake_config.toml via explicit path, env var, or cwd walk-up.

        Parameters
        ----------
        explicit_path:
            If provided and not ``None``, ``LakeConfig.load`` is used directly.

        Returns
        -------
        LakeConfig

        Raises
        ------
        LakeConfigNotFound
            When no config can be located.
        """
        # 1. Explicit path
        if explicit_path is not None:
            return cls.load(explicit_path)

        # 2. Environment variable
        env_val = os.environ.get(ENV_VAR)
        if env_val:
            return cls.load(env_val)

        # 3. Walk up from cwd
        cur = Path.cwd().resolve()
        for candidate_dir in (cur, *cur.parents):
            candidate = candidate_dir / CONFIG_FILENAME
            if candidate.exists():
                return cls.load(candidate)

        raise LakeConfigNotFound(
            f"No {CONFIG_FILENAME} found. Set ${ENV_VAR}, pass an explicit "
            f"path, or create one with `dl-init <name>`."
        )

    # ------------------------------------------------------------------
    # Internal: dict -> LakeConfig
    # ------------------------------------------------------------------

    @classmethod
    def _from_dict(cls, data: dict[str, Any], source_path: Path | None) -> "LakeConfig":
        schema_version = str(data.get("schema_version", SCHEMA_VERSION))
        if schema_version != SCHEMA_VERSION:
            raise LakeConfigInvalid(
                f"schema_version={schema_version!r} not supported by this library "
                f"(expected {SCHEMA_VERSION!r}). Upgrade data-lake or migrate the config."
            )

        if "lake" not in data:
            raise LakeConfigInvalid("Missing required [lake] section.")

        lake_in = data["lake"]
        for required in ("name", "root"):
            if required not in lake_in:
                raise LakeConfigInvalid(f"Missing required key: lake.{required}")

        root = Path(str(lake_in["root"])).expanduser()
        # Resolve root relative to the config file directory (if relative)
        if not root.is_absolute() and source_path is not None:
            root = (source_path.parent / root).resolve()
        elif root.is_absolute():
            root = root.resolve()

        lake = _Lake(
            name=str(lake_in["name"]),
            root=root,
            description=str(lake_in.get("description", "")),
            created_utc=str(lake_in.get("created_utc", "")),
        )

        def _merge(section: str, dc_cls):
            d = {**_DEFAULTS[section], **data.get(section, {})}
            return dc_cls(**d)

        try:
            partitioning = _merge("partitioning", _Partitioning)
            defaults     = _merge("defaults",     _Defaults)
            paths        = _merge("paths",        _Paths)
            ingest       = _merge("ingest",       _Ingest)
            guardrails   = _merge("guardrails",   _Guardrails)
        except TypeError as exc:
            raise LakeConfigInvalid(f"Unknown key in {source_path}: {exc}") from exc

        return cls(
            lake=lake,
            partitioning=partitioning,
            defaults=defaults,
            paths=paths,
            ingest=ingest,
            guardrails=guardrails,
            schema_version=schema_version,
            source_path=source_path,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_toml(self) -> str:
        """Render this config back to a TOML string (round-trippable)."""
        def _q(s: str) -> str:
            # Use double-quoted form; escape backslashes/quotes
            return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

        lines: list[str] = [
            f'schema_version = {_q(self.schema_version)}',
            "",
            "[lake]",
            f'name        = {_q(self.lake.name)}',
            f'description = {_q(self.lake.description)}',
            f'root        = {_q(str(self.lake.root))}',
            f'created_utc = {_q(self.lake.created_utc)}',
            "",
            "[partitioning]",
            f"hats_order       = {self.partitioning.hats_order}",
            f"chunks_per_shard = {self.partitioning.chunks_per_shard}",
            "",
            "[defaults]",
            f'wavelength_mode  = {_q(self.defaults.wavelength_mode)}',
            f'mask_dtype       = {_q(self.defaults.mask_dtype)}',
            f"with_resolution  = {str(self.defaults.with_resolution).lower()}",
            "",
            "[paths]",
            f'catalogs = {_q(self.paths.catalogs)}',
            f'spectra  = {_q(self.paths.spectra)}',
            f'cutouts  = {_q(self.paths.cutouts)}',
            f'shared   = {_q(self.paths.shared)}',
            "",
            "[ingest]",
            f'num_workers = {_q(str(self.ingest.num_workers)) if isinstance(self.ingest.num_workers, str) else self.ingest.num_workers}',
            f'log_level   = {_q(self.ingest.log_level)}',
            "",
            "[guardrails]",
            f"require_ingest_token = {str(self.guardrails.require_ingest_token).lower()}",
            f"ingest_token_hash  = {_q(self.guardrails.ingest_token_hash)}",
            f"ingest_token_file  = {_q(self.guardrails.ingest_token_file)}",
            "",
        ]
        return "\n".join(lines)

    def write(self, path: str | Path) -> None:
        """Write this config to a TOML file."""
        Path(path).write_text(self.to_toml())
