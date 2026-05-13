"""
Shared CLI helpers for making `dl-ingest-*` commands lake-config aware.

Each ingest CLI accepts an optional ``--config PATH`` (also honoured via
the ``$DATA_LAKE_CONFIG`` env var).  When a config is found, it supplies
defaults for parameters such as ``output_root``, ``norder``,
``wavelength_mode``, etc.  Explicit CLI flags always override the config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

import click

from .config import LakeConfig, LakeConfigNotFound

F = TypeVar("F", bound=Callable[..., Any])


def config_option(f: F) -> F:
    """Add ``--config`` (also reads ``$DATA_LAKE_CONFIG``) to a click command."""
    return click.option(
        "--config", "config_path",
        type=click.Path(path_type=Path),
        default=None,
        envvar="DATA_LAKE_CONFIG",
        help="Path to a lake_config.toml (or set $DATA_LAKE_CONFIG). "
             "Provides default values for output_root, norder, etc.",
    )(f)


def load_optional_config(config_path: Path | None) -> LakeConfig | None:
    """Discover a LakeConfig or return ``None`` when none is found."""
    try:
        return LakeConfig.discover(config_path)
    except LakeConfigNotFound:
        return None


def pick(cli_value: Any, cfg_value: Any, fallback: Any) -> Any:
    """First-non-None resolver: cli > config > fallback.

    Use ``None`` as the click default for any option you want the config
    to be able to override.  Then in the body call:

        norder = pick(norder, cfg.partitioning.hats_order if cfg else None, 5)
    """
    if cli_value is not None:
        return cli_value
    if cfg_value is not None:
        return cfg_value
    return fallback


def require_output_root(
    output_root: Path | None,
    cfg: LakeConfig | None,
    kind: str,
) -> Path:
    """Resolve OUTPUT_ROOT positional arg against the config.

    ``kind`` is one of ``"catalogs"``, ``"spectra"``, ``"cutouts"`` and
    selects which sub-root from the config is used.
    """
    if output_root is not None:
        return output_root
    if cfg is None:
        raise click.UsageError(
            "OUTPUT_ROOT is required when no lake config is provided. "
            "Either pass it explicitly, set $DATA_LAKE_CONFIG, or use --config."
        )
    attr = f"{kind}_root"
    return getattr(cfg, attr)
