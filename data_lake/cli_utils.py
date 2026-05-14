"""
Shared CLI helpers for making `dl-ingest-*` commands lake-config aware.

Each ingest CLI accepts an optional ``--config PATH`` (also honoured via
the ``$DATA_LAKE_CONFIG`` env var).  When a config is found, it supplies
defaults for parameters such as ``output_root``, ``norder``,
``wavelength_mode``, etc.  Explicit CLI flags always override the config.
"""

from __future__ import annotations

import logging
import os
import sys
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


_DEDUP_INSTALLED = False


def configure_warning_filters() -> None:
    """Dedup the noisiest warnings emitted by ingest readers.

    FITS-heavy workflows trigger ``astropy.units.UnitsWarning`` once per
    BINTABLE column carrying an unknown survey unit (e.g. ``nanomaggy``
    in DESI catalogs).  Naively setting ``warnings.filterwarnings("once",
    ...)`` does **not** work here: astropy's FITS unit parser calls
    ``warnings.warn_explicit(..., registry=None)`` with a fresh per-column
    registry, which short-circuits Python's ``__onceregistry__`` and
    causes every column to re-emit the same message.

    The only reliable choke point is ``warnings.showwarning``, which both
    ``warn`` and ``warn_explicit`` ultimately call.  We wrap it with a
    dedup that hashes by ``(str(message), category)`` and suppresses
    repeats of ``UnitsWarning`` only; everything else is forwarded to the
    original handler unchanged.  We also disable astropy's own
    warnings-to-logger forwarding (which prints a second ``WARNING:
    astropy:...`` copy per emission).

    Notes
    -----
    * Library code never calls this; only CLI entry points do.  This keeps
      ``import data_lake`` side-effect-free with respect to the user's
      global warning filters.
    * The ``showwarning`` hook lives in the current process only.  Pass
      this function as ``ProcessPoolExecutor(..., initializer=...)`` to
      apply it in every worker.
    * Idempotent: safe to call multiple times.
    """
    global _DEDUP_INSTALLED
    import warnings

    try:
        from astropy.units.core import UnitsWarning
    except ImportError:
        return

    if _DEDUP_INSTALLED:
        return

    try:
        from astropy import log as _astropy_log
        if _astropy_log.warnings_logging_enabled():
            _astropy_log.disable_warnings_logging()
    except Exception:
        # disable_warnings_logging() raises LoggingError if anything else
        # (pytest's warning capture, a prior hook, etc.) has already replaced
        # warnings.showwarning.  In that case astropy's duplicate emitter is
        # already out of the chain, so we can safely ignore it.
        pass

    seen: set[tuple[str, type]] = set()
    original_showwarning = warnings.showwarning

    def _dedup_showwarning(message, category, filename, lineno, file=None, line=None):
        if isinstance(category, type) and issubclass(category, UnitsWarning):
            key = (str(message), category)
            if key in seen:
                return
            seen.add(key)
        return original_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = _dedup_showwarning
    _DEDUP_INSTALLED = True


# Env keys used by ``desi_parallel_ingest`` worker processes (fork/spawn).
PARALLEL_INGEST_WORKER_FLAG = "DATA_LAKE_PARALLEL_INGEST_WORKER"
PARALLEL_WORKER_LOG_FILE_ENV = "DATA_LAKE_WORKER_LOG_FILE"
PARALLEL_WORKER_VERBOSE_ENV = "DATA_LAKE_WORKER_VERBOSE"


def init_parallel_ingest_subprocess() -> None:
    """``ProcessPoolExecutor`` initializer: mark process + apply warning filters."""
    os.environ[PARALLEL_INGEST_WORKER_FLAG] = "1"
    configure_warning_filters()


def apply_parallel_worker_logging_after_heavy_imports() -> None:
    """Detach library loggers from the TTY inside parallel-ingest worker processes.

    The parent CLI attaches ``FileHandler`` to the root logger only in the main
    process.  ``desispec`` (and similar) register their own ``StreamHandler``
    when imported inside workers, which is why ``INFO:…read_spectra`` lines
    still appeared on stderr.  This function strips TTY stream handlers and
    routes everything through the root logger again (file or ``NullHandler``).

    No-op unless ``init_parallel_ingest_subprocess`` has run in this process.
    Safe to call repeatedly (e.g. once per decoded coadd).
    """
    if os.environ.get(PARALLEL_INGEST_WORKER_FLAG) != "1":
        return

    raw = os.environ.get(PARALLEL_WORKER_LOG_FILE_ENV, "")
    log_file = raw.strip() or None
    verbose = os.environ.get(PARALLEL_WORKER_VERBOSE_ENV) == "1"
    level = logging.DEBUG if verbose else logging.INFO

    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(process)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    def _is_tty_stream_handler(handler: logging.Handler) -> bool:
        if isinstance(handler, logging.StreamHandler):
            stream = getattr(handler, "stream", None)
            return stream in (sys.stdout, sys.stderr)
        return False

    # Remove every stderr/stdout stream handler (desispec, astropy, etc.).
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if not isinstance(name, str):
            continue
        lg = logging.getLogger(name)
        for h in lg.handlers[:]:
            if _is_tty_stream_handler(h):
                lg.removeHandler(h)

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    if log_file:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.setLevel(level)
    else:
        root.addHandler(logging.NullHandler())
        root.setLevel(logging.WARNING)

    for prefix in ("desispec", "desi"):
        lg = logging.getLogger(prefix)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(level if log_file else logging.WARNING)


def require_output_root(
    output_root: Path | None,
    cfg: LakeConfig | None,
    kind: str | None = None,  # kept for forward-compat; currently unused
) -> Path:
    """Resolve OUTPUT_ROOT positional arg against the config.

    OUTPUT_ROOT is the **lake root** (``cfg.lake.root``); each ingest
    function appends its own per-kind subdirectory (``catalogs/``,
    ``spectra/``, or ``cutouts/``) underneath.  This matches the layout
    created by ``dl-init``.

    ``kind`` is accepted for forward compatibility (e.g. if we later
    expose ``[paths]`` customisation) but is currently unused.
    """
    del kind  # not used in v1
    if output_root is not None:
        return output_root
    if cfg is None:
        raise click.UsageError(
            "OUTPUT_ROOT is required when no lake config is provided. "
            "Either pass it explicitly, set $DATA_LAKE_CONFIG, or use --config."
        )
    return cfg.lake.root
