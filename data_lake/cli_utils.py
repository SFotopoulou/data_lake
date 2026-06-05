"""
Shared CLI helpers for making `dl-ingest-*` commands lake-config aware.

Each ingest CLI accepts an optional ``--config PATH`` (also honoured via
the ``$DATA_LAKE_CONFIG`` env var).  When a config is found, it supplies
defaults for parameters such as ``output_root``, ``norder``,
``wavelength_mode``, etc.  Explicit CLI flags always override the config.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import click

from .config import LakeConfig, LakeConfigNotFound

INGEST_TOKEN_ENV = "LAKE_INGEST_TOKEN"
INGEST_TOKEN_HASH_FILENAME = ".ingest_token_hash"

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


def ingest_token_option(f: F) -> F:
    """Add ``--ingest-token`` (also reads ``$LAKE_INGEST_TOKEN``) to a write ingest CLI."""
    return click.option(
        "--ingest-token",
        "ingest_token",
        default=None,
        envvar=INGEST_TOKEN_ENV,
        help="Ingest operator token (deployment hash in .ingest_token_hash; "
             f"prefer ${INGEST_TOKEN_ENV} in batch jobs).",
    )(f)


def hash_ingest_token(token: str) -> str:
    """Return the SHA-256 hex digest stored in the deployment ``.ingest_token_hash``."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def ingest_token_hash_path(deployment_dir: Path, filename: str | None = None) -> Path:
    """Path to the hidden sidecar file that stores the expected token hash."""
    return deployment_dir / (filename or INGEST_TOKEN_HASH_FILENAME)


def write_ingest_token_hash(
    deployment_dir: Path,
    token: str,
    *,
    filename: str | None = None,
) -> Path:
    """Write ``.ingest_token_hash`` (mode 0600) next to ``lake_config.toml``.

    Only the digest is stored; the plaintext token is never written to disk.
    """
    path = ingest_token_hash_path(deployment_dir, filename)
    path.write_text(hash_ingest_token(token) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass  # best-effort on non-Unix
    return path


def resolve_expected_ingest_hash(cfg: LakeConfig) -> str:
    """Expected SHA-256 hex from the deployment sidecar next to ``lake_config.toml``."""
    if cfg.source_path is None:
        return ""

    rel = (cfg.guardrails.ingest_token_file or INGEST_TOKEN_HASH_FILENAME).strip()
    if not rel:
        return ""

    sidecar = cfg.source_path.parent / rel
    if not sidecar.is_file():
        return ""

    return sidecar.read_text(encoding="utf-8").strip().lower()


def require_ingest_permission(
    cfg: LakeConfig | None,
    ingest_token: str | None = None,
) -> None:
    """Require a valid ingest token before any ``dl-ingest-*`` write.

    Rudimentary operator authentication only: anyone with shell access, the token,
    and write permission on the data path can still ingest.  Real separation uses
    filesystem ACLs (read-only lake for analysts) and future auth integration.
    """
    if cfg is None:
        raise click.ClickException(
            "Ingest requires a deployment config.  Set $DATA_LAKE_CONFIG or pass "
            "--config, then provide an ingest token (dl-init / dl-set-ingest-token)."
        )

    expected = resolve_expected_ingest_hash(cfg)
    if not expected:
        raise click.ClickException(
            "No ingest token hash found for this deployment.  Run:\n"
            "  dl-init NAME PARENT --ingest-token 'your-secret'\n"
            "or: dl-set-ingest-token --ingest-token 'your-secret'"
        )

    provided = (ingest_token or os.environ.get(INGEST_TOKEN_ENV) or "").strip()
    if not provided:
        raise click.ClickException(
            "This deployment requires an ingest token.  Export "
            f"${INGEST_TOKEN_ENV} or pass --ingest-token before running ingest."
        )

    got = hash_ingest_token(provided).lower()
    if not secrets.compare_digest(got, expected):
        raise click.ClickException("Invalid ingest token for this deployment.")


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
    # Import desispec once per worker (not once per coadd) and silence TTY loggers.
    try:
        from data_lake.ingest.fits_to_spectra_zarr import _import_desispec

        _import_desispec()
        apply_parallel_worker_logging_after_heavy_imports()
    except ImportError:
        pass


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


# ---------------------------------------------------------------------------
# Logging helpers (shared across all dl-* CLIs)
# ---------------------------------------------------------------------------

_LOG_FORMAT = "[%(asctime)s] %(name)-30s %(levelname)-7s %(message)s"
_LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def resolve_log_level(
    *,
    quiet: bool,
    verbose: bool,
    config_level: str | None = None,
) -> int:
    """Resolve the effective terminal log level from CLI flags and config.

    Priority: ``-v`` / ``-q`` CLI flag > ``config_level`` > INFO default.
    ``quiet`` and ``verbose`` must not both be True (call site should
    validate with ``click.UsageError``).
    """
    if verbose:
        return logging.DEBUG
    if quiet:
        return logging.WARNING
    if config_level:
        return getattr(logging, config_level.upper(), logging.INFO)
    return logging.INFO


def configure_cli_logging(
    *,
    level: int = logging.INFO,
    log_file: Path | None = None,
    quiet: bool = False,
) -> None:
    """Set up logging for a ``dl-*`` CLI entry point.

    Terminal (stderr) receives messages at *level*.  When a *log_file* is
    provided, a ``FileHandler`` is added:

    - In quiet mode the file always receives INFO (preserving the audit
      trail even when the terminal shows only WARNING+).
    - Otherwise the file uses the same *level* as the terminal.

    Noisy library loggers (``desispec``, ``astropy``) are silenced on the
    terminal when *quiet* or *level* >= WARNING.

    Always calls :func:`configure_warning_filters` to deduplicate
    ``astropy.units.UnitsWarning``.
    """
    fmt = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATE_FMT)
    handlers: list[logging.Handler] = []

    # Terminal handler
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    handlers.append(sh)

    # Optional file handler — always at least INFO for audit trail
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_level = logging.INFO if quiet else level
        fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(file_level)
        handlers.append(fh)

    logging.basicConfig(level=min(level, logging.INFO), handlers=handlers, force=True)
    # Let root level be low enough so FileHandler always gets INFO
    logging.getLogger().setLevel(min(level, logging.INFO))

    # Silence noisy library loggers on the terminal when quiet
    if quiet or level >= logging.WARNING:
        for prefix in ("desispec", "desi", "astropy", "fitsio"):
            lg = logging.getLogger(prefix)
            lg.setLevel(logging.WARNING)

    configure_warning_filters()


def configure_file_only_logging(log_file: Path, *, verbose: bool) -> None:
    """Send *all* logging to *log_file* and detach stderr entirely.

    Used by batch parallel CLIs (DESI, spPlate) where the tqdm progress
    bar owns stderr.  ``verbose=True`` sets DEBUG; otherwise INFO.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATE_FMT))
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, handlers=[fh], force=True)


def logging_options(f: F) -> F:
    """Add ``-q`` / ``--quiet`` and ``-v`` / ``--verbose`` to a click command."""
    f = click.option(
        "-q", "--quiet",
        is_flag=True,
        default=False,
        help="Only print warnings and errors on the terminal.  "
             "Use with --log-file to keep a full INFO audit trail.",
    )(f)
    f = click.option(
        "-v", "--verbose",
        is_flag=True,
        default=False,
        help="Print DEBUG-level output (overrides -q).",
    )(f)
    return f


def validate_quiet_verbose(quiet: bool, verbose: bool) -> None:
    """Raise ``click.UsageError`` when ``-q`` and ``-v`` are both set."""
    if quiet and verbose:
        raise click.UsageError("--quiet and --verbose are mutually exclusive.")


class HeartbeatReporter:
    """Emit a one-line progress summary on a fixed interval.

    Designed for long ingest loops where per-file INFO is suppressed.
    Call :meth:`tick` once per file; a summary is printed to stderr at
    most once per *interval_s* seconds (uses ``time.monotonic``).

    Parameters
    ----------
    total:
        Total number of items expected, or 0 if unknown.
    interval_s:
        Minimum seconds between heartbeat lines.  0 disables.
    label:
        Short label for the run (e.g. ``"ingest"``).
    """

    def __init__(
        self,
        total: int = 0,
        *,
        interval_s: float = 300.0,
        label: str = "ingest",
    ) -> None:
        self._total = total
        self._interval = interval_s
        self._label = label
        self._t_start = time.monotonic()
        self._t_last = self._t_start
        self._n_done = 0
        self._n_fail = 0
        self._n_spectra = 0

    @property
    def enabled(self) -> bool:
        return self._interval > 0

    def update(
        self,
        *,
        done: int = 0,
        failed: int = 0,
        spectra: int = 0,
        force: bool = False,
    ) -> None:
        """Accumulate counters and emit a heartbeat if the interval has elapsed."""
        self._n_done += done
        self._n_fail += failed
        self._n_spectra += spectra

        if not self.enabled:
            return
        now = time.monotonic()
        if not force and (now - self._t_last) < self._interval:
            return

        elapsed = now - self._t_start
        rate = self._n_done / elapsed if elapsed > 0 else 0.0
        parts: list[str] = []
        if self._total > 0:
            pct = 100 * self._n_done / self._total
            parts.append(f"{self._n_done}/{self._total} files ({pct:.0f}%)")
            remaining = (self._total - self._n_done) / rate if rate > 0 else float("inf")
            if remaining < float("inf"):
                eta_h = remaining / 3600
                parts.append(f"ETA ~{eta_h:.1f}h" if eta_h >= 0.1 else f"ETA ~{remaining:.0f}s")
        else:
            parts.append(f"{self._n_done} files")
        if self._n_spectra:
            parts.append(f"{self._n_spectra:,} spectra")
        if self._n_fail:
            parts.append(f"{self._n_fail} failed")
        parts.append(f"{rate:.2f} files/s")

        click.echo(
            f"[{self._label}] {', '.join(parts)} ({elapsed / 60:.1f} min elapsed)",
            err=True,
        )
        self._t_last = now

    def final(self) -> None:
        """Emit one last heartbeat regardless of interval."""
        self.update(force=True)


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
