"""
Tests for ``data_lake.cli_utils``.

Covers ``configure_warning_filters`` (dedup of astropy UnitsWarning),
``resolve_log_level``, ``configure_cli_logging``, ``validate_quiet_verbose``,
and ``HeartbeatReporter``.
"""

from __future__ import annotations

import importlib
import io
import logging
import time
import warnings

import pytest


def _reset_module_state():
    """Re-import cli_utils so ``_DEDUP_INSTALLED`` starts False each test."""
    import data_lake.cli_utils as mod

    mod._DEDUP_INSTALLED = False
    return mod


class TestConfigureWarningFilters:
    def test_installs_showwarning_hook(self):
        """After calling, ``warnings.showwarning`` must be wrapped."""
        pytest.importorskip("astropy")
        mod = _reset_module_state()

        original = warnings.showwarning
        try:
            mod.configure_warning_filters()
            assert warnings.showwarning is not original, (
                "configure_warning_filters() must replace warnings.showwarning "
                "with a dedup wrapper."
            )
        finally:
            warnings.showwarning = original
            mod._DEDUP_INSTALLED = False

    def test_repeated_unitswarning_emitted_once(self):
        """Identical UnitsWarning emissions through ``showwarning`` collapse to one."""
        pytest.importorskip("astropy")
        from astropy.units.core import UnitsWarning

        mod = _reset_module_state()

        original = warnings.showwarning
        emitted: list[tuple[str, type]] = []

        def recorder(message, category, filename, lineno, file=None, line=None):
            emitted.append((str(message), category))

        warnings.showwarning = recorder
        try:
            mod.configure_warning_filters()
            for _ in range(5):
                warnings.showwarning(
                    "'nanomaggy' did not parse as fits unit",
                    UnitsWarning,
                    "<test>",
                    1,
                )
        finally:
            warnings.showwarning = original
            mod._DEDUP_INSTALLED = False

        relevant = [e for e in emitted if issubclass(e[1], UnitsWarning)]
        assert len(relevant) == 1, (
            f"Expected exactly 1 UnitsWarning to reach the underlying "
            f"showwarning, got {len(relevant)}"
        )

    def test_distinct_unitswarning_messages_each_emit_once(self):
        """Different message strings are NOT collapsed together."""
        pytest.importorskip("astropy")
        from astropy.units.core import UnitsWarning

        mod = _reset_module_state()

        original = warnings.showwarning
        emitted: list[str] = []

        def recorder(message, category, filename, lineno, file=None, line=None):
            emitted.append(str(message))

        warnings.showwarning = recorder
        try:
            mod.configure_warning_filters()
            for msg in ["'nanomaggy' bad", "'nanomaggy^-2' bad", "'nanomaggy' bad"]:
                warnings.showwarning(msg, UnitsWarning, "<test>", 1)
        finally:
            warnings.showwarning = original
            mod._DEDUP_INSTALLED = False

        assert sorted(emitted) == ["'nanomaggy' bad", "'nanomaggy^-2' bad"], (
            "Distinct UnitsWarning messages must each be emitted once."
        )

    def test_non_unitswarning_passes_through_unchanged(self):
        """Non-UnitsWarning categories are forwarded unconditionally."""
        pytest.importorskip("astropy")

        mod = _reset_module_state()

        original = warnings.showwarning
        emitted: list[tuple[str, type]] = []

        def recorder(message, category, filename, lineno, file=None, line=None):
            emitted.append((str(message), category))

        warnings.showwarning = recorder
        try:
            mod.configure_warning_filters()
            for _ in range(3):
                warnings.showwarning("deprecated thing", DeprecationWarning, "<t>", 1)
        finally:
            warnings.showwarning = original
            mod._DEDUP_INSTALLED = False

        assert len(emitted) == 3, (
            "Non-UnitsWarning categories must not be deduplicated."
        )

    def test_idempotent(self):
        """Calling twice does not double-wrap or break dedup."""
        pytest.importorskip("astropy")
        from astropy.units.core import UnitsWarning

        mod = _reset_module_state()

        original = warnings.showwarning
        emitted: list[str] = []

        def recorder(message, *_, **__):
            emitted.append(str(message))

        warnings.showwarning = recorder
        try:
            mod.configure_warning_filters()
            hook_after_first = warnings.showwarning
            mod.configure_warning_filters()
            hook_after_second = warnings.showwarning

            assert hook_after_first is hook_after_second, (
                "Second call must not re-wrap the showwarning hook."
            )

            for _ in range(4):
                warnings.showwarning("dup msg", UnitsWarning, "<t>", 1)
        finally:
            warnings.showwarning = original
            mod._DEDUP_INSTALLED = False

        assert emitted == ["dup msg"], (
            f"Idempotent install must still dedup; got {emitted!r}"
        )

    def test_no_astropy_is_silent(self, monkeypatch):
        """Helper degrades gracefully when astropy is unavailable."""
        import sys

        mod = _reset_module_state()

        original = warnings.showwarning
        try:
            monkeypatch.setitem(sys.modules, "astropy.units.core", None)
            mod.configure_warning_filters()
            assert warnings.showwarning is original, (
                "Without astropy, configure_warning_filters() must be a no-op."
            )
            assert mod._DEDUP_INSTALLED is False
        finally:
            warnings.showwarning = original
            mod._DEDUP_INSTALLED = False


class TestResolveLogLevel:
    def test_verbose_returns_debug(self):
        from data_lake.cli_utils import resolve_log_level

        assert resolve_log_level(quiet=False, verbose=True) == logging.DEBUG

    def test_quiet_returns_warning(self):
        from data_lake.cli_utils import resolve_log_level

        assert resolve_log_level(quiet=True, verbose=False) == logging.WARNING

    def test_config_level_parsed(self):
        from data_lake.cli_utils import resolve_log_level

        assert resolve_log_level(quiet=False, verbose=False, config_level="DEBUG") == logging.DEBUG
        assert resolve_log_level(quiet=False, verbose=False, config_level="WARNING") == logging.WARNING

    def test_default_is_info(self):
        from data_lake.cli_utils import resolve_log_level

        assert resolve_log_level(quiet=False, verbose=False) == logging.INFO

    def test_cli_flag_beats_config(self):
        from data_lake.cli_utils import resolve_log_level

        assert resolve_log_level(quiet=True, verbose=False, config_level="DEBUG") == logging.WARNING
        assert resolve_log_level(quiet=False, verbose=True, config_level="WARNING") == logging.DEBUG


class TestValidateQuietVerbose:
    def test_mutual_exclusion_raises(self):
        import click

        from data_lake.cli_utils import validate_quiet_verbose

        with pytest.raises(click.UsageError):
            validate_quiet_verbose(quiet=True, verbose=True)

    def test_each_alone_is_ok(self):
        from data_lake.cli_utils import validate_quiet_verbose

        validate_quiet_verbose(quiet=True, verbose=False)
        validate_quiet_verbose(quiet=False, verbose=True)
        validate_quiet_verbose(quiet=False, verbose=False)


class TestConfigureCliLogging:
    def test_quiet_suppresses_info_on_stderr(self, tmp_path):
        """In quiet mode, INFO records must not reach stderr."""
        from data_lake.cli_utils import configure_cli_logging

        stream = io.StringIO()
        configure_cli_logging(level=logging.WARNING, log_file=None, quiet=True)
        root = logging.getLogger()
        # Patch the last handler to redirect to our StringIO
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.stream = stream
                break

        logging.getLogger("test.quiet").info("should-not-appear")
        logging.getLogger("test.quiet").warning("should-appear")

        out = stream.getvalue()
        assert "should-not-appear" not in out
        assert "should-appear" in out

        logging.basicConfig(level=logging.WARNING, force=True)  # reset

    def test_log_file_receives_info_in_quiet_mode(self, tmp_path):
        """Even with quiet mode, the log file gets INFO."""
        from data_lake.cli_utils import configure_cli_logging

        log_file = tmp_path / "test.log"
        configure_cli_logging(level=logging.WARNING, log_file=log_file, quiet=True)
        lg = logging.getLogger("test.logfile")
        lg.info("file-info-message")
        lg.warning("file-warn-message")

        # Flush handlers
        for h in logging.getLogger().handlers:
            h.flush()

        content = log_file.read_text()
        assert "file-info-message" in content
        assert "file-warn-message" in content

        logging.basicConfig(level=logging.WARNING, force=True)  # reset


class TestHeartbeatReporter:
    def test_disabled_when_interval_zero(self):
        from data_lake.cli_utils import HeartbeatReporter

        hb = HeartbeatReporter(total=10, interval_s=0)
        assert not hb.enabled

    def test_enabled_when_interval_positive(self):
        from data_lake.cli_utils import HeartbeatReporter

        hb = HeartbeatReporter(total=10, interval_s=5)
        assert hb.enabled

    def test_emits_at_most_once_per_interval(self, capsys):
        from data_lake.cli_utils import HeartbeatReporter

        hb = HeartbeatReporter(total=100, interval_s=9999, label="test")
        # Should not emit on first update (interval not elapsed)
        hb.update(done=1)
        hb.update(done=1)
        out = capsys.readouterr().err
        assert out == "", f"Expected no output, got {out!r}"

    def test_force_emits(self, capsys):
        from data_lake.cli_utils import HeartbeatReporter

        hb = HeartbeatReporter(total=100, interval_s=9999, label="hbtest")
        hb.update(done=5, spectra=50, force=True)
        out = capsys.readouterr().err
        assert "hbtest" in out
        assert "5/100" in out

    def test_final_emits(self, capsys):
        from data_lake.cli_utils import HeartbeatReporter

        hb = HeartbeatReporter(total=3, interval_s=9999, label="fintest")
        hb.update(done=3)
        hb.final()
        out = capsys.readouterr().err
        assert "fintest" in out


class TestIntegrationWithFITS:
    """End-to-end: read a FITS table whose columns carry unknown units."""

    def test_dedups_real_unitswarning_from_table_read(self, tmp_path):
        pytest.importorskip("astropy")
        import numpy as np
        from astropy.io import fits
        from astropy.table import Table
        from astropy.units.core import UnitsWarning

        mod = _reset_module_state()

        # Synthetic FITS BINTABLE with five columns all tagged 'nanomaggy'.
        cols = [
            fits.Column(
                name=f"FLUX_{i}",
                format="E",
                unit="nanomaggy",
                array=np.arange(3, dtype=np.float32),
            )
            for i in range(5)
        ]
        fits_path = tmp_path / "synth.fits"
        fits.BinTableHDU.from_columns(cols).writeto(fits_path, overwrite=True)

        original = warnings.showwarning
        emitted: list[tuple[str, type]] = []

        def recorder(message, category, filename, lineno, file=None, line=None):
            emitted.append((str(message), category))

        warnings.showwarning = recorder
        try:
            mod.configure_warning_filters()
            Table.read(fits_path)
        finally:
            warnings.showwarning = original
            mod._DEDUP_INSTALLED = False

        unit_warnings = [e for e in emitted if issubclass(e[1], UnitsWarning)]
        assert len(unit_warnings) == 1, (
            f"Reading 5 nanomaggy columns should produce exactly 1 "
            f"UnitsWarning at the showwarning boundary; got {len(unit_warnings)}.\n"
            f"Emissions: {unit_warnings}"
        )
