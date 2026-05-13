"""
Tests for ``data_lake.cli_utils.configure_warning_filters``.

The helper deduplicates astropy ``UnitsWarning`` emissions, which spam the
log once per BINTABLE column carrying an unknown survey unit (e.g.
``nanomaggy`` in DESI catalogs).

We do **not** test the obvious ``warnings.filterwarnings("once", ...)``
approach because astropy bypasses Python's once-registry by calling
``warnings.warn_explicit(..., registry=None)`` for each column.  The
helper therefore hooks ``warnings.showwarning`` instead — and these tests
exercise that hook directly, mirroring how astropy actually emits.
"""

from __future__ import annotations

import importlib
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
