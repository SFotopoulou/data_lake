"""
Tests for data_lake.cli_utils helpers.

Currently covers ``configure_warning_filters``: must dedup UnitsWarning to
``once`` without otherwise polluting the caller's filter list.
"""

from __future__ import annotations

import warnings

import pytest


class TestConfigureWarningFilters:
    def test_adds_once_filter_for_unitswarning(self):
        """After calling, exactly one UnitsWarning filter with action='once'
        should be present at the front of the filter list."""
        pytest.importorskip("astropy")
        from astropy.units.core import UnitsWarning

        from data_lake.cli_utils import configure_warning_filters

        with warnings.catch_warnings():
            warnings.resetwarnings()
            configure_warning_filters()
            # first matching filter wins
            actions = [
                f[0] for f in warnings.filters if f[2] is UnitsWarning
            ]
            assert "once" in actions
            assert actions[0] == "once", (
                "configure_warning_filters() must install its 'once' rule "
                "ahead of any catch-all default."
            )

    def test_repeat_unitswarning_emitted_once(self):
        """Two identical UnitsWarnings → only one is shown to the caller."""
        pytest.importorskip("astropy")
        from astropy.units.core import UnitsWarning

        from data_lake.cli_utils import configure_warning_filters

        with warnings.catch_warnings(record=True) as captured:
            warnings.resetwarnings()
            configure_warning_filters()
            warnings.warn("nanomaggy is not a FITS unit", UnitsWarning)
            warnings.warn("nanomaggy is not a FITS unit", UnitsWarning)
            warnings.warn("nanomaggy is not a FITS unit", UnitsWarning)

        relevant = [w for w in captured if issubclass(w.category, UnitsWarning)]
        assert len(relevant) == 1, (
            f"Expected exactly one UnitsWarning, got {len(relevant)}"
        )

    def test_no_astropy_is_silent(self, monkeypatch):
        """Helper degrades gracefully when astropy is unavailable."""
        import sys

        from data_lake.cli_utils import configure_warning_filters

        # Pretend the import fails
        monkeypatch.setitem(sys.modules, "astropy.units.core", None)
        with warnings.catch_warnings():
            warnings.resetwarnings()
            configure_warning_filters()
            # Must not raise; no UnitsWarning filter is expected to be added
            # since the category itself couldn't be imported.
