"""Tests for SED assembly and the two-panel figure (Agg backend)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from data_lake.homogenize.bandpass import BandpassRegistry
from data_lake.plot.sed import SEDPoint, SEDPoints, assemble_sed, _discover_phot_ab_columns


# ---------------------------------------------------------------------------
# _discover_phot_ab_columns
# ---------------------------------------------------------------------------


def test_discover_phot_ab_columns_with_errors():
    cols = ["_source_id", "ra", "dec", "phot_ab_w1", "phot_ab_w1_err", "phot_ab_g", "phot_ab_ks"]
    result = _discover_phot_ab_columns(cols)
    assert "phot_ab_w1" in result
    assert result["phot_ab_w1"] == "phot_ab_w1_err"
    assert "phot_ab_g" in result
    assert result["phot_ab_g"] is None  # no _err column
    assert "phot_ab_ks" in result
    assert "_source_id" not in result


def test_discover_phot_ab_columns_empty():
    result = _discover_phot_ab_columns(["ra", "dec", "redshift"])
    assert result == {}


# ---------------------------------------------------------------------------
# assemble_sed — via a mock CatalogAccessor
# ---------------------------------------------------------------------------


def _make_mock_accessor(columns: list[str], row_data: dict, source_id: int):
    """Build a mock CatalogAccessor that returns one row for get_sources_by_id."""
    import pyarrow as pa

    schema = pa.schema([(c, pa.float32() if "phot" in c else pa.int64()) for c in columns])
    arrays = [
        pa.array([row_data.get(c)], type=pa.float32() if "phot" in c else pa.int64())
        for c in columns
    ]
    table = pa.table({c: arr for c, arr in zip(columns, arrays)})

    acc = MagicMock()
    acc.survey_name = "TEST_PRODUCT"
    acc.columns = columns
    acc.get_sources_by_id.return_value = table
    return acc


def test_assemble_sed_basic():
    bp = BandpassRegistry()
    columns = ["_source_id", "phot_ab_w1", "phot_ab_w1_err", "phot_ab_g"]
    row = {"_source_id": 42, "phot_ab_w1": 15.2, "phot_ab_w1_err": 0.05, "phot_ab_g": 18.1}
    acc = _make_mock_accessor(columns, row, 42)

    sed = assemble_sed(acc, 42, bp)

    assert sed.source_id == 42
    bands = [p.band for p in sed.points]
    assert "phot_ab_g" in bands
    assert "phot_ab_w1" in bands
    # Should be sorted by wavelength: Gaia G (0.67 µm) < WISE W1 (3.37 µm)
    lams = [p.lambda_eff_um for p in sed.points]
    assert lams == sorted(lams)


def test_assemble_sed_error_propagated():
    bp = BandpassRegistry()
    columns = ["phot_ab_w1", "phot_ab_w1_err"]
    row = {"phot_ab_w1": 15.2, "phot_ab_w1_err": 0.05}
    acc = _make_mock_accessor(columns, row, 7)

    sed = assemble_sed(acc, 7, bp)
    assert len(sed.points) == 1
    assert sed.points[0].ab_mag_err == pytest.approx(0.05, abs=1e-3)


def test_assemble_sed_skips_nan():
    bp = BandpassRegistry()
    columns = ["phot_ab_w1", "phot_ab_g"]
    row = {"phot_ab_w1": float("nan"), "phot_ab_g": 18.0}
    acc = _make_mock_accessor(columns, row, 1)

    sed = assemble_sed(acc, 1, bp)
    bands = [p.band for p in sed.points]
    assert "phot_ab_w1" not in bands
    assert "phot_ab_g" in bands


def test_assemble_sed_skips_unknown_band():
    """Bands not in the registry are silently skipped."""
    bp = BandpassRegistry()
    columns = ["phot_ab_exotic_x"]  # not in bandpass.json
    row = {"phot_ab_exotic_x": 14.0}
    acc = _make_mock_accessor(columns, row, 1)

    sed = assemble_sed(acc, 1, bp)
    assert sed.points == []


def test_assemble_sed_raises_when_source_missing():
    import pyarrow as pa

    bp = BandpassRegistry()
    acc = MagicMock()
    acc.survey_name = "PROD"
    acc.columns = ["phot_ab_w1"]
    acc.get_sources_by_id.return_value = pa.table({"phot_ab_w1": pa.array([], type=pa.float32())})

    with pytest.raises(KeyError):
        assemble_sed(acc, 999, bp)


def test_sed_points_arrays():
    pts = [
        SEDPoint("phot_ab_g", 0.674, 18.0, 0.1),
        SEDPoint("phot_ab_w1", 3.37, 15.0, 0.05),
    ]
    sed = SEDPoints(source_id=1, points=pts)
    np.testing.assert_allclose(sed.lambda_eff_um, [0.674, 3.37])
    np.testing.assert_allclose(sed.ab_mag, [18.0, 15.0])
    np.testing.assert_allclose(sed.ab_mag_err, [0.1, 0.05])


# ---------------------------------------------------------------------------
# Two-panel figure smoke test (matplotlib Agg)
# ---------------------------------------------------------------------------


def _make_synthetic_spectrum():
    """Return a minimal Spectrum-like object (duck typing)."""
    from data_lake.io.spectra import Spectrum

    n = 100
    flux = np.random.default_rng(0).standard_normal(n).astype(np.float32) + 10.0
    ivar = np.full(n, 4.0, dtype=np.float32)
    mask = np.zeros(n, dtype=np.uint8)
    wave = np.linspace(3600.0, 9800.0, n)
    meta = {"z": 0.1, "instr": "TEST"}
    wcs_attrs = {"ctype": "WAVE", "crval": 3600.0, "cdelt": 62.0, "crpix": 1.0, "unit": "Angstrom"}
    return Spectrum(
        source_id=42,
        flux=flux,
        ivar=ivar,
        mask=mask,
        wavelength=wave,
        meta=meta,
        wcs_attrs=wcs_attrs,
    )


def test_plot_source_sed_spectrum_returns_figure():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from data_lake.plot.source_figure import plot_source_sed_spectrum

    bp = BandpassRegistry()
    pts = [
        SEDPoint("phot_ab_g", 0.674, 18.0, 0.1, "Gaia G"),
        SEDPoint("phot_ab_w1", 3.37, 15.0, 0.05, "WISE W1"),
    ]
    sed = SEDPoints(source_id=42, points=pts)
    spectrum = _make_synthetic_spectrum()

    fig = plot_source_sed_spectrum(sed, spectrum, bandpass_registry=bp)
    assert isinstance(fig, Figure)
    # Should have at least 2 axes (SED + spectrum; possibly more from twinx)
    assert len(fig.axes) >= 2
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_empty_sed_does_not_crash():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from data_lake.plot.source_figure import plot_source_sed_spectrum

    sed = SEDPoints(source_id=1, points=[])
    spectrum = _make_synthetic_spectrum()

    fig = plot_source_sed_spectrum(sed, spectrum)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)
