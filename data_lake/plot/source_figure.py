"""
Two-panel SED + 1D spectrum figure for a single source.

Requires the ``viz`` optional extra::

    pip install "data-lake[viz]"

Layout
------
- **Top panel**: spectral energy distribution (SED).
  AB magnitudes (y inverted) at effective wavelengths of all photometric
  bands found in the product catalog, with error bars where available.
  When a transmission-curve ECSV file is registered for a band, a
  semi-transparent shaded region is drawn on a secondary y-axis.

- **Bottom panel**: 1D spectrum.
  Observed-frame flux (native units from the Zarr store) with a 1σ
  uncertainty band shaded in grey. Shared log-wavelength x-axis in micron.

Usage
-----
>>> from data_lake.plot.source_figure import plot_source_sed_spectrum
>>> fig = plot_source_sed_spectrum(sed, spectrum, title="Source 42")
>>> fig.savefig("source_42.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from data_lake.homogenize.bandpass import BandpassRegistry
    from data_lake.io.spectra import Spectrum
    from data_lake.plot.sed import SEDPoints

# Colour cycle compatible with both light and dark themes.
_BAND_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _require_matplotlib():
    """Raise a helpful ImportError when matplotlib is not installed."""
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib.  Install it with:\n"
            "    pip install \"data-lake[viz]\"\n"
            "or:\n"
            "    pip install matplotlib"
        ) from exc


def _wave_angstrom_to_um(wave_angstrom: np.ndarray) -> np.ndarray:
    return wave_angstrom * 1e-4


def plot_source_sed_spectrum(
    sed: SEDPoints,
    spectrum: Spectrum,
    *,
    title: str | None = None,
    bandpass_registry: BandpassRegistry | None = None,
    show_curves: bool = True,
    figsize: tuple[float, float] = (10, 7),
    dpi: int = 150,
    spectrum_rest_frame: bool = False,
    extra_kwargs: dict[str, Any] | None = None,
):
    """
    Render a two-panel SED + 1D spectrum figure.

    Parameters
    ----------
    sed:
        SED points from :func:`~data_lake.plot.sed.assemble_sed`.
    spectrum:
        Spectrum from ``SpectrumAccessor.get_spectrum``.
    title:
        Figure suptitle.  Defaults to ``"Source <source_id>"``.
    bandpass_registry:
        When provided and *show_curves* is ``True``, transmission curves are
        shaded in the SED panel (requires curve ECSV files).
    show_curves:
        Draw semi-transparent bandpass transmission regions (default ``True``).
        Requires *bandpass_registry*.
    figsize:
        Matplotlib figure size in inches ``(width, height)``.
    dpi:
        Dots per inch for raster backends.
    spectrum_rest_frame:
        If ``True``, convert spectrum wavelength to rest-frame using
        ``spectrum.meta["z"]``.
    extra_kwargs:
        Passed through to ``plt.subplots``.

    Returns
    -------
    matplotlib.figure.Figure
        Caller is responsible for saving or displaying.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    subplot_kw = extra_kwargs or {}
    fig, (ax_sed, ax_spec) = plt.subplots(
        2, 1,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        gridspec_kw={"hspace": 0.05, "height_ratios": [1, 1.4]},
        **subplot_kw,
    )

    # ------------------------------------------------------------------
    # SED panel
    # ------------------------------------------------------------------
    if sed.points:
        lam = sed.lambda_eff_um
        mag = sed.ab_mag
        err = sed.ab_mag_err

        yerr = np.where(np.isnan(err), 0.0, err)
        has_err = ~np.isnan(err)

        for i, pt in enumerate(sed.points):
            colour = _BAND_COLOURS[i % len(_BAND_COLOURS)]
            ax_sed.errorbar(
                pt.lambda_eff_um,
                pt.ab_mag,
                yerr=(pt.ab_mag_err if pt.ab_mag_err is not None else 0.0),
                fmt="o",
                color=colour,
                capsize=3,
                markersize=6,
                label=pt.band.replace("phot_ab_", ""),
                zorder=3,
            )

            # Shaded transmission curve when available
            if show_curves and bandpass_registry is not None:
                curve = bandpass_registry.load_curve(pt.band)
                if curve is not None:
                    wave_um, thru = curve
                    wave_um = wave_um * 1e-4  # Å → µm
                    # Normalise to a small range around the mag point for display
                    thru_norm = thru / np.clip(thru.max(), 1e-10, None)
                    ax_sed_twin = ax_sed.twinx()
                    ax_sed_twin.fill_between(
                        wave_um, thru_norm,
                        alpha=0.12, color=colour, linewidth=0,
                    )
                    ax_sed_twin.set_ylim(0, 8)
                    ax_sed_twin.set_yticks([])
                    ax_sed_twin.set_yticklabels([])

        ax_sed.invert_yaxis()
        ax_sed.set_ylabel("AB magnitude")
        ax_sed.legend(
            fontsize="small",
            loc="best",
            title="Band",
            framealpha=0.7,
            ncol=min(4, len(sed.points)),
        )
    else:
        ax_sed.text(
            0.5, 0.5, "No photometric bands available",
            transform=ax_sed.transAxes,
            ha="center", va="center", color="gray",
        )
        ax_sed.set_ylabel("AB magnitude")

    # ------------------------------------------------------------------
    # Spectrum panel
    # ------------------------------------------------------------------
    if spectrum_rest_frame:
        wave_um = _wave_angstrom_to_um(spectrum.rest_frame_wavelength())
        xlabel = r"Rest-frame wavelength ($\mu$m)"
    else:
        wave_um = _wave_angstrom_to_um(spectrum.wavelength)
        xlabel = r"Observed wavelength ($\mu$m)"

    good = spectrum.good
    flux = spectrum.flux.copy().astype(np.float64)
    err1d = spectrum.err.astype(np.float64)

    # Plot full array dimmed, then good pixels on top
    ax_spec.plot(wave_um, flux, color="0.7", linewidth=0.5, zorder=1)
    if good.any():
        ax_spec.plot(wave_um[good], flux[good], color="#1f77b4", linewidth=0.8, zorder=2, label="flux")
        lower = flux - err1d
        upper = flux + err1d
        ax_spec.fill_between(
            wave_um[good], lower[good], upper[good],
            color="#1f77b4", alpha=0.2, linewidth=0, label=r"$\pm1\sigma$",
        )

    ax_spec.set_xlabel(xlabel)
    ax_spec.set_ylabel("Flux (native units)")
    ax_spec.legend(fontsize="small", loc="best", framealpha=0.7)

    # ------------------------------------------------------------------
    # Shared log x-axis
    # ------------------------------------------------------------------
    ax_spec.set_xscale("log")
    # Determine x-limits from both panels
    all_lam = []
    if sed.points:
        all_lam.extend([p.lambda_eff_um for p in sed.points])
    if wave_um.size:
        all_lam.append(float(wave_um.min()))
        all_lam.append(float(wave_um.max()))
    if all_lam:
        x_lo = max(0.05, min(all_lam) * 0.8)
        x_hi = max(all_lam) * 1.3
        ax_spec.set_xlim(x_lo, x_hi)

    # ------------------------------------------------------------------
    # Titles and metadata
    # ------------------------------------------------------------------
    suptitle = title or f"Source {sed.source_id}"
    z = spectrum.meta.get("z")
    instr = spectrum.meta.get("instr", "")
    subtitle_parts = []
    if z and not np.isnan(float(z)):
        subtitle_parts.append(f"z = {float(z):.4f}")
    if instr:
        subtitle_parts.append(str(instr).strip())
    if subtitle_parts:
        suptitle += "  —  " + ", ".join(subtitle_parts)
    fig.suptitle(suptitle, fontsize=11, y=1.01)

    return fig
