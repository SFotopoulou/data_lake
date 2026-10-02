"""
dl-plot-spectrum — render a 1D spectrum for one lake source.

Usage
-----
.. code-block:: bash

    export DATA_LAKE_CONFIG=/path/to/lake_config.toml
    dl-plot-spectrum DESI_DR1 --id 1234567890 -o desi_123.png

    dl-plot-spectrum DESI_DR1 /data/lake --id 1234567890 --rest-frame
"""

from __future__ import annotations

from pathlib import Path

import click

from data_lake.cli_utils import (
    config_option,
    configure_cli_logging,
    load_optional_config,
    logging_options,
    require_output_root,
    resolve_log_level,
    validate_quiet_verbose,
)


@click.command("dl-plot-spectrum")
@click.argument("survey", type=str)
@click.argument("lake_root", type=click.Path(path_type=Path), required=False)
@click.option(
    "--id",
    "source_id",
    required=True,
    type=int,
    help="Lake _source_id integer that identifies the spectrum.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output path (.png / .pdf / .svg). Default: spectrum_<survey>_<id>.png",
)
@click.option("--dpi", default=150, show_default=True, type=int, help="DPI for raster output.")
@click.option("--title", default=None, help="Figure title override.")
@click.option(
    "--rest-frame",
    is_flag=True,
    default=False,
    help="Convert wavelength to rest frame using the stored redshift.",
)
@click.option(
    "--linear-x",
    is_flag=True,
    default=False,
    help="Use a linear wavelength axis (default is log).",
)
@click.option(
    "--clip-sigma",
    default=10.0,
    show_default=True,
    type=float,
    help="Y-axis clip: median ± N×MAD-σ over good pixels. Use 0 to disable.",
)
@config_option
@logging_options
def cli(
    survey: str,
    lake_root: Path | None,
    source_id: int,
    output_path: Path | None,
    dpi: int,
    title: str | None,
    rest_frame: bool,
    linear_x: bool,
    clip_sigma: float,
    config_path: Path | None,
    quiet: bool,
    verbose: bool,
) -> None:
    """Plot the 1D spectrum for SURVEY source --id from the lake Zarr store."""
    validate_quiet_verbose(quiet, verbose)
    import logging

    log_level = resolve_log_level(quiet=quiet, verbose=verbose)
    configure_cli_logging(level=log_level, quiet=quiet)
    log = logging.getLogger(__name__)

    cfg = load_optional_config(config_path)
    root = require_output_root(lake_root, cfg)

    if output_path is None:
        safe_survey = "".join(c if c.isalnum() or c in "-_" else "_" for c in survey)
        output_path = Path(f"spectrum_{safe_survey}_{source_id}.png")

    log.info("Opening spectrum store %r under %s", survey, root)
    try:
        from data_lake.io.catalog import CatalogAccessor
        from data_lake.io.spectra import SpectrumAccessor

        catalog_acc = None
        try:
            catalog_acc = CatalogAccessor(root, survey)
        except FileNotFoundError:
            log.debug("No catalog for %r; spectrum lookup will scan Zarr tiles", survey)

        spec_acc = SpectrumAccessor(root, survey, catalog_accessor=catalog_acc)
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"Spectrum store {survey!r} not found under {root}/spectra/."
        ) from exc

    log.info("Fetching spectrum source_id=%d", source_id)
    try:
        spectrum = spec_acc.get_spectrum(source_id)
    except KeyError as exc:
        raise click.ClickException(
            f"Spectrum for source_id={source_id} not found in survey {survey!r}."
        ) from exc

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError as exc:
        raise click.ClickException(
            "matplotlib is required for plotting.\n"
            'Install it with:  uv sync --extra viz'
        ) from exc

    from data_lake.plot.source_figure import plot_spectrum

    log.info("Rendering figure → %s", output_path)
    clip = None if clip_sigma <= 0 else clip_sigma
    fig = plot_spectrum(
        spectrum,
        title=title,
        survey_name=survey,
        dpi=dpi,
        rest_frame=rest_frame,
        log_x=not linear_x,
        clip_sigma=clip,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    click.echo(str(output_path))
    log.info("Saved figure to %s", output_path)
