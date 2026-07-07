"""
dl-plot-source — render a two-panel SED + 1D spectrum figure for one source.

The source is identified by its lake ``_source_id`` integer.  Photometry is
read from a homogenized product catalog; the spectrum from a named spectra
survey directory.

Usage
-----
.. code-block:: bash

    # With a deployment config:
    export DATA_LAKE_CONFIG=/home/user/como/lake_config.toml
    dl-plot-source --id 1234567890 --product EDFF_cone_joined \\
        --spectra-survey SDSS_DR17 -o source_1234567890.png

    # Without a config (specify lake root explicitly):
    dl-plot-source --lake-root /shared/como \\
        --id 1234567890 --product EDFF_cone_joined \\
        --spectra-survey SDSS_DR17 -o source_1234567890.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from data_lake.cli_utils import (
    config_option,
    load_optional_config,
    logging_options,
    configure_cli_logging,
    resolve_log_level,
    validate_quiet_verbose,
)


@click.command("dl-plot-source")
@click.option("--id", "source_id", required=True, type=int,
              help="Lake _source_id integer that identifies the source.")
@click.option("--product", required=True,
              help="Name of the homogenized product catalog (e.g. EDFF_cone_joined).")
@click.option("--spectra-survey", "spectra_survey", required=True,
              help="Survey name under <lake_root>/spectra/ to fetch the 1D spectrum from.")
@click.option("--lake-root", "lake_root_cli", default=None, type=click.Path(path_type=Path),
              help="Lake root directory.  Not needed when --config or $DATA_LAKE_CONFIG is set.")
@click.option("-o", "--output", "output_path", default=None, type=click.Path(path_type=Path),
              help="Output file path.  Extension determines format: .png (default), .pdf, .svg. "
                   "When omitted the figure is saved as source_<id>.png in the current directory.")
@click.option("--dpi", default=150, show_default=True, type=int,
              help="DPI for raster output (PNG).")
@click.option("--title", default=None,
              help="Figure title.  Defaults to 'Source <id>'.")
@click.option("--rest-frame", is_flag=True, default=False,
              help="Convert spectrum wavelength to rest frame using the stored redshift.")
@click.option("--no-curves", is_flag=True, default=False,
              help="Suppress bandpass transmission-curve shading in the SED panel.")
@config_option
@logging_options
def cli(
    source_id: int,
    product: str,
    spectra_survey: str,
    lake_root_cli: Path | None,
    output_path: Path | None,
    dpi: int,
    title: str | None,
    rest_frame: bool,
    no_curves: bool,
    config_path: Path | None,
    quiet: bool,
    verbose: bool,
) -> None:
    """Render a two-panel SED + 1D spectrum plot for one source."""
    validate_quiet_verbose(quiet, verbose)
    import logging
    log_level = resolve_log_level(quiet=quiet, verbose=verbose)
    configure_cli_logging(level=log_level, quiet=quiet)
    log = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Resolve lake root
    # ------------------------------------------------------------------
    cfg = load_optional_config(config_path)
    if lake_root_cli is not None:
        lake_root = lake_root_cli
    elif cfg is not None:
        lake_root = cfg.lake.root
    else:
        raise click.UsageError(
            "Lake root is required.  Either pass --lake-root, set $DATA_LAKE_CONFIG, "
            "or use --config."
        )

    # ------------------------------------------------------------------
    # Resolve output path
    # ------------------------------------------------------------------
    if output_path is None:
        output_path = Path(f"source_{source_id}.png")

    # ------------------------------------------------------------------
    # Open accessors
    # ------------------------------------------------------------------
    log.info("Opening product catalog %r on lake %s", product, lake_root)
    try:
        from data_lake.io.catalog import CatalogAccessor
        product_acc = CatalogAccessor(lake_root, product)
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"Product catalog {product!r} not found under {lake_root}/catalogs/.\n"
            f"Run 'dl-describe-lake --kind product' to list available products."
        ) from exc

    log.info("Opening spectrum store %r", spectra_survey)
    try:
        from data_lake.io.spectra import SpectrumAccessor
        spec_acc = SpectrumAccessor(lake_root, spectra_survey, catalog_accessor=product_acc)
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"Spectrum store {spectra_survey!r} not found under {lake_root}/spectra/."
        ) from exc

    # ------------------------------------------------------------------
    # Assemble SED
    # ------------------------------------------------------------------
    from data_lake.homogenize.bandpass import BandpassRegistry
    from data_lake.plot.sed import assemble_sed

    bp = BandpassRegistry(lake_root=lake_root)
    log.info("Assembling SED for source_id=%d from product %r", source_id, product)
    try:
        sed = assemble_sed(product_acc, source_id, bp)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc

    if not sed.points:
        click.echo(
            f"Warning: no photometric bands with known wavelengths found for source {source_id}. "
            "The SED panel will be empty.",
            err=True,
        )
    else:
        log.info("SED: %d bands assembled (%s)", len(sed.points),
                 ", ".join(p.band for p in sed.points))

    # ------------------------------------------------------------------
    # Fetch spectrum
    # ------------------------------------------------------------------
    log.info("Fetching spectrum for source_id=%d from survey %r", source_id, spectra_survey)
    try:
        spectrum = spec_acc.get_spectrum(source_id)
    except KeyError as exc:
        raise click.ClickException(
            f"Spectrum for source_id={source_id} not found in survey {spectra_survey!r}."
        ) from exc

    # ------------------------------------------------------------------
    # Render figure
    # ------------------------------------------------------------------
    from data_lake.plot.source_figure import plot_source_sed_spectrum

    log.info("Rendering figure → %s", output_path)
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
    except ImportError as exc:
        raise click.ClickException(
            "matplotlib is required for plotting.\n"
            "Install it with:  pip install \"data-lake[viz]\""
        ) from exc

    fig = plot_source_sed_spectrum(
        sed,
        spectrum,
        title=title,
        bandpass_registry=(bp if not no_curves else None),
        show_curves=not no_curves,
        dpi=dpi,
        spectrum_rest_frame=rest_frame,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)

    click.echo(str(output_path))
    log.info("Saved figure to %s", output_path)
