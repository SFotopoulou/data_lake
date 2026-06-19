"""``dl-export-moc`` – export a survey tile footprint as a MOC."""

from __future__ import annotations

import logging
from pathlib import Path

from data_lake.discovery.moc_export import MocFormat, export_survey_moc
from data_lake.discovery.region_cli import _region_from_args
from data_lake.schema_registry import MODALITY_CATALOG, MODALITY_CUTOUT, MODALITY_SPECTRA

log = logging.getLogger(__name__)

_ALL_MODALITIES = (MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT)


try:
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

    @click.command("dl-export-moc")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option("--survey", required=True, help="Survey name (tile index / layer).")
    @click.option(
        "--modality", default=MODALITY_CATALOG, show_default=True,
        type=click.Choice(_ALL_MODALITIES),
    )
    @click.option("--moc-order", type=int, required=True,
                  help="HEALPix order for the exported MOC (NESTED).")
    @click.option("-o", "--output", "output_path", required=True,
                  type=click.Path(path_type=Path),
                  help="Output file (.fits, .json, or .txt for ascii).")
    @click.option(
        "--moc-format", "moc_format",
        type=click.Choice(["fits", "json", "ascii"]),
        default="fits", show_default=True,
    )
    @click.option("--from-area", default=None, help="Clip footprint to an area region.")
    @click.option("--cone", nargs=2, type=float, default=None,
                  help="Clip to cone: RA DEC (deg); needs --radius-arcsec.")
    @click.option("--radius-arcsec", type=float, default=None)
    @click.option("--bbox", nargs=4, type=float, default=None)
    @click.option("--npix", default=None, help="Clip to npix list (needs --norder).")
    @click.option("--norder", type=int, default=None)
    @click.option("--moc", "moc_path", type=click.Path(path_type=Path), default=None,
                  help="Clip to an input MOC file.")
    @click.option("--overwrite", is_flag=True, help="Replace an existing output file.")
    @logging_options
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        survey: str,
        modality: str,
        moc_order: int,
        output_path: Path,
        moc_format: MocFormat,
        from_area: str | None,
        cone: tuple[float, float] | None,
        radius_arcsec: float | None,
        bbox: tuple[float, float, float, float] | None,
        npix: str | None,
        norder: int | None,
        moc_path: Path | None,
        overwrite: bool,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Export populated survey tiles as an IVOA MOC (optional region clip).

        Examples:

            dl-export-moc --survey SDSS_DR17 --moc-order 5 -o sdss_footprint.fits
            dl-export-moc --survey DESI_DR1 --modality spectra \\
                --from-area MyCone --moc-order 8 -o desi_cone.moc.fits
        """
        validate_quiet_verbose(quiet, verbose)
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(
                quiet=quiet, verbose=verbose,
                config_level=cfg.ingest.log_level if cfg else None,
            ),
            quiet=quiet,
        )
        lake = require_output_root(output_root, cfg)

        region = None
        clip_selectors = [from_area, cone, bbox, npix, moc_path]
        if sum(x is not None for x in clip_selectors) > 1:
            raise click.ClickException(
                "Use at most one region clip selector "
                "(--from-area | --cone | --bbox | --npix | --moc)"
            )
        if any(x is not None for x in clip_selectors):
            try:
                region, _ = _region_from_args(
                    from_area=from_area, npix=npix, norder=norder, cone=cone,
                    radius_arcsec=radius_arcsec, bbox=bbox, moc=moc_path,
                    lake_root=lake,
                )
            except (ValueError, FileNotFoundError) as exc:
                raise click.ClickException(str(exc)) from exc

        try:
            result = export_survey_moc(
                lake, survey, output_path,
                modality=modality,
                moc_order=moc_order,
                region=region,
                fmt=moc_format,
                overwrite=overwrite,
            )
        except (ImportError, ValueError, FileNotFoundError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(
            f"Wrote MOC ({result.format}, order={result.moc_order}, "
            f"{result.n_cells:,} input cells, max_order={result.max_order}) "
            f"→ {result.output}"
        )

except ImportError:
    cli = None  # type: ignore[misc, assignment]
