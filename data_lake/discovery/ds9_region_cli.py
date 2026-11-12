"""``dl-export-ds9-region`` – export survey footprint / sources as a DS9 ``.reg``."""

from __future__ import annotations

import logging
from pathlib import Path

from data_lake.discovery.ds9_region import Ds9Mode, export_ds9_region
from data_lake.discovery.region_cli import _region_from_args

log = logging.getLogger(__name__)


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

    @click.command("dl-export-ds9-region")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option("--survey", required=True, help="Survey name (catalog layer).")
    @click.option(
        "--mode",
        type=click.Choice(["sources", "footprint", "both"]),
        default="both",
        show_default=True,
        help="Write source circles, HEALPix footprint polygons, or both.",
    )
    @click.option(
        "-o", "--output", "output_path", required=True,
        type=click.Path(path_type=Path),
        help="Output DS9 region file (.reg).",
    )
    @click.option(
        "--where", default=None,
        help="DuckDB boolean expression over catalog columns (sources only).",
    )
    @click.option(
        "--limit", type=int, default=None,
        help="Max source rows after --where (input order, no ORDER BY).",
    )
    @click.option(
        "--radius-arcsec", "circle_radius_arcsec", type=float, default=0.5,
        show_default=True,
        help="Source circle radius in arcseconds.",
    )
    @click.option("--from-area", default=None, help="Clip to an area region.")
    @click.option(
        "--cone", nargs=2, type=float, default=None,
        help="Clip to cone: RA DEC (deg); needs --cone-radius-arcsec.",
    )
    @click.option(
        "--cone-radius-arcsec", type=float, default=None,
        help="Cone radius when using --cone (kept separate from --radius-arcsec).",
    )
    @click.option("--bbox", nargs=4, type=float, default=None)
    @click.option("--npix", default=None, help="Clip to npix list (needs --norder).")
    @click.option("--norder", type=int, default=None)
    @click.option(
        "--moc", "moc_path", type=click.Path(path_type=Path), default=None,
        help="Clip to an input MOC file.",
    )
    @click.option("--ra-col", default=None, help="Override catalog RA column.")
    @click.option("--dec-col", default=None, help="Override catalog Dec column.")
    @click.option("--overwrite", is_flag=True, help="Replace an existing output file.")
    @logging_options
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        survey: str,
        mode: Ds9Mode,
        output_path: Path,
        where: str | None,
        limit: int | None,
        circle_radius_arcsec: float,
        from_area: str | None,
        cone: tuple[float, float] | None,
        cone_radius_arcsec: float | None,
        bbox: tuple[float, float, float, float] | None,
        npix: str | None,
        norder: int | None,
        moc_path: Path | None,
        ra_col: str | None,
        dec_col: str | None,
        overwrite: bool,
        log_file: Path | None,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Export catalog sources and/or survey HEALPix tiles as a DS9 region file.

        Examples:

            dl-export-ds9-region /data/lake --survey SDSS_DR17 \\
                --mode both -o sdss.reg --overwrite

            dl-export-ds9-region --survey DESI_DR1 --mode sources \\
                --from-area MyCone --where "Z > 0.5" --limit 5000 -o desi.reg
        """
        validate_quiet_verbose(quiet, verbose)
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(
                quiet=quiet, verbose=verbose,
                config_level=cfg.ingest.log_level if cfg else None,
            ),
            quiet=quiet,
            log_file=log_file,
        )
        lake = require_output_root(output_root, cfg)

        if mode == "footprint" and (where is not None or limit is not None):
            raise click.ClickException(
                "--where / --limit apply only to sources (use --mode sources or both)"
            )

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
                    from_area=from_area,
                    npix=npix,
                    norder=norder,
                    cone=cone,
                    radius_arcsec=cone_radius_arcsec,
                    bbox=bbox,
                    moc=moc_path,
                    lake_root=lake,
                )
            except (ValueError, FileNotFoundError) as exc:
                msg = str(exc).replace(
                    "--cone requires --radius-arcsec",
                    "--cone requires --cone-radius-arcsec",
                )
                raise click.ClickException(msg) from exc

        try:
            result = export_ds9_region(
                lake,
                survey,
                output_path,
                mode=mode,
                where=where,
                limit=limit,
                region=region,
                radius_arcsec=circle_radius_arcsec,
                ra_col=ra_col,
                dec_col=dec_col,
                overwrite=overwrite,
            )
        except (ValueError, FileNotFoundError, FileExistsError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(
            f"Wrote DS9 region ({result.mode}): "
            f"{result.n_footprint:,} footprint polygon(s), "
            f"{result.n_sources:,} source circle(s) → {result.output}"
        )

except ImportError:
    cli = None  # type: ignore[misc, assignment]
