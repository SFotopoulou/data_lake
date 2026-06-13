"""``dl-homogenize`` – materialise homogenized catalog products."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from data_lake.discovery.areas import load_area, resolve_region_ref
from data_lake.discovery.region import Region, parse_npix_arg
from data_lake.discovery.selection import (
    read_ids_file,
    selection_from_ids,
    selection_from_region,
    selection_from_where,
)
from data_lake.homogenize.engine import homogenize_catalog

log = logging.getLogger(__name__)


def _resolve_selection(
    lake: Path,
    survey: str,
    *,
    from_area_obj,
    region_override: Region | None,
    ids_file: Path | None,
    where: str | None,
    cone: tuple[float, float] | None,
    radius_arcsec: float | None,
    bbox: tuple[float, float, float, float] | None,
    npix: str | None,
    npix_norder: int | None,
):
    selectors = [
        ("--from-area", from_area_obj is not None),
        ("--ids", ids_file is not None),
        ("--where", where is not None),
        ("--cone", cone is not None),
        ("--bbox", bbox is not None),
        ("--npix", npix is not None),
    ]
    chosen = [n for n, present in selectors if present]
    if len(chosen) != 1:
        raise ValueError(
            "Provide exactly one selection selector "
            "(--from-area | --ids | --where | --cone | --bbox | --npix); got: "
            + (", ".join(chosen) or "none")
        )
    if from_area_obj is not None:
        region = region_override if region_override is not None else from_area_obj.region
        return selection_from_region(lake, survey, region)
    if ids_file is not None:
        return selection_from_ids(lake, survey, read_ids_file(ids_file))
    if where is not None:
        return selection_from_where(lake, survey, where)
    if cone is not None:
        if radius_arcsec is None:
            raise ValueError("--cone requires --radius-arcsec")
        region = Region.cone(cone[0], cone[1], radius_arcsec)
        return selection_from_region(lake, survey, region)
    if bbox is not None:
        return selection_from_region(lake, survey, Region.bbox(*bbox))
    if npix is not None:
        if npix_norder is None:
            raise ValueError("--npix requires --norder")
        return selection_from_region(
            lake, survey, Region.from_npix(parse_npix_arg(npix), npix_norder),
        )
    raise ValueError("no selection provided")


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

    @click.command("dl-homogenize")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option("--survey", default=None, help="Native source survey to homogenize.")
    @click.option(
        "--transform", "transform_id", default=None,
        help="Transform pack id (e.g. phot_ab_v1).",
    )
    @click.option(
        "--materialize-as", "materialize_as", default=None,
        help="Output product catalog name under catalogs/.",
    )
    @click.option(
        "--columns", default=None,
        help="Comma-separated native columns to transform (default: photometry roles + rules).",
    )
    @click.option(
        "--from-area", default=None,
        help="Use area region, or drive survey/transform/output from homogenize block.",
    )
    @click.option("--ids", "ids_file", type=click.Path(path_type=Path), default=None)
    @click.option("--where", default=None, help="Base-catalog SQL predicate.")
    @click.option("--cone", nargs=2, type=float, default=None)
    @click.option("--radius-arcsec", type=float, default=None)
    @click.option("--bbox", nargs=4, type=float, default=None)
    @click.option("--npix", default=None)
    @click.option("--norder", "npix_norder", type=int, default=None)
    @click.option(
        "--from-product", default=None,
        help="Homogenize an existing product (multi-survey); not implemented yet.",
    )
    @click.option("--check-only", is_flag=True, help="Validate rules and selection; no writes.")
    @click.option("--overwrite", is_flag=True)
    @click.option("--n-workers", type=int, default=8, show_default=True)
    @click.option("--progress", "show_progress", is_flag=True)
    @logging_options
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        survey: str | None,
        transform_id: str | None,
        materialize_as: str | None,
        columns: str | None,
        from_area: str | None,
        ids_file: Path | None,
        where: str | None,
        cone: tuple[float, float] | None,
        radius_arcsec: float | None,
        bbox: tuple[float, float, float, float] | None,
        npix: str | None,
        npix_norder: int | None,
        from_product: str | None,
        check_only: bool,
        overwrite: bool,
        n_workers: int,
        show_progress: bool,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Apply a homogenization transform to a survey selection → product catalog."""
        validate_quiet_verbose(quiet, verbose)
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(
                quiet=quiet,
                verbose=verbose,
                config_level=cfg.ingest.log_level if cfg else None,
            ),
            quiet=quiet,
        )
        lake = require_output_root(output_root, cfg)

        if from_product is not None:
            raise click.ClickException(
                "--from-product homogenization is not implemented yet; "
                "use --survey with a region selector."
            )

        area_obj = load_area(lake, from_area) if from_area else None
        hom_block = area_obj.homogenize if area_obj is not None else None
        region_override: Region | None = None

        if hom_block is not None and from_area is not None:
            survey = survey or hom_block.get("survey")
            transform_id = transform_id or hom_block.get("transform")
            materialize_as = materialize_as or hom_block.get("materialize_as")
            if hom_block.get("from_product"):
                raise click.ClickException(
                    "area homogenize block uses from_product; not implemented yet."
                )
            region_override = resolve_region_ref(
                lake, hom_block.get("region"), fallback=area_obj.region,
            )

        if not survey:
            raise click.ClickException(
                "--survey is required (or set homogenize.survey in the area)"
            )
        if not transform_id:
            raise click.ClickException(
                "--transform is required (or set homogenize.transform in the area)"
            )
        if not materialize_as and not check_only:
            raise click.ClickException(
                "--materialize-as is required (or set homogenize.materialize_as in the area)"
            )

        col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        try:
            selection = _resolve_selection(
                lake,
                survey,
                from_area_obj=area_obj,
                region_override=region_override,
                ids_file=ids_file,
                where=where,
                cone=cone,
                radius_arcsec=radius_arcsec,
                bbox=bbox,
                npix=npix,
                npix_norder=npix_norder,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        if not selection.npix:
            raise click.ClickException("Selection overlaps no populated tiles for this survey.")

        try:
            result = homogenize_catalog(
                lake,
                survey,
                transform_id,
                selection,
                materialize_as=materialize_as or "check_only",
                columns=col_list,
                overwrite=overwrite,
                check_only=check_only,
                n_workers=n_workers,
                show_progress=show_progress,
            )
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

        if check_only:
            click.echo(json.dumps({
                "check_only": True,
                "survey": survey,
                "transform": transform_id,
                "n_npix": len(selection.npix),
                "resolution": result.resolution,
            }, indent=2))
            return

        click.echo(
            f"Wrote homogenized product {result.product}: "
            f"{result.n_rows:,} rows in {result.n_tiles_written} tiles "
            f"({result.elapsed_s:.1f}s)"
        )
        click.echo(f"  rules applied: {result.resolution.get('n_applied', 0)}")

except ImportError:
    pass
