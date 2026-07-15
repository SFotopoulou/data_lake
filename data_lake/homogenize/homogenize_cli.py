"""``dl-homogenize`` – materialise homogenized catalog products.

Accepts ``--config`` / ``$DATA_LAKE_CONFIG`` for the lake deployment; when set,
the ``OUTPUT_ROOT`` positional argument may be omitted (defaults to ``lake.root``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from data_lake.discovery.areas import load_area, resolve_region_ref
from data_lake.discovery.region import Region, parse_npix_arg
from data_lake.discovery.selection import (
    BaseSelection,
    read_ids_file,
    selection_from_all_tiles,
    selection_from_ids,
    selection_from_region,
    selection_from_where,
)
from data_lake.homogenize.engine import homogenize_catalog, homogenize_product
from data_lake.homogenize.zarr_engine import homogenize_zarr
from data_lake.schema_registry import MODALITY_CATALOG, MODALITY_CUTOUT, MODALITY_SPECTRA

log = logging.getLogger(__name__)

_MODALITIES = (MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT)


def _resolve_selection(
    lake: Path,
    survey: str,
    *,
    modality: str,
    from_area_obj,
    region_override: Region | None,
    ids_file: Path | None,
    where: str | None,
    cone: tuple[float, float] | None,
    radius_arcsec: float | None,
    bbox: tuple[float, float, float, float] | None,
    npix: str | None,
    npix_norder: int | None,
    allow_all_tiles: bool = False,
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
    if len(chosen) == 0 and allow_all_tiles:
        return selection_from_all_tiles(lake, survey, modality=modality)
    if len(chosen) != 1:
        raise ValueError(
            "Provide exactly one selection selector "
            "(--from-area | --ids | --where | --cone | --bbox | --npix); got: "
            + (", ".join(chosen) or "none")
        )
    region_modality = (
        MODALITY_CATALOG if modality == MODALITY_CATALOG else modality
    )
    if from_area_obj is not None:
        region = region_override if region_override is not None else from_area_obj.region
        return selection_from_region(
            lake, survey, region, modality=region_modality,
        )
    if ids_file is not None:
        return selection_from_ids(lake, survey, read_ids_file(ids_file))
    if where is not None:
        return selection_from_where(lake, survey, where)
    if cone is not None:
        if radius_arcsec is None:
            raise ValueError("--cone requires --radius-arcsec")
        region = Region.cone(cone[0], cone[1], radius_arcsec)
        return selection_from_region(
            lake, survey, region, modality=region_modality,
        )
    if bbox is not None:
        return selection_from_region(
            lake, survey, Region.bbox(*bbox), modality=region_modality,
        )
    if npix is not None:
        if npix_norder is None:
            raise ValueError("--npix requires --norder")
        return selection_from_region(
            lake, survey,
            Region.from_npix(parse_npix_arg(npix), npix_norder),
            modality=region_modality,
        )
    raise ValueError("no selection provided")


try:
    import click

    from data_lake.cli_utils import (
        config_option,
        configure_cli_logging,
        load_optional_config,
        logging_options,
        progress_option,
        require_output_root,
        resolve_log_level,
        validate_quiet_verbose,
    )

    @click.command("dl-homogenize")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option(
        "--modality", type=click.Choice(_MODALITIES), default=MODALITY_CATALOG,
        show_default=True,
        help="Catalog (Parquet), spectra (Zarr), or cutout (Zarr).",
    )
    @click.option("--survey", default=None, help="Native source survey to homogenize.")
    @click.option(
        "--transform", "transform_id", default=None,
        help="Transform pack id (e.g. phot_ab_v1).",
    )
    @click.option(
        "--materialize-as", "materialize_as", default=None,
        help="Output product name under catalogs/, spectra/, or cutouts/.",
    )
    @click.option(
        "--columns", default=None,
        help="Comma-separated native columns to transform (default: photometry roles + rules).",
    )
    @click.option(
        "--from-area", default=None,
        help="Use area region, or drive transform/output from homogenize block.",
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
        help="Homogenize a gathered catalog product (catalog modality) or bound npix (spectra/cutout).",
    )
    @click.option("--check-only", is_flag=True, help="Validate rules and selection; no writes.")
    @click.option("--overwrite", is_flag=True)
    @click.option("--n-workers", type=int, default=8, show_default=True)
    @progress_option
    @logging_options
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        modality: str,
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
        """Apply a homogenization transform to a survey selection → product.

        OUTPUT_ROOT is optional when ``--config`` or ``$DATA_LAKE_CONFIG`` is set.
        """
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

        area_obj = load_area(lake, from_area) if from_area else None
        hom_block = area_obj.homogenize if area_obj is not None else None
        region_override: Region | None = None

        if hom_block is not None and from_area is not None:
            transform_id = transform_id or hom_block.get("transform")
            materialize_as = materialize_as or hom_block.get("materialize_as")
            if hom_block.get("from_product"):
                from_product = from_product or hom_block["from_product"]
            else:
                survey = survey or hom_block.get("survey")
                region_override = resolve_region_ref(
                    lake, hom_block.get("region"), fallback=area_obj.region,
                )

        if survey and from_product and modality == MODALITY_CATALOG:
            raise click.ClickException("Use either --survey or --from-product, not both.")

        catalog_from_product = modality == MODALITY_CATALOG and from_product is not None
        if catalog_from_product:
            target = from_product
        elif modality == MODALITY_CATALOG:
            if not survey:
                raise click.ClickException(
                    "--survey is required (or set homogenize.survey / --from-product)"
                )
            target = survey
        else:
            if not survey:
                raise click.ClickException(
                    f"--survey is required for --modality {modality}"
                )
            target = survey

        if not transform_id:
            default_transform = {
                MODALITY_CATALOG: "phot_ab_v1",
                MODALITY_SPECTRA: "spec_observed_v1",
                MODALITY_CUTOUT: "cutout_njy_v1",
            }[modality]
            transform_id = default_transform

        if not materialize_as and not check_only:
            raise click.ClickException(
                "--materialize-as is required (or set homogenize.materialize_as in the area)"
            )

        col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None

        has_spatial = any([cone, bbox, npix])
        has_other = any([ids_file, where])
        pass_area = area_obj is not None and not catalog_from_product
        allow_all_product = catalog_from_product and not (
            has_spatial or has_other or pass_area or region_override is not None
        )

        selection_survey = from_product if catalog_from_product else survey
        try:
            selection = _resolve_selection(
                lake,
                selection_survey,
                modality=modality,
                from_area_obj=area_obj if pass_area else None,
                region_override=region_override,
                ids_file=ids_file,
                where=where,
                cone=cone,
                radius_arcsec=radius_arcsec,
                bbox=bbox,
                npix=npix,
                npix_norder=npix_norder,
                allow_all_tiles=allow_all_product,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        if not selection.npix:
            raise click.ClickException("Selection overlaps no populated tiles.")

        try:
            if modality == MODALITY_CATALOG and catalog_from_product:
                result = homogenize_product(
                    lake,
                    from_product,
                    transform_id,
                    selection,
                    materialize_as=materialize_as or "check_only",
                    columns=col_list,
                    overwrite=overwrite,
                    check_only=check_only,
                    n_workers=n_workers,
                    show_progress=show_progress and not quiet,
                )
            elif modality == MODALITY_CATALOG:
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
                    show_progress=show_progress and not quiet,
                )
            else:
                zarr_selection = selection
                if from_product:
                    from data_lake.discovery import tile_index as ti

                    prod_sel = selection_from_all_tiles(lake, from_product)
                    spec_npix, spec_order = ti.survey_npix(lake, survey, modality)
                    overlap = prod_sel.npix & spec_npix
                    zarr_selection = BaseSelection(
                        base_survey=survey,
                        norder=spec_order or prod_sel.norder,
                        npix=overlap,
                        source_ids=None,
                    )
                result = homogenize_zarr(
                    lake,
                    modality,
                    survey,
                    transform_id,
                    zarr_selection,
                    materialize_as=materialize_as or "check_only",
                    overwrite=overwrite,
                    check_only=check_only,
                    n_workers=n_workers,
                    show_progress=show_progress and not quiet,
                )
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

        if check_only:
            payload = {
                "check_only": True,
                "modality": modality,
                "transform": transform_id,
                "n_npix": len(selection.npix),
                "resolution": result.resolution,
            }
            if catalog_from_product:
                payload["from_product"] = from_product
            else:
                payload["survey"] = survey
            click.echo(json.dumps(payload, indent=2))
            return

        if modality == MODALITY_CATALOG:
            click.echo(
                f"Wrote homogenized product {result.product}: "
                f"{result.n_rows:,} rows in {result.n_tiles_written} tiles "
                f"({result.elapsed_s:.1f}s)"
            )
        else:
            click.echo(
                f"Wrote homogenized {modality} product {result.product}: "
                f"{result.n_sources:,} sources in {result.n_tiles_written} tiles "
                f"({result.elapsed_s:.1f}s)"
            )
        click.echo(f"  rules applied: {result.resolution.get('n_applied', 0)}")

except ImportError:
    pass
