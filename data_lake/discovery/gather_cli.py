"""``dl-gather`` – materialise a derived multi-survey catalog product.

Two entry styles:

- ``dl-gather [LAKE] --from-area AREA`` — drive everything from the area's
  ``gather`` block (base, columns, multiplicity, materialize_as) and
  ``crossmatch_plan`` (partner radii); selection defaults to the area region.
- ad-hoc — ``--base`` + ``--columns`` (JSON ``{survey: [cols]}``) + ``--radii``
  (JSON ``{survey: arcsec}``) + a selection selector + ``--materialize-as``.

Selection selectors (exactly one): ``--from-area`` | ``--ids FILE`` |
``--where SQL`` | ``--cone RA DEC`` | ``--bbox …`` | ``--npix`` (+ ``--norder``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from data_lake.discovery.areas import Area, load_area
from data_lake.discovery.gather import (
    PartnerSpec,
    gather_product,
    partners_from_columns,
)
from data_lake.discovery.partner_tile_cache import PartnerTileCache, PartnerTileCacheConfig
from data_lake.discovery.region import Region, parse_npix_arg
from data_lake.discovery.selection import (
    BaseSelection,
    read_ids_file,
    selection_from_ids,
    selection_from_region,
    selection_from_where,
)

log = logging.getLogger(__name__)


def _radii_from_area(area: Area) -> dict[str, float]:
    from data_lake.discovery.gather import partners_meta_from_crossmatch_plan

    radii, _ = partners_meta_from_crossmatch_plan(area.crossmatch_plan)
    # gather block may also carry explicit radii
    for p in (area.gather or {}).get("partners", []):
        if p.get("survey") and p.get("radius_arcsec") is not None:
            radii[p["survey"]] = float(p["radius_arcsec"])
    return radii


def _column_partners_from_area(area: Area) -> dict[str, dict]:
    from data_lake.discovery.gather import partners_meta_from_crossmatch_plan

    _, column_partners = partners_meta_from_crossmatch_plan(area.crossmatch_plan)
    return column_partners


def _resolve_selection(
    lake: Path,
    *,
    base: str,
    norder: int,
    from_area_obj: Area | None,
    ids_file: Path | None,
    where: str | None,
    cone: tuple[float, float] | None,
    radius_arcsec: float | None,
    bbox: tuple[float, float, float, float] | None,
    npix: str | None,
    npix_norder: int | None,
) -> BaseSelection:
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
        return selection_from_region(lake, base, from_area_obj.region)
    if ids_file is not None:
        return selection_from_ids(lake, base, read_ids_file(ids_file))
    if where is not None:
        return selection_from_where(lake, base, where)
    if cone is not None:
        if radius_arcsec is None:
            raise ValueError("--cone requires --radius-arcsec")
        return selection_from_region(lake, base, Region.cone(cone[0], cone[1], radius_arcsec))
    if bbox is not None:
        return selection_from_region(lake, base, Region.bbox(*bbox))
    if npix is not None:
        if npix_norder is None:
            raise ValueError("--npix requires --norder")
        return selection_from_region(
            lake, base, Region.from_npix(parse_npix_arg(npix), npix_norder)
        )
    raise ValueError("no selection selector")  # pragma: no cover


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

    @click.command("dl-gather")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option("--from-area", default=None, help="Drive gather from an area's gather block.")
    @click.option("--base", default=None, help="Base catalog (ad-hoc mode).")
    @click.option("--columns", "columns_json", default=None,
                  help='JSON {survey: [columns]} (ad-hoc mode).')
    @click.option("--radii", "radii_json", default=None,
                  help='JSON {survey: radius_arcsec} for partner crossmatch trees.')
    @click.option("--materialize-as", "materialize_as", default=None,
                  help="Product catalog name (written to products/<name>/).")
    @click.option("--multiplicity", type=click.Choice(["nearest", "all"]),
                  default="nearest", show_default=True,
                  help="nearest match per source (default) or all matches (fan-out).")
    @click.option("--no-sep", is_flag=True, help="Omit per-partner sep_arcsec columns.")
    @click.option(
        "--matches-only",
        is_flag=True,
        help="Keep only base rows with at least one partner crossmatch (default: keep all).",
    )
    @click.option(
        "--partner-cache-mb",
        default=512,
        show_default=True,
        type=int,
        help="Max partner-tile cache size in MiB (0 disables cache).",
    )
    @click.option(
        "--partner-cache-tiles",
        default=48,
        show_default=True,
        type=int,
        help="Max partner tiles cached per gather run.",
    )
    @click.option(
        "--no-partner-cache",
        is_flag=True,
        help="Disable partner catalog tile cache.",
    )
    @click.option("--where-joined", "where_joined", default=None,
                  help="Predicate over partner columns, applied after the join.")
    # selection selectors
    @click.option("--ids", "ids_file", type=click.Path(path_type=Path), default=None,
                  help="Base source-ids from a Parquet/CSV file.")
    @click.option("--where", default=None, help="Base-catalog predicate (base columns only).")
    @click.option("--cone", nargs=2, type=float, default=None, help="Cone centre RA DEC (deg).")
    @click.option("--radius-arcsec", type=float, default=None, help="Cone radius (arcsec).")
    @click.option("--bbox", nargs=4, type=float, default=None,
                  help="Box RA_MIN RA_MAX DEC_MIN DEC_MAX (deg).")
    @click.option("--npix", default=None, help="npix list/ranges (needs --norder).")
    @click.option("--norder", "npix_norder", type=int, default=None, help="Order for --npix.")
    @click.option("--overwrite", is_flag=True, help="Overwrite an existing product catalog.")
    @progress_option
    @click.option("--extract-modalities", "extract_modalities", default=None,
                  help="Comma list (spectra,cutout) to export for product sources.")
    @click.option("--output-dir", "extract_output_dir", type=click.Path(path_type=Path),
                  default=None, help="Destination dir for --extract-modalities (outside the lake).")
    @click.option("--extract-survey", "extract_survey", default=None,
                  help="Survey to extract modalities from (default: product base).")
    @logging_options
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        from_area: str | None,
        base: str | None,
        columns_json: str | None,
        radii_json: str | None,
        materialize_as: str | None,
        multiplicity: str,
        no_sep: bool,
        matches_only: bool,
        partner_cache_mb: int,
        partner_cache_tiles: int,
        no_partner_cache: bool,
        where_joined: str | None,
        ids_file: Path | None,
        where: str | None,
        cone: tuple[float, float] | None,
        radius_arcsec: float | None,
        bbox: tuple[float, float, float, float] | None,
        npix: str | None,
        npix_norder: int | None,
        overwrite: bool,
        show_progress: bool,
        extract_modalities: str | None,
        extract_output_dir: Path | None,
        extract_survey: str | None,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Materialise a derived product catalog (base x partners over a selection).

        Examples:

            dl-gather --from-area Wide_Field_47
            dl-gather --base EUCLID --columns '{"EUCLID":["ra","dec"],"DESI_DR1":["z"]}' \\
                      --radii '{"DESI_DR1":1.0}' --cone 150.1 2.2 --radius-arcsec 600 \\
                      --materialize-as EUCLID_desi_wide47
        """
        validate_quiet_verbose(quiet, verbose)
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(quiet=quiet, verbose=verbose,
                                    config_level=cfg.ingest.log_level if cfg else None),
            quiet=quiet,
        )
        lake = require_output_root(output_root, cfg, kind="catalogs")

        area_obj: Area | None = None
        base_columns: list[str] = []
        partners: list[PartnerSpec]
        keep_all = not matches_only

        try:
            if from_area is not None:
                area_obj = load_area(lake, from_area)
                if area_obj.path is not None:
                    click.echo(f"Area config: {area_obj.path.resolve()}")
                g = area_obj.gather
                if g is None:
                    raise click.ClickException(
                        f"area {from_area!r} has no 'gather' block"
                    )
                base = g.get("base") or base
                if not base:
                    raise click.ClickException("gather block missing 'base'")
                columns = g.get("columns") or {}
                radii = _radii_from_area(area_obj)
                column_partners = _column_partners_from_area(area_obj)
                base_columns = list(columns.get(base, []))
                partners = partners_from_columns(
                    columns, radii, base, column_partners=column_partners,
                )
                multiplicity = g.get("multiplicity", multiplicity)
                if g.get("include_sep") is not None:
                    no_sep = not g["include_sep"]
                if not matches_only and g.get("keep_all") is not None:
                    keep_all = bool(g["keep_all"])
                if materialize_as is None:
                    materialize_as = g.get("materialize_as")
                where_joined = where_joined or g.get("where_joined")
            else:
                if not base:
                    raise click.ClickException("--base is required in ad-hoc mode")
                if not columns_json:
                    raise click.ClickException("--columns JSON is required in ad-hoc mode")
                columns = json.loads(columns_json)
                radii = json.loads(radii_json) if radii_json else {}
                base_columns = list(columns.get(base, []))
                partners = partners_from_columns(columns, {k: float(v) for k, v in radii.items()}, base)

            if not materialize_as:
                raise click.ClickException(
                    "--materialize-as is required (or set gather.materialize_as in the area)"
                )

            # base order from the base catalog (selection resolves it too).
            from data_lake.discovery import tile_index as ti

            _, base_order = ti.survey_npix(lake, base, "catalog")
            if base_order is None:
                from data_lake.io.catalog import CatalogAccessor

                with CatalogAccessor(lake, base) as acc:
                    base_order = acc.norder

            if partner_cache_mb < 0:
                raise click.ClickException("--partner-cache-mb must be >= 0")
            if partner_cache_tiles < 0:
                raise click.ClickException("--partner-cache-tiles must be >= 0")
            cache_enabled = not no_partner_cache and partner_cache_mb > 0 and partner_cache_tiles > 0
            partner_cache = PartnerTileCache(
                PartnerTileCacheConfig(
                    max_bytes=partner_cache_mb * 1024 * 1024,
                    max_tiles=partner_cache_tiles,
                    enabled=cache_enabled,
                )
            )

            selection = _resolve_selection(
                lake,
                base=base,
                norder=base_order,
                from_area_obj=area_obj,
                ids_file=ids_file,
                where=where,
                cone=cone,
                radius_arcsec=radius_arcsec,
                bbox=bbox,
                npix=npix,
                npix_norder=npix_norder,
            )
            out_path = lake / "products" / materialize_as
            click.echo(
                f"Gather {materialize_as}: base={base}, {len(partners)} partner(s), "
                f"{len(selection.npix)} tile(s) → {out_path}/"
            )
            result = gather_product(
                lake,
                base,
                partners,
                selection,
                base_columns=base_columns,
                multiplicity=multiplicity,
                include_sep=not no_sep,
                materialize_as=materialize_as,
                where_joined=where_joined,
                keep_all=keep_all,
                partner_cache=partner_cache,
                overwrite=overwrite,
                show_progress=show_progress and not quiet,
            )
        except (ValueError, FileNotFoundError, FileExistsError) as exc:
            raise click.ClickException(str(exc))

        click.echo(
            f"Gathered product '{result.product}': {result.n_rows:,} row(s) in "
            f"{result.n_tiles_written:,} tile(s) ({result.multiplicity} match) "
            f"→ {result.output_root} ({result.elapsed_s:.1f} s)"
        )

        if extract_modalities:
            if extract_output_dir is None:
                raise click.ClickException(
                    "--extract-modalities requires --output-dir"
                )
            mods = [m.strip() for m in extract_modalities.split(",") if m.strip()]
            from data_lake.discovery.gather import extract_modalities_for_product

            try:
                ext = extract_modalities_for_product(
                    lake,
                    result.product,
                    mods,
                    extract_output_dir,
                    survey=extract_survey,
                )
            except (ValueError, FileNotFoundError) as exc:
                raise click.ClickException(str(exc))
            click.echo(
                f"Extracted modalities {mods} for {ext.get('n_sources', 0):,} "
                f"source(s) → {extract_output_dir}"
            )

except ImportError:
    cli = None  # type: ignore[misc, assignment]
