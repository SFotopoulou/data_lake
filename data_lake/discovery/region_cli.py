"""``dl-region`` – discover what data falls in a sky region.

Resolve a region (saved area or ad-hoc selector) to the overlapping
survey x modality tiles, with rounded row estimates (``--count`` for exact
catalog counts). Optionally persist an ad-hoc region as an area (``--save-as``).

Named ``dl-region`` (not ``dl-discover``) to avoid colliding with the existing
"discovery" vocabulary (registry inventory, config-file lookup, column
discovery).
"""

from __future__ import annotations

import logging
from pathlib import Path

from data_lake.discovery.areas import Area, load_area, make_area, save_area
from data_lake.discovery.engine import DiscoveryRow, resolve_region, round_count
from data_lake.discovery.region import Region, parse_npix_arg
from data_lake.schema_registry import (
    MODALITY_CATALOG,
    MODALITY_CUTOUT,
    MODALITY_SPECTRA,
)

log = logging.getLogger(__name__)

_ALL_MODALITIES = (MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT)


def _region_from_args(
    *,
    from_area: str | None,
    npix: str | None,
    norder: int | None,
    cone: tuple[float, float] | None,
    radius_arcsec: float | None,
    bbox: tuple[float, float, float, float] | None,
    moc: Path | None,
    lake_root: Path,
) -> tuple[Region, Area | None]:
    """Build a Region from mutually-exclusive selectors. Returns (region, area_or_None)."""
    selectors = [
        ("--from-area", from_area is not None),
        ("--npix", npix is not None),
        ("--cone", cone is not None),
        ("--bbox", bbox is not None),
        ("--moc", moc is not None),
    ]
    chosen = [name for name, present in selectors if present]
    if len(chosen) != 1:
        raise ValueError(
            "Provide exactly one region selector "
            "(--from-area | --npix | --cone | --bbox | --moc); got: "
            + (", ".join(chosen) or "none")
        )

    if from_area is not None:
        area = load_area(lake_root, from_area)
        return area.region, area
    if npix is not None:
        if norder is None:
            raise ValueError("--npix requires --norder (the source HEALPix order)")
        return Region.from_npix(parse_npix_arg(npix), norder), None
    if cone is not None:
        if radius_arcsec is None:
            raise ValueError("--cone requires --radius-arcsec")
        return Region.cone(cone[0], cone[1], radius_arcsec), None
    if bbox is not None:
        return Region.bbox(*bbox), None
    if moc is not None:
        return Region.from_moc(path=moc), None
    raise ValueError("no region selector provided")  # pragma: no cover


def _format_rows(rows: list[DiscoveryRow], *, exact: bool) -> str:
    if not rows:
        return "(no surveys/modalities overlap this region)"
    header = f"{'Survey':<24} {'Modality':<10} {'Order':>5} {'Tiles':>8} {'Rows':>14}"
    lines = [header, "-" * len(header)]
    for r in sorted(rows, key=lambda x: (x.survey, x.modality)):
        if exact and r.exact_rows is not None:
            rows_str = f"{r.exact_rows:,}"
        else:
            rows_str = round_count(r.est_rows)
        lines.append(
            f"{r.survey:<24} {r.modality:<10} {str(r.hats_order):>5} "
            f"{r.n_tiles_overlap:>8,} {rows_str:>14}"
        )
    return "\n".join(lines)


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

    @click.command("dl-region")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option("--from-area", default=None, help="Use a saved area's region (areas/<id>.json).")
    @click.option("--npix", default=None, help="Comma-separated npix and a-b ranges (needs --norder).")
    @click.option("--norder", type=int, default=None, help="Source HEALPix order for --npix.")
    @click.option("--cone", nargs=2, type=float, default=None,
                  help="Cone centre: RA DEC (deg); needs --radius-arcsec.")
    @click.option("--radius-arcsec", type=float, default=None, help="Cone radius in arcsec.")
    @click.option("--bbox", nargs=4, type=float, default=None,
                  help="Box: RA_MIN RA_MAX DEC_MIN DEC_MAX (deg); RA may wrap.")
    @click.option("--moc", type=click.Path(path_type=Path), default=None,
                  help="MOC FITS file (requires the optional mocpy dependency).")
    @click.option("--surveys", default=None,
                  help="Comma-separated survey names, or 'all' (default).")
    @click.option("--modalities", default=None,
                  help="Comma-separated modalities (default: catalog).")
    @click.option("--count", "exact", is_flag=True,
                  help="Exact catalog counts via Parquet footers (slower than the estimate).")
    @click.option("--save-as", "save_as", default=None,
                  help="Persist this region as areas/<id>.json.")
    @click.option("--overwrite", is_flag=True, help="Overwrite an existing area on --save-as.")
    @logging_options
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        from_area: str | None,
        npix: str | None,
        norder: int | None,
        cone: tuple[float, float] | None,
        radius_arcsec: float | None,
        bbox: tuple[float, float, float, float] | None,
        moc: Path | None,
        surveys: str | None,
        modalities: str | None,
        exact: bool,
        save_as: str | None,
        overwrite: bool,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Discover surveys x modalities overlapping a sky region.

        Examples:

            dl-region --from-area Wide_Field_47
            dl-region --cone 150.1 2.2 --radius-arcsec 600 --modalities catalog,spectra
            dl-region --npix 1002198,1002199 --norder 5 --save-as Wide_Field_47
            dl-region --bbox 149.5 150.5 1.8 2.6 --count
        """
        validate_quiet_verbose(quiet, verbose)
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(quiet=quiet, verbose=verbose,
                                    config_level=cfg.ingest.log_level if cfg else None),
            quiet=quiet,
        )
        lake = require_output_root(output_root, cfg)

        try:
            region, area = _region_from_args(
                from_area=from_area, npix=npix, norder=norder, cone=cone,
                radius_arcsec=radius_arcsec, bbox=bbox, moc=moc, lake_root=lake,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc))

        # Resolve survey/modality scope: explicit flags win, then area.discover,
        # then defaults (all surveys, catalog only).
        if surveys is not None:
            survey_scope: str | list[str] = (
                "all" if surveys.strip() == "all"
                else [s.strip() for s in surveys.split(",") if s.strip()]
            )
        elif area is not None:
            survey_scope = area.discover_surveys
        else:
            survey_scope = "all"

        if modalities is not None:
            mods = [m.strip() for m in modalities.split(",") if m.strip()]
        elif area is not None:
            mods = area.discover_modalities
        else:
            mods = [MODALITY_CATALOG]
        bad = [m for m in mods if m not in _ALL_MODALITIES]
        if bad:
            raise click.ClickException(f"unknown modalities: {', '.join(bad)}")

        try:
            rows = resolve_region(
                lake, region, surveys=survey_scope, modalities=mods, count=exact,
            )
        except ImportError as exc:
            raise click.ClickException(str(exc))

        click.echo(_format_rows(rows, exact=exact))
        if not exact and rows:
            click.echo("\n(row counts are rounded estimates; use --count for exact catalog counts)")

        if save_as is not None:
            blocks: dict = {}
            if survey_scope != "all" or mods != [MODALITY_CATALOG]:
                blocks["discover"] = {"surveys": survey_scope, "modalities": mods}
            new_area = make_area(save_as, region, **blocks)
            try:
                path = save_area(lake, new_area, overwrite=overwrite)
            except FileExistsError as exc:
                raise click.ClickException(str(exc) + " (use --overwrite)")
            click.echo(f"\nSaved area -> {path}")

except ImportError:
    cli = None  # type: ignore[misc, assignment]
