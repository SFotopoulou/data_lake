"""``dl-area`` – manage saved area definitions and attach crossmatch/gather plans."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from data_lake.discovery.area_plan import (
    build_crossmatch_plan,
    build_gather_block,
    build_homogenize_block,
    load_plan_fragment,
    merge_plan_blocks,
    parse_columns_json,
    parse_partner_spec,
    update_area,
)
from data_lake.discovery.areas import (
    area_path,
    list_areas,
    load_area,
    validate_area,
)

log = logging.getLogger(__name__)


def _format_area_summary(area_id: str, data: dict) -> str:
    lines = [f"area_id: {area_id}"]
    region = data.get("region") or {}
    lines.append(f"region: {region.get('type', '?')}")
    if data.get("crossmatch_plan"):
        plan = data["crossmatch_plan"]
        partners = ", ".join(
            f"{p['survey']}@{p['radius_arcsec']}″"
            for p in plan.get("partners", [])
        )
        lines.append(f"crossmatch: {plan.get('base_catalog')} × [{partners}]")
    if data.get("gather"):
        g = data["gather"]
        lines.append(
            f"gather: base={g.get('base')} → {g.get('materialize_as')} "
            f"({len(g.get('columns', {}))} surveys)"
        )
    if data.get("homogenize"):
        h = data["homogenize"]
        src = h.get("from_product") or h.get("survey")
        lines.append(
            f"homogenize: {src} → {h.get('materialize_as')} ({h.get('transform')})"
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

    @click.group("dl-area")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.pass_context
    def cli(ctx: click.Context, output_root: Path | None, config_path: Path | None) -> None:
        """Manage ``areas/<id>.json`` definitions (crossmatch, gather, homogenize).

        Examples:

            dl-area list
            dl-area show MyCone
            dl-area set-crossmatch MyCone --base EUCLID_DR1 \\
                --partner ALLWISE:2.0 \\
                --partner DESI_DR1:col:desi_tid:TARGETID:TARGETID
            dl-area set-gather MyCone --base EUCLID_DR1 \\
                --columns '{"EUCLID_DR1":["ra","dec"],"DESI_DR1":["z"]}' \\
                --materialize-as euclid_north_native_v1
            dl-area import MyCone --from-file examples/areas/multi_survey_cone.example.json
        """
        cfg = load_optional_config(config_path)
        ctx.ensure_object(dict)
        ctx.obj["lake"] = require_output_root(output_root, cfg)
        ctx.obj["cfg"] = cfg

    def _lake(ctx: click.Context) -> Path:
        return ctx.obj["lake"]

    @cli.command("list")
    @click.pass_context
    def cmd_list(ctx: click.Context) -> None:
        """List saved area ids."""
        lake = _lake(ctx)
        ids = list_areas(lake)
        if not ids:
            click.echo(f"(no areas under {lake / 'areas'})")
            return
        for area_id in ids:
            click.echo(area_id)

    @cli.command("show")
    @click.argument("area_id")
    @click.option("--json", "as_json", is_flag=True, help="Print full JSON.")
    @click.pass_context
    def cmd_show(ctx: click.Context, area_id: str, as_json: bool) -> None:
        """Show an area definition."""
        lake = _lake(ctx)
        area = load_area(lake, area_id)
        if as_json:
            click.echo(json.dumps(area.data, indent=2))
        else:
            click.echo(_format_area_summary(area.area_id, area.data))
            click.echo(f"path: {area.path or area_path(lake, area.area_id)}")

    @cli.command("validate")
    @click.argument("area_id")
    @click.pass_context
    def cmd_validate(ctx: click.Context, area_id: str) -> None:
        """Validate an area file (errors and warnings)."""
        lake = _lake(ctx)
        area = load_area(lake, area_id)
        msgs = validate_area(area.data)
        if not msgs:
            click.echo(f"OK: {area_id}")
            return
        for m in msgs:
            click.echo(m)
        if any(m.startswith("ERROR") for m in msgs):
            raise click.ClickException(f"area {area_id!r} has validation errors")

    @cli.command("set-crossmatch")
    @click.argument("area_id")
    @click.option("--base", "base_catalog", required=True,
                  help="Base catalog survey (defines HEALPix partitioning).")
    @click.option(
        "--partner", "partners", multiple=True, required=True,
        help=(
            "Partner spec (repeatable). Sky: SURVEY:RADIUS_ARCSEC "
            "(e.g. ALLWISE:2.0). Column equality: SURVEY:col:MATCH_ID:COL_A:COL_B "
            "(e.g. DESI_DR1:col:desi_tid:TARGETID:TARGETID)."
        ),
    )
    @click.option(
        "--no-reuse-existing", is_flag=True,
        help="Set crossmatch_plan.reuse_existing to false.",
    )
    @click.pass_context
    def cmd_set_crossmatch(
        ctx: click.Context,
        area_id: str,
        base_catalog: str,
        partners: tuple[str, ...],
        no_reuse_existing: bool,
    ) -> None:
        """Attach or replace ``crossmatch_plan`` on an area."""
        lake = _lake(ctx)
        try:
            parsed = [parse_partner_spec(p) for p in partners]
            plan = build_crossmatch_plan(
                base_catalog, parsed, reuse_existing=not no_reuse_existing,
            )
            path = update_area(lake, area_id, crossmatch_plan=plan)
        except (ValueError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Updated crossmatch_plan → {path}")

    @cli.command("set-gather")
    @click.argument("area_id")
    @click.option("--base", required=True, help="Base catalog for the wide product.")
    @click.option(
        "--columns", "columns_json", required=True,
        help='JSON mapping {survey: [col, ...]} (check dl-describe-survey).',
    )
    @click.option("--materialize-as", required=True,
                  help="Product name under catalogs/.")
    @click.option(
        "--multiplicity", type=click.Choice(["nearest", "all"]),
        default="nearest", show_default=True,
    )
    @click.option("--matches-only", is_flag=True,
                  help="Set keep_all=false (drop base rows without partners).")
    @click.option("--no-sep", is_flag=True, help="Omit partner sep_arcsec columns.")
    @click.option("--where-joined", default=None,
                  help="SQL predicate on joined partner columns.")
    @click.pass_context
    def cmd_set_gather(
        ctx: click.Context,
        area_id: str,
        base: str,
        columns_json: str,
        materialize_as: str,
        multiplicity: str,
        matches_only: bool,
        no_sep: bool,
        where_joined: str | None,
    ) -> None:
        """Attach or replace ``gather`` block on an area."""
        lake = _lake(ctx)
        try:
            columns = parse_columns_json(columns_json)
            block = build_gather_block(
                base, columns, materialize_as,
                multiplicity=multiplicity,
                include_sep=not no_sep,
                keep_all=not matches_only,
                where_joined=where_joined,
            )
            path = update_area(lake, area_id, gather=block)
        except (ValueError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Updated gather → {path}")

    @cli.command("set-homogenize")
    @click.argument("area_id")
    @click.option("--survey", default=None, help="Native survey to homogenize.")
    @click.option("--from-product", default=None,
                  help="Gathered product to homogenize (typical after dl-gather).")
    @click.option("--transform", required=True, help="Transform pack id (e.g. phot_ab_v1).")
    @click.option("--materialize-as", required=True,
                  help="Homogenized product name under catalogs/.")
    @click.pass_context
    def cmd_set_homogenize(
        ctx: click.Context,
        area_id: str,
        survey: str | None,
        from_product: str | None,
        transform: str,
        materialize_as: str,
    ) -> None:
        """Attach or replace ``homogenize`` block on an area."""
        lake = _lake(ctx)
        try:
            block = build_homogenize_block(
                survey=survey,
                from_product=from_product,
                transform=transform,
                materialize_as=materialize_as,
                area_id=load_area(lake, area_id).area_id,
            )
            path = update_area(lake, area_id, homogenize=block)
        except (ValueError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Updated homogenize → {path}")

    @cli.command("import")
    @click.argument("area_id")
    @click.option(
        "--from-file", "from_file", required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Full area JSON or partial fragment with plan blocks.",
    )
    @click.option(
        "--create-region/--no-create-region", default=False,
        help="Allow importing region when the area file does not exist yet.",
    )
    @click.pass_context
    def cmd_import(
        ctx: click.Context,
        area_id: str,
        from_file: Path,
        create_region: bool,
    ) -> None:
        """Merge plan blocks from a JSON file into an area."""
        lake = _lake(ctx)
        fragment = load_plan_fragment(from_file)
        try:
            area = load_area(lake, area_id)
        except FileNotFoundError:
            if not create_region or "region" not in fragment:
                raise click.ClickException(
                    f"area {area_id!r} not found; use --create-region with a file "
                    "that includes a region block, or run dl-region --save-as first"
                ) from None
            from data_lake.discovery.areas import make_area
            from data_lake.discovery.region import Region

            region = Region.from_dict(fragment["region"])
            area = make_area(area_id, region)
            merge_plan_blocks(area, fragment)
            fatal = [m for m in validate_area(area.data) if m.startswith("ERROR")]
            if fatal:
                raise click.ClickException("; ".join(fatal))
            from data_lake.discovery.areas import save_area
            path = save_area(lake, area, overwrite=False)
            click.echo(f"Created area → {path}")
            return

        merge_plan_blocks(area, fragment)
        fatal = [m for m in validate_area(area.data) if m.startswith("ERROR")]
        if fatal:
            raise click.ClickException("; ".join(fatal))
        from data_lake.discovery.areas import save_area
        path = save_area(lake, area, overwrite=True)
        click.echo(f"Imported blocks → {path}")

except ImportError:
    cli = None  # type: ignore[misc, assignment]
