"""``dl-crossmatch`` / ``dl-describe-crossmatch`` Click entry points."""

from __future__ import annotations

import json
from pathlib import Path

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
from data_lake.knobs import (
    PARTNER_TILE_CACHE_MAX_BYTES,
    PARTNER_TILE_CACHE_MAX_TILES,
)
from data_lake.discovery.area_plan import suggest_crossmatch_plan
from data_lake.io.crossmatch import (
    CrossmatchResult,
    build_column_crossmatch,
    build_crossmatch,
    describe_crossmatch_tree,
    execute_crossmatch_plan,
    format_crossmatch_list,
    format_crossmatch_perf_summary,
    format_crossmatch_report,
    list_crossmatch_descriptions,
    load_crossmatch_plan,
    parse_column_crossmatch_dirname,
    parse_crossmatch_dirname,
    resolve_crossmatch_describe_target,
    validate_match_column,
)


def _echo_crossmatch_result_line(msg: str, result: CrossmatchResult) -> None:
    """Print the end-of-run summary (always; survives ``-q``)."""
    perf_line = format_crossmatch_perf_summary(result.perf)
    if perf_line:
        msg = f"{msg}; {perf_line}"
    click.echo(msg)

_cli_b_cache_mib = PARTNER_TILE_CACHE_MAX_BYTES // (1024 * 1024)


@click.command("dl-crossmatch")
@click.argument("survey_a", required=False)
@click.argument("survey_b", required=False)
@click.argument("output_root", type=click.Path(path_type=Path), required=False)
@config_option
@click.option(
    "--from-area",
    default=None,
    help="Run the area's crossmatch_plan (base x partners), region-bounded.",
)
@click.option(
    "--plan",
    "plan_file",
    type=click.Path(path_type=Path),
    default=None,
    help="Run a crossmatch plan from a JSON file (no region restriction).",
)
@click.option(
    "--suggest-plan",
    is_flag=True,
    help=(
        "Write a draft crossmatch_plan JSON (auto sky vs column) and exit; "
        "does not run matching. Requires --base and repeatable --partner."
    ),
)
@click.option(
    "--base",
    "suggest_base",
    default=None,
    help="[suggest-plan] Base catalog survey name.",
)
@click.option(
    "--partner",
    "suggest_partners",
    multiple=True,
    help="[suggest-plan] Partner survey name (repeatable). Bare name only.",
)
@click.option(
    "-o",
    "--output",
    "suggest_output",
    type=click.Path(path_type=Path),
    default=None,
    help="[suggest-plan] Write draft JSON here (default: stdout).",
)
@click.option(
    "--radius-arcsec",
    default=1.0,
    show_default=True,
    type=float,
    help="Maximum match radius in arcseconds (nearest neighbour within radius).",
)
@click.option(
    "--norder",
    "norder_a",
    type=int,
    default=None,
    help="Survey-A HEALPix order (default: catalog_info.json).",
)
@click.option("--norder-b", type=int, default=None, help="Survey-B HEALPix order.")
@click.option("--ra-col", "ra_col_a", default=None, help="Survey-A RA column override.")
@click.option("--dec-col", "dec_col_a", default=None, help="Survey-A Dec column override.")
@click.option("--ra-col-b", default=None, help="Survey-B RA column override.")
@click.option("--dec-col-b", default=None, help="Survey-B Dec column override.")
@click.option("--overwrite", is_flag=True, help="Rebuild cross-match tiles that already exist.")
@click.option(
    "--all-tiles",
    is_flag=True,
    help="Scan every HEALPix pixel (slow); default is survey-A populated tiles only.",
)
@progress_option
@click.option(
    "--n-workers",
    default=1,
    show_default=True,
    type=int,
    help="Parallel worker processes.",
)
@click.option(
    "--tiles-per-worker",
    default=1,
    show_default=True,
    type=int,
    help=(
        "Heartbeat interval inside each worker strip (log every N tiles). "
        "1 → log about every 50 tiles."
    ),
)
@click.option(
    "--strips-per-worker",
    default=4,
    show_default=True,
    type=int,
    help=(
        "With --n-workers > 1, split pending tiles into n-workers × N contiguous "
        "strips of about equal estimated cost (A rows + B rows in covering "
        "tiles), handed out heaviest first as workers free up. Workers keep "
        "their B-tile cache across strips. 1 → one strip per worker."
    ),
)
@click.option(
    "--match-backend",
    type=click.Choice(["astropy", "rapids"], case_sensitive=False),
    default="astropy",
    show_default=True,
    help="Sky matcher: astropy (CPU) or rapids (cuML on GPU; requires [rapids] extra).",
)
@click.option(
    "--gpu-id",
    default=0,
    show_default=True,
    type=int,
    help="CUDA device index when --match-backend=rapids.",
)
@click.option(
    "--export-parquet",
    type=click.Path(path_type=Path),
    default=None,
    help="Also write all match rows to a single Parquet file.",
)
@click.option(
    "--export-fits",
    type=click.Path(path_type=Path),
    default=None,
    help="Also write all match rows to a single FITS BINTABLE.",
)
@click.option(
    "--match-mode",
    type=click.Choice(["sky", "column"], case_sensitive=False),
    default="sky",
    show_default=True,
    help="Match strategy: sky (nearest-neighbour within radius) or column (equality join).",
)
@click.option(
    "--match-col-a",
    default=None,
    help="[column mode] Column in survey A to join on. Required with --match-mode=column.",
)
@click.option(
    "--match-col-b",
    default=None,
    help="[column mode] Column in survey B to join on. Required with --match-mode=column.",
)
@click.option(
    "--b-cache/--no-b-cache",
    default=True,
    show_default=True,
    help=(
        "LRU-cache full survey-B HEALPix tiles across neighbouring A tiles "
        "(sky and column modes; same defaults as dl-gather partner cache)."
    ),
)
@click.option(
    "--b-cache-tiles",
    default=PARTNER_TILE_CACHE_MAX_TILES,
    show_default=True,
    type=int,
    help="Max survey-B tiles in the per-worker LRU.",
)
@click.option(
    "--b-cache-mib",
    default=_cli_b_cache_mib,
    show_default=True,
    type=int,
    help="Max survey-B tile cache size in MiB per worker.",
)
@click.option(
    "--memory-budget-gb",
    type=float,
    default=None,
    help=(
        "Total memory for this run, split across --n-workers to size each "
        "worker's DuckDB memory_limit. Default: smallest of physical RAM, "
        "the cgroup limit, and the Slurm allocation. DuckDB threads are "
        "always split as available CPUs / --n-workers. Override either with "
        "$DATA_LAKE_DUCKDB_THREADS / $DATA_LAKE_DUCKDB_MEMORY_LIMIT_MIB."
    ),
)
@logging_options
def cli(
    survey_a: str | None,
    survey_b: str | None,
    output_root: Path | None,
    config_path: Path | None,
    from_area: str | None,
    plan_file: Path | None,
    suggest_plan: bool,
    suggest_base: str | None,
    suggest_partners: tuple[str, ...],
    suggest_output: Path | None,
    radius_arcsec: float,
    norder_a: int | None,
    norder_b: int | None,
    ra_col_a: str | None,
    dec_col_a: str | None,
    ra_col_b: str | None,
    dec_col_b: str | None,
    overwrite: bool,
    all_tiles: bool,
    show_progress: bool,
    n_workers: int,
    tiles_per_worker: int,
    strips_per_worker: int,
    match_backend: str,
    gpu_id: int,
    export_parquet: Path | None,
    export_fits: Path | None,
    match_mode: str,
    match_col_a: str | None,
    match_col_b: str | None,
    b_cache: bool,
    b_cache_tiles: int,
    b_cache_mib: int,
    memory_budget_gb: float | None,
    log_file: Path | None,
    quiet: bool,
    verbose: bool,
) -> None:
    """Build an in-lake cross-match between two ingested catalogs.

    Sky mode (default): nearest-neighbour within ``--radius-arcsec``.
    Output: ``crossmatch/<survey_a>_x_<survey_b>__r<radius>/``.

    Column mode (--match-mode column): equality join on ``--match-col-a`` /
    ``--match-col-b``.  Output: ``crossmatch/<A>_x_<B>__col_<cola>__<colb>/``.
    All matching pairs are written (many-to-many); sep_arcsec is 0.

    Draft a multi-partner plan without running::

        dl-crossmatch --suggest-plan --base EUCLID_Q1_MER \\
          --partner GAIA_DR3_source --partner ALLWISE -o plan.json
    """
    validate_quiet_verbose(quiet, verbose)
    cfg = load_optional_config(config_path)
    configure_cli_logging(
        level=resolve_log_level(quiet=quiet, verbose=verbose,
                                config_level=cfg.ingest.log_level if cfg else None),
        quiet=quiet,
        log_file=log_file,
    )

    if suggest_plan:
        if from_area is not None or plan_file is not None:
            raise click.ClickException(
                "--suggest-plan cannot be combined with --from-area or --plan"
            )
        if not suggest_base:
            raise click.ClickException("--suggest-plan requires --base SURVEY")
        if not suggest_partners:
            raise click.ClickException(
                "--suggest-plan requires at least one --partner SURVEY"
            )
        # Positional may be the lake root when config is unset.
        lake_arg = output_root
        if lake_arg is None and survey_a is not None and survey_b is None:
            lake_arg = Path(survey_a)
        lake = require_output_root(lake_arg, cfg, kind="catalogs")
        try:
            draft = suggest_crossmatch_plan(
                lake,
                suggest_base,
                list(suggest_partners),
                default_sky_radius_arcsec=radius_arcsec,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        reasons = draft.get("_suggestions") or {}
        if isinstance(reasons, dict):
            for name, reason in reasons.items():
                click.echo(f"  {name}: {reason}", err=True)
        text = json.dumps(draft, indent=2) + "\n"
        if suggest_output is not None:
            suggest_output.parent.mkdir(parents=True, exist_ok=True)
            suggest_output.write_text(text, encoding="utf-8")
            click.echo(
                f"Wrote draft crossmatch_plan → {suggest_output} "
                "(edit, then dl-crossmatch --plan …)",
                err=True,
            )
        else:
            click.echo(text, nl=False)
        return

    if n_workers < 1:
        raise click.ClickException("--n-workers must be >= 1")
    if tiles_per_worker < 1:
        raise click.ClickException("--tiles-per-worker must be >= 1")
    if strips_per_worker < 1:
        raise click.ClickException("--strips-per-worker must be >= 1")
    if gpu_id < 0:
        raise click.ClickException("--gpu-id must be >= 0")

    effective_progress = show_progress and not quiet
    if b_cache_tiles < 1:
        raise click.ClickException("--b-cache-tiles must be >= 1")
    if b_cache_mib < 1:
        raise click.ClickException("--b-cache-mib must be >= 1")
    if memory_budget_gb is not None and memory_budget_gb <= 0:
        raise click.ClickException("--memory-budget-gb must be > 0")
    memory_budget_bytes = (
        int(memory_budget_gb * 1024**3) if memory_budget_gb is not None else None
    )

    # ---- column mode: dl-crossmatch A B [LAKE] --match-mode column ... ----
    if match_mode == "column":
        if from_area is not None or plan_file is not None:
            raise click.ClickException(
                "--match-mode=column cannot be combined with --from-area or --plan "
                "(use a crossmatch_plan in the area instead)."
            )
        for flag, val in (("--match-col-a", match_col_a), ("--match-col-b", match_col_b)):
            if not val:
                raise click.ClickException(f"{flag} is required with --match-mode=column")
        if not survey_a or not survey_b:
            raise click.ClickException(
                "SURVEY_A and SURVEY_B are required for --match-mode=column."
            )
        try:
            validate_match_column(match_col_a, "--match-col-a")  # type: ignore[arg-type]
            validate_match_column(match_col_b, "--match-col-b")  # type: ignore[arg-type]
        except ValueError as exc:
            raise click.ClickException(str(exc))
        from data_lake.schema_registry import resolve_catalog_root as _rcr
        lake = require_output_root(output_root, cfg, kind="catalogs")
        for name, label in ((survey_a, "survey A"), (survey_b, "survey B")):
            cat_root = _rcr(lake, name)
            if not cat_root.is_dir():
                raise click.ClickException(f"{label} catalog not found: {cat_root}")
        try:
            result = build_column_crossmatch(
                lake,
                survey_a,
                survey_b,
                match_col_a,  # type: ignore[arg-type]
                match_col_b,  # type: ignore[arg-type]
                overwrite=overwrite,
                norder_a=norder_a,
                norder_b=norder_b,
                populated_tiles_only=not all_tiles,
                show_progress=effective_progress,
                n_workers=n_workers,
                tiles_per_worker=tiles_per_worker,
                export_parquet=export_parquet,
                export_fits=export_fits,
                b_cache=b_cache,
                b_cache_max_tiles=b_cache_tiles,
                b_cache_max_mib=b_cache_mib,
                memory_budget_bytes=memory_budget_bytes,
                strips_per_worker=strips_per_worker,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc))
        msg = (
            f"Column crossmatch {result.crossmatch_name}: "
            f"{result.n_match_rows:,} association row(s) in {result.n_tiles_written:,} tile(s) "
            f"→ {result.output_root} ({result.elapsed_s:.1f} s)"
        )
        if result.export_parquet is not None:
            msg += f"; parquet → {result.export_parquet}"
        if result.export_fits is not None:
            msg += f"; fits → {result.export_fits}"
        _echo_crossmatch_result_line(msg, result)
        return

    # ---- plan mode: dl-crossmatch [LAKE] --from-area AREA | --plan FILE ----
    if from_area is not None or plan_file is not None:
        if from_area is not None and plan_file is not None:
            raise click.ClickException("Use only one of --from-area or --plan.")
        if from_area is not None and overwrite:
            raise click.ClickException(
                "--overwrite cannot be combined with --from-area "
                "(region runs only gap-fill missing tiles; leave existing matches intact). "
                "Use unrestricted pair mode to rebuild a tree."
            )
        # In plan mode the positional args are not survey names; the first
        # positional (if any) is the lake root.
        plan_output_root = output_root
        if plan_output_root is None and survey_a is not None and survey_b is None:
            plan_output_root = Path(survey_a)
        lake = require_output_root(plan_output_root, cfg, kind="catalogs")
        try:
            plan, region = load_crossmatch_plan(
                lake, from_area=from_area, plan_file=plan_file
            )
            # Region-bounded runs never overwrite existing tiles.
            plan_overwrite = False if region is not None else overwrite
            results = execute_crossmatch_plan(
                lake,
                plan,
                region=region,
                overwrite=plan_overwrite,
                n_workers=n_workers,
                tiles_per_worker=tiles_per_worker,
                match_backend=match_backend,  # type: ignore[arg-type]
                gpu_id=gpu_id,
                show_progress=effective_progress,
                b_cache=b_cache,
                b_cache_max_tiles=b_cache_tiles,
                b_cache_max_mib=b_cache_mib,
                memory_budget_bytes=memory_budget_bytes,
                strips_per_worker=strips_per_worker,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc))
        run_rows = sum(r.n_match_rows for r in results)
        tree_rows = sum(
            (r.tree_total_rows if r.tree_total_rows is not None else r.n_match_rows)
            for r in results
        )
        click.echo(
            f"Crossmatch plan: {len(results)} partner tree(s); "
            f"this run accounted for {run_rows:,} row(s) in scope; "
            f"on-disk tree total(s) {tree_rows:,} match row(s)"
        )
        for r in results:
            tree_note = ""
            if r.tree_total_rows is not None:
                tree_note = (
                    f" (tree on disk: {r.tree_total_rows:,} row(s) "
                    f"in {r.tree_n_tiles:,} tile(s))"
                )
            _echo_crossmatch_result_line(
                f"  {r.crossmatch_name}: {r.n_match_rows:,} row(s) in "
                f"{r.n_tiles_written:,} tile(s) this run → {r.output_root}"
                f"{tree_note}",
                r,
            )
        return

    if not survey_a or not survey_b:
        raise click.ClickException(
            "SURVEY_A and SURVEY_B are required (or use --from-area / --plan)."
        )
    from data_lake.schema_registry import resolve_catalog_root as _rcr
    lake = require_output_root(output_root, cfg, kind="catalogs")

    for name, label in ((survey_a, "survey A"), (survey_b, "survey B")):
        cat_root = _rcr(lake, name)
        if not cat_root.is_dir():
            raise click.ClickException(f"{label} catalog not found: {cat_root}")

    result = build_crossmatch(
        lake,
        survey_a,
        survey_b,
        radius_arcsec=radius_arcsec,
        norder_a=norder_a,
        norder_b=norder_b,
        ra_col_a=ra_col_a,
        dec_col_a=dec_col_a,
        ra_col_b=ra_col_b,
        dec_col_b=dec_col_b,
        overwrite=overwrite,
        populated_tiles_only=not all_tiles,
        show_progress=effective_progress,
        n_workers=n_workers,
        tiles_per_worker=tiles_per_worker,
        match_backend=match_backend,  # type: ignore[arg-type]
        gpu_id=gpu_id,
        export_parquet=export_parquet,
        export_fits=export_fits,
        b_cache=b_cache,
        b_cache_max_tiles=b_cache_tiles,
        b_cache_max_mib=b_cache_mib,
        memory_budget_bytes=memory_budget_bytes,
        strips_per_worker=strips_per_worker,
    )
    backend_note = result.match_backend
    if result.match_backend == "rapids":
        backend_note = f"rapids (gpu {result.gpu_id})"
    msg = (
        f"Cross-match {result.crossmatch_name}: "
        f"{result.n_match_rows:,} match row(s) in {result.n_tiles_written:,} tile(s) "
        f"→ {result.output_root} ({result.elapsed_s:.1f} s, "
        f"{result.n_workers} process worker(s), {backend_note}"
    )
    if result.tree_total_rows is not None and (
        result.tree_total_rows != result.n_match_rows
        or (result.tree_n_tiles or 0) != result.n_tiles_written
    ):
        msg += (
            f"; tree on disk: {result.tree_total_rows:,} row(s) "
            f"in {result.tree_n_tiles:,} tile(s)"
        )
    if result.export_parquet is not None:
        msg += f"; parquet → {result.export_parquet}"
    if result.export_fits is not None:
        msg += f"; fits → {result.export_fits}"
    if result.n_workers == 1:
        msg += "; DuckDB may use additional CPU threads for parquet I/O"
    msg += ")"
    _echo_crossmatch_result_line(msg, result)

@click.command("dl-describe-crossmatch")
@click.argument("survey_a", required=False)
@click.argument("survey_b", required=False)
@click.argument("output_root", type=click.Path(path_type=Path), required=False)
@config_option
@click.option(
    "--name",
    "tree_name",
    default=None,
    help="Tree dirname under crossmatch/ (e.g. A_x_B__r1.0 or A_x_B__col_ID__ID).",
)
@click.option(
    "--radius-arcsec",
    type=float,
    default=None,
    help="Disambiguate a sky tree when multiple radii exist for the pair.",
)
@click.option(
    "--match-mode",
    type=click.Choice(["sky", "column"], case_sensitive=False),
    default=None,
    help="Select sky vs column tree when both exist for the pair.",
)
@click.option("--match-col-a", default=None, help="Column-equality key on survey A.")
@click.option("--match-col-b", default=None, help="Column-equality key on survey B.")
@click.option(
    "--recount",
    is_flag=True,
    help="Recount total_rows / n_tiles from Parquet footers (slower, authoritative).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit description as JSON.")
@logging_options
def cli_describe_crossmatch(
    survey_a: str | None,
    survey_b: str | None,
    output_root: Path | None,
    config_path: Path | None,
    tree_name: str | None,
    radius_arcsec: float | None,
    match_mode: str | None,
    match_col_a: str | None,
    match_col_b: str | None,
    recount: bool,
    as_json: bool,
    log_file: Path | None,
    quiet: bool,
    verbose: bool,
) -> None:
    """List or describe crossmatch trees (sky and column).

    With no survey args: list every tree under ``crossmatch/``.
    With SURVEY_A SURVEY_B (or ``--name`` / a tree dirname): show one tree.
    """
    validate_quiet_verbose(quiet, verbose)
    configure_cli_logging(
        level=resolve_log_level(quiet=quiet, verbose=verbose),
        quiet=quiet,
        log_file=log_file,
    )
    cfg = load_optional_config(config_path)

    def _is_tree_dirname(token: str) -> bool:
        return (
            parse_crossmatch_dirname(token) is not None
            or parse_column_crossmatch_dirname(token) is not None
        )

    def _looks_like_lake(token: str) -> bool:
        p = Path(token)
        return p.is_dir() and (p / "crossmatch").is_dir()

    a, b, lake_arg = survey_a, survey_b, output_root

    if tree_name is None and a and _is_tree_dirname(a):
        tree_name = a
        a = None
        if b is not None and lake_arg is None:
            lake_arg = Path(b)
            b = None
    elif tree_name is not None and a is not None and b is None and lake_arg is None:
        # --name TREE LAKE_ROOT → Click binds LAKE to survey_a
        lake_arg = Path(a)
        a = None
    elif (
        tree_name is None
        and a
        and b is None
        and lake_arg is None
        and _looks_like_lake(a)
    ):
        lake_arg = Path(a)
        a = None

    lake = require_output_root(lake_arg, cfg, kind="catalogs")

    list_mode = tree_name is None and not a and not b
    if list_mode:
        descs = list_crossmatch_descriptions(
            lake,
            recount=recount,
            include_columns=False,
        )
        if as_json:
            click.echo(json.dumps({"crossmatches": descs}, indent=2, default=str))
        else:
            click.echo(format_crossmatch_list(descs))
        return

    if tree_name is None and a and not b:
        raise click.UsageError(
            "Provide SURVEY_A SURVEY_B, a tree dirname / --name, "
            "or omit surveys to list all trees."
        )

    try:
        target = resolve_crossmatch_describe_target(
            lake,
            name=tree_name,
            survey_a=a,
            survey_b=b,
            radius_arcsec=radius_arcsec,
            match_mode=match_mode.lower() if match_mode else None,
            match_col_a=match_col_a,
            match_col_b=match_col_b,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    desc = describe_crossmatch_tree(
        target, recount=recount, include_columns=True
    )
    if as_json:
        click.echo(json.dumps(desc, indent=2, default=str))
    else:
        click.echo(format_crossmatch_report(desc))
