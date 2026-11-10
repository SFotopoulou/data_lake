"""``dl-ingest-catalog`` / ``dl-finalize-catalog`` Click entry points."""

from __future__ import annotations

from pathlib import Path

import click

from data_lake.cli_utils import (
    config_option,
    configure_cli_logging,
    fits_memmap_option,
    hdu_option,
    ingest_token_option,
    load_optional_config,
    logging_options,
    pick,
    require_ingest_permission,
    require_output_root,
    resolve_log_level,
    validate_quiet_verbose,
)
from data_lake.ingest.fits_to_parquet import (
    CatalogParquetOptions,
    finalize_catalog_survey,
    ingest_catalog,
    parse_set_column_spec,
)


@click.command("dl-ingest-catalog")
@click.argument("source_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_root", type=click.Path(path_type=Path), required=False)
@config_option
@ingest_token_option
@click.option("--survey", "survey_name", required=True, help="Short survey name.")
@click.option("--ra-col", default="ra", show_default=True)
@click.option("--dec-col", default="dec", show_default=True)
@click.option("--norder", default=None, type=int,
              help="HEALPix order (overrides config; default 5).")
@click.option(
    "--link-id-col",
    required=True,
    help="Survey object ID column (e.g. TARGETID, SOURCE_ID). Required.",
)
@click.option(
    "--allow-incomplete-link-id",
    is_flag=True,
    help="For composite or string --link-id-col: leave _source_id null when "
    "any link part is missing (row stays in catalog, unlinked to spectra).",
)
@click.option(
    "--tile-mode",
    type=click.Choice(["skip", "overwrite", "append"], case_sensitive=False),
    default=None,
    help="Existing Npix tile: skip (default), replace, or read-concat-write.",
)
@click.option(
    "--on-duplicate-id",
    type=click.Choice(["skip", "error", "last"], case_sensitive=False),
    default="skip",
    show_default=True,
    help="When --tile-mode=append and an ID column exists.",
)
@click.option(
    "--streaming/--no-streaming", default=False, show_default=True,
    help="FITS-only: memmap the input and write one tile at a time. "
         "Bounds memory peak to ~one tile's worth of rows (tens of MB) "
         "instead of holding the full table + sorted copy in RAM. "
         "Recommended for catalogs >~ 50 M rows.",
)
@click.option(
    "--streaming-parallel",
    default=0,
    show_default=True,
    type=int,
    help="FITS-only: shard streaming ingest row ranges across N worker processes "
         "(implies --streaming when N > 0; requires --tile-mode append).",
)
@fits_memmap_option
@hdu_option
@click.option(
    "--columns",
    default=None,
    help="Comma-separated FITS columns to keep (sky, ID, HEALPix, and index "
         "columns are always added). Omit rarely used columns to cut disk use.",
)
@click.option(
    "--compact",
    is_flag=True,
    help="Smaller Parquet tiles: ZSTD-9, no per-column statistics, no "
         "dictionary encoding, narrow string type per tile.",
)
@click.option(
    "--compression-level",
    default=None,
    type=int,
    help="ZSTD level for catalog tiles (default 3; ignored if --compact).",
)
@click.option(
    "--defer-finalize",
    is_flag=True,
    help="Skip the per-ingest _metadata rebuild (the O(all tiles) step). "
         "Records lifecycle=live and finalized=false. Run dl-finalize-catalog "
         "once after the last tile-by-tile append. Ideal for live surveys.",
)
@click.option(
    "--lifecycle",
    type=click.Choice(["static", "live"], case_sensitive=False),
    default=None,
    help="Dataset lifecycle recorded in catalog_info.json (default: static; "
         "--defer-finalize implies live).",
)
@click.option(
    "--set-column",
    "set_column_specs",
    multiple=True,
    metavar="NAME=VALUE",
    help="Inject a constant column before identity resolution.  Repeatable. "
         "Format: NAME=VALUE (auto-inferred type) or NAME:TYPE=VALUE where "
         "TYPE is int, float, str, or bool.  Example: --set-column VISIT=7.  "
         "The column must not already exist in the source file.  All ingest "
         "runs for a given survey must use the same set of injected columns.",
)
@logging_options
def cli(
    source_path: Path,
    output_root: Path | None,
    config_path: Path | None,
    ingest_token: str | None,
    survey_name: str,
    ra_col: str,
    dec_col: str,
    norder: int | None,
    link_id_col: str | None,
    allow_incomplete_link_id: bool,
    tile_mode: str | None,
    on_duplicate_id: str,
    streaming: bool,
    streaming_parallel: int,
    fits_memmap: str,
    fits_hdu: str | None,
    columns: str | None,
    compact: bool,
    compression_level: int | None,
    defer_finalize: bool,
    lifecycle: str | None,
    set_column_specs: tuple[str, ...],
    log_file: Path | None,
    quiet: bool,
    verbose: bool,
) -> None:
    """Ingest FITS/VOTable SOURCE_PATH into HATS-partitioned Parquet.

    OUTPUT_ROOT is optional when a lake config is available
    (via --config or $DATA_LAKE_CONFIG); in that case it defaults to
    ``<lake.root>/<paths.catalogs>``.
    """
    validate_quiet_verbose(quiet, verbose)
    cfg = load_optional_config(config_path)
    configure_cli_logging(
        level=resolve_log_level(quiet=quiet, verbose=verbose,
                                config_level=cfg.ingest.log_level if cfg else None),
        log_file=log_file,
        quiet=quiet,
    )
    require_ingest_permission(cfg, ingest_token)
    resolved_output = require_output_root(output_root, cfg, kind="catalogs")

    col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
    set_columns_dict: dict[str, pa.Scalar] = {}
    for spec in set_column_specs:
        name, scalar = parse_set_column_spec(spec)
        if name in set_columns_dict:
            raise click.UsageError(
                f"--set-column: duplicate column name {name!r}. "
                "Each column may only be specified once."
            )
        set_columns_dict[name] = scalar
    if streaming_parallel > 0:
        ctx = click.get_current_context(silent=True)
        if ctx is not None:
            from click.core import ParameterSource

            if (
                ctx.get_parameter_source("streaming") == ParameterSource.COMMANDLINE
                and not streaming
            ):
                raise click.UsageError(
                    "--streaming-parallel cannot be combined with --no-streaming."
                )
        streaming = True
    if streaming_parallel > 1 and (tile_mode or "skip").lower() != "append":
        raise click.UsageError(
            "--streaming-parallel requires --tile-mode append."
        )
    pq_opts = None
    if compression_level is not None and not compact:
        pq_opts = CatalogParquetOptions(compression_level=compression_level)

    ingest_catalog(
        source_path=source_path,
        output_root=resolved_output,
        survey_name=survey_name,
        ra_col=ra_col,
        dec_col=dec_col,
        norder=pick(norder,
                    cfg.partitioning.hats_order if cfg else None, 5),
        link_id_col=link_id_col,
        allow_incomplete_link_id=allow_incomplete_link_id,
        tile_mode=tile_mode.lower() if tile_mode else None,  # type: ignore[arg-type]
        on_duplicate_id=on_duplicate_id.lower(),  # type: ignore[arg-type]
        streaming=streaming,
        streaming_parallel=streaming_parallel,
        columns=col_list,
        parquet_options=pq_opts,
        compact=compact,
        fits_memmap=fits_memmap.lower(),
        fits_hdu=fits_hdu,
        defer_finalize=defer_finalize,
        lifecycle=(
            "live" if (defer_finalize and lifecycle is None)
            else (lifecycle.lower() if lifecycle else None)
        ),
        set_columns=set_columns_dict or None,
    )

@click.command("dl-finalize-catalog")
@click.argument("output_root", type=click.Path(path_type=Path), required=False)
@config_option
@click.option("--survey", "survey_name", required=True)
@click.option("--ra-col", default="ra", show_default=True)
@click.option("--dec-col", default="dec", show_default=True)
@click.option("--norder", default=None, type=int)
@logging_options
def cli_finalize(
    output_root: Path | None,
    config_path: Path | None,
    survey_name: str,
    ra_col: str,
    dec_col: str,
    norder: int | None,
    log_file: Path | None,
    quiet: bool,
    verbose: bool,
) -> None:
    """Rebuild ``catalog_info.json``, ``_common_metadata``, ``_metadata``, and ``schema_manifest.json`` from tiles."""
    validate_quiet_verbose(quiet, verbose)
    cfg = load_optional_config(config_path)
    configure_cli_logging(
        level=resolve_log_level(quiet=quiet, verbose=verbose,
                                config_level=cfg.ingest.log_level if cfg else None),
        quiet=quiet,
        log_file=log_file,
    )
    lake = require_output_root(output_root, cfg, kind="catalogs")
    n = pick(norder, cfg.partitioning.hats_order if cfg else None, 5)
    catalog_root = lake / "catalogs" / survey_name
    if not finalize_catalog_survey(
        catalog_root,
        survey_name,
        n,
        ra_col=ra_col,
        dec_col=dec_col,
    ):
        raise click.ClickException(f"No Parquet tiles under {catalog_root}")
    click.echo(f"Finalized {catalog_root}")

