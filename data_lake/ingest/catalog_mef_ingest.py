"""dl-ingest-catalog-mef – join row-aligned FITS table extensions, then ingest.

Survey-agnostic: any MEF whose table HDUs share columns and the same row
count (FastSpec-style METADATA + SPECPHOT + FASTSPEC, and similar) is joined
horizontally and written with the catalog tile path.
"""

from __future__ import annotations

import sys
from pathlib import Path

from data_lake.ingest.fits_to_parquet import ingest_catalog_mef


try:
    import click

    from data_lake.cli_utils import (
        HeartbeatReporter,
        config_option,
        configure_cli_logging,
        fits_memmap_option,
        ingest_token_option,
        load_optional_config,
        logging_options,
        pick,
        progress_option,
        require_ingest_permission,
        require_output_root,
        resolve_log_level,
        validate_quiet_verbose,
    )
    from data_lake.ingest.catalog_cli_options import (
        allow_incomplete_link_id_option,
        tile_mode_option,
    )
    from data_lake.ingest.file_list_ingest import _run_file_list

    @click.command("dl-ingest-catalog-mef")
    @click.argument("source_path", type=click.Path(exists=True, path_type=Path), required=False)
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @ingest_token_option
    @click.option(
        "--file-list",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Text file, one FITS path per line (instead of SOURCE_PATH).",
    )
    @click.option("--survey", "survey_name", required=True, help="Short survey name.")
    @click.option("--ra-col", default="ra", show_default=True)
    @click.option("--dec-col", default="dec", show_default=True)
    @click.option(
        "--norder", default=None, type=int,
        help="HEALPix order (overrides config; default 5).",
    )
    @click.option(
        "--link-id-col",
        required=True,
        help="Survey object ID column or composite spec (e.g. TARGETID,SURVEY,PROGRAM).",
    )
    @allow_incomplete_link_id_option
    @tile_mode_option(
        default="append",
        help="Existing Npix tile: skip, overwrite, or append (default; use for file lists).",
    )
    @click.option(
        "--on-duplicate-id",
        type=click.Choice(["skip", "error", "last"], case_sensitive=False),
        default="skip",
        show_default=True,
        help="When --tile-mode=append and an ID column exists.",
    )
    @click.option(
        "--hdu",
        "hdu_names",
        multiple=True,
        metavar="NAME",
        help="Table extension to join (index, EXTNAME, or EXTNAME,EXTVER). "
             "Repeatable; omit to join every non-empty table HDU. At least two required.",
    )
    @click.option(
        "--key-cols",
        default=None,
        help="Comma-separated shared columns that must match row by row "
             "(default: every column name present on all selected HDUs). "
             "Any other name that still appears on more than one HDU is an error.",
    )
    @click.option(
        "--columns",
        default=None,
        help="Comma-separated columns to keep (sky, ID, HEALPix, and index "
             "columns are always added).",
    )
    @click.option(
        "--compact", is_flag=True,
        help="Smaller Parquet tiles: ZSTD-9, no stats/dictionary.",
    )
    @fits_memmap_option
    @click.option(
        "--checkpoint",
        type=click.Path(path_type=Path),
        default=None,
        help="Checkpoint JSON for --file-list "
             "(default: catalogs/<survey>/.ingest_checkpoint.json).",
    )
    @click.option(
        "--failures-log",
        type=click.Path(path_type=Path),
        default=None,
        help="Append-only JSONL for per-file failures (--file-list).",
    )
    @progress_option
    @click.option(
        "--no-skip-completed",
        is_flag=True,
        help="Ignore checkpoint when deciding which files to run.",
    )
    @click.option(
        "--log-file",
        "log_file",
        type=click.Path(path_type=Path),
        default=None,
        help="File for INFO+ logs (useful with --quiet for an audit trail).",
    )
    @click.option(
        "--heartbeat-interval",
        default=None,
        type=int,
        help="Heartbeat summary every N seconds (default: 300 with -q, off otherwise).",
    )
    @logging_options
    def cli(
        source_path: Path | None,
        output_root: Path | None,
        config_path: Path | None,
        ingest_token: str | None,
        file_list: Path | None,
        survey_name: str,
        ra_col: str,
        dec_col: str,
        norder: int | None,
        link_id_col: str | None,
        allow_incomplete_link_id: bool,
        tile_mode: str,
        on_duplicate_id: str,
        hdu_names: tuple[str, ...],
        key_cols: str | None,
        columns: str | None,
        compact: bool,
        fits_memmap: str,
        checkpoint: Path | None,
        failures_log: Path | None,
        show_progress: bool,
        no_skip_completed: bool,
        log_file: Path | None,
        heartbeat_interval: int | None,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Join row-aligned FITS table extensions and ingest as one catalog.

        SOURCE_PATH is one MEF FITS file, or pass --file-list instead.
        OUTPUT_ROOT is optional when a lake config is available.

        Extensions that share columns and the same row count (e.g. FastSpec
        METADATA + SPECPHOT + FASTSPEC) are joined horizontally: shared
        columns are kept from the first HDU, the rest are appended.
        """
        validate_quiet_verbose(quiet, verbose)
        extensions = list(hdu_names) if hdu_names else None
        if extensions is not None and len(extensions) < 2:
            raise click.UsageError(
                "Pass --hdu at least twice, or omit it to join every table HDU."
            )
        if source_path is not None and file_list is not None:
            raise click.UsageError("Pass SOURCE_PATH or --file-list, not both.")
        if source_path is None and file_list is None:
            raise click.UsageError("Provide a FITS SOURCE_PATH or --file-list.")
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(quiet=quiet, verbose=verbose,
                                    config_level=cfg.ingest.log_level if cfg else None),
            log_file=log_file,
            quiet=quiet,
        )
        require_ingest_permission(cfg, ingest_token)
        lake = require_output_root(output_root, cfg, kind="catalogs")
        n = pick(norder, cfg.partitioning.hats_order if cfg else None, 5)
        col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        keys = [c.strip() for c in key_cols.split(",") if c.strip()] if key_cols else None

        kwargs = dict(
            output_root=lake,
            survey_name=survey_name,
            ra_col=ra_col,
            dec_col=dec_col,
            norder=n,
            link_id_col=link_id_col,
            allow_incomplete_link_id=allow_incomplete_link_id,
            tile_mode=tile_mode.lower(),
            on_duplicate_id=on_duplicate_id.lower(),
            columns=col_list,
            compact=compact,
            fits_memmap=fits_memmap.lower(),
            extensions=extensions,
            key_cols=keys,
        )

        if file_list is None:
            ingest_catalog_mef(source_path=source_path, **kwargs)
            return

        default_ck = lake / "catalogs" / survey_name / ".ingest_checkpoint.json"
        hb_interval = heartbeat_interval if heartbeat_interval is not None else (300 if quiet else 0)
        hb = HeartbeatReporter(interval_s=hb_interval, label="catalog-mef-ingest")

        def one(p: Path) -> None:
            ingest_catalog_mef(source_path=p, **kwargs)

        code = _run_file_list(
            paths_file=file_list,
            survey_name=survey_name,
            lake_root=lake,
            checkpoint=checkpoint,
            failures_log=failures_log,
            show_progress=show_progress,
            skip_completed=not no_skip_completed,
            ingest_one=one,
            default_checkpoint=default_ck,
            heartbeat=hb,
        )
        if hb.enabled:
            hb.final()
        sys.exit(code)

except ImportError:
    cli = None  # type: ignore[assignment]
