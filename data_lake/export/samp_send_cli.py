"""``dl-samp-send-catalog`` – extract a catalog subset and load it in TOPCAT via SAMP."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

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
    from data_lake.export.catalog_extract import (
        expand_column_args,
        extract_catalog,
        infer_output_format,
        parse_column_spec,
        write_catalog_extract,
    )
    from data_lake.export.samp_send import SampSendError, send_votable_to_topcat
    from data_lake.ingest.fits_to_parquet import _read_source_table

    def _resolve_region(
        *,
        lake_root: Path,
        from_area: str | None,
        cone: tuple[float, float] | None,
        radius_arcsec: float | None,
        bbox: tuple[float, float, float, float] | None,
        npix: str | None,
        norder: int | None,
        moc_path: Path | None,
    ):
        clip_selectors = [from_area, cone, bbox, npix, moc_path]
        if sum(x is not None for x in clip_selectors) > 1:
            raise click.ClickException(
                "Use at most one region clip selector "
                "(--from-area | --cone | --bbox | --npix | --moc)"
            )
        if not any(x is not None for x in clip_selectors):
            return None
        from data_lake.discovery.region_cli import _region_from_args

        try:
            region, _ = _region_from_args(
                from_area=from_area,
                npix=npix,
                norder=norder,
                cone=cone,
                radius_arcsec=radius_arcsec,
                bbox=bbox,
                moc=moc_path,
                lake_root=lake_root,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc)) from exc
        return region

    def _table_to_votable(table_path: Path, vot_path: Path) -> None:
        fmt = infer_output_format(table_path, None)
        if fmt == "votable":
            if table_path.resolve() != vot_path.resolve():
                vot_path.write_bytes(table_path.read_bytes())
            return
        if fmt == "parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(table_path)
            write_catalog_extract(table, vot_path, output_format="votable")
            return
        table = _read_source_table(table_path)
        write_catalog_extract(table, vot_path, output_format="votable")

    @click.command("dl-samp-send-catalog")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option("--survey", default=None, help="Survey name under catalogs/.")
    @click.option(
        "--input",
        "input_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Existing table file (.vot/.fits/.parquet/.csv) to send instead of extracting.",
    )
    @click.option(
        "-c",
        "--column",
        "columns",
        multiple=True,
        help="Column to export (repeatable; lake extract only).",
    )
    @click.option(
        "--all-columns",
        is_flag=True,
        help="Export every catalog column (lake extract only).",
    )
    @click.option("--where", "where_sql", default=None)
    @click.option("--limit", "row_limit", type=int, default=None)
    @click.option(
        "--target-list",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
    )
    @click.option("--target-id-col", default="TARGETID", show_default=True)
    @click.option("--via-crossmatch", default=None)
    @click.option("--from-area", default=None, help="Clip extract to an area region.")
    @click.option(
        "--cone", nargs=2, type=float, default=None,
        help="Clip to cone: RA DEC (deg); needs --radius-arcsec.",
    )
    @click.option("--radius-arcsec", type=float, default=None)
    @click.option("--bbox", nargs=4, type=float, default=None)
    @click.option("--npix", default=None)
    @click.option("--norder", type=int, default=None)
    @click.option(
        "--moc", "moc_path", type=click.Path(path_type=Path), default=None,
    )
    @click.option("--valid-sky-only", is_flag=True)
    @click.option("--ra-col", default=None)
    @click.option("--dec-col", default=None)
    @click.option(
        "--name",
        "table_name",
        default=None,
        help="Table name shown in TOPCAT (default: survey or input stem).",
    )
    @click.option(
        "--timeout",
        type=float,
        default=30.0,
        show_default=True,
        help="Seconds to wait for TOPCAT to acknowledge the load.",
    )
    @logging_options
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        survey: str | None,
        input_path: Path | None,
        columns: tuple[str, ...],
        all_columns: bool,
        where_sql: str | None,
        row_limit: int | None,
        target_list: Path | None,
        target_id_col: str,
        via_crossmatch: str | None,
        from_area: str | None,
        cone: tuple[float, float] | None,
        radius_arcsec: float | None,
        bbox: tuple[float, float, float, float] | None,
        npix: str | None,
        norder: int | None,
        moc_path: Path | None,
        valid_sky_only: bool,
        ra_col: str | None,
        dec_col: str | None,
        table_name: str | None,
        timeout: float,
        log_file: Path | None,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Extract a catalog subset and load it into a running TOPCAT via SAMP.

        Start TOPCAT first so a SAMP hub is available. Examples:

            dl-samp-send-catalog /data/lake --survey DESI_DR1 \\
                -c TARGETID,RA,DEC --from-area MyCone --limit 5000

            dl-samp-send-catalog --input desi_subset.vot --name desi_subset
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

        if input_path is not None and survey is not None:
            raise click.ClickException("Use either --input or --survey, not both.")
        if input_path is None and survey is None:
            raise click.ClickException("Pass --survey (lake extract) or --input FILE.")

        display_name = table_name
        with tempfile.TemporaryDirectory(prefix="dl_samp_") as tmp:
            tmp_dir = Path(tmp)
            vot_path = tmp_dir / "send.vot"

            if input_path is not None:
                if any(
                    x is not None
                    for x in (
                        from_area, cone, bbox, npix, moc_path, where_sql,
                        row_limit, target_list, via_crossmatch,
                    )
                ) or columns or all_columns:
                    raise click.ClickException(
                        "With --input, do not pass extract filters "
                        "(--where / --from-area / -c / …); extract first, then send."
                    )
                try:
                    _table_to_votable(input_path, vot_path)
                except (ValueError, OSError) as exc:
                    raise click.ClickException(str(exc)) from exc
                if display_name is None:
                    display_name = input_path.stem
            else:
                lake = require_output_root(output_root, cfg)
                try:
                    column_args = tuple(expand_column_args(columns))
                except ValueError as exc:
                    raise click.ClickException(str(exc)) from exc
                if all_columns and column_args:
                    raise click.ClickException("Use --all-columns or -c, not both.")
                if not all_columns and not column_args:
                    raise click.ClickException("Pass -c or --all-columns.")
                if target_list is not None and where_sql:
                    raise click.ClickException("Use --target-list or --where, not both.")
                if via_crossmatch is not None and target_list is None:
                    raise click.ClickException("--via-crossmatch requires --target-list.")
                if row_limit is not None and row_limit < 1:
                    raise click.ClickException("--limit must be a positive integer.")

                if where_sql:
                    from data_lake.export.row_filter import validate_predicate

                    try:
                        where_sql = validate_predicate(where_sql)
                    except ValueError as exc:
                        raise click.ClickException(str(exc)) from exc

                region = _resolve_region(
                    lake_root=lake,
                    from_area=from_area,
                    cone=cone,
                    radius_arcsec=radius_arcsec,
                    bbox=bbox,
                    npix=npix,
                    norder=norder,
                    moc_path=moc_path,
                )

                source_ids = None
                if target_list is not None:
                    from data_lake.export.target_id_bridge import resolve_target_bridge

                    try:
                        source_ids, _bridge = resolve_target_bridge(
                            target_list,
                            target_id_col,
                            lake_root=lake,
                            via_crossmatch=via_crossmatch,
                            extract_survey=survey,
                        )
                    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
                        raise click.ClickException(str(exc)) from exc

                column_specs: list[str] | None = (
                    list(column_args) if column_args else None
                )
                if (
                    target_list is not None
                    and not all_columns
                    and column_specs is not None
                    and not any(
                        parse_column_spec(s)[0] == "_source_id"
                        or parse_column_spec(s)[1] == "_source_id"
                        for s in column_specs
                    )
                ):
                    column_specs = ["_source_id", *column_specs]

                pq_path = tmp_dir / "extract.parquet"
                try:
                    result = extract_catalog(
                        output=pq_path,
                        specs=column_specs,
                        all_columns=all_columns,
                        lake_root=lake,
                        survey=survey,
                        norder=norder,
                        valid_sky_only=valid_sky_only,
                        ra_col=ra_col,
                        dec_col=dec_col,
                        where_sql=where_sql,
                        row_limit=row_limit,
                        source_ids=source_ids,
                        region=region,
                    )
                except (ValueError, FileNotFoundError, OSError) as exc:
                    raise click.ClickException(str(exc)) from exc

                n_rows = (
                    result.n_rows if hasattr(result, "n_rows") else int(result.num_rows)
                )
                if n_rows == 0:
                    raise click.ClickException(
                        "Extract produced 0 rows; nothing to send to TOPCAT."
                    )
                try:
                    _table_to_votable(pq_path, vot_path)
                except (ValueError, OSError) as exc:
                    raise click.ClickException(str(exc)) from exc
                if display_name is None:
                    display_name = survey

            try:
                client_id = send_votable_to_topcat(
                    vot_path, name=display_name, timeout_sec=timeout,
                )
            except SampSendError as exc:
                raise click.ClickException(str(exc)) from exc

            click.echo(
                f"Loaded {display_name!r} into TOPCAT via SAMP "
                f"(client {client_id})"
            )

except ImportError:
    cli = None  # type: ignore[misc, assignment]
