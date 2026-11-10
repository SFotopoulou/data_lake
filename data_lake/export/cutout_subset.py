"""
cutout_subset – export a curated subset of cutout images from the lake.

CLI: ``dl-extract-cutout-subset``

Output formats
--------------
zarr    One flat Zarr group (images + WCS + _source_id).  Memory-bounded,
        preferred for large batches.
fits    One FITS file per cutout in a directory, each with full WCS header.
hdf5    Single HDF5 file (stacked images + per-row WCS scalars as datasets).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target-list reader (reuse spectra_subset helpers)
# ---------------------------------------------------------------------------

def _read_target_ids(path: Path, column: str) -> np.ndarray:
    """Read source IDs from FITS / CSV / Parquet / plain-text."""
    from data_lake.export.spectra_subset import _read_target_ids as _spec_read
    return _spec_read(path, column)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    from ..cli_utils import (
        config_option,
        configure_cli_logging,
        load_optional_config,
        logging_options,
        progress_option,
        resolve_log_level,
        validate_quiet_verbose,
    )

    @click.command("dl-extract-cutout-subset")
    @click.option(
        "--survey", "survey_name", required=True,
        help="Survey name (must match the ingested directory under cutouts/).",
    )
    @click.option(
        "--target-list", "target_list", required=False, default=None,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="File of source IDs to extract (FITS / CSV / Parquet / plain-text int). "
             "Mutually exclusive with --where.",
    )
    @click.option(
        "--where", "where_sql", default=None,
        help=(
            "DuckDB boolean expression over the survey catalog "
            "(e.g. \"upper(trim(SPECTYPE)) = 'STAR'\"). "
            "Trusted-operator input only (see docs/security.md). "
            "Mutually exclusive with --target-list."
        ),
    )
    @click.option(
        "--limit", "row_limit", type=int, default=None,
        help="Maximum cutouts to extract after --where (or from --target-list).",
    )
    @click.option(
        "--target-id-col", default="_source_id", show_default=True,
        help=(
            "Column carrying source IDs in --target-list, or a comma-separated "
            "composite matching catalog --link-id-col "
            "(e.g. TARGETID,SURVEY,PROGRAM). Ignored for plain-text files."
        ),
    )
    @click.option(
        "--format", "output_format",
        type=click.Choice(["zarr", "fits", "hdf5"], case_sensitive=False),
        default="zarr", show_default=True,
        help=(
            "Output format.\n"
            "zarr – flat Zarr v3 group (images + WCS bytes + _source_id).\n"
            "fits – one FITS per cutout in a directory, full WCS header preserved.\n"
            "hdf5 – single HDF5 file, WCS scalars as separate datasets."
        ),
    )
    @click.option(
        "--output", "output_path", required=True,
        type=click.Path(path_type=Path),
        help="Destination: .zarr directory, output directory (fits), or .h5 file.",
    )
    @click.option(
        "--fits-filename-template", default="cutout_{source_id}.fits", show_default=True,
        help="Filename template for --format fits; {source_id} is substituted.",
    )
    @click.option(
        "--id-hdu-key", default="SOURCE_ID", show_default=True,
        help="FITS header keyword written for the source ID (--format fits).",
    )
    @click.option(
        "--missing",
        type=click.Choice(["skip", "error"], case_sensitive=False),
        default="skip", show_default=True,
        help="Behaviour when a requested source_id is not in the lake.",
    )
    @click.option("--overwrite", is_flag=True, help="Overwrite existing output.")
    @click.option(
        "--yes", is_flag=True,
        help="Skip confirmation prompt when --overwrite would delete existing output.",
    )
    @click.option(
        "--with-catalog/--no-with-catalog",
        default=True,
        show_default=True,
        help=(
            "Use the Parquet catalog for fast tile lookup and to embed native "
            "link-id columns (TARGETID, SURVEY, …) into Zarr/HDF5 output."
        ),
    )
    @click.option("--lake-root", default=None, type=click.Path(path_type=Path))
    @config_option
    @progress_option
    @logging_options
    def cli(
        survey_name: str,
        target_list: Path | None,
        where_sql: str | None,
        row_limit: int | None,
        target_id_col: str,
        output_format: str,
        output_path: Path,
        fits_filename_template: str,
        id_hdu_key: str,
        missing: str,
        overwrite: bool,
        yes: bool,
        with_catalog: bool,
        lake_root: Path | None,
        config_path: Path | None,
        show_progress: bool,
        log_file: Path | None,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Export a curated cutout subset from the lake.

        SOURCE IDs may be supplied as a plain-text file (one int per line),
        FITS / CSV / Parquet with --target-id-col, a composite spec such
        as TARGETID,SURVEY,PROGRAM (matched using the same hash as ingest),
        or a catalog --where predicate (mutually exclusive with --target-list).
        """
        validate_quiet_verbose(quiet, verbose)
        configure_cli_logging(
            level=resolve_log_level(quiet=quiet, verbose=verbose),
            quiet=quiet,
            log_file=log_file,
        )

        cfg = load_optional_config(config_path)
        if lake_root is None:
            if cfg is None:
                raise click.UsageError(
                    "--lake-root is required when no lake config is provided. "
                    "Either pass it explicitly, set $DATA_LAKE_CONFIG, or use --config."
                )
            lake_root = cfg.lake.root

        from data_lake.export.row_filter import (
            CUTOUT_INDEX_COLUMN,
            load_extract_source_ids,
        )

        if target_list is not None and where_sql:
            raise click.UsageError("Use --target-list or --where, not both.")
        if target_list is None and not where_sql:
            raise click.UsageError("Pass --target-list or --where.")
        if where_sql and not with_catalog:
            raise click.UsageError(
                "--where requires the survey catalog (do not pass --no-with-catalog)."
            )

        catalog_accessor = None
        try:
            if with_catalog:
                try:
                    from data_lake.io.catalog import CatalogAccessor

                    catalog_accessor = CatalogAccessor(lake_root, survey_name)
                    log.info(
                        "Using catalog at %s for lookup + link-id embedding.",
                        catalog_accessor._catalog_root,
                    )
                except FileNotFoundError:
                    if where_sql:
                        raise click.ClickException(
                            f"No catalog found for survey {survey_name!r}; "
                            "--where requires an ingested catalog."
                        ) from None
                    log.info(
                        "No catalog for survey %r — tile-scan lookup; "
                        "link-id columns will not be embedded.",
                        survey_name,
                    )

            from data_lake.io.cutouts import CutoutAccessor

            with CutoutAccessor(
                lake_root, survey_name, catalog_accessor=catalog_accessor,
            ) as acc:
                try:
                    source_ids_raw = load_extract_source_ids(
                        catalog_accessor=catalog_accessor,
                        target_list=target_list,
                        target_id_col=target_id_col,
                        where_sql=where_sql,
                        row_limit=row_limit,
                        index_column=CUTOUT_INDEX_COLUMN,
                        read_target_ids=_read_target_ids,
                    )
                except ValueError as exc:
                    raise click.UsageError(str(exc)) from exc
                source_ids = source_ids_raw.tolist()
                if where_sql:
                    click.echo(f"Selected {len(source_ids):,} source IDs via --where")
                else:
                    assert target_list is not None
                    click.echo(
                        f"Loaded {len(source_ids):,} source IDs from {target_list.name} "
                        f"(column: {target_id_col!r})"
                    )

                if output_path.exists() and not overwrite:
                    raise click.ClickException(
                        f"{output_path} already exists. Pass --overwrite to replace it."
                    )
                if output_path.exists() and overwrite and not yes:
                    click.confirm(f"Delete existing {output_path}?", abort=True)

                if output_format == "zarr":
                    result = acc.extract_subset_to_zarr(
                        source_ids,
                        output_path,
                        missing=missing,
                        show_progress=show_progress,
                        overwrite=overwrite,
                    )
                    click.echo(
                        f"Wrote {result['n_written']:,} cutout(s) → {output_path} "
                        f"(skipped {len(result['missing_ids'])} missing)."
                    )

                elif output_format == "fits":
                    output_path.mkdir(parents=True, exist_ok=True)
                    result = acc.extract_subset_to_fits(
                        source_ids,
                        output_path,
                        missing=missing,
                        show_progress=show_progress,
                        overwrite=overwrite,
                        filename_template=fits_filename_template,
                        id_hdu_key=id_hdu_key,
                    )
                    click.echo(
                        f"Wrote {result['n_written']:,} FITS cutout(s) → {output_path}/ "
                        f"(skipped {len(result['missing_ids'])} missing)."
                    )

                elif output_format == "hdf5":
                    result = acc.extract_subset_to_hdf5(
                        source_ids,
                        output_path,
                        missing=missing,
                        show_progress=show_progress,
                        overwrite=overwrite,
                    )
                    click.echo(
                        f"Wrote {result['n_written']:,} cutout(s) → {output_path} "
                        f"(skipped {len(result['missing_ids'])} missing)."
                    )

                if result.get("missing_ids") and missing == "skip":
                    click.echo(
                        f"  Missing IDs ({len(result['missing_ids'])}): "
                        f"{result['missing_ids'][:5]}"
                        f"{'…' if len(result['missing_ids']) > 5 else ''}",
                        err=True,
                    )
        finally:
            if catalog_accessor is not None:
                catalog_accessor.close()

except ImportError:
    cli = None  # type: ignore[assignment,misc]
