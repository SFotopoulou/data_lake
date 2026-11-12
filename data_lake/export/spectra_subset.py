"""
spectra_subset – extract a curated subset of spectra into Zarr, Parquet, HDF5, or FITS.

This module exposes ``dl-extract-spectra-subset``: given an already-ingested
survey in the data lake and a list of source IDs (a.k.a. TARGETIDs for DESI),
it writes only the requested spectra.  Useful when you want to materialise a
curated training set or a science sample without copying the full lake.

Source-ID list formats supported (auto-detected via
``astropy.table.Table.read`` + a Parquet/CSV fallback):

* FITS table (e.g. ``zall-pix-iron.fits``)
* CSV / TSV
* Parquet
* Plain text (one integer per line)

Example
-------
::

    dl-extract-spectra-subset \\
        --survey desi_dr1 \\
        --target-list zall-pix-iron-qso.fits \\
        --target-id-col TARGETID \\
        --format zarr \\
        --output /scratch/qso_subset.zarr

    dl-extract-spectra-subset \\
        --survey DESI_DR1 \\
        --where "upper(trim(SPECTYPE)) = 'STAR'" \\
        --limit 1000 \\
        --format zarr \\
        --output /scratch/desi_stars_1000.zarr

    dl-extract-spectra-subset ... --format parquet --output /scratch/qso.parquet
    dl-extract-spectra-subset ... --format hdf5 --output /scratch/qso_subset.h5
    dl-extract-spectra-subset ... --format fits --output /scratch/qso_fits/

    # One multi-row FITS catalog (BINTABLE + WAVELENGTH HDU)
    dl-extract-spectra-subset ... --format fits --fits-layout catalog \\
        --output /scratch/qso_spectra.fits

    # Apply bundled SDSS/DESI flux calibration (10^-17 → cgs erg/s/cm2/Angstrom)
    dl-extract-spectra-subset ... --survey SDSS_DR17 --apply-survey-calibration \\
        --output /scratch/qso_ab.zarr

    # Explicit scale factor
    dl-extract-spectra-subset ... --flux-scale 1e-17 --output /scratch/qso_ab.zarr
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from data_lake.io.spectra import FitsLayout, SubsetFormat

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target-list readers
# ---------------------------------------------------------------------------


def _try_numeric_id_array(column) -> np.ndarray | None:
    """Return int64 IDs from a numeric Arrow column, or ``None`` if not safe."""
    import pyarrow as pa
    import pyarrow.compute as pc

    arr = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    if pa.types.is_dictionary(arr.type):
        arr = pc.dictionary_decode(arr)
    if not (
        pa.types.is_integer(arr.type)
        or pa.types.is_floating(arr.type)
    ):
        return None
    try:
        return np.asarray(arr, dtype=np.int64).ravel()
    except (TypeError, ValueError, OverflowError):
        return None


def read_target_list_table(
    path: Path,
    column: str,
) -> tuple[np.ndarray, "pa.Table"]:
    """Read ``--target-list`` identity columns plus resolved lake join IDs.

    Returns
    -------
    resolved_ids:
        int64 lake ``_source_id`` values (one per input row; not uniqued).
    identity:
        Arrow table of the ``--target-id-col`` component columns (same row
        order).  Used so extract outputs can be matched back to the file.
    """
    import pyarrow as pa

    from data_lake.ingest.fits_to_parquet import (
        parse_link_id_column_spec,
        resolve_link_object_ids,
    )

    parts = parse_link_id_column_spec(column)
    if not parts:
        raise ValueError(f"Empty --target-id-col spec: {column!r}")

    suffix = path.suffix.lower()

    if suffix in {".txt", ".lst", ".list"}:
        if len(parts) > 1:
            raise ValueError(
                f"Plain-text target lists cannot use composite --target-id-col "
                f"{column!r}; use FITS/CSV/Parquet with the component columns."
            )
        ids = np.loadtxt(str(path), dtype=np.int64, ndmin=1)
        identity = pa.table({parts[0]: pa.array(ids, type=pa.int64())})
        return ids, identity

    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        tbl = pq.read_table(str(path), columns=parts)
        missing = [p for p in parts if p not in tbl.column_names]
        if missing:
            raise ValueError(
                f"Column(s) {missing} not in {path.name}. "
                f"Available columns: {tbl.column_names[:20]}"
                f"{'…' if len(tbl.column_names) > 20 else ''}"
            )
        identity = tbl.select(parts)
        if len(parts) == 1:
            fast = _try_numeric_id_array(identity.column(parts[0]))
            if fast is not None:
                return fast, identity
        cols = {p: identity.column(p).to_pylist() for p in parts}
        return resolve_link_object_ids(column, cols, context=path.name), identity

    from astropy.table import Table

    tbl = Table.read(str(path))
    missing = [p for p in parts if p not in tbl.colnames]
    if missing:
        preview = tbl.colnames[:20]
        ellipsis = "…" if len(tbl.colnames) > 20 else ""
        raise ValueError(
            f"Column(s) {missing} not in {path.name} "
            f"(spec={column!r}). Available columns: {preview}{ellipsis}"
        )
    arrays: dict[str, pa.Array] = {}
    for p in parts:
        col = tbl[p]
        try:
            arrays[p] = pa.array(np.asarray(col))
        except (TypeError, ValueError):
            arrays[p] = pa.array(list(col))
    identity = pa.table(arrays)
    if len(parts) == 1:
        try:
            fast = np.asarray(tbl[parts[0]], dtype=np.int64).ravel()
            if fast.size == len(tbl):
                return fast, identity
        except (TypeError, ValueError, OverflowError):
            pass
    cols = {p: list(tbl[p]) for p in parts}
    return resolve_link_object_ids(column, cols, context=path.name), identity


def _read_target_ids(path: Path, column: str) -> np.ndarray:
    """Read a 1-D int64 array of target IDs from a variety of file formats.

    *column* may be a single name or a comma-separated composite spec matching
    catalog / DESI ``--link-id-col`` (e.g. ``TARGETID,SURVEY,PROGRAM``).
    Uses :func:`~data_lake.ingest.fits_to_parquet.resolve_link_object_ids` so
    the hash is identical to catalog and spectra ingest.
    """
    ids, _identity = read_target_list_table(path, column)
    return ids


def _validate_output_path(
    output: Path,
    fmt: SubsetFormat,
    *,
    fits_layout: FitsLayout = "per-file",
) -> None:
    """Raise if OUTPUT is incompatible with the chosen format."""
    if fmt == "fits":
        if fits_layout == "catalog":
            if output.suffix.lower() not in {".fits", ".fit"}:
                raise ValueError(
                    "For --format fits --fits-layout catalog, --output must be "
                    "a .fits file path."
                )
        elif output.suffix.lower() in {".fits", ".fit"}:
            raise ValueError(
                "For --format fits --fits-layout per-file, --output must be a "
                "directory, not a .fits file."
            )
        return
    if fmt == "parquet":
        if output.suffix.lower() not in {".parquet", ".pq", ""}:
            log.warning(
                "Output %s does not end in .parquet; writing Parquet anyway.", output,
            )
        return
    if fmt == "hdf5":
        if output.suffix.lower() not in {".h5", ".hdf5", ".hdf"}:
            log.warning(
                "Output %s does not end in .h5/.hdf5; writing HDF5 anyway.", output,
            )
        return
    if fmt == "zarr":
        if output.suffix.lower() not in {".zarr", ""}:
            log.warning(
                "Output %s does not end in .zarr; writing Zarr anyway.", output,
            )


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
    from data_lake.knobs import DEFAULT_CHUNKS_PER_SHARD, DEFAULT_FITS_CHUNK_ROWS

    @click.command("dl-extract-spectra-subset")
    @click.option(
        "--survey", "survey_name", required=True,
        help="Source survey name (must match the ingested directory under spectra/).",
    )
    @click.option(
        "--target-list", "target_list", required=False, default=None,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="File containing the source IDs to extract "
             "(FITS / CSV / Parquet / plain-text). "
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
        help="Maximum spectra to extract after --where (or from --target-list).",
    )
    @click.option(
        "--target-id-col", default="TARGETID", show_default=True,
        help=(
            "Column carrying source IDs, or a comma-separated composite matching "
            "catalog/DESI --link-id-col (e.g. TARGETID,SURVEY,PROGRAM). "
            "With --via-crossmatch, this is the *partner* survey link-id. "
            "Ignored for plain-text files."
        ),
    )
    @click.option(
        "--via-crossmatch",
        default=None,
        help=(
            "Existing crossmatch tree name under crossmatch/ (or absolute path). "
            "Maps partner-side --target-list IDs to --survey _source_id, then "
            "extracts spectra. Requires --target-list."
        ),
    )
    @click.option(
        "--format", "output_format",
        type=click.Choice(["zarr", "parquet", "hdf5", "fits"], case_sensitive=False),
        default="zarr", show_default=True,
        help="Output format: flat Zarr group, Parquet, HDF5, or FITS.",
    )
    @click.option(
        "--output", "output_path", required=True,
        type=click.Path(path_type=Path),
        help="Destination: .zarr, .parquet, .h5, FITS catalog file, or FITS directory.",
    )
    @click.option(
        "--fits-layout",
        type=click.Choice(["per-file", "catalog"], case_sensitive=False),
        default="per-file",
        show_default=True,
        help="FITS: one file per spectrum (directory) or one catalog BINTABLE.",
    )
    @click.option(
        "--fits-filename-template", default="spec_{source_id}.fits", show_default=True,
        help="Per-spectrum FITS name when --format fits --fits-layout per-file.",
    )
    @click.option(
        "--fits-chunk-rows",
        default=DEFAULT_FITS_CHUNK_ROWS,
        show_default=True,
        type=int,
        help="Rows per temporary catalog-FITS part when --fits-layout catalog "
             "(merged into --output at the end).",
    )
    @click.option(
        "--yes", "-y", "assume_yes", is_flag=True, default=False,
        help="Skip confirmation prompt for multi-part catalog FITS extracts.",
    )
    @click.option(
        "--keep-part-files",
        is_flag=True,
        default=False,
        help="Keep temporary *_partNNNNN.fits after merging catalog FITS.",
    )
    @click.option(
        "--force-many-fits-parts",
        is_flag=True,
        default=False,
        help="Allow more than 100 temporary catalog-FITS parts (still capped at 10000).",
    )
    @click.option(
        "--lake-root", default=None,
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help="Lake root.  Defaults to <lake.root> from $DATA_LAKE_CONFIG.",
    )
    @config_option
    @click.option(
        "--with-catalog/--no-with-catalog", default=True,
        help="Use the Parquet catalog's _spectrum_index column for fast lookup "
             "when available (default: yes; falls back to tile scan).",
    )
    @click.option(
        "--missing",
        type=click.Choice(["skip", "error"]),
        default="skip", show_default=True,
        help="Behaviour for source_ids not found in the lake.",
    )
    @click.option(
        "--chunks-per-shard", default=DEFAULT_CHUNKS_PER_SHARD, show_default=True, type=int,
        help="Output Zarr shard rows (ignored for parquet/fits).",
    )
    @click.option(
        "--overwrite/--no-overwrite", default=False, show_default=True,
        help="Replace existing output.",
    )
    @click.option(
        "--flux-scale", "flux_scale", type=float, default=None,
        help="Multiply extracted flux by this factor (ivar /= scale²). Overrides registry.",
    )
    @click.option(
        "--apply-survey-calibration/--no-apply-survey-calibration",
        default=False, show_default=True,
        help="Apply spectra.flux_calibration from homogenize/surveys/<SURVEY>.json.",
    )
    @click.option(
        "--no-calibration-sidecar", is_flag=True, default=False,
        help="Do not write *.calibration.json beside the output.",
    )
    @progress_option
    @logging_options
    def cli(
        survey_name: str,
        target_list: Path | None,
        where_sql: str | None,
        row_limit: int | None,
        target_id_col: str,
        via_crossmatch: str | None,
        output_format: str,
        output_path: Path,
        fits_layout: str,
        fits_filename_template: str,
        fits_chunk_rows: int,
        assume_yes: bool,
        keep_part_files: bool,
        force_many_fits_parts: bool,
        lake_root: Path | None,
        config_path: Path | None,
        with_catalog: bool,
        missing: str,
        chunks_per_shard: int,
        overwrite: bool,
        flux_scale: float | None,
        apply_survey_calibration: bool,
        no_calibration_sidecar: bool,
        show_progress: bool,
        log_file: Path | None,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Extract a curated subset of spectra (Zarr, Parquet, HDF5, or FITS).

        Pass --target-list or --where (catalog predicate), not both.
        """
        validate_quiet_verbose(quiet, verbose)
        configure_cli_logging(
            level=resolve_log_level(quiet=quiet, verbose=verbose),
            quiet=quiet,
            log_file=log_file,
        )

        fmt: SubsetFormat = output_format.lower()  # type: ignore[assignment]
        layout: FitsLayout = fits_layout.lower()  # type: ignore[assignment]
        _validate_output_path(output_path, fmt, fits_layout=layout)

        cfg = load_optional_config(config_path)
        if lake_root is None:
            if cfg is None:
                raise click.UsageError(
                    "--lake-root is required when no lake config is provided. "
                    "Either pass it explicitly, set $DATA_LAKE_CONFIG, or use --config."
                )
            lake_root = cfg.lake.root

        from data_lake.export.spectra_calibration import (
            resolve_flux_calibration,
            write_calibration_sidecar,
        )

        try:
            calibration = resolve_flux_calibration(
                lake_root,
                survey_name,
                flux_scale=flux_scale,
                apply_survey_calibration=apply_survey_calibration,
            )
        except (LookupError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

        scale = calibration.flux_scale if calibration is not None else None
        if calibration is not None:
            log.info(
                "Flux calibration: scale=%g (%s → %s)",
                calibration.flux_scale,
                calibration.native_flux_unit or "native",
                calibration.output_flux_unit or "scaled",
            )

        from data_lake.export.row_filter import (
            SPECTRUM_INDEX_COLUMN,
            load_extract_source_ids,
        )

        if target_list is not None and where_sql:
            raise click.UsageError("Use --target-list or --where, not both.")
        if target_list is None and not where_sql:
            raise click.UsageError("Pass --target-list or --where.")
        if via_crossmatch is not None and target_list is None:
            raise click.UsageError("--via-crossmatch requires --target-list.")
        if via_crossmatch is not None and where_sql:
            raise click.UsageError("Use --via-crossmatch with --target-list, not --where.")
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
                        "Using catalog at %s for fast index lookup.",
                        catalog_accessor._catalog_root,
                    )
                except FileNotFoundError:
                    if where_sql:
                        raise click.ClickException(
                            f"No catalog found for survey {survey_name!r}; "
                            "--where requires an ingested catalog."
                        ) from None
                    log.info(
                        "No catalog found for survey %r — using tile-scan lookup.",
                        survey_name,
                    )

            target_bridge = None
            try:
                if target_list is not None:
                    from data_lake.export.target_id_bridge import resolve_target_bridge

                    source_ids, target_bridge = resolve_target_bridge(
                        target_list,
                        target_id_col,
                        lake_root=lake_root,
                        via_crossmatch=via_crossmatch,
                        extract_survey=survey_name,
                    )
                    if row_limit is not None:
                        source_ids = np.asarray(source_ids, dtype=np.int64)[: int(row_limit)]
                else:
                    source_ids = load_extract_source_ids(
                        catalog_accessor=catalog_accessor,
                        target_list=None,
                        target_id_col=target_id_col,
                        where_sql=where_sql,
                        row_limit=row_limit,
                        index_column=SPECTRUM_INDEX_COLUMN,
                        read_target_ids=_read_target_ids,
                    )
            except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
                raise click.ClickException(str(exc)) from exc
            n_unique = int(np.unique(source_ids).size)
            if where_sql:
                log.info(
                    "Selected %d source IDs (%d unique) via --where.",
                    source_ids.size, n_unique,
                )
            else:
                log.info(
                    "Loaded %d source IDs (%d unique) from %s (column=%s).",
                    source_ids.size, n_unique, target_list, target_id_col,
                )

            from data_lake.io.spectra import SpectrumAccessor

            def _confirm_fits(plan_msg: str) -> bool:
                click.echo(plan_msg)
                if assume_yes:
                    return True
                return click.confirm("Proceed?", default=False)

            confirm_arg: bool | None
            if fmt == "fits" and layout == "catalog":
                confirm_arg = True if assume_yes else _confirm_fits
            else:
                confirm_arg = None

            try:
                with SpectrumAccessor(
                    lake_root=lake_root,
                    survey_name=survey_name,
                    catalog_accessor=catalog_accessor,
                ) as acc:
                    result = acc.extract_subset(
                        source_ids=source_ids,
                        output=output_path,
                        fmt=fmt,
                        missing=missing,
                        chunks_per_shard=chunks_per_shard,
                        show_progress=show_progress,
                        overwrite=overwrite,
                        fits_filename_template=fits_filename_template,
                        fits_layout=layout,
                        flux_scale=scale,
                        fits_chunk_rows=fits_chunk_rows,
                        confirm=confirm_arg,
                        keep_part_files=keep_part_files,
                        force_many_fits_parts=force_many_fits_parts,
                    )
            except (ValueError, RuntimeError, ImportError) as exc:
                raise click.ClickException(str(exc)) from exc

            out = result.get("output", result.get("output_zarr", output_path))
            if calibration is not None and not no_calibration_sidecar:
                sidecar = write_calibration_sidecar(out, calibration)
                click.echo(f"  calibration → {sidecar}")
            if target_bridge is not None and target_bridge.num_rows > 0:
                from data_lake.export.target_id_bridge import write_target_id_sidecar

                tid_side = write_target_id_sidecar(out, target_bridge)
                click.echo(f"  target-id bridge → {tid_side}")
            fmt_label = f"{fmt}" + (f", {layout}" if fmt == "fits" else "")
            click.echo(
                f"Wrote {result['n_written']}/{result['n_requested']} spectra "
                f"({fmt_label}) → {out}  (missing: {len(result['missing_ids'])})"
            )
        finally:
            if catalog_accessor is not None:
                catalog_accessor.close()

except ImportError:
    cli = None  # type: ignore[assignment]
