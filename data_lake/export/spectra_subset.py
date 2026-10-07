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


def _read_target_ids(path: Path, column: str) -> np.ndarray:
    """Read a 1-D int64 array of target IDs from a variety of file formats.

    *column* may be a single name or a comma-separated composite spec matching
    catalog / DESI ``--link-id-col`` (e.g. ``TARGETID,SURVEY,PROGRAM``).
    Uses :func:`~data_lake.ingest.fits_to_parquet.resolve_link_object_ids` so
    the hash is identical to catalog and spectra ingest.
    """
    from data_lake.ingest.fits_to_parquet import (
        parse_link_id_column_spec,
        resolve_link_object_ids,
    )

    parts = parse_link_id_column_spec(column)
    if not parts:
        raise ValueError(f"Empty --target-id-col spec: {column!r}")

    suffix = path.suffix.lower()

    # Plain text: one int per line (composite specs are not supported)
    if suffix in {".txt", ".lst", ".list"}:
        if len(parts) > 1:
            raise ValueError(
                f"Plain-text target lists cannot use composite --target-id-col "
                f"{column!r}; use FITS/CSV/Parquet with the component columns."
            )
        return np.loadtxt(str(path), dtype=np.int64, ndmin=1)

    # Parquet
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
        cols = {p: tbl.column(p).to_pylist() for p in parts}
        return resolve_link_object_ids(column, cols, context=path.name)

    # CSV / TSV / FITS / VOTable / IPAC – let astropy figure it out
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
    cols = {p: list(tbl[p]) for p in parts}
    return resolve_link_object_ids(column, cols, context=path.name)


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
        load_optional_config,
        progress_option,
    )

    @click.command("dl-extract-spectra-subset")
    @click.option(
        "--survey", "survey_name", required=True,
        help="Source survey name (must match the ingested directory under spectra/).",
    )
    @click.option(
        "--target-list", "target_list", required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="File containing the source IDs to extract "
             "(FITS / CSV / Parquet / plain-text).",
    )
    @click.option(
        "--target-id-col", default="TARGETID", show_default=True,
        help=(
            "Column carrying source IDs, or a comma-separated composite matching "
            "catalog/DESI --link-id-col (e.g. TARGETID,SURVEY,PROGRAM). "
            "Ignored for plain-text files."
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
        default=50_000,
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
        "--chunks-per-shard", default=512, show_default=True, type=int,
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
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        survey_name: str,
        target_list: Path,
        target_id_col: str,
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
        verbose: bool,
    ) -> None:
        """Extract a curated subset of spectra (Zarr, Parquet, HDF5, or FITS)."""
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="[%(asctime)s] %(name)-22s %(levelname)-7s %(message)s",
            datefmt="%H:%M:%S",
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

        log.info("Reading source IDs from %s (column=%s) …", target_list, target_id_col)
        source_ids = _read_target_ids(target_list, target_id_col)
        n_unique = int(np.unique(source_ids).size)
        log.info("Loaded %d source IDs (%d unique).", source_ids.size, n_unique)

        catalog_accessor = None
        if with_catalog:
            try:
                from data_lake.io.catalog import CatalogAccessor
                catalog_accessor = CatalogAccessor(lake_root, survey_name)
                log.info("Using catalog at %s for fast index lookup.",
                         catalog_accessor._catalog_root)
            except FileNotFoundError:
                log.info(
                    "No catalog found for survey %r — using tile-scan lookup.",
                    survey_name,
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
        fmt_label = f"{fmt}" + (f", {layout}" if fmt == "fits" else "")
        click.echo(
            f"Wrote {result['n_written']}/{result['n_requested']} spectra "
            f"({fmt_label}) → {out}  (missing: {len(result['missing_ids'])})"
        )

except ImportError:
    cli = None  # type: ignore[assignment]
