"""
spectra_subset – extract a curated subset of spectra into Zarr, Parquet, or FITS.

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
    dl-extract-spectra-subset ... --format fits --output /scratch/qso_fits/

    # One multi-row FITS catalog (BINTABLE + WAVELENGTH HDU)
    dl-extract-spectra-subset ... --format fits --fits-layout catalog \\
        --output /scratch/qso_spectra.fits
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
    """Read a 1-D int64 array of target IDs from a variety of file formats."""
    suffix = path.suffix.lower()

    # Plain text: one int per line
    if suffix in {".txt", ".lst", ".list"}:
        return np.loadtxt(str(path), dtype=np.int64, ndmin=1)

    # Parquet
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq
        tbl = pq.read_table(str(path), columns=[column])
        return np.asarray(tbl.column(column).to_numpy(zero_copy_only=False), dtype=np.int64)

    # CSV / TSV / FITS / VOTable / IPAC – let astropy figure it out
    from astropy.table import Table
    tbl = Table.read(str(path))
    if column not in tbl.colnames:
        preview = tbl.colnames[:20]
        ellipsis = "…" if len(tbl.colnames) > 20 else ""
        raise ValueError(
            f"Column {column!r} not in {path.name}. "
            f"Available columns: {preview}{ellipsis}"
        )
    return np.asarray(tbl[column], dtype=np.int64)


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
        help="Column name carrying the source IDs (ignored for plain-text files).",
    )
    @click.option(
        "--format", "output_format",
        type=click.Choice(["zarr", "parquet", "fits"], case_sensitive=False),
        default="zarr", show_default=True,
        help="Output format: flat Zarr group, single Parquet file, or FITS.",
    )
    @click.option(
        "--output", "output_path", required=True,
        type=click.Path(path_type=Path),
        help="Destination: .zarr, .parquet, FITS catalog file, or FITS directory.",
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
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        survey_name: str,
        target_list: Path,
        target_id_col: str,
        output_format: str,
        output_path: Path,
        fits_layout: str,
        fits_filename_template: str,
        lake_root: Path | None,
        config_path: Path | None,
        with_catalog: bool,
        missing: str,
        chunks_per_shard: int,
        overwrite: bool,
        verbose: bool,
    ) -> None:
        """Extract a curated subset of spectra (Zarr, Parquet, or FITS)."""
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
        acc = SpectrumAccessor(
            lake_root=lake_root,
            survey_name=survey_name,
            catalog_accessor=catalog_accessor,
        )

        result = acc.extract_subset(
            source_ids=source_ids,
            output=output_path,
            fmt=fmt,
            missing=missing,
            chunks_per_shard=chunks_per_shard,
            show_progress=True,
            overwrite=overwrite,
            fits_filename_template=fits_filename_template,
            fits_layout=layout,
        )

        out = result.get("output", result.get("output_zarr", output_path))
        fmt_label = f"{fmt}" + (f", {layout}" if fmt == "fits" else "")
        click.echo(
            f"Wrote {result['n_written']}/{result['n_requested']} spectra "
            f"({fmt_label}) → {out}  (missing: {len(result['missing_ids'])})"
        )

except ImportError:
    cli = None  # type: ignore[assignment]
