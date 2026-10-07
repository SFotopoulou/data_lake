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
        load_optional_config,
        progress_option,
    )

    @click.command("dl-extract-cutout-subset")
    @click.option(
        "--survey", "survey_name", required=True,
        help="Survey name (must match the ingested directory under cutouts/).",
    )
    @click.option(
        "--target-list", "target_list", required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="File of source IDs to extract (FITS / CSV / Parquet / plain-text int).",
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
    @click.option("--lake-root", default=None, type=click.Path(path_type=Path))
    @config_option
    @progress_option
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        survey_name: str,
        target_list: Path,
        target_id_col: str,
        output_format: str,
        output_path: Path,
        fits_filename_template: str,
        id_hdu_key: str,
        missing: str,
        overwrite: bool,
        yes: bool,
        lake_root: Path | None,
        config: Path | None,
        show_progress: bool,
        verbose: bool,
    ) -> None:
        """Export a curated cutout subset from the lake.

        SOURCE IDs may be supplied as a plain-text file (one int per line),
        FITS / CSV / Parquet with --target-id-col, or a composite spec such
        as TARGETID,SURVEY,PROGRAM (matched using the same hash as ingest).
        """
        import sys

        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

        cfg = load_optional_config(config)
        if lake_root is None:
            lake_root = Path(cfg.get("lake", {}).get("root", "."))

        from data_lake.io.cutouts import CutoutAccessor

        acc = CutoutAccessor(lake_root, survey_name)

        source_ids_raw = _read_target_ids(target_list, target_id_col)
        source_ids = source_ids_raw.tolist()
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
                f"{result['missing_ids'][:5]}{'…' if len(result['missing_ids']) > 5 else ''}",
                err=True,
            )

except ImportError:
    cli = None  # type: ignore[assignment,misc]
