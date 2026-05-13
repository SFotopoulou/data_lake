"""
spectra_subset – extract a curated subset of spectra into a single flat Zarr.

This module exposes ``dl-extract-spectra-subset``: given an already-ingested
survey in the data lake and a list of source IDs (a.k.a. TARGETIDs for DESI),
it writes one self-contained Zarr v3 group containing only the requested
spectra.  Useful when you want to materialise a curated training set or a
science sample without copying the full lake.

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
        --output /scratch/qso_subset.zarr
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

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
        "--output", "output_zarr", required=True,
        type=click.Path(path_type=Path),
        help="Destination .zarr directory.",
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
        help="Output Zarr shard rows.",
    )
    @click.option(
        "--overwrite/--no-overwrite", default=False, show_default=True,
        help="Replace OUTPUT if it already exists.",
    )
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        survey_name: str,
        target_list: Path,
        target_id_col: str,
        output_zarr: Path,
        lake_root: Path | None,
        config_path: Path | None,
        with_catalog: bool,
        missing: str,
        chunks_per_shard: int,
        overwrite: bool,
        verbose: bool,
    ) -> None:
        """Extract a curated subset of spectra into a single flat Zarr group."""
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="[%(asctime)s] %(name)-22s %(levelname)-7s %(message)s",
            datefmt="%H:%M:%S",
        )

        # Resolve lake root (CLI > config)
        cfg = load_optional_config(config_path)
        if lake_root is None:
            if cfg is None:
                raise click.UsageError(
                    "--lake-root is required when no lake config is provided. "
                    "Either pass it explicitly, set $DATA_LAKE_CONFIG, or use --config."
                )
            lake_root = cfg.lake.root

        # Read source IDs
        log.info("Reading source IDs from %s (column=%s) …", target_list, target_id_col)
        source_ids = _read_target_ids(target_list, target_id_col)
        n_unique = int(np.unique(source_ids).size)
        log.info("Loaded %d source IDs (%d unique).", source_ids.size, n_unique)

        # Wire optional catalog accessor
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

        result = acc.extract_subset_to_zarr(
            source_ids=source_ids,
            output_zarr=output_zarr,
            missing=missing,
            chunks_per_shard=chunks_per_shard,
            show_progress=True,
            overwrite=overwrite,
        )

        click.echo(
            f"Wrote {result['n_written']}/{result['n_requested']} spectra to "
            f"{result['output_zarr']}  (missing: {len(result['missing_ids'])})"
        )

except ImportError:
    cli = None  # type: ignore[assignment]
