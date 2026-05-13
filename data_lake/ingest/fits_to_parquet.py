"""
fits_to_parquet – ingest FITS / VOTable survey catalogs into HATS-partitioned Parquet.

Layout produced
---------------
<root>/
  Norder=<order>/Dir=<dir>/Npix=<pix>.parquet
  _metadata                    ← Parquet aggregate footer (pyarrow)
  catalog_info.json            ← HATS-style descriptor
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Iterator, Sequence

import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HATS_DIR_STRIDE = 10_000  # tiles per Dir= folder (HATS convention)
_ZSTD_LEVEL = 3             # good balance of speed vs ratio


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def healpix_dir(norder: int, npix: int) -> str:
    """Return the HATS directory path fragment for a pixel (e.g. ``Norder=5/Dir=0``)."""
    dir_index = (npix // _HATS_DIR_STRIDE) * _HATS_DIR_STRIDE
    return f"Norder={norder}/Dir={dir_index}"


def assign_healpix(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    norder: int = 5,
) -> np.ndarray:
    """Return HEALPix NESTED pixel indices for arrays of RA/Dec (degrees)."""
    nside = hp.order2nside(norder)
    theta = np.radians(90.0 - dec_deg)
    phi = np.radians(ra_deg)
    return hp.ang2pix(nside, theta, phi, nest=True).astype(np.int64)


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


def _read_source_table(path: Path) -> pa.Table:
    """Read a FITS or VOTable file into a PyArrow Table."""
    suffix = path.suffix.lower()
    if suffix in {".fit", ".fits", ".fits.gz", ".fz"}:
        fmt = "fits"
    elif suffix in {".xml", ".vot", ".votable"}:
        fmt = "votable"
    else:
        # Try FITS first, fall back to auto-detect
        fmt = "fits"

    log.info("Reading %s as %s …", path.name, fmt)
    astropy_table = Table.read(str(path), format=fmt)
    return pa.Table.from_pandas(astropy_table.to_pandas(index=False), preserve_index=False)


def _add_healpix_columns(
    table: pa.Table,
    ra_col: str,
    dec_col: str,
    norder: int,
) -> pa.Table:
    """Append ``_healpix_order<N>`` and ``_cutout_index`` placeholder columns."""
    ra = table.column(ra_col).to_pylist()
    dec = table.column(dec_col).to_pylist()
    pix = assign_healpix(np.array(ra, dtype=np.float64), np.array(dec, dtype=np.float64), norder)
    col_name = f"_healpix_norder{norder}"
    table = table.append_column(col_name, pa.array(pix, type=pa.int64()))
    # cutout_index and spectrum_index are filled by the respective ingest steps;
    # initialise both to -1 (sentinel meaning "not yet ingested").
    if "_cutout_index" not in table.schema.names:
        table = table.append_column(
            "_cutout_index", pa.array(np.full(len(table), -1, dtype=np.int64), type=pa.int64())
        )
    if "_spectrum_index" not in table.schema.names:
        table = table.append_column(
            "_spectrum_index", pa.array(np.full(len(table), -1, dtype=np.int64), type=pa.int64())
        )
    return table


def ingest_catalog(
    source_path: Path | str,
    output_root: Path | str,
    survey_name: str,
    ra_col: str = "ra",
    dec_col: str = "dec",
    norder: int = 5,
    source_id_col: str | None = None,
    overwrite: bool = False,
) -> None:
    """
    Ingest a single FITS/VOTable file into HATS-partitioned Parquet.

    Parameters
    ----------
    source_path:
        Path to the input FITS or VOTable file.
    output_root:
        Root of the data lake (e.g. ``/data/lake``). Survey files are written
        under ``<output_root>/catalogs/<survey_name>/``.
    survey_name:
        Short name used for the output directory (e.g. ``"des_dr2"``).
    ra_col / dec_col:
        Column names for right ascension and declination in degrees.
    norder:
        HEALPix order for partitioning (default 5 → ~12k tiles of ~3.7 deg²).
    source_id_col:
        If provided, used as the stable ``source_id``; otherwise a sequential ID
        is generated.
    overwrite:
        If False (default), skip tiles that already exist.
    """
    source_path = Path(source_path)
    output_root = Path(output_root)
    catalog_root = output_root / "catalogs" / survey_name

    table = _read_source_table(source_path)
    log.info("Loaded %d rows × %d columns", len(table), len(table.schema))

    # Ensure a stable source_id column
    if source_id_col and source_id_col in table.schema.names:
        if table.schema.field(source_id_col).type != pa.int64():
            table = table.set_column(
                table.schema.get_field_index(source_id_col),
                source_id_col,
                table.column(source_id_col).cast(pa.int64()),
            )
    else:
        table = table.append_column(
            "source_id",
            pa.array(np.arange(len(table), dtype=np.int64), type=pa.int64()),
        )

    table = _add_healpix_columns(table, ra_col, dec_col, norder)
    hp_col = f"_healpix_norder{norder}"

    # Sort by healpix for locality
    sort_indices = pa.compute.sort_indices(table, sort_keys=[(hp_col, "ascending")])
    table = table.take(sort_indices)

    pix_array = table.column(hp_col).to_pylist()
    unique_pixels = sorted(set(pix_array))
    log.info("Writing %d HEALPix tiles at Norder=%d …", len(unique_pixels), norder)

    # Map pixel -> row range
    pix_np = np.array(pix_array, dtype=np.int64)
    writer_meta: list[pq.FileMetaData] = []

    compression = pq.ParquetWriter  # just a reference check
    _ = compression  # suppress lint

    t0 = time.perf_counter()
    for npix in unique_pixels:
        mask = pix_np == npix
        tile_table = table.filter(pa.array(mask))

        out_dir = catalog_root / healpix_dir(norder, npix)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"Npix={npix}.parquet"

        if out_file.exists() and not overwrite:
            log.debug("Skip existing tile %s", out_file)
            continue

        writer = pq.ParquetWriter(
            str(out_file),
            tile_table.schema,
            compression="zstd",
            compression_level=_ZSTD_LEVEL,
            write_statistics=True,
            use_dictionary=True,
        )
        writer.write_table(tile_table)
        writer.close()

        # Collect metadata for aggregate _metadata file
        meta = pq.read_metadata(str(out_file))
        writer_meta.append(meta)

    elapsed = time.perf_counter() - t0
    log.info("Wrote %d tiles in %.1f s", len(unique_pixels), elapsed)

    # Write aggregate _metadata
    _write_aggregate_metadata(catalog_root, writer_meta, table.schema)
    _write_catalog_info(catalog_root, survey_name, norder, len(table), len(table.schema))
    log.info("Catalog written to %s", catalog_root)


def _write_aggregate_metadata(
    catalog_root: Path,
    file_metadata: list[pq.FileMetaData],
    schema: pa.Schema,
) -> None:
    if not file_metadata:
        return
    combined = file_metadata[0]
    for m in file_metadata[1:]:
        combined.append_row_groups(m)
    combined.write_metadata_file(str(catalog_root / "_metadata"))


def _write_catalog_info(
    catalog_root: Path,
    survey_name: str,
    norder: int,
    n_rows: int,
    n_cols: int,
) -> None:
    info = {
        "catalog_name": survey_name,
        "catalog_type": "object",
        "hats_order": norder,
        "total_rows": n_rows,
        "total_columns": n_cols,
        "schema_version": "1",
        "epoch": "J2000",
        "ra_column": "ra",
        "dec_column": "dec",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(catalog_root / "catalog_info.json", "w") as fh:
        json.dump(info, fh, indent=2)


# ---------------------------------------------------------------------------
# Batch ingest: multiple files → single survey
# ---------------------------------------------------------------------------


def ingest_catalog_batch(
    source_paths: Sequence[Path | str],
    output_root: Path | str,
    survey_name: str,
    ra_col: str = "ra",
    dec_col: str = "dec",
    norder: int = 5,
    source_id_col: str | None = None,
) -> None:
    """Ingest multiple source files into the same survey catalog."""
    for path in source_paths:
        ingest_catalog(
            source_path=path,
            output_root=output_root,
            survey_name=survey_name,
            ra_col=ra_col,
            dec_col=dec_col,
            norder=norder,
            source_id_col=source_id_col,
            overwrite=True,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    @click.command("dl-ingest-catalog")
    @click.argument("source_path", type=click.Path(exists=True, path_type=Path))
    @click.argument("output_root", type=click.Path(path_type=Path))
    @click.option("--survey", "survey_name", required=True, help="Short survey name.")
    @click.option("--ra-col", default="ra", show_default=True)
    @click.option("--dec-col", default="dec", show_default=True)
    @click.option("--norder", default=5, show_default=True, type=int)
    @click.option("--source-id-col", default=None)
    @click.option("--overwrite", is_flag=True)
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        source_path: Path,
        output_root: Path,
        survey_name: str,
        ra_col: str,
        dec_col: str,
        norder: int,
        source_id_col: str | None,
        overwrite: bool,
        verbose: bool,
    ) -> None:
        """Ingest FITS/VOTable SOURCE_PATH into HATS-partitioned Parquet at OUTPUT_ROOT."""
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        ingest_catalog(
            source_path=source_path,
            output_root=output_root,
            survey_name=survey_name,
            ra_col=ra_col,
            dec_col=dec_col,
            norder=norder,
            source_id_col=source_id_col,
            overwrite=overwrite,
        )

except ImportError:
    cli = None  # type: ignore[assignment]
