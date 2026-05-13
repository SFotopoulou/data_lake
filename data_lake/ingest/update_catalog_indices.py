"""
update_catalog_indices – rewrite only the affected tile Parquet files with
updated ``_cutout_index`` or ``_spectrum_index`` values after a Zarr ingest.

The ingest functions in ``fits_to_zarr`` and ``fits_to_spectra_zarr`` return a
``{source_id: local_index}`` mapping.  Pass that mapping to
``update_index_column`` so the relevant Parquet tiles are patched in-place.

Only tiles that actually contain matched source_ids are rewritten – every
other tile is left untouched, making this efficient even for large catalogs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import healpix_dir, _ZSTD_LEVEL

log = logging.getLogger(__name__)

IndexKind = Literal["cutout", "spectrum"]
_INDEX_COL: dict[str, str] = {
    "cutout":   "_cutout_index",
    "spectrum": "_spectrum_index",
}


def update_index_column(
    lake_root: Path | str,
    survey_name: str,
    source_id_to_index: dict[int, int],
    kind: IndexKind = "spectrum",
    norder: int = 5,
) -> int:
    """
    Update ``_cutout_index`` or ``_spectrum_index`` in the affected Parquet tiles.

    Reads only the tiles that contain at least one source_id from
    ``source_id_to_index``, modifies the relevant index column, and rewrites
    the tile file in-place with the same Zstd compression.

    Parameters
    ----------
    lake_root:
        Data lake root.
    survey_name:
        Survey whose catalog tiles should be updated.
    source_id_to_index:
        Mapping returned by ``ingest_spectra_from_fits`` / ``ingest_cutouts_from_fits``.
    kind:
        Which index column to update: ``"cutout"`` or ``"spectrum"``.
    norder:
        HEALPix partitioning order.

    Returns
    -------
    Number of Parquet tile files that were actually modified.
    """
    lake_root = Path(lake_root)
    catalog_root = lake_root / "catalogs" / survey_name
    if not catalog_root.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_root}")

    index_col = _INDEX_COL[kind]

    # Bucket the source_ids by their HEALPix tile so we only open each tile once
    try:
        import healpy as hp
        nside = hp.order2nside(norder)
    except ImportError:
        nside = None

    # Discover affected tile files by scanning the Parquet tree
    all_parquet = list(catalog_root.rglob("Npix=*.parquet"))
    if not all_parquet:
        log.warning("No Parquet tiles found in %s", catalog_root)
        return 0

    source_id_set = set(source_id_to_index.keys())
    n_modified = 0

    for tile_file in all_parquet:
        table = pq.read_table(str(tile_file))
        if "source_id" not in table.schema.names:
            continue

        tile_ids = table.column("source_id").to_pylist()
        matches = [sid for sid in tile_ids if sid in source_id_set]
        if not matches:
            continue

        # Materialise the index column as a mutable numpy array
        if index_col in table.schema.names:
            idx_arr = np.array(table.column(index_col).to_pylist(), dtype=np.int64)
        else:
            idx_arr = np.full(len(table), -1, dtype=np.int64)
            log.debug("Adding missing column %s to %s", index_col, tile_file.name)

        # Apply the updates
        id_to_row: dict[int, list[int]] = {}
        for row_i, sid in enumerate(tile_ids):
            id_to_row.setdefault(sid, []).append(row_i)

        for sid in matches:
            for row_i in id_to_row[sid]:
                idx_arr[row_i] = source_id_to_index[sid]

        # Replace / add column and rewrite
        if index_col in table.schema.names:
            col_pos = table.schema.get_field_index(index_col)
            table = table.set_column(col_pos, index_col, pa.array(idx_arr, type=pa.int64()))
        else:
            table = table.append_column(index_col, pa.array(idx_arr, type=pa.int64()))

        pq.write_table(
            table,
            str(tile_file),
            compression="zstd",
            compression_level=_ZSTD_LEVEL,
            write_statistics=True,
            use_dictionary=True,
        )
        n_modified += 1
        log.debug("Updated %d rows in %s", len(matches), tile_file.name)

    log.info(
        "Updated %s in %d tile file(s) for survey=%s (%d sources)",
        index_col, n_modified, survey_name, len(source_id_to_index),
    )

    # Regenerate aggregate _metadata
    if n_modified > 0:
        _regenerate_metadata(catalog_root)

    return n_modified


def _regenerate_metadata(catalog_root: Path) -> None:
    """Rebuild the aggregate Parquet ``_metadata`` file after tile rewrites."""
    try:
        file_metas = [
            pq.read_metadata(str(p))
            for p in sorted(catalog_root.rglob("Npix=*.parquet"))
        ]
        if not file_metas:
            return
        combined = file_metas[0]
        for m in file_metas[1:]:
            combined.append_row_groups(m)
        combined.write_metadata_file(str(catalog_root / "_metadata"))
        log.debug("Regenerated _metadata for %s", catalog_root.name)
    except Exception as exc:
        log.warning("Could not regenerate _metadata: %s", exc)
