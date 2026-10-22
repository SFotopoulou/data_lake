"""Embed catalog link-id columns into spectrum / cutout subset exports.

At extract time we pull the native identifier columns that define
``_source_id`` (e.g. ``TARGETID``, ``SURVEY``, ``PROGRAM``) from the
Parquet catalog and write them as peer arrays alongside ``_source_id``.
Exported Zarr / HDF5 files are then self-contained for TOPCAT joins
without needing the lake.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Attr keys written on the subset root / file
ATTR_LINK_ID_COLUMNS = "link_id_columns"
ATTR_LINK_ID_MODE = "link_id_mode"


def values_to_numpy(values: Sequence[object]) -> np.ndarray:
    """Convert a sequence of Python values to a NumPy array for Zarr/HDF5."""
    sample = next((v for v in values if v is not None), None)
    if sample is None:
        return np.asarray([""] * len(values), dtype="U1")
    if isinstance(sample, (bool, np.bool_)):
        return np.asarray(
            [bool(v) if v is not None else False for v in values], dtype=np.bool_
        )
    if isinstance(sample, (int, np.integer)) and not isinstance(sample, (bool, np.bool_)):
        return np.asarray(
            [int(v) if v is not None else -1 for v in values], dtype=np.int64
        )
    if isinstance(sample, (float, np.floating)):
        return np.asarray(
            [float(v) if v is not None else np.nan for v in values], dtype=np.float64
        )
    # Strings / mixed → fixed-width UTF-8 bytes (stable under Zarr v3;
    # FixedLengthUTF32 has no Zarr V3 spec yet).
    texts = ["" if v is None else str(v) for v in values]
    encoded = [t.encode("utf-8") for t in texts]
    max_b = max((len(b) for b in encoded), default=1)
    return np.asarray(encoded, dtype=f"|S{max(max_b, 1)}")


def align_column_maps(
    ordered_sids: np.ndarray,
    column_maps: Mapping[str, Mapping[int, object]],
) -> dict[str, np.ndarray]:
    """Build length-``N`` arrays for each column, ordered like *ordered_sids*."""
    out: dict[str, np.ndarray] = {}
    sids = [int(s) for s in np.asarray(ordered_sids).tolist()]
    for col, mapping in column_maps.items():
        values = [mapping.get(s) for s in sids]
        out[col] = values_to_numpy(values)
        n_hit = sum(1 for v in values if v is not None)
        if n_hit < len(sids):
            log.warning(
                "Link-id column %r: %d/%d source_ids found in catalog",
                col, n_hit, len(sids),
            )
    return out


def load_link_id_column_maps(
    catalog,
    source_ids: Sequence[int] | np.ndarray,
) -> tuple[list[str], dict[str, dict[int, object]]]:
    """Resolve component columns + per-id maps from a :class:`CatalogAccessor`.

    Returns ``([], {})`` when the catalog has no recoverable native ID columns.
    """
    if catalog is None:
        return [], {}
    cols = list(catalog.link_id_component_columns)
    if not cols:
        return [], {}
    maps = catalog.values_for_source_ids(source_ids, cols)
    log.info(
        "Embedding link-id columns %s from catalog %r (%s).",
        cols,
        getattr(catalog, "survey_name", "?"),
        getattr(catalog, "link_id_mode", "") or "unknown mode",
    )
    return cols, maps


def write_link_id_arrays_zarr(
    root,
    ordered_sids: np.ndarray,
    columns: Sequence[str],
    column_maps: Mapping[str, Mapping[int, object]],
    *,
    link_id_mode: str = "",
) -> None:
    """Create one Zarr array per link-id column + attrs on *root*."""
    if not columns:
        return
    aligned = align_column_maps(ordered_sids, column_maps)
    for col in columns:
        arr = aligned[col]
        if col in root:
            del root[col]
        root.create_array(
            col,
            data=arr,
            chunks=(min(4096, max(len(arr), 1)),),
        )
    root.attrs[ATTR_LINK_ID_COLUMNS] = list(columns)
    if link_id_mode:
        root.attrs[ATTR_LINK_ID_MODE] = link_id_mode


def write_link_id_arrays_hdf5(
    handle,
    ordered_sids: np.ndarray,
    columns: Sequence[str],
    column_maps: Mapping[str, Mapping[int, object]],
    *,
    link_id_mode: str = "",
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> None:
    """Create one HDF5 dataset per link-id column + attrs on *handle*."""
    if not columns:
        return
    aligned = align_column_maps(ordered_sids, column_maps)
    ck: dict[str, Any] = {}
    if compression:
        ck = {"compression": compression, "compression_opts": compression_opts}
    for col in columns:
        arr = aligned[col]
        if col in handle:
            del handle[col]
        # h5py needs encode for unicode; use variable-length UTF-8 for strings
        if arr.dtype.kind in ("S", "U", "O"):
            import h5py

            dt = h5py.string_dtype(encoding="utf-8")
            ds = handle.create_dataset(col, shape=(len(arr),), dtype=dt, **ck)
            if arr.dtype.kind == "S":
                ds[:] = [b.decode("utf-8") for b in arr.tolist()]
            else:
                ds[:] = np.asarray(arr, dtype=object)
        else:
            handle.create_dataset(col, data=arr, **ck)
    handle.attrs[ATTR_LINK_ID_COLUMNS] = list(columns)
    if link_id_mode:
        handle.attrs[ATTR_LINK_ID_MODE] = link_id_mode
