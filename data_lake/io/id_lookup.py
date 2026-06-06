"""
Bulk source_id → (npix, local_index) resolution for spectra and cutouts.

Combines batched DuckDB catalog queries with optional vectorised Zarr tile scans.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from data_lake.io.catalog import CatalogAccessor

log = logging.getLogger(__name__)

ModalityKind = Literal["spectrum", "cutout"]


def _index_columns(kind: ModalityKind, norder: int) -> tuple[str, str, str]:
    """Return (index_col, npix_col, fallback_hp_col) for a modality."""
    index_col = f"_{kind}_index"
    npix_col = f"_{kind}_npix"
    hp_col = f"_healpix_norder{norder}"
    return index_col, npix_col, hp_col


def bulk_tile_index_from_catalog(
    catalog: CatalogAccessor,
    source_ids: Sequence[int] | np.ndarray,
    kind: ModalityKind,
    *,
    batch_size: int = 10_000,
    show_progress: bool = True,
) -> dict[int, tuple[int, int]]:
    """Resolve IDs via batched SQL against the catalog view."""
    requested_int = np.unique(np.asarray(source_ids, dtype=np.int64).ravel())
    result: dict[int, tuple[int, int]] = {}
    if requested_int.size == 0:
        return result

    cat_cols = catalog.columns
    index_col, npix_col, hp_col = _index_columns(kind, catalog.norder)
    if index_col not in cat_cols:
        return result

    if npix_col in cat_cols:
        tile_col = npix_col
    elif hp_col in cat_cols:
        tile_col = hp_col
    else:
        return result

    sid_col = catalog.link_id_column

    try:
        from tqdm.auto import tqdm
    except ImportError:

        def tqdm(x, **_kw):  # type: ignore[misc]
            return x

    try:
        for start in tqdm(
            range(0, len(requested_int), batch_size),
            desc=f"catalog {kind} lookup",
            disable=not show_progress,
            unit="batch",
        ):
            chunk = requested_int[start : start + batch_size]
            ids_csv = ",".join(str(int(s)) for s in chunk)
            sql = (
                f"SELECT {sid_col}, {tile_col}, {index_col} "
                f"FROM catalog WHERE {sid_col} IN ({ids_csv}) "
                f"AND {index_col} >= 0"
            )
            rows = catalog._con.execute(sql).fetchall()
            for sid, npix, lidx in rows:
                result[int(sid)] = (int(npix), int(lidx))
    except Exception:  # pragma: no cover - defensive
        log.warning(
            "Catalog bulk %s lookup failed; tile scan may be needed.",
            kind,
            exc_info=True,
        )
        result.clear()

    return result


def bulk_tile_index_with_scan(
    source_ids: Sequence[int] | np.ndarray,
    *,
    catalog: CatalogAccessor | None = None,
    kind: ModalityKind = "spectrum",
    scan_tile: Callable[[int, np.ndarray], dict[int, tuple[int, int]]],
    available_tiles: Callable[[], Iterable[int]],
    show_progress: bool = True,
) -> dict[int, tuple[int, int]]:
    """Resolve source IDs using catalog SQL then vectorised tile scan for misses."""
    try:
        from tqdm.auto import tqdm
    except ImportError:

        def tqdm(x, **_kw):  # type: ignore[misc]
            return x

    requested_int = np.unique(np.asarray(source_ids, dtype=np.int64).ravel())
    result: dict[int, tuple[int, int]] = {}

    if catalog is not None:
        result = bulk_tile_index_from_catalog(
            catalog, requested_int, kind, show_progress=show_progress,
        )

    remaining = np.setdiff1d(
        requested_int,
        np.fromiter(result.keys(), dtype=np.int64, count=len(result)),
        assume_unique=True,
    )

    if remaining.size > 0:
        for npix in tqdm(
            available_tiles(),
            desc=f"{kind} tile scan",
            disable=not show_progress,
            unit="tile",
        ):
            if remaining.size == 0:
                break
            hits = scan_tile(int(npix), remaining)
            if not hits:
                continue
            result.update(hits)
            remaining = np.setdiff1d(
                remaining,
                np.fromiter(hits.keys(), dtype=np.int64, count=len(hits)),
                assume_unique=False,
            )

    return result
