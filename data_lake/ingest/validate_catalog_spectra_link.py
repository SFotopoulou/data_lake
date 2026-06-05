"""
Cross-validate catalog Parquet ↔ spectrum Zarr linkage for one survey.

Checks that ``_source_id``, ``_healpix_norder{N}``, and ``_spectrum_index`` on
each catalog row agree with the paired ``Npix=*.zarr`` tile's ``_source_id``
array (or legacy ``source_id``).

Performance
-----------
When the catalog has ``_spectrum_npix`` (post-v0.2 format), the previous
implementation re-read every catalog Parquet tile for every Zarr tile, giving
O(n_zarr × n_catalog) total reads.  :func:`build_catalog_link_index` does a
single pass over the catalog and builds a ``spec_npix``-keyed lookup so each
Zarr tile only touches the catalog rows that claim to link into it.

With 6.5 M spectra and 5.8 M catalog rows at Norder=5 this reduces runtime
from hours to a few minutes (dominated by sequential Zarr shard reads).  Use
``--n-workers`` to parallelise the Zarr scan further.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import (
    LAKE_JOIN_ID_COLUMN,
    healpix_dir,
    normalize_object_id,
    resolve_link_id_column,
)
from data_lake.ingest.zarr_ids import zarr_join_array

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stats / report (public API — unchanged)
# ---------------------------------------------------------------------------

@dataclass
class LinkValidationStats:
    n_tiles_checked: int = 0
    n_linked: int = 0
    n_stale_index: int = 0
    n_wrong_id: int = 0
    n_wrong_healpix: int = 0
    n_orphan_zarr: int = 0
    n_unpatched_catalog: int = 0
    n_missing_catalog_tile: int = 0
    n_empty_zarr: int = 0
    n_null_source_id: int = 0
    n_null_source_id_linked: int = 0


@dataclass
class LinkValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: LinkValidationStats = field(default_factory=LinkValidationStats)

    def ok(self, *, strict: bool) -> bool:
        if self.errors:
            return False
        if strict:
            if self.warnings:
                return False
            st = self.stats
            if st.n_orphan_zarr > 0 or st.n_unpatched_catalog > 0:
                return False
        return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_spectrum_zarr_tiles(spectra_root: Path) -> Iterator[Path]:
    yield from sorted(spectra_root.rglob("Npix=*.zarr"))


def _npix_from_tile_name(name: str) -> int:
    return int(name.split("=")[-1].split(".")[0])


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_catalog_tile_columns(tile_path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    pf = pq.ParquetFile(tile_path)
    available = set(pf.schema_arrow.names)
    missing = [c for c in columns if c not in available]
    if missing:
        raise KeyError(
            f"{tile_path}: missing column(s) {missing!r} "
            f"(available: {sorted(available)[:30]})"
        )
    table = pf.read(columns=columns)
    out: dict[str, np.ndarray] = {}
    for name in columns:
        col = table.column(name).combine_chunks()
        if pa.types.is_dictionary(col.type):
            col = pc.cast(col, col.type.value_type)
        if pa.types.is_integer(col.type) and col.null_count > 0:
            out[name] = np.array(col.to_pylist(), dtype=object)
        else:
            out[name] = np.asarray(col.to_numpy(zero_copy_only=False))
    return out


def _resolve_norders(
    catalog_root: Path,
    spectra_root: Path,
    norder: int | None,
    rep: LinkValidationReport,
) -> tuple[int, int]:
    """Return ``(cat_order, spec_order)``; each defaults independently.

    Orders may differ when catalog and spectra were partitioned at different
    resolutions — this is now supported via ``_spectrum_npix`` on catalog rows.
    A warning (not an error) is emitted when orders differ.
    """
    cat_order: int | None = norder
    spec_order: int | None = norder

    cat_info = catalog_root / "catalog_info.json"
    if cat_info.is_file():
        cat_order = int(_load_json(cat_info).get("hats_order", cat_order or 5))
    spec_info = spectra_root / "spectrum_info.json"
    if spec_info.is_file():
        spec_order = int(_load_json(spec_info).get("hats_order", spec_order or 5))

    cat_order = cat_order or 5
    spec_order = spec_order or 5

    if cat_order != spec_order:
        rep.warnings.append(
            f"hats_order differs: catalog {cat_order} vs spectra {spec_order} "
            f"(OK when _spectrum_npix linkage is present)"
        )
    return cat_order, spec_order


def _catalog_tile_path(catalog_root: Path, norder: int, npix: int) -> Path:
    return catalog_root / healpix_dir(norder, npix) / f"Npix={npix}.parquet"


def _optional_catalog_sid(value: object) -> int | None:
    """Return normalized int64 ID, or None for null / missing link parts."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return int(normalize_object_id(value))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Catalog link index
# ---------------------------------------------------------------------------

@dataclass
class _TileLinkData:
    """Pre-indexed forward-check data for one catalog tile scoped to one spec_npix."""

    cat_tile: Path
    row_indices: np.ndarray   # int64[k]: local row offsets with spec_index >= 0
    spec_indices: np.ndarray  # int64[k]: _spectrum_index values
    sids: np.ndarray          # int64[k]: normalized source_ids (0 where null)
    null_mask: np.ndarray     # bool[k]: True where source_id was null


@dataclass
class CatalogLinkIndex:
    """Pre-built index from a single pass over all catalog Parquet tiles.

    Eliminates the O(n_zarr × n_catalog) re-read pattern: each Zarr tile only
    touches the catalog rows that name it via ``_spectrum_npix``.  The global
    reverse-check sets (``all_catalog_sids``, ``unpatched_sids``) are built
    once and reused across all tiles.

    Only built when the catalog has the ``_spectrum_npix`` column.  The legacy
    path continues to use direct per-tile Parquet reads (already O(1) tiles per
    Zarr tile).
    """

    by_spec_npix: dict[int, list[_TileLinkData]]
    all_catalog_sids: set[int]
    unpatched_sids: set[int]
    n_null_source_id: int


def _decode_sid_col(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(int64 sids, bool null_mask)`` from a raw source-id column array.

    Handles both object arrays (nullable int column) and plain int arrays.
    """
    if raw.dtype == object:
        null_mask = np.array(
            [v is None or (isinstance(v, float) and np.isnan(v)) for v in raw],
            dtype=bool,
        )
        sids = np.zeros(len(raw), dtype=np.int64)
        valid = ~null_mask
        if valid.any():
            sids[valid] = np.fromiter(
                (int(normalize_object_id(v)) for v in raw[valid]),
                dtype=np.int64,
                count=int(valid.sum()),
            )
    else:
        null_mask = np.zeros(len(raw), dtype=bool)
        sids = raw.astype(np.int64)
    return sids, null_mask


def build_catalog_link_index(
    all_parquet: list[Path],
    sid_col: str,
    cat_order: int,
    has_npix_col: bool,
    *,
    show_progress: bool = False,
) -> CatalogLinkIndex:
    """Scan every catalog Parquet tile once and return a ``spec_npix``-keyed index.

    Each tile is read with only three columns (``sid_col``, ``_spectrum_index``,
    ``_spectrum_npix``).  The resulting index lets each Zarr tile validation
    look up only the catalog rows that point at it, rather than scanning the
    whole catalog.

    Parameters
    ----------
    all_parquet:
        Sorted list of all ``Npix=*.parquet`` paths under the catalog root.
    sid_col:
        Catalog column holding the source identifier (e.g. ``_source_id``).
    cat_order:
        HEALPix order of the catalog (used when ``has_npix_col=False`` to name
        the hp column).
    has_npix_col:
        ``True`` when catalog has ``_spectrum_npix``; ``False`` for the legacy
        path (groups by ``_healpix_norder{N}`` instead).
    show_progress:
        Show a tqdm progress bar while scanning catalog tiles.
    """
    index_col = "_spectrum_index"
    ref_col = "_spectrum_npix" if has_npix_col else f"_healpix_norder{cat_order}"
    read_cols = [sid_col, index_col, ref_col]

    by_spec_npix: dict[int, list[_TileLinkData]] = {}
    all_catalog_sids: set[int] = set()
    unpatched_sids: set[int] = set()
    n_null_source_id = 0

    tiles_iter: Any = all_parquet
    if show_progress:
        try:
            from tqdm.auto import tqdm
            tiles_iter = tqdm(all_parquet, unit="tile", desc="indexing catalog")
        except ImportError:
            pass

    for tile_path in tiles_iter:
        if not tile_path.is_file():
            continue
        try:
            pf = pq.ParquetFile(tile_path)
            available = set(pf.schema_arrow.names)
            missing = [c for c in read_cols if c not in available]
            if missing:
                log.warning("%s: skipping in index — missing columns %r", tile_path, missing)
                continue
            table = pf.read(columns=read_cols)
        except Exception as exc:
            log.warning("Cannot read catalog tile %s during indexing: %s", tile_path, exc)
            continue

        # Decode source-id column (may have nulls)
        raw_sid = table.column(sid_col).combine_chunks()
        if pa.types.is_dictionary(raw_sid.type):
            raw_sid = pc.cast(raw_sid, raw_sid.type.value_type)
        if pa.types.is_integer(raw_sid.type) and raw_sid.null_count > 0:
            raw_sid_np: np.ndarray = np.array(raw_sid.to_pylist(), dtype=object)
        else:
            raw_sid_np = np.asarray(raw_sid.to_numpy(zero_copy_only=False))

        sids, null_mask = _decode_sid_col(raw_sid_np)

        # Decode integer index/ref columns (no nulls expected)
        cat_idx = np.asarray(
            table.column(index_col).to_numpy(zero_copy_only=False), dtype=np.int64
        )
        cat_ref = np.asarray(
            table.column(ref_col).to_numpy(zero_copy_only=False), dtype=np.int64
        )

        # Build global reverse sets (vectorized)
        n_null_source_id += int(null_mask.sum())
        valid_mask = ~null_mask
        if valid_mask.any():
            all_catalog_sids.update(sids[valid_mask].tolist())
            unpatched_mask = valid_mask & (cat_idx < 0)
            if unpatched_mask.any():
                unpatched_sids.update(sids[unpatched_mask].tolist())

        # Build forward index: group linked rows by spec_npix / hp_npix
        linked_positions = np.nonzero(cat_idx >= 0)[0]
        if linked_positions.size == 0:
            continue

        linked_ref = cat_ref[linked_positions]
        for npix_val in np.unique(linked_ref):
            npix = int(npix_val)
            sel = linked_ref == npix_val
            rows = linked_positions[sel].astype(np.int64)
            by_spec_npix.setdefault(npix, []).append(
                _TileLinkData(
                    cat_tile=tile_path,
                    row_indices=rows,
                    spec_indices=cat_idx[rows],
                    sids=sids[rows],
                    null_mask=null_mask[rows],
                )
            )

    return CatalogLinkIndex(
        by_spec_npix=by_spec_npix,
        all_catalog_sids=all_catalog_sids,
        unpatched_sids=unpatched_sids,
        n_null_source_id=n_null_source_id,
    )


# ---------------------------------------------------------------------------
# Per-tile validation (fast indexed path)
# ---------------------------------------------------------------------------

def _validate_tile_with_index(
    *,
    zarr_tile: Path,
    zarr_npix: int,
    cat_order: int,
    sid_col: str,
    rep: LinkValidationReport,
    cat_index: CatalogLinkIndex,
    has_npix_col: bool,
    sample: int | None,
    rng: random.Random | None,
    quiet: bool,
) -> None:
    """Validate one Zarr tile using the pre-built catalog link index."""
    import zarr as _zarr

    stats = rep.stats
    stats.n_tiles_checked += 1

    try:
        root = _zarr.open_group(
            store=_zarr.storage.LocalStore(str(zarr_tile)),
            mode="r",
            zarr_format=3,
        )
    except Exception as exc:
        rep.errors.append(f"Cannot open Zarr tile {zarr_tile}: {exc}")
        return

    if LAKE_JOIN_ID_COLUMN not in root:
        rep.errors.append(f"{zarr_tile}: missing join array")
        return

    zarr_ids = np.asarray(zarr_join_array(root)[:], dtype=np.int64)
    n_zarr = int(zarr_ids.shape[0])
    if n_zarr == 0:
        stats.n_empty_zarr += 1
        return

    index_col = "_spectrum_index"
    ref_col = "_spectrum_npix" if has_npix_col else f"_healpix_norder{cat_order}"

    tile_links: list[_TileLinkData] = cat_index.by_spec_npix.get(zarr_npix, [])
    zarr_idx_linked: set[int] = set()

    for tld in tile_links:
        rows = tld.row_indices
        spec_idxs = tld.spec_indices
        sids = tld.sids
        nulls = tld.null_mask

        # Optional sampling (per catalog-tile / spec_npix group)
        if sample is not None and sample > 0 and len(rows) > sample:
            pick = rng or random.Random(0)
            chosen = np.array(pick.sample(range(len(rows)), sample), dtype=np.int64)
            rows = rows[chosen]
            spec_idxs = spec_idxs[chosen]
            sids = sids[chosen]
            nulls = nulls[chosen]

        # Null-sid linked rows → errors
        if nulls.any():
            for k in np.nonzero(nulls)[0]:
                stats.n_null_source_id_linked += 1
                rep.errors.append(
                    f"{tld.cat_tile.name} row {int(rows[k])}: {index_col}={int(spec_idxs[k])} "
                    f"but {sid_col} is null (cannot verify Zarr linkage)"
                )

        valid = ~nulls
        if not valid.any():
            continue

        v_rows = rows[valid]
        v_idx = spec_idxs[valid]
        v_sids = sids[valid]

        # Range check (vectorized)
        in_range = (v_idx >= 0) & (v_idx < n_zarr)
        if (~in_range).any():
            for k in np.nonzero(~in_range)[0]:
                stats.n_stale_index += 1
                rep.errors.append(
                    f"{tld.cat_tile.name} row {int(v_rows[k])}: {index_col}={int(v_idx[k])} "
                    f"out of range (Zarr rows={n_zarr} in Npix={zarr_npix})"
                )

        # ID check (vectorized)
        v_idx_ok = v_idx[in_range]
        v_sids_ok = v_sids[in_range]
        v_rows_ok = v_rows[in_range]
        zarr_at = zarr_ids[v_idx_ok]
        match = zarr_at == v_sids_ok

        for k in np.nonzero(match)[0]:
            stats.n_linked += 1
            zarr_idx_linked.add(int(v_idx_ok[k]))

        for k in np.nonzero(~match)[0]:
            stats.n_wrong_id += 1
            rep.errors.append(
                f"{tld.cat_tile.name} row {int(v_rows_ok[k])}: "
                f"{sid_col}={int(v_sids_ok[k])} but "
                f"Zarr {index_col}={int(v_idx_ok[k])} has _source_id={int(zarr_at[k])}"
            )

    # Reverse check: classify unlinked Zarr rows as "unpatched" or "orphan".
    # Build unlinked index mask efficiently.
    if zarr_idx_linked:
        linked_arr = np.fromiter(zarr_idx_linked, dtype=np.int64, count=len(zarr_idx_linked))
        unlinked_mask = np.ones(n_zarr, dtype=bool)
        unlinked_mask[linked_arr] = False
        unlinked_indices = np.nonzero(unlinked_mask)[0]
    else:
        unlinked_indices = np.arange(n_zarr, dtype=np.int64)

    for j in unlinked_indices.tolist():
        sid = int(zarr_ids[j])
        if sid in cat_index.unpatched_sids:
            stats.n_unpatched_catalog += 1
            if not quiet:
                rep.warnings.append(
                    f"{zarr_tile.name} row {j}: _source_id={sid} in catalog but "
                    f"{index_col}=-1 (run dl-rebuild-catalog-indices)"
                )
        elif sid not in cat_index.all_catalog_sids:
            stats.n_orphan_zarr += 1
            if not quiet:
                rep.warnings.append(
                    f"{zarr_tile.name} row {j}: _source_id={sid} not linked by any "
                    f"catalog row with {ref_col}={zarr_npix} "
                    f"(run dl-rebuild-catalog-indices)"
                )


# ---------------------------------------------------------------------------
# Per-tile validation (public entry point — dispatches to fast or legacy path)
# ---------------------------------------------------------------------------

def validate_tile_link(
    *,
    zarr_tile: Path,
    zarr_npix: int,
    cat_order: int,
    cat_tiles: list[Path] | None = None,
    sid_col: str,
    rep: LinkValidationReport,
    has_npix_col: bool,
    sample: int | None = None,
    rng: random.Random | None = None,
    quiet: bool = False,
    cat_index: CatalogLinkIndex | None = None,
) -> None:
    """Validate one Zarr tile against its linked catalog rows.

    When ``cat_index`` is provided (and ``has_npix_col`` is True), the fast
    indexed path is used: only the pre-extracted rows for this ``zarr_npix``
    are touched, with no additional Parquet reads.

    When ``cat_index`` is ``None`` (or ``has_npix_col`` is False), falls back
    to the original per-tile Parquet read path using ``cat_tiles``.
    """
    if cat_index is not None:
        _validate_tile_with_index(
            zarr_tile=zarr_tile,
            zarr_npix=zarr_npix,
            cat_order=cat_order,
            sid_col=sid_col,
            rep=rep,
            cat_index=cat_index,
            has_npix_col=has_npix_col,
            sample=sample,
            rng=rng,
            quiet=quiet,
        )
        return

    # Legacy / fallback path: read catalog tiles directly.
    import zarr as _zarr

    stats = rep.stats
    stats.n_tiles_checked += 1

    try:
        root = _zarr.open_group(
            store=_zarr.storage.LocalStore(str(zarr_tile)),
            mode="r",
            zarr_format=3,
        )
    except Exception as exc:
        rep.errors.append(f"Cannot open Zarr tile {zarr_tile}: {exc}")
        return

    if LAKE_JOIN_ID_COLUMN not in root:
        rep.errors.append(f"{zarr_tile}: missing join array")
        return

    zarr_ids = np.asarray(zarr_join_array(root)[:], dtype=np.int64)
    n_zarr = int(zarr_ids.shape[0])
    if n_zarr == 0:
        stats.n_empty_zarr += 1
        return

    index_col = "_spectrum_index"
    npix_col = "_spectrum_npix"
    hp_col = f"_healpix_norder{cat_order}"

    zarr_idx_linked: set[int] = set()
    zarr_sid_in_catalog: set[int] = set()
    zarr_sid_unpatched: set[int] = set()

    if not has_npix_col:
        candidate_tiles = [t for t in (cat_tiles or []) if f"Npix={zarr_npix}.parquet" in t.name]
        if not candidate_tiles:
            stats.n_missing_catalog_tile += 1
            rep.warnings.append(
                f"{zarr_tile.name}: no catalog tile Npix={zarr_npix} "
                f"(legacy mode, orders must match)"
            )
            stats.n_orphan_zarr += n_zarr
            return
    else:
        candidate_tiles = cat_tiles or []

    for cat_tile in candidate_tiles:
        if not cat_tile.is_file():
            continue

        read_cols = [sid_col, index_col, npix_col if has_npix_col else hp_col]
        try:
            cols = _read_catalog_tile_columns(cat_tile, read_cols)
        except KeyError as exc:
            rep.errors.append(f"{cat_tile}: {exc}")
            continue

        sid_raw = cols[sid_col]
        sid_iter = sid_raw.tolist() if isinstance(sid_raw, np.ndarray) else list(sid_raw)
        cat_sids: list[int | None] = [_optional_catalog_sid(x) for x in sid_iter]
        cat_idx = np.asarray(cols[index_col], dtype=np.int64)
        ref_col = npix_col if has_npix_col else hp_col
        cat_ref = np.asarray(cols[ref_col], dtype=np.int64)

        if has_npix_col:
            candidate_rows = np.nonzero((cat_idx >= 0) & (cat_ref == zarr_npix))[0]
        else:
            candidate_rows = np.nonzero(cat_idx >= 0)[0]

        if sample is not None and sample > 0 and candidate_rows.size > sample:
            pick = rng or random.Random(0)
            candidate_rows = np.array(
                pick.sample(candidate_rows.tolist(), sample), dtype=np.int64,
            )

        for row_i in candidate_rows.tolist():
            idx = int(cat_idx[row_i])
            sid_opt = cat_sids[row_i]
            if sid_opt is None:
                stats.n_null_source_id_linked += 1
                rep.errors.append(
                    f"{cat_tile.name} row {row_i}: {index_col}={idx} but "
                    f"{sid_col} is null (cannot verify Zarr linkage)"
                )
                continue
            sid = sid_opt

            if not has_npix_col:
                hp = int(cat_ref[row_i])
                if hp != zarr_npix:
                    continue

            if idx < 0 or idx >= n_zarr:
                stats.n_stale_index += 1
                rep.errors.append(
                    f"{cat_tile.name} row {row_i}: {index_col}={idx} out of range "
                    f"(Zarr rows={n_zarr} in Npix={zarr_npix})"
                )
                continue

            zarr_sid = int(normalize_object_id(int(zarr_ids[idx])))
            if zarr_sid != sid:
                stats.n_wrong_id += 1
                rep.errors.append(
                    f"{cat_tile.name} row {row_i}: {sid_col}={sid} but "
                    f"Zarr {index_col}={idx} has _source_id={zarr_sid}"
                )
            else:
                stats.n_linked += 1
                zarr_idx_linked.add(idx)

        for row_i, sid_opt in enumerate(cat_sids):
            if sid_opt is None:
                stats.n_null_source_id += 1
                continue
            zarr_sid_in_catalog.add(sid_opt)
            if int(cat_idx[row_i]) < 0:
                zarr_sid_unpatched.add(sid_opt)

    for j, sid_raw in enumerate(zarr_ids.tolist()):
        if j in zarr_idx_linked:
            continue
        sid = int(normalize_object_id(int(sid_raw)))
        if sid in zarr_sid_unpatched:
            stats.n_unpatched_catalog += 1
            if not quiet:
                rep.warnings.append(
                    f"{zarr_tile.name} row {j}: _source_id={sid} in catalog but "
                    f"{index_col}=-1 (run dl-rebuild-catalog-indices)"
                )
        elif sid not in zarr_sid_in_catalog:
            stats.n_orphan_zarr += 1
            if not quiet:
                rep.warnings.append(
                    f"{zarr_tile.name} row {j}: _source_id={sid} not linked by any "
                    f"catalog row with {npix_col if has_npix_col else hp_col}={zarr_npix} "
                    f"(run dl-rebuild-catalog-indices)"
                )


# ---------------------------------------------------------------------------
# Parallel worker support
# ---------------------------------------------------------------------------

_WORKER_INDEX: CatalogLinkIndex | None = None


def _init_worker(index: CatalogLinkIndex) -> None:
    """ProcessPoolExecutor initializer: install the catalog link index in workers."""
    global _WORKER_INDEX
    _WORKER_INDEX = index


@dataclass
class _ValidationTileConfig:
    """Pickle-friendly per-tile config for parallel validation workers."""

    zarr_tile: str        # str, not Path — slightly faster pickling
    zarr_npix: int
    cat_order: int
    sid_col: str
    has_npix_col: bool
    sample: int | None
    seed: int
    quiet: bool
    # Legacy path only: pre-resolved matching catalog tile path (or "" if absent)
    legacy_cat_tile: str = ""


@dataclass
class _ValidationTileResult:
    stats: LinkValidationStats
    errors: list[str]
    warnings: list[str]


def _validation_tile_worker(config: _ValidationTileConfig) -> _ValidationTileResult:
    """Validate one Zarr tile in a worker process."""
    rep = LinkValidationReport()
    # Use a per-tile seed so workers don't all sample the same rows.
    rng = random.Random(config.seed + config.zarr_npix) if config.sample else None

    if _WORKER_INDEX is not None:
        validate_tile_link(
            zarr_tile=Path(config.zarr_tile),
            zarr_npix=config.zarr_npix,
            cat_order=config.cat_order,
            sid_col=config.sid_col,
            rep=rep,
            has_npix_col=config.has_npix_col,
            sample=config.sample,
            rng=rng,
            quiet=config.quiet,
            cat_index=_WORKER_INDEX,
        )
    else:
        # Legacy parallel path: read the one matching catalog tile from disk.
        legacy_tiles = [Path(config.legacy_cat_tile)] if config.legacy_cat_tile else []
        validate_tile_link(
            zarr_tile=Path(config.zarr_tile),
            zarr_npix=config.zarr_npix,
            cat_order=config.cat_order,
            cat_tiles=legacy_tiles,
            sid_col=config.sid_col,
            rep=rep,
            has_npix_col=config.has_npix_col,
            sample=config.sample,
            rng=rng,
            quiet=config.quiet,
        )
    return _ValidationTileResult(
        stats=rep.stats,
        errors=rep.errors,
        warnings=rep.warnings,
    )


def _merge_tile_result(rep: LinkValidationReport, result: _ValidationTileResult) -> None:
    s, r = rep.stats, result.stats
    s.n_tiles_checked += r.n_tiles_checked
    s.n_linked += r.n_linked
    s.n_stale_index += r.n_stale_index
    s.n_wrong_id += r.n_wrong_id
    s.n_wrong_healpix += r.n_wrong_healpix
    s.n_orphan_zarr += r.n_orphan_zarr
    s.n_unpatched_catalog += r.n_unpatched_catalog
    s.n_missing_catalog_tile += r.n_missing_catalog_tile
    s.n_empty_zarr += r.n_empty_zarr
    # n_null_source_id is set globally from the index (not per tile)
    s.n_null_source_id_linked += r.n_null_source_id_linked
    rep.errors.extend(result.errors)
    rep.warnings.extend(result.warnings)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_validation(
    lake_root: Path,
    survey: str,
    *,
    norder: int | None = None,
    link_id_col: str | None = None,
    max_tiles: int | None = None,
    sample: int | None = None,
    seed: int = 0,
    quiet: bool = False,
    n_workers: int = 1,
    show_progress: bool = False,
) -> LinkValidationReport:
    """Cross-check catalog ``_spectrum_index`` / ``_spectrum_npix`` against Zarr tiles.

    Catalog and spectrum may use different HEALPix orders; linkage is validated
    via ``_spectrum_npix`` (new format) or by same-Npix pairing (legacy).

    Parameters
    ----------
    n_workers:
        Number of parallel worker processes for Zarr tile validation.
        Default 1 (serial).  Use ``os.cpu_count() - 1`` for maximum
        throughput on large lakes.  Each worker validates disjoint tiles so
        there is no locking.
    show_progress:
        Show a tqdm progress bar while validating Zarr tiles (and a second bar
        while scanning catalog tiles when ``_spectrum_npix`` is present).
    """
    lake_root = Path(lake_root)
    catalog_root = lake_root / "catalogs" / survey
    spectra_root = lake_root / "spectra" / survey
    rep = LinkValidationReport()

    if not catalog_root.is_dir():
        rep.errors.append(f"Catalog not found: {catalog_root}")
        return rep
    if not spectra_root.is_dir():
        rep.errors.append(f"Spectrum store not found: {spectra_root}")
        return rep

    cat_order, _spec_order = _resolve_norders(catalog_root, spectra_root, norder, rep)

    all_parquet = sorted(catalog_root.rglob("Npix=*.parquet"))
    schema_names: list[str] | None = None
    if all_parquet:
        schema_names = pq.read_schema(str(all_parquet[0])).names
    try:
        sid_col = resolve_link_id_column(
            catalog_root,
            schema_names=schema_names,
            override=link_id_col,
        )
    except Exception as exc:
        rep.errors.append(f"Cannot resolve join ID column: {exc}")
        return rep

    if "_spectrum_index" not in (schema_names or []):
        rep.errors.append(
            f"Catalog {survey!r} has no _spectrum_index column "
            "(re-ingest catalog or run dl-rebuild-catalog-indices)"
        )
        return rep

    has_npix_col = "_spectrum_npix" in (schema_names or [])

    tiles = list(_iter_spectrum_zarr_tiles(spectra_root))
    if not tiles:
        rep.warnings.append(f"No Npix=*.zarr tiles under {spectra_root}")
        return rep

    if max_tiles is not None:
        tiles = tiles[: max(0, max_tiles)]

    # Build catalog link index (single pass, fast path for _spectrum_npix).
    cat_index: CatalogLinkIndex | None = None
    if has_npix_col:
        cat_index = build_catalog_link_index(
            all_parquet,
            sid_col,
            cat_order,
            has_npix_col,
            show_progress=show_progress,
        )
        # n_null_source_id is counted once globally; workers don't re-count it.
        rep.stats.n_null_source_id = cat_index.n_null_source_id

    if n_workers > 1:
        _run_parallel(
            tiles=tiles,
            catalog_root=catalog_root,
            cat_index=cat_index,
            cat_order=cat_order,
            sid_col=sid_col,
            has_npix_col=has_npix_col,
            sample=sample,
            seed=seed,
            quiet=quiet,
            n_workers=n_workers,
            show_progress=show_progress,
            rep=rep,
        )
    else:
        _run_serial(
            tiles=tiles,
            catalog_root=catalog_root,
            all_parquet=all_parquet,
            cat_index=cat_index,
            cat_order=cat_order,
            sid_col=sid_col,
            has_npix_col=has_npix_col,
            sample=sample,
            seed=seed,
            quiet=quiet,
            show_progress=show_progress,
            rep=rep,
        )

    return rep


def _run_serial(
    *,
    tiles: list[Path],
    catalog_root: Path,
    all_parquet: list[Path],
    cat_index: CatalogLinkIndex | None,
    cat_order: int,
    sid_col: str,
    has_npix_col: bool,
    sample: int | None,
    seed: int,
    quiet: bool,
    show_progress: bool,
    rep: LinkValidationReport,
) -> None:
    tiles_iter: Any = tiles
    if show_progress:
        try:
            from tqdm.auto import tqdm
            tiles_iter = tqdm(tiles, unit="tile", desc="validating")
        except ImportError:
            pass

    rng = random.Random(seed) if sample else None
    for zarr_tile in tiles_iter:
        npix = _npix_from_tile_name(zarr_tile.name)
        validate_tile_link(
            zarr_tile=zarr_tile,
            zarr_npix=npix,
            cat_order=cat_order,
            cat_tiles=all_parquet,
            sid_col=sid_col,
            rep=rep,
            has_npix_col=has_npix_col,
            sample=sample,
            rng=rng,
            quiet=quiet,
            cat_index=cat_index,
        )


def _run_parallel(
    *,
    tiles: list[Path],
    catalog_root: Path,
    cat_index: CatalogLinkIndex | None,
    cat_order: int,
    sid_col: str,
    has_npix_col: bool,
    sample: int | None,
    seed: int,
    quiet: bool,
    n_workers: int,
    show_progress: bool,
    rep: LinkValidationReport,
) -> None:
    configs = []
    for zarr_tile in tiles:
        npix = _npix_from_tile_name(zarr_tile.name)
        legacy_cat = ""
        if not has_npix_col:
            p = _catalog_tile_path(catalog_root, cat_order, npix)
            legacy_cat = str(p) if p.is_file() else ""
        configs.append(
            _ValidationTileConfig(
                zarr_tile=str(zarr_tile),
                zarr_npix=npix,
                cat_order=cat_order,
                sid_col=sid_col,
                has_npix_col=has_npix_col,
                sample=sample,
                seed=seed,
                quiet=quiet,
                legacy_cat_tile=legacy_cat,
            )
        )

    init_fn = _init_worker if cat_index is not None else None
    init_args: tuple = (cat_index,) if cat_index is not None else ()

    pbar: Any = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(configs), unit="tile", desc="validating")
        except ImportError:
            pass

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_fn,
        initargs=init_args,
    ) as pool:
        futures = {pool.submit(_validation_tile_worker, cfg): cfg for cfg in configs}
        for fut in as_completed(futures):
            if pbar is not None:
                pbar.update(1)
            cfg = futures[fut]
            try:
                result = fut.result()
                _merge_tile_result(rep, result)
            except Exception as exc:
                rep.errors.append(f"Worker error for Npix={cfg.zarr_npix}: {exc}")

    if pbar is not None:
        pbar.close()


# ---------------------------------------------------------------------------
# Survey discovery
# ---------------------------------------------------------------------------

def discover_surveys_for_spectra_link_validation(lake_root: Path | str) -> list[str]:
    """Survey names that have both a catalog tree and a spectrum store."""
    from data_lake.ingest.validate_cli import discover_catalog_spectra_link_surveys

    return discover_catalog_spectra_link_surveys(lake_root)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _report_validation(
    rep: LinkValidationReport,
    survey_name: str,
    *,
    strict: bool,
    quiet: bool = False,
) -> bool:
    """Print one survey's report; return whether validation passed."""
    st = rep.stats
    for msg in rep.errors:
        click.echo(f"ERROR:   {msg}", err=True)
    if not quiet:
        for msg in rep.warnings:
            click.echo(f"WARNING: {msg}", err=True)

    click.echo(
        f"Tiles checked: {st.n_tiles_checked}  linked rows verified: {st.n_linked}  "
        f"stale index: {st.n_stale_index}  wrong id: {st.n_wrong_id}  "
        f"wrong healpix: {st.n_wrong_healpix}  orphan zarr: {st.n_orphan_zarr}  "
        f"unpatched catalog: {st.n_unpatched_catalog}  "
        f"missing catalog tile: {st.n_missing_catalog_tile}  "
        f"null source_id: {st.n_null_source_id}  "
        f"null source_id linked: {st.n_null_source_id_linked}"
    )

    if st.n_unpatched_catalog > 0:
        click.echo(
            f"Hint: run dl-rebuild-catalog-indices --survey {survey_name!r} "
            f"--kind spectrum (uses hats_order from catalog_info.json unless --norder is set)"
        )

    if rep.ok(strict=strict):
        n_warn = len(rep.warnings)
        if quiet and (st.n_orphan_zarr or st.n_unpatched_catalog):
            n_warn = st.n_orphan_zarr + st.n_unpatched_catalog
        if n_warn and not strict:
            click.echo(
                f"OK (with {n_warn} warning(s)): "
                f"catalog ↔ spectra link for {survey_name!r}."
            )
        else:
            click.echo(f"OK: catalog ↔ spectra link for {survey_name!r}.")
        return True

    click.echo(f"Link validation failed for {survey_name!r}.", err=True)
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    from ..cli_utils import (
        config_option,
        configure_cli_logging,
        load_optional_config,
        require_output_root,
    )
    from .validate_cli import (
        discover_catalog_spectra_link_surveys,
        echo_multi_survey_footer,
        echo_survey_banner,
        resolve_validation_survey_names,
        validation_survey_options,
    )

    @click.command("dl-validate-catalog-spectra-link")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @validation_survey_options
    @click.option("--norder", type=int, default=None, help="HEALPix order (default: info JSON).")
    @click.option(
        "--link-id-col",
        default=None,
        help="Override catalog join column (default: resolve from catalog_info.json).",
    )
    @click.option(
        "--max-tiles",
        type=int,
        default=None,
        help="Validate only the first N Zarr tiles (sorted path order).",
    )
    @click.option(
        "--sample",
        type=int,
        default=None,
        help="Check at most N random catalog-linked rows per tile (smoke test).",
    )
    @click.option("--seed", type=int, default=0, show_default=True, help="RNG seed for --sample.")
    @click.option(
        "--strict",
        is_flag=True,
        help="Treat warnings (orphan Zarr, unpatched catalog) as errors.",
    )
    @click.option(
        "-q",
        "--quiet",
        is_flag=True,
        default=False,
        help="Summary only: do not print per-row WARNING lines (use for surveys "
             "with many orphan spectra). Counts still appear in the stats line.",
    )
    @click.option(
        "--n-workers",
        default=1,
        show_default=True,
        type=int,
        help="Parallel worker processes for Zarr tile validation. "
             "Default 1 (serial). Use cpu_count()-1 for large surveys.",
    )
    @click.option(
        "--progress",
        "show_progress",
        is_flag=True,
        default=False,
        help="Show tqdm progress bars (catalog index scan + Zarr tile validation).",
    )
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        surveys: tuple[str, ...],
        validate_all: bool,
        norder: int | None,
        link_id_col: str | None,
        max_tiles: int | None,
        sample: int | None,
        seed: int,
        strict: bool,
        quiet: bool,
        n_workers: int,
        show_progress: bool,
    ) -> None:
        """Verify catalog _spectrum_index matches spectrum Zarr _source_id tiles."""
        import logging as _logging

        configure_cli_logging(
            level=_logging.WARNING if quiet else _logging.INFO,
            quiet=quiet,
        )
        cfg = load_optional_config(config_path)
        lake = require_output_root(output_root, cfg, kind="spectra")

        if n_workers < 1:
            raise click.ClickException("--n-workers must be >= 1")

        names = resolve_validation_survey_names(
            surveys=surveys,
            validate_all=validate_all,
            discovered=discover_catalog_spectra_link_surveys(lake),
            empty_message=(
                "No surveys with both catalogs/ and spectra/ found under the lake root."
            ),
        )

        all_ok = True
        for i, survey_name in enumerate(names):
            echo_survey_banner(i, survey_name, total=len(names))

            rep = run_validation(
                lake,
                survey_name,
                norder=norder,
                link_id_col=link_id_col,
                max_tiles=max_tiles,
                sample=sample,
                seed=seed,
                quiet=quiet,
                n_workers=n_workers,
                show_progress=show_progress,
            )
            if not _report_validation(rep, survey_name, strict=strict, quiet=quiet):
                all_ok = False

        echo_multi_survey_footer(
            all_ok=all_ok,
            n_surveys=len(names),
            ok_message=f"OK: catalog ↔ spectra link for all {len(names)} survey(s).",
        )

        sys.exit(0 if all_ok else 1)

except ImportError:
    cli = None  # type: ignore[assignment]
