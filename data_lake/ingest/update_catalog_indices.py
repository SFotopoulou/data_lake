"""
update_catalog_indices – rewrite only the affected tile Parquet files with
updated ``_cutout_index`` / ``_cutout_npix`` or ``_spectrum_index`` /
``_spectrum_npix`` values after a Zarr ingest.

The ingest functions return a ``{source_id: (zarr_npix, local_index)}``
mapping.  Pass that mapping to ``update_index_column`` so the relevant
Parquet tiles are patched in-place.

The modality-specific npix column (``_spectrum_npix``, ``_cutout_npix``)
allows catalog and spectra/cutout layers to use **different HEALPix orders
and different coordinates** while still joining on ``_source_id``.

Only tiles that actually contain matched source_ids are rewritten – every
other tile is left untouched, making this efficient even for large catalogs.

The ID column name is read automatically from ``catalog_info.json``
(``link_id_mode``), so catalogs using native columns such as ``TARGETID``
are handled correctly without any manual configuration.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import (
    LAKE_JOIN_ID_COLUMN,
    _ZSTD_LEVEL,
    _regenerate_metadata_from_all_tiles,
    healpix_dir,
    normalize_object_id,
    resolve_link_id_column,
    warn_if_id_column_unsafe,
)
from data_lake.ingest.zarr_ids import zarr_join_array

log = logging.getLogger(__name__)

IndexKind = Literal["cutout", "spectrum"]
_INDEX_COL: dict[str, str] = {
    "cutout":   "_cutout_index",
    "spectrum": "_spectrum_index",
}
_NPIX_COL: dict[str, str] = {
    "cutout":   "_cutout_npix",
    "spectrum": "_spectrum_npix",
}
_KIND_DIR: dict[str, str] = {"spectrum": "spectra", "cutout": "cutouts"}
_MODALITY_INFO: dict[str, str] = {"spectrum": "spectrum_info.json", "cutout": "cutout_info.json"}


def _resolve_catalog_norder(
    catalog_root: Path,
    modality_root: Path | None = None,
    *,
    kind: IndexKind = "spectrum",
    override: int | None = None,
) -> int:
    """Return HEALPix order for catalog tile paths (metadata, then default 5)."""
    if override is not None:
        return override

    cat_order: int | None = None
    info_path = catalog_root / "catalog_info.json"
    if info_path.is_file():
        with open(info_path) as fh:
            info = json.load(fh)
        if "hats_order" in info:
            cat_order = int(info["hats_order"])

    if cat_order is None:
        log.warning(
            "catalog_info.json missing hats_order under %s; defaulting to norder=5",
            catalog_root,
        )
        return 5

    if modality_root is not None:
        mod_info_path = modality_root / _MODALITY_INFO.get(kind, "spectrum_info.json")
        if mod_info_path.is_file():
            with open(mod_info_path) as fh:
                mod_info = json.load(fh)
            mod_order = mod_info.get("hats_order")
            if mod_order is not None and int(mod_order) != cat_order:
                log.warning(
                    "hats_order mismatch: catalog %d vs %s %d (using catalog order)",
                    cat_order,
                    mod_info_path.name,
                    int(mod_order),
                )

    return cat_order


def _catalog_tile_for_npix(
    catalog_root: Path,
    norder: int,
    npix: int,
) -> Path | None:
    """Resolve catalog Parquet path for one HEALPix pixel (with rglob fallback)."""
    primary = catalog_root / healpix_dir(norder, npix) / f"Npix={npix}.parquet"
    if primary.is_file():
        return primary
    fallback = next(catalog_root.rglob(f"Npix={npix}.parquet"), None)
    return fallback


def _read_id_column_as_int64(col: "pa.Array") -> "tuple[np.ndarray, np.ndarray]":
    """Return (ids_int64, valid_mask) for a source-ID column.

    Fast path for integer-typed columns (avoids per-element Python calls to
    ``normalize_object_id``).  Falls back to the generic scalar normaliser for
    string / bytes columns.
    """
    if pa.types.is_integer(col.type):
        valid_mask: np.ndarray = col.is_valid().to_numpy(zero_copy_ok=False)
        ids: np.ndarray = col.cast(pa.int64()).fill_null(0).to_numpy(zero_copy_ok=False)
        return ids, valid_mask

    # Fallback: generic per-element path (string IDs etc.)
    py_list = col.to_pylist()
    ids_out = np.zeros(len(py_list), dtype=np.int64)
    valid_out = np.zeros(len(py_list), dtype=bool)
    for i, x in enumerate(py_list):
        if x is not None:
            ids_out[i] = normalize_object_id(x)
            valid_out[i] = True
    return ids_out, valid_out


def _patch_catalog_parquet_file(
    tile_file: Path,
    source_id_to_tile: dict[int, tuple[int, int]],
    *,
    sid_col: str,
    index_col: str,
    npix_col: str,
    reset_unmatched: bool = False,
) -> bool:
    """Patch one catalog Parquet tile; return True if the file was rewritten.

    ``source_id_to_tile`` maps ``source_id -> (zarr_npix, local_index)``.

    Both ``index_col`` (local Zarr row) and ``npix_col`` (which Zarr tile)
    are updated so that readers can open the correct tile at any HEALPix order.

    When *reset_unmatched* is True (full Zarr-tile rebuild), rows whose
    ``sid_col`` value is absent from *source_id_to_tile* get ``-1`` for both
    columns.

    Two-phase I/O: reads only [sid_col, index_col, npix_col] to decide whether
    a write is needed, then reads the full table only when a change is required.
    For large catalogs (many columns) this avoids deserialising every column on
    the common "tile already correct" path, which can be >90 % of tiles on
    re-runs.
    """
    schema = pq.read_schema(str(tile_file))
    if sid_col not in schema.names:
        log.warning(
            "Source-ID column %r not found in %s. "
            "Check survey name, catalog ingest, or pass --link-id-col.",
            sid_col, tile_file.name,
        )
        return False

    # --- Phase 1: lightweight column read (3 cols instead of all columns) ---
    probe_cols = [sid_col]
    if index_col in schema.names:
        probe_cols.append(index_col)
    if npix_col in schema.names:
        probe_cols.append(npix_col)

    probe = pq.read_table(str(tile_file), columns=probe_cols)

    warn_if_id_column_unsafe(
        sid_col, probe.schema.field(sid_col).type, context="catalog",
    )

    ids, valid_mask = _read_id_column_as_int64(probe.column(sid_col))
    # Build tile_ids list (None for nulls) – needed for id_to_row and reset loop.
    tile_ids: list[int | None] = [
        int(ids[i]) if valid_mask[i] else None for i in range(len(ids))
    ]

    source_id_set = set(source_id_to_tile.keys())
    matches = [sid for sid in tile_ids if sid is not None and sid in source_id_set]
    if not matches and not reset_unmatched:
        return False

    # Compute what the new arrays should look like (still using probe columns).
    n_rows = len(probe)
    if index_col in schema.names:
        idx_arr = probe.column(index_col).cast(pa.int64()).fill_null(-1).to_numpy(
            zero_copy_ok=False
        ).copy()
    else:
        idx_arr = np.full(n_rows, -1, dtype=np.int64)
        log.debug("Adding missing column %s to %s", index_col, tile_file.name)

    if npix_col in schema.names:
        npix_arr = probe.column(npix_col).cast(pa.int64()).fill_null(-1).to_numpy(
            zero_copy_ok=False
        ).copy()
    else:
        npix_arr = np.full(n_rows, -1, dtype=np.int64)
        log.debug("Adding missing column %s to %s", npix_col, tile_file.name)

    id_to_row: dict[int, list[int]] = {}
    for row_i, sid in enumerate(tile_ids):
        if sid is None:
            continue
        id_to_row.setdefault(sid, []).append(row_i)

    for sid in matches:
        zarr_npix, local_idx = source_id_to_tile[sid]
        for row_i in id_to_row[sid]:
            idx_arr[row_i] = local_idx
            npix_arr[row_i] = zarr_npix

    if reset_unmatched:
        for row_i, sid in enumerate(tile_ids):
            if sid is None or sid not in source_id_set:
                idx_arr[row_i] = -1
                npix_arr[row_i] = -1

    # --- Early exit: skip the expensive full-table read + write when the
    # existing on-disk values are already identical to what we'd write. ---
    cols_already_present = (
        index_col in schema.names and npix_col in schema.names
    )
    if cols_already_present:
        old_idx = probe.column(index_col).cast(pa.int64()).fill_null(-1).to_numpy(
            zero_copy_ok=False
        )
        old_npix = probe.column(npix_col).cast(pa.int64()).fill_null(-1).to_numpy(
            zero_copy_ok=False
        )
        if np.array_equal(idx_arr, old_idx) and np.array_equal(npix_arr, old_npix):
            return False  # tile is already up-to-date

    # --- Phase 2: read full table and rewrite with updated columns ---
    table = pq.read_table(str(tile_file))

    for col_name, arr in ((index_col, idx_arr), (npix_col, npix_arr)):
        if col_name in table.schema.names:
            col_pos = table.schema.get_field_index(col_name)
            table = table.set_column(col_pos, col_name, pa.array(arr, type=pa.int64()))
        else:
            table = table.append_column(col_name, pa.array(arr, type=pa.int64()))

    pq.write_table(
        table,
        str(tile_file),
        compression="zstd",
        compression_level=_ZSTD_LEVEL,
        write_statistics=True,
        use_dictionary=True,
    )
    log.debug("Updated %d rows in %s", len(matches), tile_file.name)
    return True


def _npix_from_tile_name(name: str) -> int:
    return int(name.split("=")[-1].split(".")[0])


# Process-pool worker state for parallel catalog patching.
_WORKER_INDEX_MAP: dict[int, tuple[int, int]] | None = None
_WORKER_PATCH_COLS: tuple[str, str, str] | None = None  # sid_col, index_col, npix_col


@dataclass
class _CatalogPatchConfig:
    """Pickle-friendly per-tile config for parallel catalog patching."""

    tile_file: str


def _init_catalog_patch_worker(
    index_map: dict[int, tuple[int, int]],
    sid_col: str,
    index_col: str,
    npix_col: str,
) -> None:
    global _WORKER_INDEX_MAP, _WORKER_PATCH_COLS
    _WORKER_INDEX_MAP = index_map
    _WORKER_PATCH_COLS = (sid_col, index_col, npix_col)


def _catalog_patch_worker(config: _CatalogPatchConfig) -> bool:
    if _WORKER_INDEX_MAP is None or _WORKER_PATCH_COLS is None:
        raise RuntimeError("catalog patch worker not initialized")
    sid_col, index_col, npix_col = _WORKER_PATCH_COLS
    return _patch_catalog_parquet_file(
        Path(config.tile_file),
        _WORKER_INDEX_MAP,
        sid_col=sid_col,
        index_col=index_col,
        npix_col=npix_col,
        reset_unmatched=True,
    )


def _patch_all_catalog_tiles(
    all_parquet: list[Path],
    index_map: dict[int, tuple[int, int]],
    *,
    sid_col: str,
    index_col: str,
    npix_col: str,
    n_workers: int = 1,
    show_progress: bool = False,
) -> int:
    """Patch every catalog tile from a full Zarr index map (single catalog pass).

    Uses ``reset_unmatched=True`` so rows absent from *index_map* get ``-1`` for
    both index and npix columns — equivalent to the old reset-then-patch flow.
    """
    if not all_parquet:
        return 0

    if n_workers > 1:
        configs = [_CatalogPatchConfig(tile_file=str(p)) for p in all_parquet]
        pbar: Any = None
        if show_progress:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(total=len(configs), unit="tile", desc="patching catalog")
            except ImportError:
                pass

        n_modified = 0
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_catalog_patch_worker,
            initargs=(index_map, sid_col, index_col, npix_col),
        ) as pool:
            futures = {pool.submit(_catalog_patch_worker, cfg): cfg for cfg in configs}
            for fut in as_completed(futures):
                if pbar is not None:
                    pbar.update(1)
                try:
                    if fut.result():
                        n_modified += 1
                except Exception as exc:
                    cfg = futures[fut]
                    log.warning("Failed to patch catalog tile %s: %s", cfg.tile_file, exc)
        if pbar is not None:
            pbar.close()
        return n_modified

    tiles_iter: Any = all_parquet
    if show_progress:
        try:
            from tqdm.auto import tqdm
            tiles_iter = tqdm(all_parquet, unit="tile", desc="patching catalog")
        except ImportError:
            pass

    n_modified = 0
    for cat_tile in tiles_iter:
        if _patch_catalog_parquet_file(
            cat_tile,
            index_map,
            sid_col=sid_col,
            index_col=index_col,
            npix_col=npix_col,
            reset_unmatched=True,
        ):
            n_modified += 1
    return n_modified


def update_index_column_from_zarr_tiles(
    lake_root: Path | str,
    survey_name: str,
    kind: IndexKind = "spectrum",
    norder: int | None = None,
    link_id_col: str | None = None,
    *,
    n_workers: int = 1,
    show_progress: bool = False,
) -> int:
    """Rebuild catalog index + npix columns from on-disk Zarr tiles.

    Scans every ``Npix=*.zarr`` once to build ``{source_id: (zarr_npix,
    local_index)}``, then patches each catalog Parquet tile in a single pass.
    Rows with no matching Zarr entry are reset to ``-1``.

    Complexity is O(n_zarr_tiles + n_catalog_tiles) instead of the previous
    O(n_zarr × n_catalog) pattern (one full catalog scan per Zarr tile).

    For surveys with millions of spectra the in-memory map is typically
    100–200 MB (one dict entry per Zarr row); well within RAM on modern hosts.
    """
    lake_root = Path(lake_root)
    catalog_root = lake_root / "catalogs" / survey_name
    if not catalog_root.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_root}")

    zarr_root = lake_root / _KIND_DIR[kind] / survey_name
    if not zarr_root.exists():
        raise FileNotFoundError(f"No {kind} tiles found at {zarr_root}")

    index_col = _INDEX_COL[kind]
    npix_col = _NPIX_COL[kind]
    schema_names: list[str] | None = None
    all_parquet = sorted(catalog_root.rglob("Npix=*.parquet"))
    if all_parquet:
        schema_names = pq.read_schema(str(all_parquet[0])).names
    sid_col = resolve_link_id_column(
        catalog_root, schema_names=schema_names, override=link_id_col,
    )
    resolved_cat_norder = _resolve_catalog_norder(
        catalog_root, zarr_root, kind=kind, override=norder,
    )
    log.info(
        "Rebuilding %s/%s using catalog hats_order=%d (id_col=%r)",
        index_col, npix_col, resolved_cat_norder, sid_col,
    )

    index_map = build_index_map_from_zarr(
        lake_root,
        survey_name,
        kind=kind,
        show_progress=show_progress,
    )
    if not index_map:
        log.warning(
            "No source IDs found in %s tiles for survey=%r; catalog indices unchanged",
            kind, survey_name,
        )
        return 0

    n_modified = _patch_all_catalog_tiles(
        all_parquet,
        index_map,
        sid_col=sid_col,
        index_col=index_col,
        npix_col=npix_col,
        n_workers=n_workers,
        show_progress=show_progress,
    )

    if index_map and n_modified == 0:
        log.warning(
            "No catalog tiles were modified for survey=%r (kind=%r, %d Zarr source IDs). "
            "Possible causes: survey name mismatch, no catalog ingested yet, "
            "or wrong source-ID column (resolved to %r).",
            survey_name, kind, len(index_map), sid_col,
        )

    log.info(
        "Patched %s in %d catalog tile(s) from %d Zarr entries for survey=%r (id_col=%r)",
        index_col, n_modified, len(index_map), survey_name, sid_col,
    )
    if n_modified > 0:
        _regenerate_metadata_from_all_tiles(catalog_root)
    return n_modified


def update_index_column(
    lake_root: Path | str,
    survey_name: str,
    source_id_to_index: dict[int, tuple[int, int]] | dict[int, int],
    kind: IndexKind = "spectrum",
    norder: int = 5,
    link_id_col: str | None = None,
) -> int:
    """
    Update modality index + npix columns in the affected Parquet tiles.

    ``source_id_to_index`` is the mapping returned by
    ``ingest_spectra_from_fits`` / ``ingest_cutouts_from_fits``.  Each value
    is a ``(zarr_npix, local_index)`` 2-tuple; the legacy plain ``int`` form
    (``local_index`` only) is still accepted for backward compatibility and is
    treated as ``(catalog_Npix, local_index)`` with a deprecation warning.

    Both ``_spectrum_index`` / ``_cutout_index`` (local Zarr row) and
    ``_spectrum_npix`` / ``_cutout_npix`` (which Zarr tile) are updated.

    The ID column name is resolved automatically from ``catalog_info.json``:
    catalogs ingested with ``--link-id-col TARGETID`` (or any other native
    column) are handled without extra configuration.

    Parameters
    ----------
    lake_root:
        Data lake root.
    survey_name:
        Survey whose catalog tiles should be updated.
    source_id_to_index:
        ``{source_id: (zarr_npix, local_index)}`` mapping.
    kind:
        Which columns to update: ``"cutout"`` or ``"spectrum"``.
    norder:
        Kept for backward compatibility; ignored when map values are tuples.
    link_id_col:
        Force the catalog ID column.  When omitted, resolved from
        ``catalog_info.json``.

    Returns
    -------
    Number of Parquet tile files that were actually modified.
    """
    lake_root = Path(lake_root)
    catalog_root = lake_root / "catalogs" / survey_name
    if not catalog_root.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_root}")

    index_col = _INDEX_COL[kind]
    npix_col = _NPIX_COL[kind]

    # Discover affected tile files by scanning the Parquet tree
    all_parquet = list(catalog_root.rglob("Npix=*.parquet"))
    schema_names: list[str] | None = None
    if all_parquet:
        schema_names = pq.read_schema(str(all_parquet[0])).names
    sid_col = resolve_link_id_column(
        catalog_root, schema_names=schema_names, override=link_id_col,
    )
    if not all_parquet:
        log.warning("No Parquet tiles found in %s", catalog_root)
        return 0

    # Normalise keys and coerce legacy int values to (npix, local_index) tuples.
    normalised: dict[int, tuple[int, int]] = {}
    _legacy_warned = False
    for k, v in source_id_to_index.items():
        nk = normalize_object_id(k)
        if isinstance(v, (list, tuple)):
            normalised[nk] = (int(v[0]), int(v[1]))
        else:
            if not _legacy_warned:
                log.warning(
                    "update_index_column received plain-int index values (legacy format). "
                    "Ingest should return (zarr_npix, local_index) tuples; "
                    "_spectrum_npix / _cutout_npix will not be set correctly.",
                )
                _legacy_warned = True
            normalised[nk] = (-1, int(v))

    # First pass: patch matched tiles (adds npix_col when absent on patched tiles).
    n_modified = 0
    patched: set[Path] = set()
    for tile_file in all_parquet:
        if _patch_catalog_parquet_file(
            tile_file,
            normalised,
            sid_col=sid_col,
            index_col=index_col,
            npix_col=npix_col,
        ):
            n_modified += 1
            patched.add(tile_file)

    # Second pass: add npix_col=-1 to any tiles that were NOT patched but are
    # missing the column, so all tiles share the same schema.
    for tile_file in all_parquet:
        if tile_file in patched:
            continue
        schema = pq.read_schema(str(tile_file))
        if npix_col not in schema.names:
            table = pq.ParquetFile(str(tile_file)).read()
            table = table.append_column(
                npix_col,
                pa.array(np.full(len(table), -1, dtype=np.int64), type=pa.int64()),
            )
            pq.write_table(
                table,
                str(tile_file),
                compression="zstd",
                compression_level=_ZSTD_LEVEL,
                write_statistics=True,
                use_dictionary=True,
            )
            log.debug("Added placeholder %s to %s", npix_col, tile_file.name)

    log.info(
        "Updated %s/%s in %d tile file(s) for survey=%s (%d sources, id_col=%r)",
        index_col, npix_col, n_modified, survey_name, len(normalised), sid_col,
    )

    if normalised and n_modified == 0:
        log.warning(
            "No catalog tiles were modified for survey=%r (kind=%r, %d source IDs given). "
            "Possible causes: survey name mismatch, no catalog ingested yet, "
            "or wrong source-ID column (resolved to %r from catalog_info.json).",
            survey_name, kind, len(normalised), sid_col,
        )

    # Regenerate aggregate _metadata
    if n_modified > 0:
        try:
            _regenerate_metadata_from_all_tiles(catalog_root)
            log.debug("Regenerated _metadata for %s", catalog_root.name)
        except Exception as exc:
            log.warning("Could not regenerate _metadata: %s", exc)

    return n_modified


def build_index_map_from_zarr(
    lake_root: Path | str,
    survey_name: str,
    kind: IndexKind = "spectrum",
    *,
    show_progress: bool = False,
) -> dict[int, tuple[int, int]]:
    """Scan existing Zarr tiles and return ``{source_id: (zarr_npix, local_index)}``.

    Reads the ``source_id`` array from every tile under
    ``<lake_root>/<kind>s/<survey_name>/`` without touching FITS files, so
    this is safe to run against a partially-ingested or resumed deployment.

    Parameters
    ----------
    lake_root:
        Data lake root.
    survey_name:
        Survey whose Zarr tiles should be scanned.
    kind:
        ``"spectrum"`` scans ``spectra/<survey>/`` tiles;
        ``"cutout"`` scans ``cutouts/<survey>/`` tiles.
    """
    import zarr

    lake_root = Path(lake_root)
    zarr_root = lake_root / _KIND_DIR[kind] / survey_name
    if not zarr_root.exists():
        raise FileNotFoundError(f"No {kind} tiles found at {zarr_root}")

    index_map: dict[int, tuple[int, int]] = {}
    tile_paths = sorted(zarr_root.rglob("Npix=*.zarr"))
    if not tile_paths:
        log.warning("No Zarr tiles found under %s", zarr_root)
        return index_map

    tiles_iter: Any = tile_paths
    if show_progress:
        try:
            from tqdm.auto import tqdm
            tiles_iter = tqdm(tile_paths, unit="tile", desc=f"scanning {kind}")
        except ImportError:
            pass

    for tile_path in tiles_iter:
        try:
            root = zarr.open_group(
                store=zarr.storage.LocalStore(str(tile_path)),
                mode="r",
                zarr_format=3,
            )
            if LAKE_JOIN_ID_COLUMN not in root:
                continue
            zarr_npix = _npix_from_tile_name(tile_path.name)
            sids = zarr_join_array(root)[:]
            for local_i, sid in enumerate(sids.tolist()):
                index_map[normalize_object_id(sid)] = (zarr_npix, local_i)
        except Exception as exc:
            log.warning("Could not read tile %s: %s", tile_path.name, exc)

    log.info(
        "Built index_map with %d entries from %d %s tile(s) for survey=%r",
        len(index_map), len(tile_paths), kind, survey_name,
    )
    return index_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    from ..cli_utils import (
        config_option,
        configure_cli_logging,
        ingest_token_option,
        load_optional_config,
        logging_options,
        require_ingest_permission,
        require_output_root,
        resolve_log_level,
        validate_quiet_verbose,
    )

    @click.command("dl-rebuild-catalog-indices")
    @click.option("--survey", "survey_name", required=True,
                  help="Survey name (must match both catalog and spectra/cutouts directories).")
    @click.option(
        "--kind",
        type=click.Choice(["spectrum", "cutout"]),
        default="spectrum",
        show_default=True,
        help="Which index column to rebuild.",
    )
    @click.option(
        "--norder",
        default=None,
        type=int,
        help="HEALPix order for catalog tile paths (default: hats_order from catalog_info.json).",
    )
    @click.option(
        "--link-id-col",
        default=None,
        help="Override catalog ID column (e.g. TARGETID). "
             "Auto-detected from Parquet schema when omitted.",
    )
    @click.option(
        "--n-workers",
        default=1,
        show_default=True,
        type=int,
        help="Parallel worker processes for catalog tile patching. "
             "Default 1 (serial). Use cpu_count()-1 for large surveys.",
    )
    @click.option(
        "--progress/--no-progress",
        "show_progress",
        default=True,
        show_default=True,
        help="Show tqdm progress bars (Zarr scan + catalog patch). On by default.",
    )
    @click.argument("lake_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @ingest_token_option
    @logging_options
    def cli_rebuild(
        survey_name: str,
        kind: str,
        norder: int | None,
        link_id_col: str | None,
        n_workers: int,
        show_progress: bool,
        lake_root: Path | None,
        config_path: Path | None,
        ingest_token: str | None,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Rebuild _spectrum_index / _cutout_index in catalog tiles from existing Zarr data.

        Scans the Zarr tiles already on disk for SURVEY and patches the matching
        Parquet catalog tiles — no FITS re-ingestion required.  Rows with no
        spectrum in the current Zarr tile get ``index=-1`` (stale indices cleared).
        Use this to repair a deployment where spectra or cutouts were ingested
        without catalog patching, or after Zarr tiles were replaced.
        """
        validate_quiet_verbose(quiet, verbose)
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(quiet=quiet, verbose=verbose,
                                    config_level=cfg.ingest.log_level if cfg else None),
            quiet=quiet,
        )
        require_ingest_permission(cfg, ingest_token)
        resolved_root = require_output_root(lake_root, cfg)

        if n_workers < 1:
            raise click.ClickException("--n-workers must be >= 1")

        # Warn early when the job is likely to be slow.
        import os as _os
        cpu_count = _os.cpu_count() or 1
        if n_workers == 1 and cpu_count > 2:
            click.echo(
                f"[hint] Running single-threaded. For large surveys consider "
                f"--n-workers {max(1, cpu_count - 1)} to parallelise tile patching.",
                err=True,
            )

        effective_progress = show_progress and not quiet
        click.echo(f"Scanning {kind} tiles for survey={survey_name!r} …")
        n_modified = update_index_column_from_zarr_tiles(
            lake_root=resolved_root,
            survey_name=survey_name,
            kind=kind,  # type: ignore[arg-type]
            norder=norder,
            link_id_col=link_id_col,
            n_workers=n_workers,
            show_progress=effective_progress,
        )
        click.echo(f"Done: patched _{kind}_index in {n_modified} catalog tile(s).")

except ImportError:
    cli_rebuild = None  # type: ignore[assignment,misc]


def main() -> None:
    """Console entry point for ``dl-rebuild-catalog-indices`` (always importable)."""
    if cli_rebuild is None:
        raise SystemExit(
            "dl-rebuild-catalog-indices requires 'click'. "
            "Install the package in this environment: pip install -e ."
        )
    cli_rebuild()


if __name__ == "__main__":
    main()
