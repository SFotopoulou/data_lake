"""
update_catalog_indices – rewrite only the affected tile Parquet files with
updated ``_cutout_index`` or ``_spectrum_index`` values after a Zarr ingest.

The ingest functions in ``fits_to_zarr`` and ``fits_to_spectra_zarr`` return a
``{source_id: local_index}`` mapping.  Pass that mapping to
``update_index_column`` so the relevant Parquet tiles are patched in-place.

Only tiles that actually contain matched source_ids are rewritten – every
other tile is left untouched, making this efficient even for large catalogs.

The ID column name is read automatically from ``catalog_info.json``
(``link_id_mode``), so catalogs using native columns such as ``TARGETID``
are handled correctly without any manual configuration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

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


def _patch_catalog_parquet_file(
    tile_file: Path,
    source_id_to_index: dict[int, int],
    *,
    sid_col: str,
    index_col: str,
) -> bool:
    """Patch one catalog Parquet tile; return True if the file was rewritten."""
    table = pq.ParquetFile(str(tile_file)).read()
    if sid_col not in table.schema.names:
        log.warning(
            "Source-ID column %r not found in %s. "
            "Check survey name, catalog ingest, or pass --link-id-col.",
            sid_col, tile_file.name,
        )
        return False

    warn_if_id_column_unsafe(
        sid_col, table.schema.field(sid_col).type, context="catalog",
    )

    tile_ids = [normalize_object_id(x) for x in table.column(sid_col).to_pylist()]
    source_id_set = set(source_id_to_index.keys())
    matches = [sid for sid in tile_ids if sid in source_id_set]
    if not matches:
        return False

    if index_col in table.schema.names:
        idx_arr = np.array(table.column(index_col).to_pylist(), dtype=np.int64)
    else:
        idx_arr = np.full(len(table), -1, dtype=np.int64)
        log.debug("Adding missing column %s to %s", index_col, tile_file.name)

    id_to_row: dict[int, list[int]] = {}
    for row_i, sid in enumerate(tile_ids):
        id_to_row.setdefault(sid, []).append(row_i)

    for sid in matches:
        for row_i in id_to_row[sid]:
            idx_arr[row_i] = source_id_to_index[sid]

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
    log.debug("Updated %d rows in %s", len(matches), tile_file.name)
    return True


def _npix_from_tile_name(name: str) -> int:
    return int(name.split("=")[-1].split(".")[0])


def update_index_column_from_zarr_tiles(
    lake_root: Path | str,
    survey_name: str,
    kind: IndexKind = "spectrum",
    norder: int | None = None,
    link_id_col: str | None = None,
) -> int:
    """Patch catalog indices one Zarr tile at a time (bounded memory).

    For large surveys (millions of spectra), building a single in-memory
    ``{source_id: index}`` map can exhaust RAM.  This walks each
    ``Npix=*.zarr``, patches the matching catalog Parquet tile, and discards
    the per-tile map before opening the next Zarr group.
    """
    import zarr

    lake_root = Path(lake_root)
    catalog_root = lake_root / "catalogs" / survey_name
    if not catalog_root.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_root}")

    zarr_root = lake_root / _KIND_DIR[kind] / survey_name
    if not zarr_root.exists():
        raise FileNotFoundError(f"No {kind} tiles found at {zarr_root}")

    index_col = _INDEX_COL[kind]
    schema_names: list[str] | None = None
    sample_parquet = next(catalog_root.rglob("Npix=*.parquet"), None)
    if sample_parquet is not None:
        schema_names = pq.read_schema(str(sample_parquet)).names
    sid_col = resolve_link_id_column(
        catalog_root, schema_names=schema_names, override=link_id_col,
    )
    resolved_norder = _resolve_catalog_norder(
        catalog_root, zarr_root, kind=kind, override=norder,
    )
    log.info(
        "Rebuilding %s using catalog hats_order=%d (id_col=%r)",
        index_col, resolved_norder, sid_col,
    )

    n_modified = 0
    tile_paths = sorted(zarr_root.rglob("Npix=*.zarr"))
    for tile_path in tile_paths:
        try:
            root = zarr.open_group(
                store=zarr.storage.LocalStore(str(tile_path)),
                mode="r",
                zarr_format=3,
            )
            if LAKE_JOIN_ID_COLUMN not in root:
                continue
            sids = np.asarray(zarr_join_array(root)[:], dtype=np.int64)
            if sids.size == 0:
                continue
            partial_map = {
                normalize_object_id(int(sid)): int(i)
                for i, sid in enumerate(sids.tolist())
            }
        except Exception as exc:
            log.warning("Could not read Zarr tile %s: %s", tile_path.name, exc)
            continue

        npix = _npix_from_tile_name(tile_path.name)
        catalog_tile = _catalog_tile_for_npix(catalog_root, resolved_norder, npix)
        if catalog_tile is None:
            log.warning(
                "Zarr %s: no catalog tile for Npix=%d at hats_order=%d under %s",
                tile_path.name,
                npix,
                resolved_norder,
                catalog_root,
            )
            continue
        if _patch_catalog_parquet_file(
            catalog_tile,
            partial_map,
            sid_col=sid_col,
            index_col=index_col,
        ):
            n_modified += 1
        else:
            log.warning(
                "Zarr %s: catalog tile %s has no matching %s values "
                "(%d Zarr row(s); id_col=%r)",
                tile_path.name,
                catalog_tile.name,
                sid_col,
                len(partial_map),
                sid_col,
            )

    log.info(
        "Patched %s from %d Zarr tile(s) for survey=%r (id_col=%r)",
        index_col, n_modified, survey_name, sid_col,
    )
    if n_modified > 0:
        _regenerate_metadata_from_all_tiles(catalog_root)
    return n_modified


def update_index_column(
    lake_root: Path | str,
    survey_name: str,
    source_id_to_index: dict[int, int],
    kind: IndexKind = "spectrum",
    norder: int = 5,
    link_id_col: str | None = None,
) -> int:
    """
    Update ``_cutout_index`` or ``_spectrum_index`` in the affected Parquet tiles.

    Reads only the tiles that contain at least one source_id from
    ``source_id_to_index``, modifies the relevant index column, and rewrites
    the tile file in-place with the same Zstd compression.

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
        Mapping returned by ``ingest_spectra_from_fits`` / ``ingest_cutouts_from_fits``.
    kind:
        Which index column to update: ``"cutout"`` or ``"spectrum"``.
    norder:
        HEALPix partitioning order.
    link_id_col:
        Force the catalog ID column (e.g. ``TARGETID``).  When omitted, resolved
        from ``catalog_info.json`` and validated against the Parquet schema.

    Returns
    -------
    Number of Parquet tile files that were actually modified.
    """
    lake_root = Path(lake_root)
    catalog_root = lake_root / "catalogs" / survey_name
    if not catalog_root.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_root}")

    index_col = _INDEX_COL[kind]

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

    source_id_to_index = {
        normalize_object_id(k): int(v) for k, v in source_id_to_index.items()
    }
    n_modified = 0
    for tile_file in all_parquet:
        if _patch_catalog_parquet_file(
            tile_file,
            source_id_to_index,
            sid_col=sid_col,
            index_col=index_col,
        ):
            n_modified += 1

    log.info(
        "Updated %s in %d tile file(s) for survey=%s (%d sources, id_col=%r)",
        index_col, n_modified, survey_name, len(source_id_to_index), sid_col,
    )

    if source_id_to_index and n_modified == 0:
        log.warning(
            "No catalog tiles were modified for survey=%r (kind=%r, %d source IDs given). "
            "Possible causes: survey name mismatch, no catalog ingested yet, "
            "or wrong source-ID column (resolved to %r from catalog_info.json).",
            survey_name, kind, len(source_id_to_index), sid_col,
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
) -> dict[int, int]:
    """Scan existing Zarr tiles and return a ``{source_id: local_index}`` map.

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

    index_map: dict[int, int] = {}
    tile_paths = sorted(zarr_root.rglob("Npix=*.zarr"))
    if not tile_paths:
        log.warning("No Zarr tiles found under %s", zarr_root)
        return index_map

    for tile_path in tile_paths:
        try:
            root = zarr.open_group(
                store=zarr.storage.LocalStore(str(tile_path)),
                mode="r",
                zarr_format=3,
            )
            if LAKE_JOIN_ID_COLUMN not in root:
                continue
            sids = zarr_join_array(root)[:]
            for local_i, sid in enumerate(sids.tolist()):
                index_map[normalize_object_id(sid)] = local_i
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
        ingest_token_option,
        load_optional_config,
        require_ingest_permission,
        require_output_root,
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
    @click.argument("lake_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @ingest_token_option
    def cli_rebuild(
        survey_name: str,
        kind: str,
        norder: int | None,
        link_id_col: str | None,
        lake_root: Path | None,
        config_path: Path | None,
        ingest_token: str | None,
    ) -> None:
        """Rebuild _spectrum_index / _cutout_index in catalog tiles from existing Zarr data.

        Scans the Zarr tiles already on disk for SURVEY and patches the matching
        Parquet catalog tiles — no FITS re-ingestion required.  Use this to repair
        a deployment where spectra or cutouts were ingested without catalog patching.
        """
        import logging
        logging.basicConfig(level=logging.INFO)

        cfg = load_optional_config(config_path)
        require_ingest_permission(cfg, ingest_token)
        resolved_root = require_output_root(lake_root, cfg)

        click.echo(f"Scanning {kind} tiles for survey={survey_name!r} …")
        n_modified = update_index_column_from_zarr_tiles(
            lake_root=resolved_root,
            survey_name=survey_name,
            kind=kind,  # type: ignore[arg-type]
            norder=norder,
            link_id_col=link_id_col,
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
