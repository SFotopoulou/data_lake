"""
update_catalog_indices – rewrite only the affected tile Parquet files with
updated ``_cutout_index`` or ``_spectrum_index`` values after a Zarr ingest.

The ingest functions in ``fits_to_zarr`` and ``fits_to_spectra_zarr`` return a
``{source_id: local_index}`` mapping.  Pass that mapping to
``update_index_column`` so the relevant Parquet tiles are patched in-place.

Only tiles that actually contain matched source_ids are rewritten – every
other tile is left untouched, making this efficient even for large catalogs.

The ID column name is read automatically from ``catalog_info.json``
(``source_id_mode``), so catalogs using native columns such as ``TARGETID``
are handled correctly without any manual configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import (
    _ZSTD_LEVEL,
    healpix_dir,
    normalize_object_id,
    resolve_source_id_column,
    warn_if_id_column_unsafe,
)

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
    source_id_col: str | None = None,
) -> int:
    """
    Update ``_cutout_index`` or ``_spectrum_index`` in the affected Parquet tiles.

    Reads only the tiles that contain at least one source_id from
    ``source_id_to_index``, modifies the relevant index column, and rewrites
    the tile file in-place with the same Zstd compression.

    The ID column name is resolved automatically from ``catalog_info.json``:
    catalogs ingested with ``--source-id-col TARGETID`` (or any other native
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
    source_id_col:
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
    sid_col = resolve_source_id_column(
        catalog_root, schema_names=schema_names, override=source_id_col,
    )
    if not all_parquet:
        log.warning("No Parquet tiles found in %s", catalog_root)
        return 0

    # Normalize keys once so Zarr/catalog IDs match regardless of numpy scalar type.
    source_id_to_index = {
        normalize_object_id(k): int(v) for k, v in source_id_to_index.items()
    }
    source_id_set = set(source_id_to_index.keys())
    n_modified = 0
    _warned_missing_col = False
    _warned_id_dtype = False

    for tile_file in all_parquet:
        # Use ParquetFile.read() to avoid PyArrow's Hive partition discovery,
        # which would inject Norder/Dir path-based columns into the table and
        # corrupt the schema when the tile is rewritten.
        table = pq.ParquetFile(str(tile_file)).read()
        if sid_col not in table.schema.names:
            if not _warned_missing_col:
                log.warning(
                    "Source-ID column %r not found in %s (and possibly other tiles). "
                    "Check survey name, catalog ingest, or pass --source-id-col.",
                    sid_col, tile_file.name,
                )
                _warned_missing_col = True
            continue

        if not _warned_id_dtype:
            warn_if_id_column_unsafe(
                sid_col, table.schema.field(sid_col).type, context="catalog",
            )
            _warned_id_dtype = True

        tile_ids = [normalize_object_id(x) for x in table.column(sid_col).to_pylist()]
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

    _KIND_DIR: dict[str, str] = {"spectrum": "spectra", "cutout": "cutouts"}
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
            if "source_id" not in root:
                continue
            sids = root["source_id"][:]
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

    from ..cli_utils import config_option, load_optional_config, require_output_root

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
    @click.option("--norder", default=5, show_default=True, type=int,
                  help="HEALPix order used for catalog partitioning.")
    @click.option(
        "--source-id-col",
        default=None,
        help="Override catalog ID column (e.g. TARGETID). "
             "Auto-detected from Parquet schema when omitted.",
    )
    @click.argument("lake_root", type=click.Path(path_type=Path), required=False)
    @config_option
    def cli_rebuild(
        survey_name: str,
        kind: str,
        norder: int,
        source_id_col: str | None,
        lake_root: Path | None,
        config_path: Path | None,
    ) -> None:
        """Rebuild _spectrum_index / _cutout_index in catalog tiles from existing Zarr data.

        Scans the Zarr tiles already on disk for SURVEY and patches the matching
        Parquet catalog tiles — no FITS re-ingestion required.  Use this to repair
        a deployment where spectra or cutouts were ingested without catalog patching.
        """
        import logging
        logging.basicConfig(level=logging.INFO)

        cfg = load_optional_config(config_path)
        resolved_root = require_output_root(lake_root, cfg)

        click.echo(f"Scanning {kind} tiles for survey={survey_name!r} …")
        index_map = build_index_map_from_zarr(
            lake_root=resolved_root,
            survey_name=survey_name,
            kind=kind,  # type: ignore[arg-type]
        )
        if not index_map:
            click.echo("No source IDs found in Zarr tiles — nothing to patch.")
            return

        click.echo(f"Found {len(index_map)} source IDs.  Patching catalog …")
        n_modified = update_index_column(
            lake_root=resolved_root,
            survey_name=survey_name,
            source_id_to_index=index_map,
            kind=kind,  # type: ignore[arg-type]
            norder=norder,
            source_id_col=source_id_col,
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
