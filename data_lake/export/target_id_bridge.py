"""Keep ``--target-id-col`` identity columns on ``dl-extract*`` outputs.

Target-list rows resolve to lake join IDs (and optionally through a
crossmatch).  Extract outputs are keyed by the *extract* survey
``_source_id``; without the original identity columns, callers cannot
match rows back to the target-list file.  This module builds a bridge
table and either merges it into a catalog extract or writes a sidecar
Parquet next to spectra / cutout exports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".target_ids.parquet"


def target_id_sidecar_path(output: Path | str) -> Path:
    """Path of the target-id sidecar next to an extract output."""
    output = Path(output)
    return output.with_name(output.name + SIDECAR_SUFFIX)


def build_target_id_bridge(
    resolved_ids: np.ndarray,
    identity: "pa.Table",
    *,
    via_pairs: "pa.Table | None" = None,
) -> "pa.Table":
    """Build ``_source_id`` (extract survey) + target-list identity columns.

    Parameters
    ----------
    resolved_ids:
        Per-row lake join IDs from the target-list (partner side when
        *via_pairs* is set; extract-survey IDs otherwise).
    identity:
        Arrow table of ``--target-id-col`` component columns (same order).
    via_pairs:
        Optional ``partner_id`` / ``extract_id`` pairs from a crossmatch.
        When set, ``_source_id`` is the extract-survey ID and identity
        columns are expanded for every match.
    """
    import pyarrow as pa

    ids = np.asarray(resolved_ids, dtype=np.int64).ravel()
    if identity.num_rows != int(ids.size):
        raise ValueError(
            f"identity rows ({identity.num_rows}) != resolved_ids "
            f"({ids.size})"
        )
    if identity.num_rows == 0:
        cols = {"_source_id": pa.array([], type=pa.int64())}
        for name in identity.column_names:
            cols[name] = identity.column(name)
        return pa.table(cols)

    id_names = list(identity.column_names)
    # Avoid clashing with the extract join key in the bridge itself.
    rename = {}
    if "_source_id" in id_names:
        rename["_source_id"] = "target__source_id"
        id_names = [rename.get(n, n) for n in identity.column_names]
        identity = identity.rename_columns(id_names)

    if via_pairs is None:
        cols: dict[str, pa.Array] = {
            "_source_id": pa.array(ids, type=pa.int64()),
        }
        for name in identity.column_names:
            cols[name] = identity.column(name)
        return pa.table(cols)

    if set(via_pairs.column_names) < {"partner_id", "extract_id"}:
        raise ValueError(
            "via_pairs must have columns partner_id and extract_id "
            f"(got {via_pairs.column_names})"
        )

    import duckdb

    partner = identity.append_column(
        "_partner_id", pa.array(ids, type=pa.int64())
    )
    con = duckdb.connect()
    try:
        con.register("identity", partner)
        con.register("pairs", via_pairs)
        id_select = ", ".join(
            f'i.{_sql_ident(n)}' for n in identity.column_names
        )
        select_sql = (
            "p.extract_id AS _source_id"
            + (f", {id_select}" if id_select else "")
        )
        return con.execute(
            f"""
            SELECT {select_sql}
            FROM identity i
            INNER JOIN pairs p ON i._partner_id = p.partner_id
            """
        ).to_arrow_table()
    finally:
        con.close()


def prepare_bridge_columns(
    bridge: "pa.Table",
    occupied: Sequence[str],
    *,
    via_crossmatch: bool,
) -> "pa.Table":
    """Drop or rename identity columns that collide with the extract schema.

    Without ``--via-crossmatch``, columns already present in the extract
    (e.g. native ``TARGETID``) are dropped as redundant.  With
    via-crossmatch, colliding names become ``target_<name>``.
    """
    occupied_set = set(occupied)
    keep_names: list[str] = []
    new_names: list[str] = []
    for name in bridge.column_names:
        if name == "_source_id":
            keep_names.append(name)
            new_names.append(name)
            continue
        if name in occupied_set:
            if not via_crossmatch:
                continue
            renamed = f"target_{name}"
            keep_names.append(name)
            new_names.append(renamed)
        else:
            keep_names.append(name)
            new_names.append(name)
    if not keep_names:
        return bridge.select([])
    out = bridge.select(keep_names)
    if new_names != keep_names:
        out = out.rename_columns(new_names)
    return out


def attach_bridge_to_table(
    extract: "pa.Table",
    bridge: "pa.Table",
    *,
    via_crossmatch: bool = False,
    join_key: str = "_source_id",
) -> "pa.Table":
    """Left-join *bridge* onto *extract* on *join_key* (row-expanding)."""
    import duckdb

    if bridge.num_rows == 0 or set(bridge.column_names) <= {join_key}:
        return extract
    if join_key not in extract.column_names:
        raise ValueError(
            f"Cannot attach target-id columns: extract is missing {join_key!r}"
        )

    prepared = prepare_bridge_columns(
        bridge, extract.column_names, via_crossmatch=via_crossmatch
    )
    identity_cols = [n for n in prepared.column_names if n != join_key]
    if not identity_cols:
        return extract

    con = duckdb.connect()
    try:
        con.register("extract_tbl", extract)
        con.register("bridge_tbl", prepared)
        id_select = ", ".join(f"b.{_sql_ident(n)}" for n in identity_cols)
        return con.execute(
            f"""
            SELECT e.*, {id_select}
            FROM extract_tbl e
            LEFT JOIN bridge_tbl b
              ON e.{_sql_ident(join_key)} = b.{_sql_ident(join_key)}
            """
        ).to_arrow_table()
    finally:
        con.close()


def write_target_id_sidecar(output: Path | str, bridge: "pa.Table") -> Path:
    """Write bridge Parquet next to *output*; return the sidecar path."""
    import pyarrow.parquet as pq

    path = target_id_sidecar_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(bridge, str(path), compression="zstd", compression_level=3)
    log.info(
        "Wrote target-id sidecar (%d row(s), %d col(s)) → %s",
        bridge.num_rows,
        bridge.num_columns,
        path,
    )
    return path


def apply_bridge_to_catalog_file(
    output: Path,
    bridge: "pa.Table",
    *,
    via_crossmatch: bool = False,
    output_format: str | None = None,
) -> tuple[Path, int, int]:
    """Rewrite a single-file catalog extract with attached identity columns.

    Returns ``(path_written, n_rows_after, n_cols_after)``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from data_lake.export.catalog_extract import (
        infer_output_format,
        write_catalog_extract,
    )

    fmt = infer_output_format(output, output_format)  # type: ignore[arg-type]
    if fmt == "parquet":
        extract = pq.read_table(str(output))
    elif fmt == "csv":
        import pyarrow.csv as pacsv

        extract = pacsv.read_csv(str(output))
    else:
        from astropy.table import Table

        at = Table.read(str(output))
        arrays: dict[str, pa.Array] = {}
        for name in at.colnames:
            col = at[name]
            try:
                arrays[name] = pa.array(np.asarray(col))
            except (TypeError, ValueError):
                arrays[name] = pa.array(list(col))
        extract = pa.table(arrays)

    if "_source_id" not in extract.column_names:
        sidecar = write_target_id_sidecar(output, bridge)
        log.warning(
            "Extract %s has no _source_id; wrote target-id sidecar → %s",
            output.name,
            sidecar,
        )
        return sidecar, extract.num_rows, extract.num_columns

    merged = attach_bridge_to_table(
        extract, bridge, via_crossmatch=via_crossmatch
    )
    write_catalog_extract(merged, output, output_format=fmt)
    return output, merged.num_rows, merged.num_columns


def resolve_target_bridge(
    target_list: Path,
    target_id_col: str,
    *,
    lake_root: Path | str | None = None,
    via_crossmatch: str | None = None,
    extract_survey: str | None = None,
) -> tuple[np.ndarray, "pa.Table"]:
    """Load target-list identity + resolve extract ``_source_id`` values.

    Returns ``(unique_extract_ids, bridge)`` where *bridge* has
    ``_source_id`` plus the ``--target-id-col`` component columns.
    """
    from data_lake.export.spectra_subset import read_target_list_table

    resolved, identity = read_target_list_table(target_list, target_id_col)
    if via_crossmatch is None:
        bridge = build_target_id_bridge(resolved, identity)
        extract_ids = np.unique(np.asarray(resolved, dtype=np.int64))
        return extract_ids, bridge

    if lake_root is None or extract_survey is None:
        raise ValueError(
            "--via-crossmatch requires lake_root and extract_survey"
        )
    from data_lake.export.catalog_extract import map_id_pairs_via_crossmatch

    pairs = map_id_pairs_via_crossmatch(
        lake_root,
        via_crossmatch,
        extract_survey=extract_survey,
        source_ids=resolved,
    )
    bridge = build_target_id_bridge(resolved, identity, via_pairs=pairs)
    if bridge.num_rows == 0:
        extract_ids = np.asarray([], dtype=np.int64)
    else:
        extract_ids = np.unique(
            np.asarray(bridge.column("_source_id"), dtype=np.int64)
        )
    return extract_ids, bridge


def _sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
