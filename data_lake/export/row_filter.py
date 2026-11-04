"""Catalog row predicates for ``dl-extract-*`` ``--where`` / ``--limit``.

``--where`` is a DuckDB boolean expression over ingested catalog columns.
It is always wrapped as ``WHERE (<predicate>)`` and is never executed as its
own statement.  ``--limit`` caps rows after the predicate (tile order, no
``ORDER BY``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from data_lake.io.catalog import CatalogAccessor

log = logging.getLogger(__name__)

SPECTRUM_INDEX_COLUMN = "_spectrum_index"
CUTOUT_INDEX_COLUMN = "_cutout_index"


def _quote_sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def validate_predicate(where: str) -> str:
    """Return a stripped boolean expression, or raise ``ValueError``.

    Rejects empty text, statement separators (``;``), and comment markers
    (``--``, ``/*``) so the fragment cannot introduce extra SQL.
    """
    text = where.strip()
    if text[:5].upper() == "WHERE" and (len(text) == 5 or text[5].isspace()):
        text = text[5:].strip()
    if not text:
        raise ValueError("--where is empty")
    if ";" in text:
        raise ValueError(
            "--where must be a single boolean expression (no ';')"
        )
    if "--" in text or "/*" in text:
        raise ValueError("--where must not contain SQL comments")
    return text


def compose_where_clause(
    where: str | None,
    extra_predicates: Sequence[str] = (),
) -> str:
    """Return `` WHERE (pred) AND (extra) …`` or an empty string."""
    parts: list[str] = []
    if where is not None:
        parts.append(f"({validate_predicate(where)})")
    for extra in extra_predicates:
        clause = extra.strip()
        if not clause:
            continue
        if clause.upper().startswith("WHERE"):
            raise ValueError("extra predicate must not include WHERE")
        parts.append(f"({clause})")
    if not parts:
        return ""
    return " WHERE " + " AND ".join(parts)


def format_limit(limit: int | None) -> str:
    """Return `` LIMIT n`` or an empty string."""
    if limit is None:
        return ""
    if int(limit) < 1:
        raise ValueError("--limit must be a positive integer")
    return f" LIMIT {int(limit)}"


def source_ids_from_catalog(
    accessor: CatalogAccessor,
    *,
    where: str | None = None,
    limit: int | None = None,
    index_column: str | None = None,
) -> np.ndarray:
    """Select ``_source_id`` values matching *where* / *limit*.

    When *index_column* is present on the catalog (``_spectrum_index`` or
    ``_cutout_index``), rows with index ``< 0`` are dropped so ``--limit``
    counts objects that can actually be read from Zarr.
    """
    extra: list[str] = []
    if index_column and index_column in accessor.columns:
        extra.append(f"{_quote_sql_ident(index_column)} >= 0")

    sid_col = accessor.link_id_column
    sql = (
        f"SELECT {_quote_sql_ident(sid_col)} FROM catalog"
        f"{compose_where_clause(where, extra)}"
        f"{format_limit(limit)}"
    )
    log.info("Catalog extract filter: %s", sql)
    try:
        table = accessor.query(sql, fmt="arrow")
    except Exception as exc:
        raise ValueError(f"invalid --where: {exc}") from exc

    if table.num_rows == 0:
        return np.array([], dtype=np.int64)
    col = table.column(0).combine_chunks()
    return np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.int64)


def load_extract_source_ids(
    *,
    catalog_accessor: CatalogAccessor | None,
    target_list: Path | None,
    target_id_col: str,
    where_sql: str | None,
    row_limit: int | None,
    index_column: str | None,
    read_target_ids,
) -> np.ndarray:
    """Resolve source IDs from ``--target-list`` or ``--where`` (not both)."""
    if target_list is not None and where_sql:
        raise ValueError("Use --target-list or --where, not both.")
    if target_list is None and not where_sql:
        raise ValueError("Pass --target-list or --where.")
    if row_limit is not None and int(row_limit) < 1:
        raise ValueError("--limit must be a positive integer")

    if where_sql:
        if catalog_accessor is None:
            raise ValueError(
                "--where requires an ingested catalog for this survey "
                "(do not pass --no-with-catalog)."
            )
        return source_ids_from_catalog(
            catalog_accessor,
            where=where_sql,
            limit=row_limit,
            index_column=index_column,
        )

    ids = np.asarray(read_target_ids(target_list, target_id_col), dtype=np.int64)
    if row_limit is not None:
        ids = ids[: int(row_limit)]
    return ids
