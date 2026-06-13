"""Generalised base-source selection for ``dl-gather``.

Gather does not care how the base subset is chosen; it only needs a set of base
``_source_id`` values (and the HEALPix pixels they fall in, to bound downstream
tile reads). A :class:`BaseSelection` is produced from one of three selectors:

- **region** – spatial (npix/cone/bbox/moc); tile-granular (selects all base
  sources in the overlapping tiles). ``source_ids`` is ``None``.
- **ids** – an explicit list / external file of base source-ids.
- **query** – a DuckDB predicate over the base catalog (``--where``).

The npix set always bounds the crossmatch and modality tiles read downstream, so
even id/predicate selections stay tile-scoped (never full-tree scans).

Predicates that reference partner columns (``--where-joined``) are applied by
``dl-gather`` *after* the crossmatch join, not here (this module resolves the
base catalog only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from data_lake.discovery import tile_index as ti
from data_lake.discovery.region import Region
from data_lake.io.catalog import CatalogAccessor
from data_lake.schema_registry import MODALITY_CATALOG

log = logging.getLogger(__name__)


@dataclass
class BaseSelection:
    """A resolved selection of base-catalog sources."""

    base_survey: str
    norder: int
    npix: set[int]
    #: Explicit base source-ids, or ``None`` for a tile-granular region selection.
    source_ids: list[int] | None

    @property
    def is_tile_granular(self) -> bool:
        return self.source_ids is None


def _base_hp_column(acc: CatalogAccessor) -> str:
    return f"_healpix_norder{acc.norder}"


def selection_from_region(
    lake_root: Path | str,
    base_survey: str,
    region: Region,
    *,
    allow_scan: bool = True,
    modality: str = MODALITY_CATALOG,
) -> BaseSelection:
    """Tile-granular base selection from a spatial region.

    The base npix are the region resolved at the base catalog order intersected
    with populated tiles. ``source_ids`` is ``None`` (downstream filters by tile).
    """
    npix_set, hats_order = ti.survey_npix(
        lake_root, base_survey, modality, allow_scan=allow_scan
    )
    if hats_order is None:
        # Fall back to the accessor's order if the index lacks it.
        with CatalogAccessor(lake_root, base_survey) as acc:
            hats_order = acc.norder
    overlap = region.to_npix(hats_order) & npix_set
    return BaseSelection(
        base_survey=base_survey,
        norder=hats_order,
        npix=set(overlap),
        source_ids=None,
    )


def selection_from_ids(
    lake_root: Path | str,
    base_survey: str,
    source_ids: Sequence[int],
) -> BaseSelection:
    """Base selection from an explicit list of source-ids.

    Looks up the tile (``_healpix_norder<N>``) of each id to bound downstream
    reads.
    """
    ids = [int(s) for s in source_ids]
    if not ids:
        with CatalogAccessor(lake_root, base_survey) as acc:
            return BaseSelection(base_survey, acc.norder, set(), [])
    with CatalogAccessor(lake_root, base_survey) as acc:
        hp_col = _base_hp_column(acc)
        sid_col = acc.link_id_column
        npix: set[int] = set()
        found: list[int] = []
        # Chunk the IN (...) to keep SQL manageable for large id lists.
        for start in range(0, len(ids), 10_000):
            chunk = ids[start : start + 10_000]
            id_csv = ", ".join(str(i) for i in chunk)
            sql = (
                f"SELECT {sid_col}, {hp_col} FROM catalog "
                f"WHERE {sid_col} IN ({id_csv})"
            )
            for sid_raw, npix_raw in acc._con.execute(sql).fetchall():
                found.append(int(sid_raw))
                npix.add(int(npix_raw))
        return BaseSelection(base_survey, acc.norder, npix, found)


def selection_from_where(
    lake_root: Path | str,
    base_survey: str,
    where_sql: str,
) -> BaseSelection:
    """Base selection from a DuckDB predicate over the base catalog (base columns only).

    Runs ``SELECT <id>, <hp_col> FROM catalog WHERE <where_sql>`` so DuckDB
    applies projection + predicate pushdown over the base tiles.
    """
    predicate = where_sql.strip()
    if predicate.lower().startswith("where "):
        predicate = predicate[6:]
    with CatalogAccessor(lake_root, base_survey) as acc:
        hp_col = _base_hp_column(acc)
        sid_col = acc.link_id_column
        sql = f"SELECT {sid_col}, {hp_col} FROM catalog WHERE {predicate}"
        rows = acc._con.execute(sql).fetchall()
        ids = [int(r[0]) for r in rows]
        npix = {int(r[1]) for r in rows}
        return BaseSelection(base_survey, acc.norder, npix, ids)


def selection_from_all_tiles(
    lake_root: Path | str,
    base_survey: str,
    *,
    allow_scan: bool = True,
    modality: str = MODALITY_CATALOG,
) -> BaseSelection:
    """Select every populated tile in a catalog (used for full-product homogenize)."""
    npix_set, hats_order = ti.survey_npix(
        lake_root, base_survey, modality, allow_scan=allow_scan,
    )
    if hats_order is None:
        with CatalogAccessor(lake_root, base_survey) as acc:
            hats_order = acc.norder
    return BaseSelection(
        base_survey=base_survey,
        norder=hats_order,
        npix=set(npix_set),
        source_ids=None,
    )


def read_ids_file(path: str | Path, id_col: str | None = None) -> list[int]:
    """Read source-ids from a Parquet or CSV file.

    The column is ``id_col`` when given; otherwise the first column for CSV, or a
    column named like an id (``*_id`` / ``source_id`` / ``targetid``) for Parquet,
    falling back to the first column.
    """
    import pyarrow as pa

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        import pyarrow.parquet as pq

        table = pq.read_table(str(path))
    elif suffix in (".csv", ".txt", ".tsv"):
        from pyarrow import csv as pacsv

        delimiter = "\t" if suffix == ".tsv" else ","
        table = pacsv.read_csv(
            str(path), parse_options=pacsv.ParseOptions(delimiter=delimiter)
        )
    else:
        raise ValueError(f"unsupported id file type {suffix!r}; use .parquet or .csv")

    names = table.column_names
    if id_col is not None:
        if id_col not in names:
            raise KeyError(f"column {id_col!r} not in {path} (have {names})")
        col = id_col
    else:
        col = _guess_id_column(names)
    arr = table.column(col)
    return [int(v) for v in arr.to_pylist() if v is not None]


def _guess_id_column(names: Iterable[str]) -> str:
    names = list(names)
    for n in names:
        low = n.lower()
        if low in ("source_id", "targetid", "objid") or low.endswith("_id"):
            return n
    if not names:
        raise ValueError("id file has no columns")
    return names[0]
