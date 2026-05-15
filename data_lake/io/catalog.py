"""
catalog – DuckDB-backed accessor for HATS-partitioned Parquet catalogs.

Exposes results as Polars DataFrame, Astropy Table, or PyArrow Table.

Usage
-----
>>> from data_lake.io.catalog import CatalogAccessor
>>> cat = CatalogAccessor("/data/lake", "des_dr2")
>>> df = cat.query("SELECT ra, dec, mag_i FROM catalog WHERE mag_i < 22.0")
>>> cutout_ids = cat.sources_in_cone(ra=150.0, dec=2.2, radius_deg=0.5)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import resolve_source_id_column

log = logging.getLogger(__name__)

ReturnFormat = Literal["polars", "astropy", "arrow"]


def _arrow_array_to_numpy_or_sequence(arr: pa.Array):
    """Convert a single Arrow array to something Astropy Table columns accept."""
    if pa.types.is_dictionary(arr.type):
        arr = pc.dictionary_decode(arr)
    if pa.types.is_null(arr.type):
        return np.array([], dtype=np.float64)
    if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        return np.array(arr.to_pylist(), dtype=object)
    if pa.types.is_binary(arr.type) or pa.types.is_large_binary(arr.type):
        return np.array(arr.to_pylist(), dtype=object)
    if pa.types.is_nested(arr.type):
        return arr.to_pylist()
    try:
        return arr.to_numpy(zero_copy_only=False)
    except (pa.ArrowInvalid, TypeError, ValueError):
        return np.array(arr.to_pylist(), dtype=object)


def _arrow_table_to_astropy(table: pa.Table):
    """Build an ``astropy.table.Table`` from PyArrow without pandas."""
    from astropy.table import Table

    cols = {}
    for name in table.column_names:
        arr = table[name].combine_chunks()
        cols[name] = _arrow_array_to_numpy_or_sequence(arr)
    return Table(cols)


class CatalogAccessor:
    """
    Query interface for a single HATS-partitioned survey catalog.

    The underlying storage is Parquet; queries are executed via DuckDB so
    column projection and predicate pushdown happen transparently.

    Parameters
    ----------
    lake_root:
        Data lake root directory (contains ``catalogs/`` sub-tree).
    survey_name:
        Survey identifier (must match the directory under ``catalogs/``).
    norder:
        HEALPix order used for partitioning.  Read from ``catalog_info.json``
        if not supplied.
    """

    def __init__(
        self,
        lake_root: Path | str,
        survey_name: str,
        norder: int | None = None,
    ) -> None:
        self.lake_root = Path(lake_root)
        self.survey_name = survey_name
        self._catalog_root = self.lake_root / "catalogs" / survey_name
        if not self._catalog_root.exists():
            raise FileNotFoundError(f"Catalog not found: {self._catalog_root}")

        self._info = self._load_info()
        self.norder: int = norder if norder is not None else int(self._info.get("hats_order", 5))
        self._source_id_column: str = resolve_source_id_column(
            self._catalog_root,
            schema_names=list(self.schema.names) if self._catalog_root.exists() else None,
        )

        # DuckDB in-process connection (no file, memory only)
        self._con = duckdb.connect(database=":memory:")
        # Register the full Parquet glob as a view once so queries are cheap
        self._glob = str(self._catalog_root / f"Norder={self.norder}" / "**" / "*.parquet")
        self._con.execute(f"CREATE OR REPLACE VIEW catalog AS SELECT * FROM parquet_scan('{self._glob}')")
        log.debug("Registered view 'catalog' → %s", self._glob)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _load_info(self) -> dict:
        info_path = self._catalog_root / "catalog_info.json"
        if info_path.exists():
            with open(info_path) as fh:
                return json.load(fh)
        return {}

    @property
    def info(self) -> dict:
        return self._info

    @property
    def source_id_column(self) -> str:
        """Column name that holds the integer object identifier (e.g. ``TARGETID`` for DESI)."""
        return self._source_id_column

    @property
    def schema(self) -> pa.Schema:
        """PyArrow schema of the catalog (reads aggregate _metadata if present)."""
        meta_path = self._catalog_root / "_metadata"
        if meta_path.exists():
            return pq.read_metadata(str(meta_path)).schema.to_arrow_schema()
        # Fall back to reading one file
        first_file = next(self._catalog_root.rglob("*.parquet"), None)
        if first_file:
            return pq.read_schema(str(first_file))
        raise RuntimeError("No Parquet files found in catalog.")

    @property
    def columns(self) -> list[str]:
        return self.schema.names

    # ------------------------------------------------------------------
    # SQL query
    # ------------------------------------------------------------------

    def query(
        self,
        sql: str,
        fmt: ReturnFormat = "polars",
    ):
        """
        Execute an arbitrary SQL query against the catalog view.

        The table is exposed as ``catalog`` inside the SQL string.

        Parameters
        ----------
        sql:
            DuckDB SQL statement.  Use ``catalog`` as the table name.
        fmt:
            Return type: ``"polars"`` (default), ``"astropy"``, or ``"arrow"``.
        """
        arrow_result: pa.Table = self._con.execute(sql).arrow()
        return self._convert(arrow_result, fmt)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def sources_in_tile(
        self,
        npix: int,
        columns: list[str] | None = None,
        fmt: ReturnFormat = "polars",
    ):
        """Return all sources in a HEALPix tile."""
        col_expr = ", ".join(columns) if columns else "*"
        hp_col = f"_healpix_norder{self.norder}"
        sql = f"SELECT {col_expr} FROM catalog WHERE {hp_col} = {npix}"
        return self.query(sql, fmt=fmt)

    def sources_in_cone(
        self,
        ra: float,
        dec: float,
        radius_deg: float,
        columns: list[str] | None = None,
        fmt: ReturnFormat = "polars",
    ):
        """
        Return sources within a cone.

        Uses a fast HEALPix tile pre-filter plus a per-row angular separation
        check via DuckDB's haversine-equivalent SQL.
        """
        try:
            import healpy as hp
            nside = hp.order2nside(self.norder)
            vec = hp.ang2vec(np.radians(90.0 - dec), np.radians(ra))
            radius_rad = np.radians(radius_deg)
            tiles = hp.query_disc(nside, vec, radius_rad, nest=True, inclusive=True).tolist()
        except ImportError:
            tiles = None

        col_expr = ", ".join(columns) if columns else "*"
        hp_col = f"_healpix_norder{self.norder}"

        if tiles is not None:
            tile_list = ", ".join(str(t) for t in tiles)
            tile_filter = f"AND {hp_col} IN ({tile_list})"
        else:
            tile_filter = ""

        # Haversine angular separation in degrees using DuckDB functions
        sql = f"""
            SELECT {col_expr} FROM catalog
            WHERE 1=1 {tile_filter}
              AND (
                degrees(
                  acos(
                    LEAST(1.0,
                      sin(radians(dec)) * sin(radians({dec}))
                      + cos(radians(dec)) * cos(radians({dec}))
                        * cos(radians(ra - {ra}))
                    )
                  )
                )
              ) <= {radius_deg}
        """
        return self.query(sql.strip(), fmt=fmt)

    def get_sources_by_id(
        self,
        source_ids: list[int],
        columns: list[str] | None = None,
        fmt: ReturnFormat = "polars",
    ):
        """Fetch rows by a list of source_ids (using the catalog's actual ID column)."""
        col_expr = ", ".join(columns) if columns else "*"
        id_list = ", ".join(str(i) for i in source_ids)
        sid_col = self._source_id_column
        sql = f"SELECT {col_expr} FROM catalog WHERE {sid_col} IN ({id_list})"
        return self.query(sql, fmt=fmt)

    def get_tile_index(
        self,
        source_id: int,
        kind: str = "cutout",
    ) -> tuple[int, int]:
        """
        Return ``(healpix_npix, local_index)`` for a given source_id.

        Parameters
        ----------
        source_id:
            Source identifier (value of the catalog's ID column, e.g. ``TARGETID``).
        kind:
            Which index column to return: ``"cutout"`` (default) reads
            ``_cutout_index``; ``"spectrum"`` reads ``_spectrum_index``.

        Raises KeyError if the source is not found.
        """
        if kind not in ("cutout", "spectrum"):
            raise ValueError(f"kind must be 'cutout' or 'spectrum', got {kind!r}")
        index_col = f"_{kind}_index"
        hp_col = f"_healpix_norder{self.norder}"
        sid_col = self._source_id_column

        # _spectrum_index may not exist in catalogs ingested before this change
        cols_available = self.columns
        if index_col not in cols_available:
            raise KeyError(
                f"Column {index_col!r} not found in catalog '{self.survey_name}'. "
                f"Re-run catalog ingest or call update_index_column() first."
            )

        sql = f"SELECT {hp_col}, {index_col} FROM catalog WHERE {sid_col} = {source_id} LIMIT 1"
        result = self._con.execute(sql).fetchone()
        if result is None:
            raise KeyError(f"source_id={source_id} not found in catalog")
        return int(result[0]), int(result[1])

    # ------------------------------------------------------------------
    # Format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert(table: pa.Table, fmt: ReturnFormat):
        if fmt == "arrow":
            return table
        if fmt == "polars":
            try:
                import polars as pl
                return pl.from_arrow(table)
            except ImportError as e:
                raise ImportError("polars is not installed.") from e
        if fmt == "astropy":
            try:
                return _arrow_table_to_astropy(table)
            except ImportError as e:
                raise ImportError("astropy is not installed.") from e
        raise ValueError(f"Unknown fmt={fmt!r}. Choose from: polars, astropy, arrow.")

    # ------------------------------------------------------------------
    # Multi-survey cross-catalog helper
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return (
            f"CatalogAccessor(survey={self.survey_name!r}, "
            f"norder={self.norder}, root={self.lake_root})"
        )


# ---------------------------------------------------------------------------
# Multi-survey accessor
# ---------------------------------------------------------------------------


class MultiCatalogAccessor:
    """
    Convenience wrapper that holds multiple CatalogAccessor instances and
    provides cross-survey joins via a shared DuckDB connection.

    Parameters
    ----------
    lake_root:
        Data lake root.
    surveys:
        List of survey names to load.
    """

    def __init__(self, lake_root: Path | str, surveys: list[str]) -> None:
        self.lake_root = Path(lake_root)
        self._accessors: dict[str, CatalogAccessor] = {}
        self._con = duckdb.connect(":memory:")
        for name in surveys:
            acc = CatalogAccessor(lake_root, name)
            glob = str(acc._catalog_root / f"Norder={acc.norder}" / "**" / "*.parquet")
            self._con.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM parquet_scan('{glob}')"
            )
            self._accessors[name] = acc

    def query(self, sql: str, fmt: ReturnFormat = "polars"):
        """Run SQL referencing any survey by its name as a table."""
        arrow_result = self._con.execute(sql).arrow()
        return CatalogAccessor._convert(arrow_result, fmt)

    def __getitem__(self, survey: str) -> CatalogAccessor:
        return self._accessors[survey]

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
