"""
catalog_extract – export selected catalog columns for catalog–catalog association.

Use before positional matching (STILTS, ``build_crossmatch``, etc.): pull only the
ID and sky columns (plus any extras) from survey catalog files or from an
already-ingested lake catalog.  This is separate from catalog↔spectrum linkage,
which uses ``_spectrum_index`` / native spectrum keys on catalog rows.

Large surveys (tens–hundreds of millions of rows, or more) must **not** be
materialised in RAM.  Lake exports stream one HEALPix tile at a time; optional
``--output-dir`` writes a HATS tree for parallel downstream tools.  For
catalog↔catalog matching without a monolithic export, prefer in-lake
``build_crossmatch`` (``data_lake.io.crossmatch``).

Example
-------
::

    dl-extract-catalog survey_a.fits -o a_positions.parquet \\
        -c TARGETID,RA,DEC

    dl-extract-catalog --lake-root /data/lake --survey DESI_DR1 \\
        -c source_id,ra,dec -o desi_sky.parquet

    dl-extract-catalog --lake-root /data/lake --survey DESI_DR1 \\
        -c TARGETID -c Z --where "upper(trim(SPECTYPE)) = 'STAR'" \\
        --limit 1000 -o desi_stars.parquet

    dl-extract-catalog --lake-root /data/lake --survey DESI_DR1 \\
        --target-list qso_ids.fits --target-id-col TARGETID \\
        --all-columns -o desi_qso_catalog.parquet

    dl-extract-catalog zall-pix-iron.fits -c TARGETID -c Z \\
        --where "upper(trim(SPECTYPE)) = 'STAR'" --limit 1000 \\
        -o desi_stars.parquet

    # 290M+ rows: tiled export (bounded RAM, parallel-friendly)
    dl-extract-catalog --lake-root /data/lake --survey GAIA_DR3 \\
        --output-dir /scratch/gaia_sky/ -c source_id -c ra -c dec
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import (
    _read_source_table,
    _ZSTD_LEVEL,
    resolve_catalog_column_name,
)

log = logging.getLogger(__name__)

OutputFormat = Literal["parquet", "csv", "fits", "votable"]
LakeEngine = Literal["auto", "tiles", "duckdb"]
_SKY_RA_ALIASES = ("ra", "ra_deg", "radeg", "raj2000", "ra_obj")
_SKY_DEC_ALIASES = ("dec", "dec_deg", "decdeg", "dej2000", "dec_obj")
_LARGE_OUTPUT_ROW_WARN = 5_000_000
_FITS_STREAM_BATCH_ROWS = 1_000_000
_TARGET_IDS_RELATION = "extract_target_ids"


@contextmanager
def _extract_staging_dir(parent: Path | str | None = None) -> Iterator[Path]:
    """Create a staging directory under the process cwd (not system ``/tmp``).

    FITS/VOTable lake exports write an intermediate Parquet here, then convert
    and delete the directory.  *parent* is accepted for API compatibility but
    ignored: staging always uses :func:`Path.cwd` so the temp tree appears in
    the directory where the command was launched.  Cwd is created with
    ``mkdir(parents=True, exist_ok=True)`` if missing (unusual, but safe).
    """
    del parent  # callers may pass output.parent; staging is always cwd
    base = Path.cwd()
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dl_extract_", dir=str(base)) as tmp:
        yield Path(tmp)


def _staging_parquet_path(staging_dir: Path, output: Path) -> Path:
    """Absolute staging Parquet path derived from the final output name."""
    return (staging_dir / f"{Path(output).stem}.dl_extract.parquet").resolve()


@dataclass(frozen=True)
class ExtractResult:
    """Summary of a streaming extract (no in-memory table retained)."""

    n_rows: int
    column_names: tuple[str, ...]
    output: Path | None = None
    output_dir: Path | None = None
    n_tiles: int = 0


def expand_column_args(args: Sequence[str]) -> list[str]:
    """Expand ``-c`` values that may be comma-separated lists.

    Each ``-c`` / ``--column`` value may be a single ``NAME`` / ``NAME:alias``
    or a comma-separated list of those (e.g. ``RA,DEC,TARGETID`` or
    ``ra:RA,dec:DEC``). Repeated ``-c`` flags are concatenated in order.
    Empty segments (``a,,b`` or a trailing comma) raise ``ValueError``.
    """
    out: list[str] = []
    for raw in args:
        for part in str(raw).split(","):
            text = part.strip()
            if not text:
                raise ValueError(
                    f"empty column spec in {raw!r} "
                    "(use comma-separated names with no empty segments)"
                )
            out.append(text)
    return out


def parse_column_spec(spec: str) -> tuple[str, str | None]:
    """Parse ``NAME`` or ``NAME:alias`` from a ``-c`` argument."""
    text = spec.strip()
    if not text:
        raise ValueError("empty column spec")
    if ":" in text:
        name, alias = text.split(":", 1)
        name, alias = name.strip(), alias.strip()
        if not name:
            raise ValueError(f"invalid column spec {spec!r}")
        return name, alias or None
    return text, None


def resolve_column_specs(
    available: Sequence[str],
    specs: Sequence[str],
) -> list[tuple[str, str]]:
    """Map user specs to ``(catalog_column, output_name)`` pairs."""
    out: list[tuple[str, str]] = []
    for spec in specs:
        requested, alias = parse_column_spec(spec)
        resolved = resolve_catalog_column_name(available, requested)
        out_name = alias if alias is not None else resolved
        out.append((resolved, out_name))
    return out


def specs_from_schema(
    names: Sequence[str],
    *,
    all_columns: bool,
    specs: Sequence[str],
) -> list[str]:
    """Resolve export column specs from a table schema."""
    if all_columns:
        return list(names)
    if not specs:
        raise ValueError("pass column specs or all_columns=True")
    return list(specs)


def select_catalog_columns(
    table: pa.Table,
    specs: Sequence[str],
) -> pa.Table:
    """Project *table* to the requested columns (with optional renames)."""
    if not specs:
        raise ValueError("at least one column is required")
    mapping = resolve_column_specs(table.schema.names, specs)
    arrays = []
    names: list[str] = []
    for src, out_name in mapping:
        arrays.append(table.column(src).combine_chunks())
        names.append(out_name)
    return pa.table(dict(zip(names, arrays)))


def _resolve_sky_output_names(
    table: pa.Table,
    *,
    ra_col: str | None,
    dec_col: str | None,
) -> tuple[str, str]:
    ra_name = ra_col
    dec_name = dec_col
    if ra_name is None:
        by_lower = {n.lower(): n for n in table.schema.names}
        for cand in _SKY_RA_ALIASES:
            if cand in by_lower:
                ra_name = by_lower[cand]
                break
    if dec_name is None:
        by_lower = {n.lower(): n for n in table.schema.names}
        for cand in _SKY_DEC_ALIASES:
            if cand in by_lower:
                dec_name = by_lower[cand]
                break
    if ra_name is None or dec_name is None:
        raise ValueError(
            "valid-sky filter needs RA/Dec columns; pass --ra-col and --dec-col "
            f"or include them in the export (columns: {table.schema.names})"
        )
    return ra_name, dec_name


def filter_valid_sky_rows(
    table: pa.Table,
    *,
    ra_col: str | None = None,
    dec_col: str | None = None,
) -> pa.Table:
    """Drop rows whose RA/Dec are not usable for HEALPix / cone matching."""
    ra_name, dec_name = _resolve_sky_output_names(table, ra_col=ra_col, dec_col=dec_col)

    ra_np = np.asarray(
        pc.cast(table.column(ra_name).combine_chunks(), pa.float64()),
        dtype=np.float64,
    )
    dec_np = np.asarray(
        pc.cast(table.column(dec_name).combine_chunks(), pa.float64()),
        dtype=np.float64,
    )
    mask = (
        np.isfinite(ra_np)
        & np.isfinite(dec_np)
        & (dec_np >= -90.0)
        & (dec_np <= 90.0)
        & (ra_np > -9000.0)
        & (dec_np > -9000.0)
    )
    n_before = table.num_rows
    out = table.filter(pa.array(mask, type=pa.bool_()))
    n_after = out.num_rows
    if n_after < n_before:
        log.debug(
            "Dropped %d row(s) with invalid sky position (kept %d)",
            n_before - n_after,
            n_after,
        )
    return out


def _sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _prepare_source_id_filter(
    catalog_root: Path,
    schema_names: Sequence[str],
    source_ids: np.ndarray | None,
) -> tuple[np.ndarray | None, str | None, str | None]:
    """Return unique ``_source_id`` values, a DuckDB membership predicate, and link col.

    *source_ids* are lake join ids (the same int64 values ``--target-list``
    resolves).  The predicate references a relation registered with
    :func:`_register_target_ids`.
    """
    if source_ids is None:
        return None, None, None
    from data_lake.ingest.fits_to_parquet import resolve_link_id_column

    try:
        link_col = resolve_link_id_column(catalog_root, schema_names=list(schema_names))
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    ids = np.unique(np.asarray(source_ids, dtype=np.int64))
    predicate = (
        f"{_sql_ident(link_col)} IN (SELECT id FROM {_TARGET_IDS_RELATION})"
    )
    log.info(
        "Target-list filter: %d unique id(s) on column %s",
        int(ids.size),
        link_col,
    )
    return ids, predicate, link_col


def _register_target_ids(connection, source_ids: np.ndarray) -> None:
    """Register *source_ids* as ``extract_target_ids(id)`` on a DuckDB connection."""
    connection.register(
        _TARGET_IDS_RELATION,
        pa.table({"id": pa.array(np.asarray(source_ids, dtype=np.int64), type=pa.int64())}),
    )


def _connect_extract_duckdb(source_ids: np.ndarray | None):
    """Open an extract DuckDB connection with process/env limits and metadata cache."""
    from data_lake.io.duckdb_limits import connect as connect_duckdb
    from data_lake.io.duckdb_limits import parquet_metadata_cache_default

    connection = connect_duckdb(
        ":memory:",
        parquet_metadata_cache=parquet_metadata_cache_default(),
    )
    if source_ids is not None:
        _register_target_ids(connection, source_ids)
    return connection


def _npix_from_tile_path(tile_path: Path) -> int | None:
    """Parse ``Npix=<int>`` from a lake tile filename."""
    stem = tile_path.stem
    if not stem.startswith("Npix="):
        return None
    try:
        return int(stem.split("=", 1)[1])
    except ValueError:
        return None


def _prune_tiles_for_source_ids(
    lake_root: Path | str,
    survey: str,
    tiles: Sequence[Path],
    source_ids: np.ndarray,
    *,
    norder: int,
) -> list[Path]:
    """Keep only tiles that contain any of *source_ids* (via ``_healpix_norder{N}``).

    Falls back to the full *tiles* list when the HEALPix column is missing or
    the lightweight catalog scan fails.
    """
    ids = np.unique(np.asarray(source_ids, dtype=np.int64).ravel())
    if ids.size == 0:
        return []
    try:
        from data_lake.io.catalog import CatalogAccessor

        cat = CatalogAccessor(lake_root, survey, norder=norder)
        hp_col = f"_healpix_norder{cat.norder}"
        if hp_col not in cat.columns:
            log.info(
                "Tile prune skipped: %s not in catalog schema; scanning all %d tile(s).",
                hp_col,
                len(tiles),
            )
            return list(tiles)
        sid_sql = _sql_ident(cat.link_id_column)
        hp_sql = _sql_ident(hp_col)
        hit_npix: set[int] = set()
        for ids_table in cat._id_tables(ids, 1_000_000):
            sql = (
                f"SELECT DISTINCT {hp_sql} FROM catalog "
                f"WHERE {sid_sql} IN (SELECT id FROM {ids_table})"
            )
            rows = cat._con.execute(sql).fetchnumpy()
            if not rows:
                continue
            arr = np.asarray(next(iter(rows.values())), dtype=np.int64)
            hit_npix.update(int(v) for v in arr.tolist() if v >= 0)
        if not hit_npix:
            log.info(
                "Tile prune: 0 hit tile(s) for %d id(s) (of %d surveyed).",
                int(ids.size),
                len(tiles),
            )
            return []
        pruned = [
            p for p in tiles
            if (npix := _npix_from_tile_path(p)) is not None and npix in hit_npix
        ]
        log.info(
            "Tile prune: %d/%d tile(s) hold %d target id(s).",
            len(pruned),
            len(tiles),
            int(ids.size),
        )
        return pruned
    except Exception:
        log.warning(
            "Tile prune failed; scanning all %d tile(s).",
            len(tiles),
            exc_info=True,
        )
        return list(tiles)


def _parquet_num_rows(path: Path) -> int:
    """Row count from Parquet footers (no data scan)."""
    pf = pq.ParquetFile(path)
    return int(pf.metadata.num_rows) if pf.metadata is not None else 0


def _lake_schema_names(catalog_root: Path, tiles: Sequence[Path]) -> list[str]:
    """Column names from ``_common_metadata`` when present, else the first tile."""
    from data_lake.schema_registry import catalog_schema_source

    source = catalog_schema_source(catalog_root)
    if source is not None:
        return list(pq.read_schema(str(source)).names)
    if tiles:
        return list(pq.read_schema(str(tiles[0])).names)
    return []


def _projected_select_sql(
    mapping: Sequence[tuple[str, str]],
    from_sql: str,
    *,
    where_sql: str | None = None,
    extra_predicates: Sequence[str] = (),
    row_limit: int | None = None,
    order_for_limit: bool = False,
) -> str:
    """``SELECT src AS out FROM … [WHERE user AND extra] [ORDER BY … LIMIT n]``.

    *from_sql* must be a ``read_parquet`` scan.  ``order_for_limit`` is only used
    when the caller explicitly wants deterministic tile/file order under LIMIT
    (costly on large semi-joins); the default lake path uses the tile engine for
    ``--limit`` instead.
    """
    from data_lake.export.row_filter import compose_where_clause, format_limit

    col_sql = ", ".join(
        f"{_sql_ident(src)} AS {_sql_ident(out)}" for src, out in mapping
    )
    order_sql = (
        " ORDER BY file_index, file_row_number"
        if row_limit is not None and order_for_limit
        else ""
    )
    return (
        f"SELECT {col_sql} FROM {from_sql}"
        f"{compose_where_clause(where_sql, extra_predicates)}"
        f"{order_sql}"
        f"{format_limit(row_limit)}"
    )


def _read_catalog_tile(
    tile_path: Path,
    columns: Sequence[str],
) -> pa.Table:
    """Read one on-disk Parquet tile without hive partition discovery.

    ``pq.read_table`` on paths under ``Norder=…/Dir=…/`` can attach partition
    columns and merge incompatible types across tiles; ``ParquetFile`` reads
  only the file payload.
    """
    pf = pq.ParquetFile(tile_path)
    available = set(pf.schema_arrow.names)
    missing = [c for c in columns if c not in available]
    if missing:
        raise KeyError(
            f"{tile_path}: column(s) not in tile: {missing!r} "
            f"(available: {sorted(available)[:30]})"
        )
    return pf.read(columns=list(columns))


def _canonicalize_column_types(table: pa.Table) -> pa.Table:
    """Decode dictionary columns and normalize chunks for stable Parquet writes."""
    cols: dict[str, pa.ChunkedArray] = {}
    for name in table.schema.names:
        col = table.column(name).combine_chunks()
        if pa.types.is_dictionary(col.type):
            col = pc.cast(col, col.type.value_type)
        cols[name] = col
    return pa.table(cols)


def _promote_arrow_type(a: pa.DataType, b: pa.DataType) -> pa.DataType:
    """Pick the wider of two Arrow types (e.g. null + float64 → float64).

    Prefer string/binary over float so legacy gather pads (float64 nulls for
    unmatched string partners) do not lock a text column as double.
    """
    if a.equals(b) or pa.types.is_null(a):
        return b
    if pa.types.is_null(b):
        return a

    def _is_text(t: pa.DataType) -> bool:
        return (
            pa.types.is_string(t)
            or pa.types.is_large_string(t)
            or pa.types.is_binary(t)
            or pa.types.is_large_binary(t)
        )

    if pa.types.is_floating(a) and _is_text(b):
        return b
    if _is_text(a) and pa.types.is_floating(b):
        return a
    if pa.types.is_floating(a) or pa.types.is_floating(b):
        if pa.types.is_floating(a) and pa.types.is_floating(b):
            return pa.float64() if (a == pa.float64() or b == pa.float64()) else pa.float32()
        return pa.float64()
    if pa.types.is_integer(a) and pa.types.is_integer(b):
        return a if a.bit_width >= b.bit_width else b
    return a


def _unified_schema_for_mapping(
    tile_paths: Sequence[Path],
    mapping: Sequence[tuple[str, str]],
    *,
    catalog_root: Path | None = None,
) -> pa.Schema:
    """Merge Parquet schemas so null-only columns unify with typed columns.

    Prefer ``_common_metadata`` / cheapest schema source under *catalog_root*
    when every mapped column is present there; otherwise fall back to scanning
    tile footers (needed when tiles disagree on null vs typed columns).
    """
    src_types: dict[str, pa.DataType] = {}
    if catalog_root is not None:
        from data_lake.schema_registry import catalog_schema_source

        source = catalog_schema_source(catalog_root)
        if source is not None:
            sch = pq.read_schema(str(source))
            names = set(sch.names)
            # Use the cheap schema only when every mapped column is present and
            # none are null-typed (null in _common_metadata can hide float tiles).
            if all(src in names for src, _ in mapping) and not any(
                pa.types.is_null(sch.field(src).type) for src, _ in mapping
            ):
                return pa.schema([
                    pa.field(out, sch.field(src).type) for src, out in mapping
                ])
    for path in tile_paths:
        sch = pq.read_schema(str(path))
        for src, _out in mapping:
            if src not in sch.names:
                continue
            t = sch.field(src).type
            if src in src_types:
                src_types[src] = _promote_arrow_type(src_types[src], t)
            else:
                src_types[src] = t
    return pa.schema([
        pa.field(out, src_types[src])
        for src, out in mapping
        if src in src_types
    ])


def _align_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Cast *table* to *schema* so a multi-tile ``ParquetWriter`` stays consistent."""
    arrays: list[pa.ChunkedArray] = []
    for field in schema:
        if field.name not in table.schema.names:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
            continue
        col = table.column(field.name).combine_chunks()
        target_type = field.type
        if col.type != target_type:
            if pa.types.is_null(col.type):
                col = pa.nulls(len(col), type=target_type)
            else:
                col = pc.cast(col, target_type, safe=False)
        arrays.append(col)
    return pa.Table.from_arrays(arrays, schema=schema)


def _lake_catalog_root(lake_root: Path | str, survey: str) -> Path:
    from data_lake.schema_registry import resolve_catalog_root
    root = resolve_catalog_root(lake_root, survey)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Catalog not found: {root} "
            f"(checked catalogs/{survey} and products/{survey})"
        )
    return root


def _lake_norder(catalog_root: Path, norder: int | None) -> int:
    if norder is not None:
        return norder
    info_path = catalog_root / "catalog_info.json"
    if info_path.is_file():
        with open(info_path) as fh:
            return int(json.load(fh).get("hats_order", 5))
    return 5


def read_lake_catalog_info(lake_root: Path | str, survey: str) -> dict[str, Any]:
    """Load ``catalog_info.json`` for a lake catalog or product."""
    info_path = _lake_catalog_root(lake_root, survey) / "catalog_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing catalog metadata: {info_path}")
    return json.loads(info_path.read_text())


def validate_homogenized_export(
    lake_root: Path | str,
    survey: str,
    *,
    require_homogenized: bool,
) -> dict[str, Any]:
    """Ensure a catalog export source meets homogenized product expectations."""
    from data_lake.schema_registry import PRODUCT_SUBTYPE_HOMOGENIZED

    info = read_lake_catalog_info(lake_root, survey)
    subtype = info.get("product_subtype")
    if require_homogenized and subtype != PRODUCT_SUBTYPE_HOMOGENIZED:
        raise ValueError(
            f"catalog {survey!r} has product_subtype={subtype!r}; "
            f"expected {PRODUCT_SUBTYPE_HOMOGENIZED!r} (use dl-homogenize first "
            "or omit --require-subtype homogenized)"
        )
    return info


def homogenized_export_provenance(catalog_info: dict[str, Any]) -> dict[str, Any]:
    """Build provenance payload for ML export sidecars."""
    prov = catalog_info.get("provenance") or {}
    return {
        "source_catalog": catalog_info.get("catalog_name"),
        "kind": catalog_info.get("kind"),
        "product_subtype": catalog_info.get("product_subtype"),
        "transform_id": prov.get("transform_id"),
        "transform_version": prov.get("transform_version"),
        "source_survey": prov.get("source_survey"),
        "source_product": prov.get("source_product"),
        "column_lineage": prov.get("column_lineage"),
    }


def write_homogenize_export_provenance(
    catalog_info: dict[str, Any],
    *,
    output: Path | None,
    output_dir: Path | None,
) -> Path | None:
    """Write ``*.homogenize_provenance.json`` beside a lake catalog export."""
    from data_lake.schema_registry import PRODUCT_SUBTYPE_HOMOGENIZED

    if catalog_info.get("product_subtype") != PRODUCT_SUBTYPE_HOMOGENIZED:
        return None
    payload = homogenized_export_provenance(catalog_info)
    if output_dir is not None:
        sidecar = Path(output_dir) / "extract_provenance.json"
    elif output is not None:
        out = Path(output)
        sidecar = out.with_name(out.name + ".homogenize_provenance.json")
    else:
        return None
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, "w") as fh:
        json.dump(payload, fh, indent=2)
    return sidecar


def iter_lake_catalog_tiles(
    catalog_root: Path,
    *,
    norder: int | None = None,
) -> Iterator[Path]:
    """Yield ``Npix=*.parquet`` tile paths under a lake catalog."""
    order = _lake_norder(catalog_root, norder)
    order_root = catalog_root / f"Norder={order}"
    if order_root.is_dir():
        yield from sorted(order_root.rglob("Npix=*.parquet"))
        return
    yield from sorted(catalog_root.rglob("Npix=*.parquet"))


def _project_tile_table(
    table: pa.Table,
    mapping: Sequence[tuple[str, str]],
) -> pa.Table:
    return pa.table(
        {out_name: table.column(src).combine_chunks() for src, out_name in mapping}
    )


def _process_tile_chunk(
    table: pa.Table,
    mapping: Sequence[tuple[str, str]],
    *,
    valid_sky_only: bool,
    ra_col: str | None,
    dec_col: str | None,
) -> pa.Table:
    chunk = _project_tile_table(table, mapping)
    if valid_sky_only:
        chunk = filter_valid_sky_rows(chunk, ra_col=ra_col, dec_col=dec_col)
    return chunk


def stream_extract_from_lake_catalog(
    lake_root: Path | str,
    survey: str,
    specs: Sequence[str],
    *,
    all_columns: bool = False,
    output: Path | None = None,
    output_dir: Path | None = None,
    norder: int | None = None,
    valid_sky_only: bool = False,
    ra_col: str | None = None,
    dec_col: str | None = None,
    output_format: OutputFormat | None = None,
    engine: LakeEngine = "auto",
    show_progress: bool = False,
    require_homogenized: bool = False,
    write_homogenize_provenance: bool = True,
    where_sql: str | None = None,
    row_limit: int | None = None,
    source_ids: np.ndarray | None = None,
) -> ExtractResult:
    """Stream column projection from lake Parquet tiles without loading the full survey."""
    if output is None and output_dir is None:
        raise ValueError("pass output or output_dir")
    if output is not None and output_dir is not None:
        raise ValueError("pass output or output_dir, not both")

    catalog_info = validate_homogenized_export(
        lake_root, survey, require_homogenized=require_homogenized,
    )
    catalog_root = _lake_catalog_root(lake_root, survey)
    order = _lake_norder(catalog_root, norder)
    tiles = list(iter_lake_catalog_tiles(catalog_root, norder=order))
    if not tiles:
        raise FileNotFoundError(f"No Parquet tiles under {catalog_root}")

    schema_names = _lake_schema_names(catalog_root, tiles)
    effective_specs = specs_from_schema(
        schema_names, all_columns=all_columns, specs=specs,
    )
    mapping = resolve_column_specs(schema_names, effective_specs)
    src_cols = [src for src, _ in mapping]
    out_names = tuple(out for _, out in mapping)
    source_ids, id_predicate, link_col = _prepare_source_id_filter(
        catalog_root, schema_names, source_ids,
    )
    if source_ids is not None:
        tiles = _prune_tiles_for_source_ids(
            lake_root, survey, tiles, source_ids, norder=order,
        )

    fmt: OutputFormat = "parquet"
    if output is not None:
        fmt = infer_output_format(Path(output), output_format)
    if output_dir is not None and fmt != "parquet":
        raise ValueError(
            "--output-dir writes a HATS Parquet tree only; use -o file.csv or file.fits "
            "for a single-file export."
        )

    # Prefer the tile engine for --limit under auto: it early-stops without a
    # global ORDER BY over the filtered set. Explicit engine=duckdb keeps ORDER BY.
    prefer_tiles_for_limit = row_limit is not None and engine == "auto"
    use_duckdb = (
        bool(tiles)
        and output is not None
        and output_dir is None
        and fmt in ("parquet", "csv", "fits", "votable")
        and (engine == "duckdb" or engine == "auto")
        and not prefer_tiles_for_limit
    )
    if use_duckdb:
        try:
            result = _stream_lake_via_duckdb(
                tiles,
                mapping,
                output=output,
                output_format=fmt,
                valid_sky_only=valid_sky_only,
                ra_col=ra_col,
                dec_col=dec_col,
                where_sql=where_sql,
                row_limit=row_limit,
                source_ids=source_ids,
                id_predicate=id_predicate,
                order_for_limit=(engine == "duckdb"),
            )
            return _finalize_lake_extract_result(
                result,
                catalog_info=catalog_info,
                output=output,
                output_dir=output_dir,
                write_homogenize_provenance=write_homogenize_provenance,
            )
        except Exception as exc:
            log.warning(
                "DuckDB export failed (%s); falling back to tile streaming.", exc
            )

    result = _stream_lake_tile_by_tile(
        catalog_root,
        tiles,
        mapping,
        src_cols=src_cols,
        out_names=out_names,
        output=output,
        output_dir=output_dir,
        output_format=fmt,
        valid_sky_only=valid_sky_only,
        ra_col=ra_col,
        dec_col=dec_col,
        show_progress=show_progress,
        where_sql=where_sql,
        row_limit=row_limit,
        source_ids=source_ids,
        id_predicate=id_predicate,
        link_col=link_col,
    )
    return _finalize_lake_extract_result(
        result,
        catalog_info=catalog_info,
        output=output,
        output_dir=output_dir,
        write_homogenize_provenance=write_homogenize_provenance,
    )


def _finalize_lake_extract_result(
    result: ExtractResult,
    *,
    catalog_info: dict[str, Any],
    output: Path | None,
    output_dir: Path | None,
    write_homogenize_provenance: bool,
) -> ExtractResult:
    if write_homogenize_provenance:
        write_homogenize_export_provenance(
            catalog_info, output=output, output_dir=output_dir,
        )
    return result


def _sky_predicate_sql(ra_name: str, dec_name: str) -> str:
    ra_sql = _sql_ident(ra_name)
    dec_sql = _sql_ident(dec_name)
    return (
        f"isfinite({ra_sql}) AND isfinite({dec_sql}) "
        f"AND {dec_sql} BETWEEN -90 AND 90 "
        f"AND {ra_sql} > -9000 AND {dec_sql} > -9000"
    )


def _duckdb_sky_predicate(
    mapping: Sequence[tuple[str, str]],
    *,
    ra_col: str | None,
    dec_col: str | None,
) -> str:
    dummy = pa.table({out: pa.array([0.0], type=pa.float64()) for _, out in mapping})
    ra_name, dec_name = _resolve_sky_output_names(dummy, ra_col=ra_col, dec_col=dec_col)
    return _sky_predicate_sql(ra_name, dec_name)


def _apply_extract_filter(
    table: pa.Table,
    mapping: Sequence[tuple[str, str]],
    *,
    where_sql: str | None,
    row_limit: int | None,
    valid_sky_only: bool,
    ra_col: str | None,
    dec_col: str | None,
) -> pa.Table:
    """Project *mapping* and apply ``--where`` / sky / ``--limit`` via DuckDB."""
    from data_lake.export.row_filter import filter_arrow_table

    extra: list[str] = []
    if valid_sky_only:
        ra_name, dec_name = _resolve_sky_output_names(
            table, ra_col=ra_col, dec_col=dec_col,
        )
        extra.append(_sky_predicate_sql(ra_name, dec_name))
    col_sql = ", ".join(
        f"{_sql_ident(src)} AS {_sql_ident(out)}" for src, out in mapping
    )
    return filter_arrow_table(
        table,
        where=where_sql,
        limit=row_limit,
        extra_predicates=extra,
        select_sql=col_sql,
    )


def _stream_lake_via_duckdb(
    tile_paths: Sequence[Path],
    mapping: Sequence[tuple[str, str]],
    *,
    output: Path,
    output_format: OutputFormat,
    valid_sky_only: bool,
    ra_col: str | None,
    dec_col: str | None,
    where_sql: str | None = None,
    row_limit: int | None = None,
    source_ids: np.ndarray | None = None,
    id_predicate: str | None = None,
    order_for_limit: bool = False,
) -> ExtractResult:
    extra: list[str] = []
    if valid_sky_only:
        extra.append(_duckdb_sky_predicate(mapping, ra_col=ra_col, dec_col=dec_col))
    if id_predicate:
        extra.append(id_predicate)
    select_sql = _projected_select_sql(
        mapping,
        "read_parquet($files, hive_partitioning=false)",
        where_sql=where_sql,
        extra_predicates=extra,
        row_limit=row_limit,
        order_for_limit=order_for_limit,
    )

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    out_names = tuple(out for _, out in mapping)
    if not tile_paths:
        empty = pa.table({name: pa.array([], type=pa.null()) for name in out_names})
        with _extract_staging_dir() as tmp:
            tmp_pq = _staging_parquet_path(tmp, output)
            pq.write_table(empty, tmp_pq, compression="zstd", compression_level=_ZSTD_LEVEL)
            if output_format == "parquet":
                shutil.copyfile(tmp_pq, output)
            else:
                write_catalog_extract_from_parquet(
                    tmp_pq, output, output_format=output_format,
                )
        return ExtractResult(n_rows=0, column_names=out_names, output=output)

    files = [str(Path(p).resolve()) for p in tile_paths]
    convert_after = output_format in ("fits", "votable")
    copy_format: OutputFormat = "parquet" if convert_after else output_format

    with _extract_staging_dir() as tmp:
        copy_target = (
            _staging_parquet_path(tmp, output) if convert_after else Path(output).resolve()
        )
        if copy_format == "parquet":
            copy_to = "TO $out (FORMAT PARQUET, COMPRESSION ZSTD)"
        elif copy_format == "csv":
            copy_to = "TO $out (HEADER, DELIMITER ',')"
        else:
            raise ValueError(f"DuckDB lake export does not support {copy_format!r}")

        con = _connect_extract_duckdb(source_ids)
        try:
            # Explicit file list avoids hive_partitioning on Norder=/Dir= paths and
            # dodges COPY placeholder ordering quirks with read vs write targets.
            # Always pass an absolute path: a relative $out would land in cwd.
            con.execute(
                f"COPY ({select_sql}) {copy_to}",
                {"files": files, "out": str(copy_target.resolve())},
            )
        finally:
            con.close()

        if copy_format == "parquet":
            n_rows = _parquet_num_rows(copy_target)
        else:
            # CSV: one pass with DuckDB is still cheaper than Python; footer N/A.
            con = _connect_extract_duckdb(None)
            try:
                n_rows = int(
                    con.execute(
                        "SELECT count(*) FROM read_csv($out, header=true)",
                        {"out": str(copy_target)},
                    ).fetchone()[0]
                )
            finally:
                con.close()

        if convert_after:
            write_catalog_extract_from_parquet(
                copy_target, output, output_format=output_format,
            )

    log.info("DuckDB wrote %d row(s) → %s", n_rows, output)
    return ExtractResult(
        n_rows=n_rows,
        column_names=out_names,
        output=output,
    )


def _filter_tile_via_duckdb(
    tile_path: Path,
    mapping: Sequence[tuple[str, str]],
    *,
    where_sql: str | None,
    row_limit: int | None,
    valid_sky_only: bool,
    ra_col: str | None,
    dec_col: str | None,
    connection,
    id_predicate: str | None = None,
) -> pa.Table:
    extra: list[str] = []
    if valid_sky_only:
        extra.append(_duckdb_sky_predicate(mapping, ra_col=ra_col, dec_col=dec_col))
    if id_predicate:
        extra.append(id_predicate)
    sql = _projected_select_sql(
        mapping,
        "read_parquet(?, hive_partitioning=false)",
        where_sql=where_sql,
        extra_predicates=extra,
        row_limit=row_limit,
    )
    raw = connection.execute(sql, [str(Path(tile_path).resolve())]).arrow()
    if isinstance(raw, pa.RecordBatchReader):
        return raw.read_all()
    return raw


def _tile_extract_chunk(
    tile_path: Path,
    mapping: Sequence[tuple[str, str]],
    *,
    src_cols: Sequence[str],
    valid_sky_only: bool,
    ra_col: str | None,
    dec_col: str | None,
    where_sql: str | None,
    remaining_limit: int | None,
    duckdb_con,
    id_predicate: str | None = None,
) -> pa.Table:
    """Project and filter one lake tile (Arrow path, or DuckDB when WHERE/LIMIT)."""
    if id_predicate is not None and duckdb_con is None:
        raise RuntimeError("target-list filter requires a DuckDB connection")
    if where_sql is not None or remaining_limit is not None or id_predicate is not None:
        chunk = _filter_tile_via_duckdb(
            tile_path,
            mapping,
            where_sql=where_sql,
            row_limit=remaining_limit,
            valid_sky_only=valid_sky_only,
            ra_col=ra_col,
            dec_col=dec_col,
            connection=duckdb_con,
            id_predicate=id_predicate,
        )
        return _canonicalize_column_types(chunk)
    raw = _canonicalize_column_types(_read_catalog_tile(tile_path, src_cols))
    return _process_tile_chunk(
        raw,
        mapping,
        valid_sky_only=valid_sky_only,
        ra_col=ra_col,
        dec_col=dec_col,
    )


def _iter_lake_extract_chunks(
    catalog_root: Path,
    tiles: Sequence[Path],
    mapping: Sequence[tuple[str, str]],
    *,
    src_cols: Sequence[str],
    valid_sky_only: bool,
    ra_col: str | None,
    dec_col: str | None,
    show_progress: bool,
    where_sql: str | None = None,
    row_limit: int | None = None,
    source_ids: np.ndarray | None = None,
    id_predicate: str | None = None,
    link_col: str | None = None,
) -> Iterator[pa.Table]:
    """Yield projected catalog chunks one HEALPix tile at a time."""
    iterator: Iterable[Path] = tiles
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(tiles, unit="tile", desc="extract")
        except ImportError:
            pass

    # Early-exit once every target ID has been seen (target-list only; --where
    # can select rows beyond the ID set so we keep scanning).
    pending_ids: set[int] | None = None
    track_out: str | None = None
    work_mapping = list(mapping)
    if (
        source_ids is not None
        and id_predicate is not None
        and where_sql is None
        and link_col is not None
    ):
        pending_ids = {int(x) for x in np.asarray(source_ids, dtype=np.int64).tolist()}
        if not any(src == link_col for src, _ in work_mapping):
            track_out = "__extract_sid__"
            work_mapping.append((link_col, track_out))
        else:
            track_out = next(out for src, out in work_mapping if src == link_col)

    duckdb_con = None
    remaining = row_limit
    try:
        if where_sql is not None or remaining is not None or id_predicate is not None:
            duckdb_con = _connect_extract_duckdb(source_ids)
        for tile_path in iterator:
            if remaining is not None and remaining <= 0:
                break
            if pending_ids is not None and not pending_ids:
                break
            chunk = _tile_extract_chunk(
                tile_path,
                work_mapping,
                src_cols=[src for src, _ in work_mapping],
                valid_sky_only=valid_sky_only,
                ra_col=ra_col,
                dec_col=dec_col,
                where_sql=where_sql,
                remaining_limit=remaining,
                duckdb_con=duckdb_con,
                id_predicate=id_predicate,
            )
            if chunk.num_rows > 0:
                if pending_ids is not None and track_out is not None and track_out in chunk.column_names:
                    seen = chunk.column(track_out).to_pylist()
                    pending_ids.difference_update(int(v) for v in seen if v is not None)
                if track_out == "__extract_sid__" and track_out in chunk.column_names:
                    chunk = chunk.drop_columns([track_out])
                if remaining is not None:
                    remaining -= chunk.num_rows
                yield chunk
    finally:
        if duckdb_con is not None:
            duckdb_con.close()


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _write_chunks_to_csv(chunks: Iterable[pa.Table], output: Path) -> int:
    import csv

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    n_rows = 0
    wrote_header = False
    with open(output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for chunk in chunks:
            if not wrote_header:
                writer.writerow(list(chunk.schema.names))
                wrote_header = True
            cols = [chunk.column(name).to_pylist() for name in chunk.schema.names]
            for row in zip(*cols):
                writer.writerow([_csv_cell(v) for v in row])
            n_rows += chunk.num_rows
    return n_rows


def _write_chunks_to_parquet(
    chunks: Iterable[pa.Table],
    output: Path,
    *,
    schema: pa.Schema | None = None,
) -> int:
    writer: pq.ParquetWriter | None = None
    writer_schema: pa.Schema | None = schema
    n_rows = 0
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    try:
        for chunk in chunks:
            if writer_schema is not None:
                chunk = _align_table_to_schema(chunk, writer_schema)
            if writer is None:
                writer_schema = writer_schema or chunk.schema
                writer = pq.ParquetWriter(
                    str(output),
                    writer_schema,
                    compression="zstd",
                    compression_level=_ZSTD_LEVEL,
                )
            elif not chunk.schema.equals(writer_schema):
                chunk = _align_table_to_schema(chunk, writer_schema)
            writer.write_table(chunk)
            n_rows += chunk.num_rows
    finally:
        if writer is not None:
            writer.close()
    return n_rows


def _arrow_table_to_astropy(table: pa.Table):
    """Convert Arrow → astropy Table with FITS-safe string columns.

    Nullable Arrow strings become NumPy ``object`` arrays (``None`` + ``<U…>``)
    via ``to_numpy``, which Astropy FITS rejects as mixed object types. Fill
    nulls with empty strings and materialise a fixed-width unicode ndarray.
    """
    from astropy.table import Table

    astropy_tbl = Table()
    for name in table.schema.names:
        col = table.column(name).combine_chunks()
        if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
            values = ["" if v is None else str(v) for v in col.to_pylist()]
            width = max((len(v) for v in values), default=1)
            astropy_tbl[name] = np.asarray(values, dtype=f"<U{max(width, 1)}")
        elif pa.types.is_nested(col.type):
            astropy_tbl[name] = col.to_pylist()
        else:
            try:
                astropy_tbl[name] = col.to_numpy(zero_copy_only=False)
            except (pa.ArrowInvalid, TypeError, ValueError):
                astropy_tbl[name] = col.to_pylist()
    return astropy_tbl


def write_catalog_extract_from_parquet(
    parquet_path: Path | str,
    output: Path | str,
    *,
    output_format: OutputFormat | None = None,
) -> int:
    """Convert a Parquet extract to CSV/FITS/VOTable without loading huge tables at once."""
    parquet_path = Path(parquet_path)
    output = Path(output)
    fmt = infer_output_format(output, output_format)
    pf = pq.ParquetFile(parquet_path)
    n_rows = pf.metadata.num_rows or 0
    _warn_large_non_parquet(n_rows, fmt)

    if fmt == "parquet":
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parquet_path, output)
        return n_rows

    if fmt == "csv":
        import csv

        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        wrote_header = False
        with open(output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            for batch in pf.iter_batches(batch_size=_FITS_STREAM_BATCH_ROWS):
                chunk = pa.Table.from_batches([batch])
                if not wrote_header:
                    writer.writerow(list(chunk.schema.names))
                    wrote_header = True
                cols = [chunk.column(name).to_pylist() for name in chunk.schema.names]
                for row in zip(*cols):
                    writer.writerow([_csv_cell(v) for v in row])
        return n_rows

    from astropy.table import vstack as vstack_tables

    parts = []
    for batch in pf.iter_batches(batch_size=_FITS_STREAM_BATCH_ROWS):
        parts.append(_arrow_table_to_astropy(pa.Table.from_batches([batch])))
    astropy_tbl = parts[0] if len(parts) == 1 else vstack_tables(parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "fits":
        astropy_tbl.write(str(output), format="fits", overwrite=True)
        return n_rows
    if fmt == "votable":
        astropy_tbl.write(str(output), format="votable", overwrite=True)
        return n_rows
    raise ValueError(f"unsupported output format {fmt!r}")


def _extract_tile_worker_count(*, row_limit: int | None = None) -> int:
    """Parallel tile workers for catalog ``--output-dir`` extracts."""
    from data_lake.knobs import extract_tile_workers

    return extract_tile_workers(force_serial=row_limit is not None)


def _write_one_output_dir_tile(
    tile_path: Path,
    *,
    catalog_root: Path,
    output_dir: Path,
    mapping: Sequence[tuple[str, str]],
    src_cols: Sequence[str],
    valid_sky_only: bool,
    ra_col: str | None,
    dec_col: str | None,
    where_sql: str | None,
    source_ids: np.ndarray | None,
    id_predicate: str | None,
) -> tuple[int, Path | None]:
    """Project one tile into ``output_dir``; return ``(n_rows, out_path|None)``."""
    duckdb_con = None
    try:
        if where_sql is not None or id_predicate is not None:
            duckdb_con = _connect_extract_duckdb(source_ids)
        chunk = _tile_extract_chunk(
            tile_path,
            mapping,
            src_cols=src_cols,
            valid_sky_only=valid_sky_only,
            ra_col=ra_col,
            dec_col=dec_col,
            where_sql=where_sql,
            remaining_limit=None,
            duckdb_con=duckdb_con,
            id_predicate=id_predicate,
        )
    finally:
        if duckdb_con is not None:
            duckdb_con.close()
    if chunk.num_rows == 0:
        return 0, None
    rel = tile_path.relative_to(catalog_root)
    out_path = output_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(chunk, out_path, compression="zstd", compression_level=_ZSTD_LEVEL)
    return int(chunk.num_rows), out_path


def _stream_lake_tile_by_tile(
    catalog_root: Path,
    tiles: Sequence[Path],
    mapping: Sequence[tuple[str, str]],
    *,
    src_cols: Sequence[str],
    out_names: Sequence[str],
    output: Path | None,
    output_dir: Path | None,
    output_format: OutputFormat,
    valid_sky_only: bool,
    ra_col: str | None,
    dec_col: str | None,
    show_progress: bool,
    where_sql: str | None = None,
    row_limit: int | None = None,
    source_ids: np.ndarray | None = None,
    id_predicate: str | None = None,
    link_col: str | None = None,
) -> ExtractResult:
    chunk_kwargs = dict(
        src_cols=src_cols,
        valid_sky_only=valid_sky_only,
        ra_col=ra_col,
        dec_col=dec_col,
        show_progress=show_progress,
        where_sql=where_sql,
        row_limit=row_limit,
        source_ids=source_ids,
        id_predicate=id_predicate,
        link_col=link_col,
    )
    chunk_iter = _iter_lake_extract_chunks(
        catalog_root,
        tiles,
        mapping,
        **chunk_kwargs,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        n_workers = _extract_tile_worker_count(row_limit=row_limit)
        if n_workers > 1 and row_limit is None:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            n_rows = 0
            n_written_tiles = 0
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [
                    pool.submit(
                        _write_one_output_dir_tile,
                        tile_path,
                        catalog_root=catalog_root,
                        output_dir=output_dir,
                        mapping=mapping,
                        src_cols=src_cols,
                        valid_sky_only=valid_sky_only,
                        ra_col=ra_col,
                        dec_col=dec_col,
                        where_sql=where_sql,
                        source_ids=source_ids,
                        id_predicate=id_predicate,
                    )
                    for tile_path in tiles
                ]
                iterator = as_completed(futures)
                if show_progress:
                    try:
                        from tqdm.auto import tqdm

                        iterator = tqdm(
                            iterator, total=len(futures), unit="tile", desc="extract",
                        )
                    except ImportError:
                        pass
                for fut in iterator:
                    rows, out_path = fut.result()
                    n_rows += rows
                    if out_path is not None:
                        n_written_tiles += 1
        else:
            n_rows = 0
            n_written_tiles = 0
            iterator: Iterable[Path] = tiles
            if show_progress:
                try:
                    from tqdm.auto import tqdm

                    iterator = tqdm(tiles, unit="tile", desc="extract")
                except ImportError:
                    pass
            duckdb_con = None
            remaining = row_limit
            pending_ids: set[int] | None = None
            if (
                source_ids is not None
                and id_predicate is not None
                and where_sql is None
                and link_col is not None
            ):
                pending_ids = {
                    int(x) for x in np.asarray(source_ids, dtype=np.int64).tolist()
                }
            try:
                if where_sql is not None or remaining is not None or id_predicate is not None:
                    duckdb_con = _connect_extract_duckdb(source_ids)
                for tile_path in iterator:
                    if remaining is not None and remaining <= 0:
                        break
                    if pending_ids is not None and not pending_ids:
                        break
                    chunk = _tile_extract_chunk(
                        tile_path,
                        mapping,
                        src_cols=src_cols,
                        valid_sky_only=valid_sky_only,
                        ra_col=ra_col,
                        dec_col=dec_col,
                        where_sql=where_sql,
                        remaining_limit=remaining,
                        duckdb_con=duckdb_con,
                        id_predicate=id_predicate,
                    )
                    if chunk.num_rows == 0:
                        continue
                    if pending_ids is not None and link_col is not None:
                        # Best-effort: only when the projected chunk still has the link col.
                        if link_col in chunk.column_names:
                            pending_ids.difference_update(
                                int(v)
                                for v in chunk.column(link_col).to_pylist()
                                if v is not None
                            )
                        elif any(out == link_col for _, out in mapping):
                            out_name = next(out for src, out in mapping if src == link_col)
                            pending_ids.difference_update(
                                int(v)
                                for v in chunk.column(out_name).to_pylist()
                                if v is not None
                            )
                    if remaining is not None:
                        remaining -= chunk.num_rows
                    rel = tile_path.relative_to(catalog_root)
                    out_path = output_dir / rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(
                        chunk, out_path, compression="zstd", compression_level=_ZSTD_LEVEL,
                    )
                    n_rows += chunk.num_rows
                    n_written_tiles += 1
            finally:
                if duckdb_con is not None:
                    duckdb_con.close()
        log.info(
            "Streamed %d row(s) from %d tile(s) → %s",
            n_rows,
            n_written_tiles,
            output_dir,
        )
        return ExtractResult(
            n_rows=n_rows,
            column_names=tuple(out_names),
            output_dir=output_dir,
            n_tiles=n_written_tiles,
        )

    assert output is not None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    unified_schema = _unified_schema_for_mapping(
        tiles, mapping, catalog_root=catalog_root,
    ) if tiles else None

    if output_format in ("fits", "votable"):
        with _extract_staging_dir() as tmp:
            tmp_pq = _staging_parquet_path(tmp, output)
            n_rows = _write_chunks_to_parquet(
                _iter_lake_extract_chunks(
                    catalog_root,
                    tiles,
                    mapping,
                    **chunk_kwargs,
                ),
                tmp_pq,
                schema=unified_schema,
            )
            write_catalog_extract_from_parquet(tmp_pq, output, output_format=output_format)
    elif output_format == "csv":
        n_rows = _write_chunks_to_csv(chunk_iter, output)
    else:
        n_rows = _write_chunks_to_parquet(chunk_iter, output, schema=unified_schema)

    log.info("Streamed %d row(s) from %d tile(s) → %s", n_rows, len(tiles), output)
    return ExtractResult(
        n_rows=n_rows,
        column_names=tuple(out_names),
        output=output,
        n_tiles=len(tiles),
    )


def extract_from_lake_catalog(
    lake_root: Path | str,
    survey: str,
    specs: Sequence[str],
    *,
    all_columns: bool = False,
    norder: int | None = None,
    valid_sky_only: bool = False,
    ra_col: str | None = None,
    dec_col: str | None = None,
    where_sql: str | None = None,
    row_limit: int | None = None,
    source_ids: np.ndarray | None = None,
) -> pa.Table:
    """In-memory lake extract (small catalogs / tests only)."""
    catalog_root = _lake_catalog_root(lake_root, survey)
    tiles = list(iter_lake_catalog_tiles(catalog_root, norder=_lake_norder(catalog_root, norder)))
    schema_names = pq.read_schema(str(tiles[0])).names
    mapping = resolve_column_specs(
        schema_names,
        specs_from_schema(schema_names, all_columns=all_columns, specs=specs),
    )

    with _extract_staging_dir() as tmp:
        out = tmp / "extract.parquet"
        result = stream_extract_from_lake_catalog(
            lake_root,
            survey,
            specs,
            all_columns=all_columns,
            output=out,
            norder=norder,
            valid_sky_only=valid_sky_only,
            ra_col=ra_col,
            dec_col=dec_col,
            engine="tiles",
            where_sql=where_sql,
            row_limit=row_limit,
            source_ids=source_ids,
        )
        if result.n_rows == 0:
            return pa.table({out_name: pa.array([], type=pa.float64()) for _, out_name in mapping})
        return pq.read_table(out)


def extract_from_catalog_path(
    path: Path | str,
    specs: Sequence[str],
    *,
    all_columns: bool = False,
    valid_sky_only: bool = False,
    ra_col: str | None = None,
    dec_col: str | None = None,
    where_sql: str | None = None,
    row_limit: int | None = None,
) -> pa.Table:
    """Read one catalog file and return the selected columns (in memory)."""
    table = _read_source_table(Path(path))
    effective_specs = specs_from_schema(
        table.schema.names, all_columns=all_columns, specs=specs,
    )
    mapping = resolve_column_specs(table.schema.names, effective_specs)
    if where_sql is not None or row_limit is not None:
        return _apply_extract_filter(
            table,
            mapping,
            where_sql=where_sql,
            row_limit=row_limit,
            valid_sky_only=valid_sky_only,
            ra_col=ra_col,
            dec_col=dec_col,
        )
    out = select_catalog_columns(table, effective_specs)
    if valid_sky_only:
        out = filter_valid_sky_rows(out, ra_col=ra_col, dec_col=dec_col)
    return out


def stream_extract_from_fits_catalog(
    path: Path | str,
    specs: Sequence[str],
    output: Path,
    *,
    all_columns: bool = False,
    valid_sky_only: bool = False,
    ra_col: str | None = None,
    dec_col: str | None = None,
    batch_rows: int = _FITS_STREAM_BATCH_ROWS,
    fits_memmap: str = "auto",
    where_sql: str | None = None,
    row_limit: int | None = None,
) -> ExtractResult:
    """Stream a large FITS BINTABLE to Parquet in row batches."""
    from astropy.table import Table

    from data_lake.ingest.fits_to_parquet import _bintable_hdu_index
    from data_lake.io.fits_read import default_fits_read_policy, open_fits

    path = Path(path)
    if path.suffix.lower() not in {".fit", ".fits", ".fz"} and not path.name.lower().endswith(".fits.gz"):
        raise ValueError(f"--streaming requires FITS input, got {path.name}")

    writer: pq.ParquetWriter | None = None
    n_rows = 0
    out_names: tuple[str, ...] = ()

    from data_lake.io.fits_read import open_fits

    with open_fits(path, default_fits_read_policy(fits_memmap)) as hdul:
        idx = _bintable_hdu_index(hdul)
        hdu = hdul[idx]
        data = hdu.data
        if data is None:
            raise ValueError(f"{path}: BINTABLE HDU {idx} has no data")
        n_total = len(data)
        names = data.dtype.names or ()
        effective_specs = specs_from_schema(
            list(names), all_columns=all_columns, specs=specs,
        )
        mapping = resolve_column_specs(list(names), effective_specs)
        src_cols = [src for src, _ in mapping]
        out_names = tuple(out for _, out in mapping)
        batch_cols = list(names) if where_sql else src_cols
        remaining = row_limit

        for start in range(0, n_total, batch_rows):
            if remaining is not None and remaining <= 0:
                break
            stop = min(start + batch_rows, n_total)
            batch_tbl = Table({col: data[col][start:stop] for col in batch_cols})
            from data_lake.ingest.fits_to_parquet import _astropy_table_to_arrow

            raw_arrow = _astropy_table_to_arrow(batch_tbl)
            if where_sql is not None or remaining is not None:
                chunk = _apply_extract_filter(
                    raw_arrow,
                    mapping,
                    where_sql=where_sql,
                    row_limit=remaining,
                    valid_sky_only=valid_sky_only,
                    ra_col=ra_col,
                    dec_col=dec_col,
                )
            else:
                chunk = _process_tile_chunk(
                    raw_arrow,
                    mapping,
                    valid_sky_only=valid_sky_only,
                    ra_col=ra_col,
                    dec_col=dec_col,
                )
            if chunk.num_rows == 0:
                continue
            if remaining is not None:
                remaining -= chunk.num_rows
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(
                    str(output),
                    chunk.schema,
                    compression="zstd",
                    compression_level=_ZSTD_LEVEL,
                )
            writer.write_table(chunk)
            n_rows += chunk.num_rows
            log.info("Streamed rows %d–%d of %d from %s", start, stop, n_total, path.name)

    if writer is not None:
        writer.close()
    elif output.exists():
        output.unlink()

    return ExtractResult(n_rows=n_rows, column_names=out_names, output=output)


def extract_from_catalog_paths(
    paths: Iterable[Path | str],
    specs: Sequence[str],
    *,
    all_columns: bool = False,
    valid_sky_only: bool = False,
    ra_col: str | None = None,
    dec_col: str | None = None,
    add_input_path: bool = False,
    streaming: bool = False,
    output: Path | None = None,
    batch_rows: int = _FITS_STREAM_BATCH_ROWS,
    where_sql: str | None = None,
    row_limit: int | None = None,
) -> pa.Table | ExtractResult:
    """Concatenate extracts from multiple catalog files."""
    path_list = [Path(p) for p in paths]
    if not path_list:
        raise ValueError("no catalog paths to read")

    if streaming:
        if output is None:
            raise ValueError("streaming file extract requires output path")
        if len(path_list) != 1:
            raise ValueError("streaming FITS extract supports one input file at a time")
        return stream_extract_from_fits_catalog(
            path_list[0],
            specs,
            output,
            all_columns=all_columns,
            valid_sky_only=valid_sky_only,
            ra_col=ra_col,
            dec_col=dec_col,
            batch_rows=batch_rows,
            where_sql=where_sql,
            row_limit=row_limit,
        )

    tables: list[pa.Table] = []
    remaining = row_limit
    for p in path_list:
        if remaining is not None and remaining <= 0:
            break
        chunk = extract_from_catalog_path(
            p,
            specs,
            all_columns=all_columns,
            valid_sky_only=valid_sky_only,
            ra_col=ra_col,
            dec_col=dec_col,
            where_sql=where_sql,
            row_limit=remaining,
        )
        if add_input_path:
            chunk = chunk.append_column(
                "input_path",
                pa.array([str(p.resolve())] * chunk.num_rows, type=pa.string()),
            )
        tables.append(chunk)
        log.info("Read %d row(s) from %s", chunk.num_rows, p.name)
        if remaining is not None:
            remaining -= chunk.num_rows

    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables, promote_options="default")


def infer_output_format(path: Path, explicit: OutputFormat | None) -> OutputFormat:
    if explicit is not None:
        return explicit
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix == ".csv" or name.endswith(".csv.gz"):
        return "csv"
    if suffix in {".fits", ".fit", ".fz"} or name.endswith(".fits.gz"):
        return "fits"
    if suffix in {".xml", ".vot", ".votable"}:
        return "votable"
    return "parquet"


def _warn_large_non_parquet(n_rows: int, fmt: OutputFormat) -> None:
    if fmt != "parquet" and n_rows >= _LARGE_OUTPUT_ROW_WARN:
        log.warning(
            "%d rows to %s may be slow or fail; prefer Parquet or --output-dir tiles.",
            n_rows,
            fmt,
        )


def write_catalog_extract(
    table: pa.Table,
    output: Path | str,
    *,
    output_format: OutputFormat | None = None,
) -> None:
    """Write an extracted table to Parquet, CSV, FITS, or VOTable."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fmt = infer_output_format(output, output_format)
    _warn_large_non_parquet(table.num_rows, fmt)

    if fmt == "parquet":
        pq.write_table(table, str(output), compression="zstd", compression_level=_ZSTD_LEVEL)
        return

    if fmt == "csv":
        import csv

        with open(output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(table.schema.names)
            cols = [table.column(n).to_pylist() for n in table.schema.names]
            for row in zip(*cols):
                writer.writerow([_csv_cell(v) for v in row])
        return

    astropy_tbl = _arrow_table_to_astropy(table)

    if fmt == "fits":
        astropy_tbl.write(str(output), format="fits", overwrite=True)
        return
    if fmt == "votable":
        astropy_tbl.write(str(output), format="votable", overwrite=True)
        return

    raise ValueError(f"unsupported output format {fmt!r}")


def extract_catalog(
    output: Path | str | None = None,
    specs: Sequence[str] | None = None,
    *,
    all_columns: bool = False,
    paths: Sequence[Path | str] | None = None,
    lake_root: Path | str | None = None,
    survey: str | None = None,
    output_dir: Path | str | None = None,
    norder: int | None = None,
    valid_sky_only: bool = False,
    ra_col: str | None = None,
    dec_col: str | None = None,
    add_input_path: bool = False,
    output_format: OutputFormat | None = None,
    engine: LakeEngine = "auto",
    streaming: bool = False,
    batch_rows: int = _FITS_STREAM_BATCH_ROWS,
    show_progress: bool = False,
    require_homogenized: bool = False,
    write_homogenize_provenance: bool = True,
    where_sql: str | None = None,
    row_limit: int | None = None,
    source_ids: np.ndarray | None = None,
) -> ExtractResult | pa.Table:
    """Extract columns and write output; returns summary or small in-memory table."""
    if all_columns and specs:
        raise ValueError("use all_columns or column specs, not both")
    if not all_columns and not specs:
        raise ValueError("pass column specs or all_columns=True")
    if row_limit is not None and int(row_limit) < 1:
        raise ValueError("--limit must be a positive integer")
    if source_ids is not None and where_sql:
        raise ValueError("use --target-list or --where, not both")
    if source_ids is not None and lake_root is None:
        raise ValueError("--target-list requires an ingested catalog (--survey)")
    column_specs: Sequence[str] = specs or ()

    if lake_root is not None:
        if not survey:
            raise ValueError("--survey is required with --lake-root")
        if paths:
            raise ValueError("pass catalog paths or --lake-root/--survey, not both")
        result = stream_extract_from_lake_catalog(
            lake_root,
            survey,
            column_specs,
            all_columns=all_columns,
            output=Path(output) if output is not None else None,
            output_dir=Path(output_dir) if output_dir is not None else None,
            norder=norder,
            valid_sky_only=valid_sky_only,
            ra_col=ra_col,
            dec_col=dec_col,
            output_format=output_format,
            engine=engine,
            show_progress=show_progress,
            require_homogenized=require_homogenized,
            write_homogenize_provenance=write_homogenize_provenance,
            where_sql=where_sql,
            row_limit=row_limit,
            source_ids=source_ids,
        )
        return result

    if not paths:
        raise ValueError("catalog path(s) or --lake-root required")
    if output_dir is not None:
        raise ValueError("--output-dir is only valid with --lake-root")

    raw_result = extract_from_catalog_paths(
        paths,
        column_specs,
        all_columns=all_columns,
        valid_sky_only=valid_sky_only,
        ra_col=ra_col,
        dec_col=dec_col,
        add_input_path=add_input_path,
        streaming=streaming,
        output=Path(output) if output is not None else None,
        batch_rows=batch_rows,
        where_sql=where_sql,
        row_limit=row_limit,
    )
    if isinstance(raw_result, ExtractResult):
        return raw_result

    if output is None:
        return raw_result
    write_catalog_extract(raw_result, output, output_format=output_format)
    return raw_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import click

from data_lake.cli_utils import config_option, load_optional_config, progress_option, configure_cli_logging, logging_options, resolve_log_level, validate_quiet_verbose


@click.command("dl-extract-catalog")
@config_option
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
@click.option(
    "--file-list",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Text file with one catalog path per line.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(path_type=Path),
    help="Single output file (Parquet recommended). Not used with --output-dir.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Write HATS tile tree (bounded RAM; best for 100M+ row lake catalogs).",
)
@click.option(
    "-c",
    "--column",
    "columns",
    multiple=True,
    help=(
        "Column to export (repeatable). Pass one NAME / NAME:alias per -c, or a "
        "comma-separated list (e.g. -c RA,DEC,TARGETID or -c ra:RA,dec:DEC)."
    ),
)
@click.option(
    "--all-columns",
    is_flag=True,
    help="Export every column in the catalog schema (mutually exclusive with -c).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["parquet", "csv", "fits", "votable"], case_sensitive=False),
    default=None,
    help="Output format (default: infer from --output suffix). Lake export supports all formats.",
)
@click.option(
    "--lake-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Data lake root (default: lake.root from $DATA_LAKE_CONFIG when --survey is set).",
)
@click.option("--survey", default=None, help="Survey name under catalogs/ (with --lake-root).")
@click.option(
    "--from-product",
    default=None,
    help="Export a homogenized catalog product (alias for --survey with subtype check).",
)
@click.option(
    "--require-subtype",
    default=None,
    help="Require catalog_info product_subtype (e.g. homogenized).",
)
@click.option(
    "--no-homogenize-provenance",
    is_flag=True,
    help="Do not write extract_provenance.json / sidecar for homogenized exports.",
)
@click.option("--norder", type=int, default=None, help="HEALPix order (default: catalog_info.json).")
@click.option(
    "--engine",
    type=click.Choice(["auto", "tiles", "duckdb"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Lake export engine: DuckDB COPY (auto) or Arrow tile streaming.",
)
@click.option(
    "--valid-sky-only",
    is_flag=True,
    help="Drop rows with non-finite or sentinel RA/Dec (SDSS -9999, |Dec|>90).",
)
@click.option("--ra-col", default=None, help="RA column for --valid-sky-only (default: auto).")
@click.option("--dec-col", default=None, help="Dec column for --valid-sky-only (default: auto).")
@click.option(
    "--target-list",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "File of source IDs to keep (FITS, CSV, Parquet, or plain text). "
        "IDs are resolved with the same link-id rule as ingest and matched to "
        "catalog _source_id. Requires --survey. Mutually exclusive with --where."
    ),
)
@click.option(
    "--target-id-col",
    default="TARGETID",
    show_default=True,
    help=(
        "Column in --target-list carrying source IDs, or a comma-separated "
        "composite matching catalog --link-id-col (e.g. TARGETID,SURVEY,PROGRAM). "
        "Ignored for plain-text files."
    ),
)
@click.option(
    "--where",
    "where_sql",
    default=None,
    help=(
        "DuckDB boolean expression over catalog columns "
        "(lake or raw files). "
        "Trusted-operator input only (see docs/security.md). "
        "Example: \"upper(trim(SPECTYPE)) = 'STAR'\"."
    ),
)
@click.option(
    "--limit",
    "row_limit",
    type=int,
    default=None,
    help="Maximum rows to write after --where (first rows in tile order).",
)
@click.option(
    "--add-input-path",
    is_flag=True,
    help="Add input_path column when merging multiple catalog files.",
)
@click.option(
    "--streaming",
    is_flag=True,
    help="Stream a large FITS BINTABLE to Parquet in row batches (one input file).",
)
@click.option(
    "--batch-rows",
    default=_FITS_STREAM_BATCH_ROWS,
    show_default=True,
    type=int,
    help="Row batch size for --streaming FITS export.",
)
@progress_option
@logging_options
def cli(
    paths: tuple[Path, ...],
    file_list: Path | None,
    output: Path | None,
    output_dir: Path | None,
    columns: tuple[str, ...],
    all_columns: bool,
    output_format: str | None,
    lake_root: Path | None,
    survey: str | None,
    from_product: str | None,
    require_subtype: str | None,
    no_homogenize_provenance: bool,
    norder: int | None,
    engine: str,
    valid_sky_only: bool,
    ra_col: str | None,
    dec_col: str | None,
    target_list: Path | None,
    target_id_col: str,
    where_sql: str | None,
    row_limit: int | None,
    add_input_path: bool,
    streaming: bool,
    batch_rows: int,
    show_progress: bool,
    log_file: Path | None,
    quiet: bool,
    verbose: bool,
    config_path: Path | None,
) -> None:
    """Export selected catalog columns for catalog–catalog association (sky matching).

    Pass --target-list to keep rows whose _source_id matches an external ID file
    (same resolution as dl-extract-spectra-subset). Mutually exclusive with --where.
    """
    validate_quiet_verbose(quiet, verbose)
    configure_cli_logging(
        level=resolve_log_level(quiet=quiet, verbose=verbose),
        quiet=quiet,
        log_file=log_file,
    )

    all_paths: list[Path] = list(paths)
    if file_list is not None:
        for line in file_list.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                all_paths.append(Path(line))

    if output is None and output_dir is None:
        raise click.ClickException("Pass -o/--output or --output-dir.")
    try:
        columns = tuple(expand_column_args(columns))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if all_columns and columns:
        raise click.ClickException("Use --all-columns or -c, not both.")
    if not all_columns and not columns:
        raise click.ClickException("Pass -c or --all-columns.")
    if target_list is not None and where_sql:
        raise click.ClickException("Use --target-list or --where, not both.")

    cfg = load_optional_config(config_path)
    if lake_root is None and survey is not None:
        if cfg is None:
            raise click.ClickException(
                "--lake-root is required when no lake config is provided. "
                "Either pass it explicitly, set $DATA_LAKE_CONFIG, or use --config."
            )
        lake_root = cfg.lake.root

    if lake_root is None and not all_paths:
        raise click.ClickException(
            "Pass catalog path(s), --file-list, or --survey "
            "(with --lake-root or $DATA_LAKE_CONFIG)."
        )
    if lake_root is not None and not survey and not from_product:
        raise click.ClickException(
            "--survey or --from-product is required with --lake-root."
        )
    if survey and from_product:
        raise click.ClickException("Use either --survey or --from-product, not both.")
    if from_product:
        survey = from_product
    require_homogenized = (
        require_subtype == "homogenized"
        or from_product is not None
    )
    if require_subtype and require_subtype != "homogenized":
        raise click.ClickException(
            f"Unsupported --require-subtype {require_subtype!r} (only homogenized)"
        )
    if lake_root is not None and all_paths:
        raise click.ClickException("Use either input paths or --lake-root/--survey, not both.")
    if output is not None and output_dir is not None:
        raise click.ClickException("Pass -o/--output or --output-dir, not both.")
    if streaming and lake_root is not None:
        raise click.ClickException("--streaming is for raw FITS input, not --lake-root.")
    if streaming and len(all_paths) != 1:
        raise click.ClickException("--streaming requires exactly one input FITS file.")
    if row_limit is not None and row_limit < 1:
        raise click.ClickException("--limit must be a positive integer.")
    if target_list is not None and lake_root is None:
        raise click.ClickException(
            "--target-list requires an ingested catalog (--survey)."
        )
    if where_sql:
        from data_lake.export.row_filter import validate_predicate

        try:
            where_sql = validate_predicate(where_sql)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    source_ids = None
    if target_list is not None:
        from data_lake.export.spectra_subset import _read_target_ids

        try:
            loaded = _read_target_ids(target_list, target_id_col)
        except (ValueError, KeyError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc
        source_ids = np.unique(np.asarray(loaded, dtype=np.int64))
        log.info(
            "Loaded %d source ID(s) (%d unique) from %s (column=%s).",
            int(np.asarray(loaded).size),
            int(source_ids.size),
            target_list,
            target_id_col,
        )

    fmt: OutputFormat | None = (
        output_format.lower()  # type: ignore[assignment]
        if output_format is not None
        else None
    )

    result = extract_catalog(
        output=output,
        specs=list(columns) if columns else None,
        all_columns=all_columns,
        paths=all_paths or None,
        lake_root=lake_root,
        survey=survey,
        output_dir=output_dir,
        norder=norder,
        valid_sky_only=valid_sky_only,
        ra_col=ra_col,
        dec_col=dec_col,
        add_input_path=add_input_path,
        output_format=fmt,
        engine=engine.lower(),  # type: ignore[arg-type]
        streaming=streaming,
        batch_rows=batch_rows,
        show_progress=show_progress,
        require_homogenized=require_homogenized,
        write_homogenize_provenance=not no_homogenize_provenance,
        where_sql=where_sql,
        row_limit=row_limit,
        source_ids=source_ids,
    )

    if isinstance(result, ExtractResult):
        dest = result.output_dir or result.output
        extra = f", {result.n_tiles} tile(s)" if result.n_tiles else ""
        click.echo(
            f"Wrote {result.n_rows} row(s), {len(result.column_names)} column(s)"
            f"{extra} → {dest}"
        )
        if lake_root is not None and survey and not no_homogenize_provenance:
            from data_lake.schema_registry import PRODUCT_SUBTYPE_HOMOGENIZED

            info = read_lake_catalog_info(lake_root, survey)
            if info.get("product_subtype") == PRODUCT_SUBTYPE_HOMOGENIZED:
                if output_dir is not None:
                    sidecar = Path(output_dir) / "extract_provenance.json"
                elif output is not None:
                    sidecar = output.with_name(output.name + ".homogenize_provenance.json")
                else:
                    sidecar = None
                if sidecar is not None and sidecar.is_file():
                    click.echo(f"  homogenize provenance → {sidecar}")
    else:
        click.echo(
            f"Wrote {result.num_rows} row(s), {len(result.schema.names)} column(s) → {output}"
        )
