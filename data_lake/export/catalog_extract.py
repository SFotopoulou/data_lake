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
        -c TARGETID -c RA -c DEC

    dl-extract-catalog --lake-root /data/lake --survey DESI_DR1 \\
        -c source_id -c ra -c dec -o desi_sky.parquet

    # 290M+ rows: tiled export (bounded RAM, parallel-friendly)
    dl-extract-catalog --lake-root /data/lake --survey GAIA_DR3 \\
        --output-dir /scratch/gaia_sky/ -c source_id -c ra -c dec
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

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


@dataclass(frozen=True)
class ExtractResult:
    """Summary of a streaming extract (no in-memory table retained)."""

    n_rows: int
    column_names: tuple[str, ...]
    output: Path | None = None
    output_dir: Path | None = None
    n_tiles: int = 0


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


def _align_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Cast *table* to *schema* so a multi-tile ``ParquetWriter`` stays consistent."""
    arrays: list[pa.ChunkedArray] = []
    for field in schema:
        col = table.column(field.name).combine_chunks()
        if col.type != field.type:
            col = pc.cast(col, field.type, safe=False)
        arrays.append(col)
    return pa.Table.from_arrays(arrays, schema=schema)


def _lake_catalog_root(lake_root: Path | str, survey: str) -> Path:
    root = Path(lake_root) / "catalogs" / survey
    if not root.is_dir():
        raise FileNotFoundError(f"Catalog not found: {root}")
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

    schema_names = pq.read_schema(str(tiles[0])).names
    mapping = resolve_column_specs(schema_names, specs)
    src_cols = [src for src, _ in mapping]
    out_names = tuple(out for _, out in mapping)

    fmt: OutputFormat = "parquet"
    if output is not None:
        fmt = infer_output_format(Path(output), output_format)
    if output_dir is not None and fmt != "parquet":
        raise ValueError(
            "--output-dir writes a HATS Parquet tree only; use -o file.csv or file.fits "
            "for a single-file export."
        )

    use_duckdb = (
        tiles
        and output is not None
        and output_dir is None
        and fmt in ("parquet", "csv")
        and (engine == "duckdb" or engine == "auto")
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


def _duckdb_sky_predicate(
    mapping: Sequence[tuple[str, str]],
    *,
    ra_col: str | None,
    dec_col: str | None,
) -> str:
    dummy = pa.table({out: pa.array([0.0], type=pa.float64()) for _, out in mapping[:1]})
    ra_name, dec_name = _resolve_sky_output_names(dummy, ra_col=ra_col, dec_col=dec_col)
    ra_sql = _sql_ident(ra_name)
    dec_sql = _sql_ident(dec_name)
    return (
        f"isfinite({ra_sql}) AND isfinite({dec_sql}) "
        f"AND {dec_sql} BETWEEN -90 AND 90 "
        f"AND {ra_sql} > -9000 AND {dec_sql} > -9000"
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
) -> ExtractResult:
    import duckdb

    col_sql = ", ".join(
        f"{_sql_ident(src)} AS {_sql_ident(out)}" for src, out in mapping
    )
    where = ""
    if valid_sky_only:
        where = f" WHERE {_duckdb_sky_predicate(mapping, ra_col=ra_col, dec_col=dec_col)}"

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    files = [str(Path(p).resolve()) for p in tile_paths]
    if output_format == "parquet":
        copy_to = "TO $out (FORMAT PARQUET, COMPRESSION ZSTD)"
        count_from = "read_parquet($out)"
    elif output_format == "csv":
        copy_to = "TO $out (HEADER, DELIMITER ',')"
        count_from = "read_csv($out, header=true)"
    else:
        raise ValueError(f"DuckDB lake export does not support {output_format!r}")

    con = duckdb.connect(database=":memory:")
    try:
        # Explicit file list avoids hive_partitioning on Norder=/Dir= paths and
        # dodges COPY placeholder ordering quirks with read vs write targets.
        con.execute(
            f"COPY (SELECT {col_sql} FROM read_parquet($files, hive_partitioning=false)"
            f"{where}) {copy_to}",
            {"files": files, "out": str(output)},
        )
        n_rows = int(
            con.execute(f"SELECT count(*) FROM {count_from}", {"out": str(output)}).fetchone()[0]
        )
    finally:
        con.close()

    out_names = tuple(out for _, out in mapping)
    log.info("DuckDB wrote %d row(s) → %s", n_rows, output)
    return ExtractResult(
        n_rows=n_rows,
        column_names=out_names,
        output=output,
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
) -> Iterator[pa.Table]:
    """Yield projected catalog chunks one HEALPix tile at a time."""
    iterator: Iterable[Path] = tiles
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(tiles, unit="tile", desc="extract")
        except ImportError:
            pass

    for tile_path in iterator:
        raw = _canonicalize_column_types(_read_catalog_tile(tile_path, src_cols))
        chunk = _process_tile_chunk(
            raw,
            mapping,
            valid_sky_only=valid_sky_only,
            ra_col=ra_col,
            dec_col=dec_col,
        )
        if chunk.num_rows > 0:
            yield chunk


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
) -> int:
    writer: pq.ParquetWriter | None = None
    writer_schema: pa.Schema | None = None
    n_rows = 0
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    try:
        for chunk in chunks:
            if writer is None:
                writer_schema = chunk.schema
                writer = pq.ParquetWriter(
                    str(output),
                    writer_schema,
                    compression="zstd",
                    compression_level=_ZSTD_LEVEL,
                )
            elif writer_schema is not None and not chunk.schema.equals(writer_schema):
                chunk = _align_table_to_schema(chunk, writer_schema)
            writer.write_table(chunk)
            n_rows += chunk.num_rows
    finally:
        if writer is not None:
            writer.close()
    return n_rows


def _arrow_table_to_astropy(table: pa.Table):
    from astropy.table import Table

    astropy_tbl = Table()
    for name in table.schema.names:
        col = table.column(name).combine_chunks()
        if pa.types.is_nested(col.type):
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
) -> ExtractResult:
    chunk_iter = _iter_lake_extract_chunks(
        catalog_root,
        tiles,
        mapping,
        src_cols=src_cols,
        valid_sky_only=valid_sky_only,
        ra_col=ra_col,
        dec_col=dec_col,
        show_progress=show_progress,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        n_rows = 0
        n_written_tiles = 0
        iterator: Iterable[Path] = tiles
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(tiles, unit="tile", desc="extract")
            except ImportError:
                pass
        for tile_path in iterator:
            raw = _canonicalize_column_types(_read_catalog_tile(tile_path, src_cols))
            chunk = _process_tile_chunk(
                raw,
                mapping,
                valid_sky_only=valid_sky_only,
                ra_col=ra_col,
                dec_col=dec_col,
            )
            if chunk.num_rows == 0:
                continue
            rel = tile_path.relative_to(catalog_root)
            out_path = output_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(chunk, out_path, compression="zstd", compression_level=_ZSTD_LEVEL)
            n_rows += chunk.num_rows
            n_written_tiles += 1
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

    if output_format in ("fits", "votable"):
        with tempfile.TemporaryDirectory(prefix="dl_extract_") as tmp:
            tmp_pq = Path(tmp) / "lake_extract.parquet"
            n_rows = _write_chunks_to_parquet(
                _iter_lake_extract_chunks(
                    catalog_root,
                    tiles,
                    mapping,
                    src_cols=src_cols,
                    valid_sky_only=valid_sky_only,
                    ra_col=ra_col,
                    dec_col=dec_col,
                    show_progress=show_progress,
                ),
                tmp_pq,
            )
            write_catalog_extract_from_parquet(tmp_pq, output, output_format=output_format)
    elif output_format == "csv":
        n_rows = _write_chunks_to_csv(chunk_iter, output)
    else:
        n_rows = _write_chunks_to_parquet(chunk_iter, output)

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
    norder: int | None = None,
    valid_sky_only: bool = False,
    ra_col: str | None = None,
    dec_col: str | None = None,
) -> pa.Table:
    """In-memory lake extract (small catalogs / tests only)."""
    import tempfile

    catalog_root = _lake_catalog_root(lake_root, survey)
    tiles = list(iter_lake_catalog_tiles(catalog_root, norder=_lake_norder(catalog_root, norder)))
    mapping = resolve_column_specs(pq.read_schema(str(tiles[0])).names, specs)

    with tempfile.TemporaryDirectory(prefix="dl_extract_") as tmp:
        out = Path(tmp) / "extract.parquet"
        result = stream_extract_from_lake_catalog(
            lake_root,
            survey,
            specs,
            output=out,
            norder=norder,
            valid_sky_only=valid_sky_only,
            ra_col=ra_col,
            dec_col=dec_col,
            engine="tiles",
        )
        if result.n_rows == 0:
            return pa.table({out_name: pa.array([], type=pa.float64()) for _, out_name in mapping})
        return pq.read_table(out)


def extract_from_catalog_path(
    path: Path | str,
    specs: Sequence[str],
    *,
    valid_sky_only: bool = False,
    ra_col: str | None = None,
    dec_col: str | None = None,
) -> pa.Table:
    """Read one catalog file and return the selected columns (in memory)."""
    table = _read_source_table(Path(path))
    out = select_catalog_columns(table, specs)
    if valid_sky_only:
        out = filter_valid_sky_rows(out, ra_col=ra_col, dec_col=dec_col)
    return out


def stream_extract_from_fits_catalog(
    path: Path | str,
    specs: Sequence[str],
    output: Path,
    *,
    valid_sky_only: bool = False,
    ra_col: str | None = None,
    dec_col: str | None = None,
    batch_rows: int = _FITS_STREAM_BATCH_ROWS,
    fits_memmap: str = "auto",
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
        mapping = resolve_column_specs(list(names), specs)
        src_cols = [src for src, _ in mapping]
        out_names = tuple(out for _, out in mapping)

        for start in range(0, n_total, batch_rows):
            stop = min(start + batch_rows, n_total)
            batch_tbl = Table({col: data[col][start:stop] for col in src_cols})
            from data_lake.ingest.fits_to_parquet import _astropy_table_to_arrow

            chunk = _process_tile_chunk(
                _astropy_table_to_arrow(batch_tbl),
                mapping,
                valid_sky_only=valid_sky_only,
                ra_col=ra_col,
                dec_col=dec_col,
            )
            if chunk.num_rows == 0:
                continue
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
    valid_sky_only: bool = False,
    ra_col: str | None = None,
    dec_col: str | None = None,
    add_input_path: bool = False,
    streaming: bool = False,
    output: Path | None = None,
    batch_rows: int = _FITS_STREAM_BATCH_ROWS,
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
            valid_sky_only=valid_sky_only,
            ra_col=ra_col,
            dec_col=dec_col,
            batch_rows=batch_rows,
        )

    tables: list[pa.Table] = []
    for p in path_list:
        chunk = extract_from_catalog_path(
            p,
            specs,
            valid_sky_only=valid_sky_only,
            ra_col=ra_col,
            dec_col=dec_col,
        )
        if add_input_path:
            chunk = chunk.append_column(
                "input_path",
                pa.array([str(p.resolve())] * chunk.num_rows, type=pa.string()),
            )
        tables.append(chunk)
        log.info("Read %d row(s) from %s", chunk.num_rows, p.name)

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
) -> ExtractResult | pa.Table:
    """Extract columns and write output; returns summary or small in-memory table."""
    if specs is None:
        raise ValueError("specs required")

    if lake_root is not None:
        if not survey:
            raise ValueError("--survey is required with --lake-root")
        if paths:
            raise ValueError("pass catalog paths or --lake-root/--survey, not both")
        result = stream_extract_from_lake_catalog(
            lake_root,
            survey,
            specs,
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
        )
        return result

    if not paths:
        raise ValueError("catalog path(s) or --lake-root required")
    if output_dir is not None:
        raise ValueError("--output-dir is only valid with --lake-root")

    raw_result = extract_from_catalog_paths(
        paths,
        specs,
        valid_sky_only=valid_sky_only,
        ra_col=ra_col,
        dec_col=dec_col,
        add_input_path=add_input_path,
        streaming=streaming,
        output=Path(output) if output is not None else None,
        batch_rows=batch_rows,
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


@click.command("dl-extract-catalog")
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
    required=True,
    help="Column to export (repeatable). Use NAME:alias to rename (e.g. ra:RA).",
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
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Data lake root; stream from catalogs/<survey>/ tiles.",
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
@click.option("--progress", "show_progress", is_flag=True, help="Show tile progress bar (lake export).")
@click.option("-v", "--verbose", is_flag=True)
def cli(
    paths: tuple[Path, ...],
    file_list: Path | None,
    output: Path | None,
    output_dir: Path | None,
    columns: tuple[str, ...],
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
    add_input_path: bool,
    streaming: bool,
    batch_rows: int,
    show_progress: bool,
    verbose: bool,
) -> None:
    """Export selected catalog columns for catalog–catalog association (sky matching)."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    all_paths: list[Path] = list(paths)
    if file_list is not None:
        for line in file_list.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                all_paths.append(Path(line))

    if output is None and output_dir is None:
        raise click.ClickException("Pass -o/--output or --output-dir.")
    if lake_root is None and not all_paths:
        raise click.ClickException(
            "Pass catalog path(s), --file-list, or --lake-root with --survey."
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

    fmt: OutputFormat | None = (
        output_format.lower()  # type: ignore[assignment]
        if output_format is not None
        else None
    )

    result = extract_catalog(
        output=output,
        specs=list(columns),
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
