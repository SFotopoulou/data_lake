"""
catalog_extract – export selected catalog columns for catalog–catalog association.

Use before positional matching (STILTS, ``build_crossmatch``, etc.): pull only the
ID and sky columns (plus any extras) from survey catalog files or from an
already-ingested lake catalog.  This is separate from catalog↔spectrum linkage,
which uses ``_spectrum_index`` / native spectrum keys on catalog rows.

Example
-------
::

    dl-extract-catalog survey_a.fits -o a_positions.parquet \\
        -c TARGETID -c RA -c DEC

    dl-extract-catalog --file-list catalogs.txt -o b.csv --format csv \\
        -c ID -c ra:RA -c dec:DEC --valid-sky-only

    dl-extract-catalog --lake-root /data/lake --survey DESI_DR1 \\
        -c source_id -c ra -c dec -o desi_dr1_sky.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Literal, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import (
    _read_source_table,
    is_valid_sky_position,
    resolve_catalog_column_name,
)

log = logging.getLogger(__name__)

OutputFormat = Literal["parquet", "csv", "fits", "votable"]
_SKY_RA_ALIASES = ("ra", "ra_deg", "radeg", "raj2000", "ra_obj")
_SKY_DEC_ALIASES = ("dec", "dec_deg", "decdeg", "dej2000", "dec_obj")


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


def filter_valid_sky_rows(
    table: pa.Table,
    *,
    ra_col: str | None = None,
    dec_col: str | None = None,
) -> pa.Table:
    """Drop rows whose RA/Dec are not usable for HEALPix / cone matching."""
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

    ra = pc.cast(table.column(ra_name).combine_chunks(), pa.float64())
    dec = pc.cast(table.column(dec_name).combine_chunks(), pa.float64())
    ra_np = ra.to_numpy(zero_copy_only=False)
    dec_np = dec.to_numpy(zero_copy_only=False)
    mask = pa.array(
        [is_valid_sky_position(float(r), float(d)) for r, d in zip(ra_np, dec_np)],
        type=pa.bool_(),
    )
    n_before = table.num_rows
    out = table.filter(mask)
    n_after = out.num_rows
    if n_after < n_before:
        log.info(
            "Dropped %d row(s) with invalid sky position (kept %d)",
            n_before - n_after,
            n_after,
        )
    return out


def _sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


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
    """Read selected columns from ``catalogs/<survey>/`` Parquet tiles."""
    from data_lake.io.catalog import CatalogAccessor

    acc = CatalogAccessor(lake_root, survey, norder=norder)
    mapping = resolve_column_specs(acc.columns, specs)
    col_sql = ", ".join(f"{_sql_ident(src)} AS {_sql_ident(out)}" for src, out in mapping)
    raw = acc.query(f"SELECT {col_sql} FROM catalog", fmt="arrow")
    if isinstance(raw, pa.RecordBatchReader):
        table = raw.read_all()
    else:
        table = raw
    if valid_sky_only:
        table = filter_valid_sky_rows(table, ra_col=ra_col, dec_col=dec_col)
    return table


def extract_from_catalog_path(
    path: Path | str,
    specs: Sequence[str],
    *,
    valid_sky_only: bool = False,
    ra_col: str | None = None,
    dec_col: str | None = None,
) -> pa.Table:
    """Read one catalog file and return the selected columns."""
    table = _read_source_table(Path(path))
    out = select_catalog_columns(table, specs)
    if valid_sky_only:
        out = filter_valid_sky_rows(out, ra_col=ra_col, dec_col=dec_col)
    return out


def extract_from_catalog_paths(
    paths: Iterable[Path | str],
    specs: Sequence[str],
    *,
    valid_sky_only: bool = False,
    ra_col: str | None = None,
    dec_col: str | None = None,
    add_input_path: bool = False,
) -> pa.Table:
    """Concatenate extracts from multiple catalog files."""
    tables: list[pa.Table] = []
    for path in paths:
        p = Path(path)
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

    if not tables:
        raise ValueError("no catalog paths to read")
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

    if fmt == "parquet":
        pq.write_table(table, str(output), compression="zstd")
        return

    if fmt == "csv":
        import csv

        with open(output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(table.schema.names)
            cols = [table.column(n).to_pylist() for n in table.schema.names]
            for row in zip(*cols):
                writer.writerow(row)
        return

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

    if fmt == "fits":
        astropy_tbl.write(str(output), format="fits", overwrite=True)
        return
    if fmt == "votable":
        astropy_tbl.write(str(output), format="votable", overwrite=True)
        return

    raise ValueError(f"unsupported output format {fmt!r}")


def extract_catalog(
    output: Path | str,
    specs: Sequence[str],
    *,
    paths: Sequence[Path | str] | None = None,
    lake_root: Path | str | None = None,
    survey: str | None = None,
    norder: int | None = None,
    valid_sky_only: bool = False,
    ra_col: str | None = None,
    dec_col: str | None = None,
    add_input_path: bool = False,
    output_format: OutputFormat | None = None,
) -> pa.Table:
    """Extract columns and write *output*; returns the table."""
    if lake_root is not None:
        if not survey:
            raise ValueError("--survey is required with --lake-root")
        if paths:
            raise ValueError("pass catalog paths or --lake-root/--survey, not both")
        table = extract_from_lake_catalog(
            lake_root,
            survey,
            specs,
            norder=norder,
            valid_sky_only=valid_sky_only,
            ra_col=ra_col,
            dec_col=dec_col,
        )
    else:
        if not paths:
            raise ValueError("catalog path(s) or --file-list required without --lake-root")
        table = extract_from_catalog_paths(
            paths,
            specs,
            valid_sky_only=valid_sky_only,
            ra_col=ra_col,
            dec_col=dec_col,
            add_input_path=add_input_path,
        )

    write_catalog_extract(table, output, output_format=output_format)
    return table


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
    required=True,
    type=click.Path(path_type=Path),
    help="Output Parquet, CSV, FITS, or VOTable (format from suffix or --format).",
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
    help="Output format (default: infer from --output suffix).",
)
@click.option(
    "--lake-root",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Data lake root; read from catalogs/<survey>/ instead of input files.",
)
@click.option("--survey", default=None, help="Survey name under catalogs/ (with --lake-root).")
@click.option("--norder", type=int, default=None, help="HEALPix order (default: catalog_info.json).")
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
@click.option("-v", "--verbose", is_flag=True)
def cli(
    paths: tuple[Path, ...],
    file_list: Path | None,
    output: Path,
    columns: tuple[str, ...],
    output_format: str | None,
    lake_root: Path | None,
    survey: str | None,
    norder: int | None,
    valid_sky_only: bool,
    ra_col: str | None,
    dec_col: str | None,
    add_input_path: bool,
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

    if lake_root is None and not all_paths:
        raise click.ClickException(
            "Pass catalog path(s), --file-list, or --lake-root with --survey."
        )
    if lake_root is not None and not survey:
        raise click.ClickException("--survey is required with --lake-root.")
    if lake_root is not None and all_paths:
        raise click.ClickException("Use either input paths or --lake-root/--survey, not both.")

    fmt: OutputFormat | None = (
        output_format.lower()  # type: ignore[assignment]
        if output_format is not None
        else None
    )

    table = extract_catalog(
        output,
        list(columns),
        paths=all_paths or None,
        lake_root=lake_root,
        survey=survey,
        norder=norder,
        valid_sky_only=valid_sky_only,
        ra_col=ra_col,
        dec_col=dec_col,
        add_input_path=add_input_path,
        output_format=fmt,
    )
    click.echo(
        f"Wrote {table.num_rows} row(s), {len(table.schema.names)} column(s) → {output}"
    )
