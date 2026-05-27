"""
Extract a plugmap catalog (ra, dec, plate, mjd, fiberid) from spPlate FITS files.

Use when spPlate plates are missing from an ingested SpecObj catalog: build a
Parquet/CSV table for inspection, sidecar lookup, or supplemental catalog ingest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits

from data_lake.ingest.sdss_specobj_lookup import (
    _FIBER_COL_ALIASES,
    _MJD_COL_ALIASES,
    _PLATE_COL_ALIASES,
    _column_as_int64_numpy,
    _resolve_column,
    spplate_plate_mjd_from_hdul,
)

log = logging.getLogger(__name__)

_CATALOG_OUT_COLUMNS = ("plate", "mjd", "fiberid", "ra", "dec")


def _iter_spplate_paths(paths: Iterable[Path | str]) -> Iterator[Path]:
    for item in paths:
        p = Path(item)
        if p.is_dir():
            yield from sorted(p.glob("spPlate-*.fits"))
            yield from sorted(p.glob("spPlate-*.FITS"))
            continue
        if any(ch in p.name for ch in "*?[]"):
            base = p.parent if str(p.parent) not in ("", ".") else Path.cwd()
            yield from sorted(base.glob(p.name))
            continue
        yield p


def extract_spplate_catalog_rows(
    hdul: fits.HDUList,
    path: Path | None = None,
    *,
    active_only: bool = True,
    ra_col: str = "RA",
    dec_col: str = "DEC",
) -> list[dict]:
    """Read PLUGMAP + header into flat rows (one dict per fiber)."""
    from data_lake.ingest.fits_to_spectra_zarr import (
        _fits_bintable_column,
        _spplate_fiber_table_hdu,
        _spplate_flux_and_calib_hdus,
    )

    plate, mjd = spplate_plate_mjd_from_hdul(hdul, path)
    flux_2d, _, _, _ = _spplate_flux_and_calib_hdus(hdul)
    n_fiber = flux_2d.shape[0]

    ftable = _spplate_fiber_table_hdu(hdul)
    if ftable is None:
        raise ValueError(f"{path}: no PLUGMAP / FIBERID BINTABLE")

    fdata = ftable.data
    fiber_col = _fits_bintable_column(fdata, "fiberid", "FIBERID")
    ra_arr = _fits_bintable_column(fdata, ra_col.lower(), ra_col, "RA")
    dec_arr = _fits_bintable_column(fdata, dec_col.lower(), dec_col, "DEC")

    holetype = None
    objtype = None
    names = fdata.dtype.names or ()
    if "HOLETYPE" in names:
        holetype = fdata["HOLETYPE"]
    if "OBJTYPE" in names:
        objtype = fdata["OBJTYPE"]

    spplate_name = path.name if path else ""
    rows: list[dict] = []
    n_table = len(fdata)
    for row_i in range(min(n_fiber, n_table)):
        fiber_id = int(fiber_col[row_i])
        if active_only:
            flux = np.asarray(flux_2d[row_i], dtype=np.float32)
            if not np.any(np.isfinite(flux)) or np.all(flux == 0):
                continue

        row: dict = {
            "plate": int(plate),
            "mjd": int(mjd),
            "fiberid": fiber_id,
            "ra": float(ra_arr[row_i]),
            "dec": float(dec_arr[row_i]),
        }
        if spplate_name:
            row["spplate_file"] = spplate_name
        if holetype is not None:
            row["holetype"] = str(holetype[row_i]).strip()
        if objtype is not None:
            row["objtype"] = str(objtype[row_i]).strip()
        rows.append(row)
    return rows


def extract_spplate_catalog_table(
    path: Path | str,
    *,
    active_only: bool = True,
    ra_col: str = "RA",
    dec_col: str = "DEC",
) -> pa.Table:
    """Extract one spPlate file as a PyArrow table."""
    p = Path(path)
    with fits.open(str(p), memmap=True) as hdul:
        rows = extract_spplate_catalog_rows(
            hdul, p, active_only=active_only, ra_col=ra_col, dec_col=dec_col,
        )
    if not rows:
        return pa.table({c: pa.array([], type=pa.float64()) for c in _CATALOG_OUT_COLUMNS})
    return pa.Table.from_pylist(rows)


def extract_spplate_catalog_from_paths(
    paths: Iterable[Path | str],
    *,
    active_only: bool = True,
    ra_col: str = "RA",
    dec_col: str = "DEC",
) -> pa.Table:
    """Concatenate plugmap rows from many spPlate files."""
    all_rows: list[dict] = []
    for p in _iter_spplate_paths(paths):
        if not p.is_file():
            log.warning("Skipping missing path: %s", p)
            continue
        try:
            with fits.open(str(p), memmap=True) as hdul:
                all_rows.extend(
                    extract_spplate_catalog_rows(
                        hdul,
                        p,
                        active_only=active_only,
                        ra_col=ra_col,
                        dec_col=dec_col,
                    )
                )
        except Exception as exc:
            log.error("Failed to read %s: %s", p, exc)
            raise
    if not all_rows:
        return pa.table({c: pa.array([], type=pa.float64()) for c in _CATALOG_OUT_COLUMNS})
    return pa.Table.from_pylist(all_rows)


def load_catalog_plate_mjd_fiber_keys(
    catalog_root: Path | str,
    survey_name: str,
    *,
    plates: set[int] | None = None,
) -> set[tuple[int, int, int]]:
    """Load ``{(plate, mjd, fiberid)}`` present in ``catalogs/<survey>/``."""
    root = Path(catalog_root) / "catalogs" / survey_name
    if not root.is_dir():
        raise FileNotFoundError(f"Catalog not found: {root}")

    tile_paths = sorted(root.rglob("Npix=*.parquet"))
    if not tile_paths:
        raise FileNotFoundError(f"No Parquet tiles under {root}")

    plate_col = mjd_col = fiber_col = None
    keys: set[tuple[int, int, int]] = set()

    for tile_path in tile_paths:
        schema = pq.read_schema(str(tile_path))
        names = schema.names
        if plate_col is None:
            plate_col = _resolve_column(names, _PLATE_COL_ALIASES)
            mjd_col = _resolve_column(names, _MJD_COL_ALIASES)
            fiber_col = _resolve_column(names, _FIBER_COL_ALIASES)
            if not all((plate_col, mjd_col, fiber_col)):
                raise ValueError(
                    f"Catalog {root} missing plate/mjd/fiber columns; "
                    f"schema sample: {names[:25]}"
                )

        cols = [plate_col, mjd_col, fiber_col]  # type: ignore[list-item]
        chunk = pq.read_table(str(tile_path), columns=cols)
        plates_arr = _column_as_int64_numpy(chunk.column(plate_col))
        mjds = _column_as_int64_numpy(chunk.column(mjd_col))
        fibers = _column_as_int64_numpy(chunk.column(fiber_col))

        if plates is not None:
            plate_mask = np.isin(plates_arr, list(plates))
            if not np.any(plate_mask):
                continue
            plates_arr = plates_arr[plate_mask]
            mjds = mjds[plate_mask]
            fibers = fibers[plate_mask]

        for p, m, f in zip(plates_arr.tolist(), mjds.tolist(), fibers.tolist()):
            if f is None or m is None or p is None:
                continue
            keys.add((int(p), int(m), int(f)))

    return keys


def filter_rows_not_in_catalog(
    table: pa.Table,
    catalog_keys: set[tuple[int, int, int]],
) -> pa.Table:
    """Keep rows whose ``(plate, mjd, fiberid)`` are absent from the lake catalog."""
    if table.num_rows == 0:
        return table
    plates = table.column("plate").to_pylist()
    mjds = table.column("mjd").to_pylist()
    fibers = table.column("fiberid").to_pylist()
    keep = [
        (int(p), int(m), int(f)) not in catalog_keys
        for p, m, f in zip(plates, mjds, fibers)
    ]
    return table.filter(pa.array(keep))


def write_catalog_table(table: pa.Table, output: Path | str) -> Path:
    """Write Parquet or CSV based on suffix."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in (".csv", ".tsv"):
        import pyarrow.csv as pacsv

        pacsv.write_csv(table, out)
    else:
        pq.write_table(table, out)
    return out


def extract_spplate_catalog(
    paths: Iterable[Path | str],
    output: Path | str,
    *,
    active_only: bool = True,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    subtract_catalog_root: Path | str | None = None,
    subtract_survey: str | None = None,
) -> pa.Table:
    """
    Build a plugmap catalog table and optionally write it to disk.

    When ``subtract_catalog_root`` and ``subtract_survey`` are set, only rows
    **not** already in ``catalogs/<survey>/`` are kept (matched on plate/mjd/fiber).
    """
    table = extract_spplate_catalog_from_paths(
        paths, active_only=active_only, ra_col=ra_col, dec_col=dec_col,
    )
    if subtract_catalog_root is not None and subtract_survey is not None:
        plates = {int(x) for x in table.column("plate").to_pylist()} if table.num_rows else set()
        keys = load_catalog_plate_mjd_fiber_keys(
            subtract_catalog_root, subtract_survey, plates=plates or None,
        )
        before = table.num_rows
        table = filter_rows_not_in_catalog(table, keys)
        log.info(
            "Catalog anti-join: %d → %d rows (survey=%r, %d catalog keys for these plates)",
            before,
            table.num_rows,
            subtract_survey,
            len(keys),
        )
    write_catalog_table(table, output)
    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import click


@click.command("dl-extract-spplate-catalog")
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
@click.option(
    "--file-list",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Text file with one spPlate path per line.",
)
@click.option("-o", "--output", required=True, type=click.Path(path_type=Path))
@click.option(
    "--active-only/--all-fibers",
    default=True,
    show_default=True,
    help="Skip fibers with all-zero flux (recommended).",
)
@click.option("--ra-col", default="RA", show_default=True)
@click.option("--dec-col", default="DEC", show_default=True)
@click.option(
    "--subtract-catalog",
    "catalog_root",
    type=click.Path(path_type=Path),
    default=None,
    help="Lake root; keep only rows missing from catalogs/<survey>/.",
)
@click.option(
    "--survey",
    "subtract_survey",
    default=None,
    help="Survey name for --subtract-catalog (required with that flag).",
)
@click.option("-v", "--verbose", is_flag=True)
def cli(
    paths: tuple[Path, ...],
    file_list: Path | None,
    output: Path,
    active_only: bool,
    ra_col: str,
    dec_col: str,
    catalog_root: Path | None,
    subtract_survey: str | None,
    verbose: bool,
) -> None:
    """Export ra/dec/plate/mjd/fiberid from spPlate PLUGMAP (HDU 5)."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    all_paths: list[Path] = list(paths)
    if file_list is not None:
        for line in file_list.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                all_paths.append(Path(line))
    if not all_paths:
        raise click.ClickException("Pass spPlate path(s) or --file-list.")

    if catalog_root is not None and not subtract_survey:
        raise click.ClickException("--survey is required with --subtract-catalog.")

    table = extract_spplate_catalog(
        all_paths,
        output,
        active_only=active_only,
        ra_col=ra_col,
        dec_col=dec_col,
        subtract_catalog_root=catalog_root,
        subtract_survey=subtract_survey,
    )
    click.echo(f"Wrote {table.num_rows} row(s) to {output}")
