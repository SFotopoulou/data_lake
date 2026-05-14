#!/usr/bin/env python3
"""
Synthetic cross-survey example: LSST-like photometry catalog + DESI spectra +
Euclid-like cutouts, joined via a *master association* Parquet and DuckDB.

Run from the repo root::

    uv run python examples/cross_survey_lsst_desi_euclid/demo.py

This writes ``examples/cross_survey_lsst_desi_euclid/synthetic_lake/`` with
minimal catalog tiles and ``associations/lsst_desi_euclid_master.parquet``,
then runs DuckDB to select LSST sources with ``g_mag < 22`` and print the
Zarr paths + row indices for spectra (DESI) and cutouts (Euclid).

The path pattern matches the data lake layout (see ``healpix_dir`` in
``data_lake.ingest.fits_to_parquet``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import healpix_dir

# ---------------------------------------------------------------------------
# Layout constants (same HEALPix order for all surveys in this toy example)
# ---------------------------------------------------------------------------

NORDER = 5
LSST_SURVEY = "lsst_dc2_photometry"
DESI_SURVEY = "desi_dr1_spectra"
EUCLID_SURVEY = "euclid_q1_cutouts"


def zarr_tile_path(lake: Path, modality: str, survey: str, norder: int, npix: int) -> Path:
    """Resolve ``…/{spectra|cutouts}/<survey>/Norder=…/Dir=…/Npix=<npix>.zarr``."""
    sub = "spectra" if modality == "spectra" else "cutouts"
    rel = healpix_dir(norder, npix)
    return lake / sub / survey / rel / f"Npix={npix}.zarr"


def write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _catalog_info(path: Path, survey: str, norder: int, n_rows: int, n_cols: int) -> None:
    info = {
        "catalog_name": survey,
        "catalog_type": "object",
        "hats_order": norder,
        "total_rows": n_rows,
        "total_columns": n_cols,
        "schema_version": "1",
        "epoch": "J2000",
        "ra_column": "ra",
        "dec_column": "dec",
        "source_id_mode": "column:source_id",
        "ingest_streaming": False,
        "created_utc": "2099-01-01T00:00:00Z",
    }
    path.write_text(json.dumps(info, indent=2))


def build_synthetic_lake(lake: Path) -> Path:
    """
    All toy sources share one HEALPix pixel so a single Zarr tile path applies.

    LSST catalog: photometry only (spectrum/cutout indices left at -1).
    DESI / Euclid catalogs: hold the per-tile Zarr row indices for *their* IDs.
    Master table: maps ``lsst_source_id`` → DESI + Euclid IDs and Zarr rows.
    """
    if lake.exists():
        shutil.rmtree(lake)
    lake.mkdir(parents=True)

    # One shared tile (any npix is fine; Dir= follows HATS stride)
    npix = 12_345
    hp_col = f"_healpix_norder{NORDER}"

    # --- LSST photometry (3 rows): wide g_mag for filtering ----------------
    lsst = pa.table({
        "source_id": pa.array([1_001, 1_002, 1_003], type=pa.int64()),
        "ra": pa.array([200.0, 200.01, 200.02], type=pa.float64()),
        "dec": pa.array([0.0, 0.0, 0.01], type=pa.float64()),
        "g_mag": pa.array([21.2, 22.4, 21.9], type=pa.float64()),
        hp_col: pa.array([npix, npix, npix], type=pa.int64()),
        "_spectrum_index": pa.array([-1, -1, -1], type=pa.int64()),
        "_cutout_index": pa.array([-1, -1, -1], type=pa.int64()),
    })
    lsst_root = lake / "catalogs" / LSST_SURVEY
    tile_dir = lsst_root / healpix_dir(NORDER, npix)
    write_parquet(tile_dir / f"Npix={npix}.parquet", lsst)
    _catalog_info(lsst_root / "catalog_info.json", LSST_SURVEY, NORDER, len(lsst), len(lsst.schema))

    # --- DESI catalog (spectra indexed): two targets in same tile ------------
    desi = pa.table({
        "source_id": pa.array([2_000_001, 2_000_002], type=pa.int64()),
        "ra": pa.array([200.0, 200.02], type=pa.float64()),
        "dec": pa.array([0.0, 0.01], type=pa.float64()),
        hp_col: pa.array([npix, npix], type=pa.int64()),
        "_spectrum_index": pa.array([0, 1], type=pa.int64()),
        "_cutout_index": pa.array([-1, -1], type=pa.int64()),
    })
    desi_root = lake / "catalogs" / DESI_SURVEY
    tile_d = desi_root / healpix_dir(NORDER, npix)
    write_parquet(tile_d / f"Npix={npix}.parquet", desi)
    _catalog_info(desi_root / "catalog_info.json", DESI_SURVEY, NORDER, len(desi), len(desi.schema))

    # --- Euclid catalog (cutouts indexed) -----------------------------------
    euclid = pa.table({
        "source_id": pa.array([3_000_001, 3_000_002], type=pa.int64()),
        "ra": pa.array([200.0, 200.02], type=pa.float64()),
        "dec": pa.array([0.0, 0.01], type=pa.float64()),
        hp_col: pa.array([npix, npix], type=pa.int64()),
        "_spectrum_index": pa.array([-1, -1], type=pa.int64()),
        "_cutout_index": pa.array([0, 1], type=pa.int64()),
    })
    eu_root = lake / "catalogs" / EUCLID_SURVEY
    tile_e = eu_root / healpix_dir(NORDER, npix)
    write_parquet(tile_e / f"Npix={npix}.parquet", euclid)
    _catalog_info(eu_root / "catalog_info.json", EUCLID_SURVEY, NORDER, len(euclid), len(euclid.schema))

    # --- Master association (built *outside* the lake in real life) ---------
    master = pa.table({
        "lsst_source_id": pa.array([1_001, 1_003], type=pa.int64()),
        "desi_targetid": pa.array([2_000_001, 2_000_002], type=pa.int64()),
        "euclid_source_id": pa.array([3_000_001, 3_000_002], type=pa.int64()),
        "healpix_npix": pa.array([npix, npix], type=pa.int64()),
        "norder": pa.array([NORDER, NORDER], type=pa.int32()),
        "desi_zarr_row": pa.array([0, 1], type=pa.int64()),
        "euclid_zarr_row": pa.array([0, 1], type=pa.int64()),
    })
    assoc = lake / "associations"
    assoc.mkdir(parents=True)
    master_path = assoc / "lsst_desi_euclid_master.parquet"
    write_parquet(master_path, master)

    return master_path


def run_duckdb_query(lake: Path, master_path: Path, sql_path: Path) -> None:
    lsst_glob = str(lake / "catalogs" / LSST_SURVEY / "**" / "*.parquet")
    raw = sql_path.read_text()
    sql = (
        raw.replace("__LAKE__", lake.as_posix())
        .replace("__LSST_PARQUET_GLOB__", lsst_glob.replace("'", "''"))
        .replace("__MASTER_PARQUET__", master_path.as_posix().replace("'", "''"))
        .replace("__DESI_SURVEY__", DESI_SURVEY)
        .replace("__EUCLID_SURVEY__", EUCLID_SURVEY)
    )
    con = duckdb.connect(database=":memory:")
    parts = [p.strip() for p in sql.split(";") if p.strip()]
    *setup, select_stmt = parts
    for stmt in setup:
        con.execute(stmt + ";")
    print("DuckDB result (LSST g_mag < 22, with DESI + Euclid associations):\n")
    rel = con.execute(select_stmt + ";")
    tbl = rel.to_arrow_table()
    names = tbl.column_names
    for i in range(tbl.num_rows):
        row = {n: tbl.column(n)[i].as_py() for n in names}
        print(row)


def main() -> None:
    here = Path(__file__).resolve().parent
    lake = here / "synthetic_lake"
    sql_path = here / "query.sql"
    master_path = build_synthetic_lake(lake)
    print(f"Wrote synthetic lake → {lake}\nMaster table → {master_path}\n")
    run_duckdb_query(lake, master_path, sql_path)
    npix = 12_345
    print("\nExpected Zarr paths (same pattern as ``healpix_dir`` in the package):")
    print("  Spectra: ", zarr_tile_path(lake, "spectra", DESI_SURVEY, NORDER, npix))
    print("  Cutouts:", zarr_tile_path(lake, "cutouts", EUCLID_SURVEY, NORDER, npix))
    print(
        "\nNext step in real code: open each unique tile once with zarr, then\n"
        "  root['flux'][desi_zarr_row]   (or SpectrumAccessor pattern)\n"
        "  root['images'][euclid_zarr_row]  (or CutoutAccessor pattern)\n"
    )


if __name__ == "__main__":
    main()
