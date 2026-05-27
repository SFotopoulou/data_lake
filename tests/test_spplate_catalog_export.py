"""Tests for spPlate plugmap catalog export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits

from data_lake.export.spplate_catalog import (
    extract_spplate_catalog,
    extract_spplate_catalog_table,
    filter_rows_not_in_catalog,
    load_catalog_plate_mjd_fiber_keys,
)


def _write_minimal_spplate(
    path: Path,
    *,
    plate: int = 1960,
    mjd: int = 53289,
    n_fiber: int = 4,
    n_pix: int = 32,
) -> None:
    flux = np.ones((n_fiber, n_pix), dtype=np.float32) * 50.0
    phdu = fits.PrimaryHDU(flux)
    phdu.header["PLATEID"] = plate
    phdu.header["MJD"] = mjd
    phdu.header["COEFF0"] = 3.58
    phdu.header["COEFF1"] = 0.0001
    fiberid = np.arange(1, n_fiber + 1, dtype=np.int32)
    cols = [
        fits.Column(name="FIBERID", format="J", array=fiberid),
        fits.Column(name="RA", format="D", array=np.linspace(120.0, 121.0, n_fiber)),
        fits.Column(name="DEC", format="D", array=np.full(n_fiber, 45.0)),
    ]
    hdus = [
        phdu,
        fits.ImageHDU(np.ones_like(flux), name="IVAR"),
        fits.ImageHDU(np.zeros_like(flux, dtype=np.int32), name="ANDMASK"),
        fits.BinTableHDU.from_columns(cols, name="PLUGMAP"),
    ]
    fits.HDUList(hdus).writeto(path, overwrite=True)


class TestSpplateCatalogExport:
    def test_extract_minimal_plate(self, tmp_path: Path) -> None:
        sp = tmp_path / "spPlate-1960-53289.fits"
        _write_minimal_spplate(sp, plate=1960, mjd=53289, n_fiber=3)
        tbl = extract_spplate_catalog_table(sp)
        assert tbl.num_rows == 3
        assert set(tbl.column_names) >= {"plate", "mjd", "fiberid", "ra", "dec"}
        assert tbl.column("plate").to_pylist() == [1960, 1960, 1960]
        assert tbl.column("fiberid").to_pylist() == [1, 2, 3]

    def test_subtract_catalog(self, tmp_path: Path) -> None:
        sp = tmp_path / "spPlate-1960-53289.fits"
        _write_minimal_spplate(sp, plate=1960, mjd=53289, n_fiber=3)
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "sdss_test" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "plate": pa.array([1960], type=pa.int32()),
                "mjd": pa.array([53289], type=pa.int64()),
                "fiber": pa.array([1], type=pa.int16()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        out = tmp_path / "missing.parquet"
        tbl = extract_spplate_catalog(
            [sp],
            out,
            subtract_catalog_root=lake,
            subtract_survey="sdss_test",
        )
        assert tbl.num_rows == 2
        assert set(tbl.column("fiberid").to_pylist()) == {2, 3}
        assert out.is_file()

    def test_catalog_keys_loader(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "x" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "plate": pa.array([1, 1], type=pa.int32()),
                "mjd": pa.array([2, 2], type=pa.int64()),
                "fiber": pa.array([10, 11], type=pa.int16()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        keys = load_catalog_plate_mjd_fiber_keys(lake, "x", plates={1})
        assert keys == {(1, 2, 10), (1, 2, 11)}

    def test_filter_not_in_catalog(self) -> None:
        tbl = pa.table({
            "plate": [1, 1],
            "mjd": [2, 2],
            "fiberid": [10, 99],
            "ra": [0.0, 1.0],
            "dec": [0.0, 1.0],
        })
        out = filter_rows_not_in_catalog(tbl, {(1, 2, 10)})
        assert out.num_rows == 1
        assert out.column("fiberid")[0].as_py() == 99
