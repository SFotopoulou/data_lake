"""Tests for SDSS spPlate spectrum ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits


def _write_minimal_spplate(
    path: Path,
    *,
    plate: int = 1960,
    mjd: int = 53289,
    n_fiber: int = 4,
    n_pix: int = 32,
) -> None:
    coeff0, coeff1 = 3.58, 0.0001
    flux = np.ones((n_fiber, n_pix), dtype=np.float32) * 50.0
    sigma = np.ones((n_fiber, n_pix), dtype=np.float32) * 5.0
    mask0 = np.zeros((n_fiber, n_pix), dtype=np.int32)

    phdu = fits.PrimaryHDU(flux)
    phdu.header["PLATEID"] = plate
    phdu.header["MJD"] = mjd
    phdu.header["COEFF0"] = coeff0
    phdu.header["COEFF1"] = coeff1
    ivar = np.where(sigma > 0, 1.0 / (sigma * sigma), 0.0).astype(np.float32)
    hdus = [
        phdu,
        fits.ImageHDU(ivar, name="IVAR"),
        fits.ImageHDU(mask0, name="ANDMASK"),
        fits.ImageHDU(mask0, name="ORMASK"),
    ]
    fiberid = np.arange(1, n_fiber + 1, dtype=np.int32)
    ra = np.linspace(120.0, 121.0, n_fiber)
    dec = np.full(n_fiber, 45.0)
    cols = [
        fits.Column(name="FIBERID", format="J", array=fiberid),
        fits.Column(name="RA", format="D", array=ra),
        fits.Column(name="DEC", format="D", array=dec),
        fits.Column(name="OBJID", format="5J", array=np.zeros((n_fiber, 5), dtype=np.int32)),
    ]
    hdus.append(fits.BinTableHDU.from_columns(cols, name="PLUGMAP"))
    phdu.header["RUN2D"] = "v5_13_2"
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _write_lookup(path: Path, plate: int, mjd: int, survey: str, fibers: dict[int, int]) -> None:
    rows = [
        {
            "survey": survey,
            "PLATE": plate,
            "MJD": mjd,
            "FIBERID": fid,
            "SPECOBJID": sid,
        }
        for fid, sid in fibers.items()
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


class TestSpplateDetectAndRead:
    def test_detect_format(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        p = tmp_path / "spPlate-1960-53289.fits"
        _write_minimal_spplate(p)
        assert _detect_format_from_path(p) == "sdss_spplate"

    def test_read_with_lookup(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_sdss_spplate
        from data_lake.ingest.sdss_specobj_lookup import build_fiber_to_specobjid_map

        plate, mjd = 1960, 53289
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=3)
        lookup = tmp_path / "lookup.parquet"
        _write_lookup(
            lookup,
            plate,
            mjd,
            "sdss_dr17",
            {1: 9001, 2: 9002},
        )
        fiber_map = build_fiber_to_specobjid_map(
            "sdss_dr17", plate, mjd, lookup_path=lookup,
        )
        with fits.open(sp) as hdul:
            records, wcs = _read_sdss_spplate(
                hdul, path=sp, fiber_to_specobjid=fiber_map,
            )
        assert len(records) == 2
        assert {r.source_id for r in records} == {
            normalize_object_id(9001),
            normalize_object_id(9002),
        }
        assert wcs["n_pix"] == 32
        assert len(records[0].flux) == 32


class TestSpplateIngest:
    def test_ingest_to_zarr(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
        from data_lake.io.spectra import SpectrumAccessor

        plate, mjd = 1960, 53289
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=2)
        lookup = tmp_path / "lookup.parquet"
        _write_lookup(lookup, plate, mjd, "sdss_test", {1: 111, 2: 222})

        lake = tmp_path / "lake"
        index_map = ingest_spectra_from_fits(
            sp,
            lake,
            "sdss_test",
            fmt="sdss_spplate",
            specobj_lookup=lookup,
            norder=5,
            on_duplicate_source_id="skip",
        )
        assert len(index_map) == 2
        acc = SpectrumAccessor(lake, "sdss_test")
        assert acc.get_spectrum(111) is not None
        assert acc.get_spectrum(222) is not None

    def test_ingest_from_plate_specobjid(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
        from data_lake.ingest.sdss_specobj_lookup import sdss_specobjid_from_plate_fiber
        from data_lake.io.spectra import SpectrumAccessor

        plate, mjd = 1960, 53289
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=2)
        lake = tmp_path / "lake"
        index_map = ingest_spectra_from_fits(
            sp,
            lake,
            "sdss_plate_id",
            fmt="sdss_spplate",
            specobj_lookup_from_plate=True,
            norder=5,
        )
        assert len(index_map) == 2
        sid1 = sdss_specobjid_from_plate_fiber(plate, 1, mjd, "v5_13_2")
        sid2 = sdss_specobjid_from_plate_fiber(plate, 2, mjd, "v5_13_2")
        assert sid1 in index_map and sid2 in index_map
        acc = SpectrumAccessor(lake, "sdss_plate_id")
        sp1 = acc.get_spectrum(sid1)
        assert sp1.ivar.max() > 0

    @pytest.fixture
    def boss_spplate_path(self) -> Path:
        path = Path(__file__).resolve().parents[1] / "data" / "spPlate-3523-55144.fits"
        if not path.is_file():
            pytest.skip(f"Example spPlate not found: {path}")
        return path

    def test_boss_spplate_read_and_ingest(
        self, tmp_path: Path, boss_spplate_path: Path,
    ) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import (
            _read_sdss_spplate,
            ingest_spectra_from_fits,
        )
        from data_lake.ingest.sdss_specobj_lookup import build_fiber_to_specobjid_from_spplate
        from data_lake.io.spectra import SpectrumAccessor

        with fits.open(boss_spplate_path) as hdul:
            fiber_map = build_fiber_to_specobjid_from_spplate(hdul, boss_spplate_path)
            records, wcs = _read_sdss_spplate(
                hdul, path=boss_spplate_path, fiber_to_specobjid=fiber_map,
            )
        assert wcs["n_pix"] == 3854
        assert len(records) >= 400
        assert records[0].ivar.max() > 1.0

        lake = tmp_path / "lake"
        n = ingest_spectra_from_fits(
            boss_spplate_path,
            lake,
            "boss_plate_test",
            fmt="sdss_spplate",
            specobj_lookup_from_plate=True,
            norder=5,
            on_duplicate_source_id="skip",
        )
        assert len(n) == len(records)
        acc = SpectrumAccessor(lake, "boss_plate_test")
        any_sid = next(iter(n))
        assert acc.get_spectrum(any_sid).flux.shape[0] == 3854

    def test_requires_lookup(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits

        sp = tmp_path / "spPlate-1-2.fits"
        _write_minimal_spplate(sp, plate=1, mjd=2, n_fiber=1)
        with pytest.raises(ValueError, match="specobj_lookup"):
            ingest_spectra_from_fits(
                sp, tmp_path / "lake", "sdss_test", fmt="sdss_spplate",
            )
