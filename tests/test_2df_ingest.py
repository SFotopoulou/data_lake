"""Tests for 2dFGRS spectrum ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits


# ---------------------------------------------------------------------------
# FITS fixture helpers
# ---------------------------------------------------------------------------

def _write_2df_spectrum(
    path: Path,
    *,
    seqnum: int = 154714,
    ra: float = 0.497,
    dec: float = -0.466,
    n_pix: int = 1024,
    z: float = 0.042,
) -> None:
    """Write a minimal 2dFGRS-style FITS file."""
    flux = np.ones(n_pix, dtype=np.float32) * 10.0
    variance = np.ones(n_pix, dtype=np.float32) * 4.0  # ivar = 0.25
    sky = np.zeros(n_pix, dtype=np.float32)
    data = np.stack([flux, variance, sky])  # shape (3, n_pix)

    primary = fits.PrimaryHDU(np.zeros((n_pix, n_pix), dtype=np.float32))
    primary.header["SEQNUM"] = seqnum
    primary.header["NAME"] = "TGS153Z170"
    primary.header["BJSEL"] = 17.735
    primary.header["RA"] = ra
    primary.header["DEC"] = dec

    spec_hdu = fits.ImageHDU(data, name="SPECTRUM")
    spec_hdu.header["NAXIS1"] = n_pix
    spec_hdu.header["NAXIS2"] = 3
    spec_hdu.header["CRVAL1"] = 5756.157
    spec_hdu.header["CRPIX1"] = 512.0
    spec_hdu.header["CDELT1"] = 4.31
    spec_hdu.header["Z"] = z
    spec_hdu.header["QUALITY"] = 5
    spec_hdu.header["ABEMMA"] = 2

    fits.HDUList([primary, spec_hdu]).writeto(path, overwrite=True)


def _write_2df_spectrum_transposed(path: Path, *, n_pix: int = 1024) -> None:
    """Write a 2dF FITS where SPECTRUM has shape (n_pix, 3) instead of (3, n_pix)."""
    flux = np.ones(n_pix, dtype=np.float32) * 5.0
    variance = np.ones(n_pix, dtype=np.float32) * 1.0
    sky = np.zeros(n_pix, dtype=np.float32)
    data = np.stack([flux, variance, sky]).T  # (n_pix, 3)

    primary = fits.PrimaryHDU(np.zeros((10, 10), dtype=np.float32))
    primary.header["SEQNUM"] = 9999
    primary.header["BJSEL"] = 18.0
    primary.header["RA"] = 10.0
    primary.header["DEC"] = -5.0

    spec_hdu = fits.ImageHDU(data, name="SPECTRUM")
    spec_hdu.header["CRVAL1"] = 5000.0
    spec_hdu.header["CRPIX1"] = 1.0
    spec_hdu.header["CDELT1"] = 3.0
    spec_hdu.header["Z"] = 0.1

    fits.HDUList([primary, spec_hdu]).writeto(path, overwrite=True)


def _write_catalog_with_serial(
    path: Path,
    *,
    survey: str,
    entries: list[dict],
) -> None:
    """Write a minimal lake catalog Parquet mapping serial → source_id + sky."""
    rows = [
        {
            "serial": int(e["serial"]),
            "source_id": int(e["source_id"]),
            "ra": float(e["ra"]),
            "dec": float(e["dec"]),
            "_healpix_norder5": 0,
            "_spectrum_index": -1,
        }
        for e in entries
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


# ---------------------------------------------------------------------------
# Unit tests: reader and detection
# ---------------------------------------------------------------------------


class TestDetect2df:
    def test_auto_detects_2df_format(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        p = tmp_path / "154714.fits"
        _write_2df_spectrum(p)
        assert _detect_format_from_path(p) == "2df"

    def test_does_not_detect_generic_as_2df(self, tmp_path: Path) -> None:
        from astropy.table import Table
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        p = tmp_path / "generic_spec.fits"
        flux = np.ones(100, dtype=np.float32)
        hdu = fits.PrimaryHDU(flux)
        hdu.header["CRVAL1"] = 4000.0
        hdu.header["CDELT1"] = 1.0
        hdu.header["CRPIX1"] = 1.0
        fits.HDUList([hdu]).writeto(p, overwrite=True)
        assert _detect_format_from_path(p) == "generic"


class TestRead2dfSpectrum:
    def test_flux_ivar_wavelength_extracted(self, tmp_path: Path) -> None:
        from astropy.io import fits as afits
        from data_lake.ingest.fits_to_spectra_zarr import _read_2df_spectrum

        p = tmp_path / "154714.fits"
        _write_2df_spectrum(p, seqnum=154714, n_pix=1024, z=0.042)
        with afits.open(str(p), memmap=True) as hdul:
            records, wcs = _read_2df_spectrum(hdul, p)

        assert len(records) == 1
        rec = records[0]
        assert rec.flux.shape == (1024,)
        assert rec.ivar.shape == (1024,)
        assert np.allclose(rec.flux, 10.0)
        assert np.allclose(rec.ivar, 0.25)  # 1/4
        assert rec.wavelength is not None and rec.wavelength.shape == (1024,)
        assert rec.meta["z"] == pytest.approx(0.042, abs=1e-4)
        assert wcs["n_pix"] == 1024

    def test_source_id_from_filename(self, tmp_path: Path) -> None:
        from astropy.io import fits as afits
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_2df_spectrum

        p = tmp_path / "154714.fits"
        _write_2df_spectrum(p)
        with afits.open(str(p), memmap=True) as hdul:
            records, _ = _read_2df_spectrum(hdul, p)

        expected_id = normalize_object_id("154714")
        assert records[0].source_id == expected_id

    def test_transposed_spectrum_shape_handled(self, tmp_path: Path) -> None:
        from astropy.io import fits as afits
        from data_lake.ingest.fits_to_spectra_zarr import _read_2df_spectrum

        p = tmp_path / "9999.fits"
        _write_2df_spectrum_transposed(p, n_pix=1024)
        with afits.open(str(p), memmap=True) as hdul:
            records, _ = _read_2df_spectrum(hdul, p)

        assert records[0].flux.shape == (1024,)
        assert np.allclose(records[0].flux, 5.0)
        assert np.allclose(records[0].ivar, 1.0)

    def test_zero_variance_gives_zero_ivar(self, tmp_path: Path) -> None:
        from astropy.io import fits as afits
        from data_lake.ingest.fits_to_spectra_zarr import _read_2df_spectrum

        p = tmp_path / "zero_var.fits"
        flux = np.ones(1024, dtype=np.float32) * 2.0
        variance = np.zeros(1024, dtype=np.float32)  # all zeros
        sky = np.zeros(1024, dtype=np.float32)
        data = np.stack([flux, variance, sky])
        primary = fits.PrimaryHDU(np.zeros((4, 4)))
        primary.header["SEQNUM"] = 1
        primary.header["BJSEL"] = 17.0
        primary.header["RA"] = 5.0
        primary.header["DEC"] = -10.0
        spec_hdu = fits.ImageHDU(data, name="SPECTRUM")
        spec_hdu.header["CRVAL1"] = 5000.0
        spec_hdu.header["CRPIX1"] = 1.0
        spec_hdu.header["CDELT1"] = 4.0
        fits.HDUList([primary, spec_hdu]).writeto(p, overwrite=True)

        with afits.open(str(p), memmap=True) as hdul:
            records, _ = _read_2df_spectrum(hdul, p)

        assert np.all(records[0].ivar == 0.0)


# ---------------------------------------------------------------------------
# Integration tests: ingest to Zarr
# ---------------------------------------------------------------------------


class TestIngest2df:
    def test_ingest_single_file(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
        from data_lake.io.spectra import SpectrumAccessor

        p = tmp_path / "154714.fits"
        _write_2df_spectrum(p, seqnum=154714, ra=0.497, dec=-0.466)

        lake = tmp_path / "lake"
        index_map = ingest_spectra_from_fits(
            p, lake, "2DFG_DR3",
            fmt="2df",
            norder=5,
            on_duplicate_source_id="skip",
        )

        expected_id = normalize_object_id("154714")
        assert expected_id in index_map

        acc = SpectrumAccessor(lake, "2DFG_DR3")
        sp = acc.get_spectrum(expected_id)
        assert sp is not None
        assert sp.flux.shape[0] == 1024
        assert np.allclose(sp.flux, 10.0)
        assert np.allclose(sp.ivar, 0.25)

    def test_ingest_auto_detection(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits

        p = tmp_path / "200001.fits"
        _write_2df_spectrum(p, seqnum=200001, ra=1.0, dec=-1.0)

        lake = tmp_path / "lake"
        index_map = ingest_spectra_from_fits(
            p, lake, "2DFG_DR3",
            # No explicit fmt — relies on auto-detection
            norder=5,
        )
        expected_id = normalize_object_id("200001")
        assert expected_id in index_map

    def test_duplicate_skipped(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
        from data_lake.io.spectra import SpectrumAccessor

        p = tmp_path / "111111.fits"
        _write_2df_spectrum(p, seqnum=111111, ra=2.0, dec=-2.0)

        lake = tmp_path / "lake"
        ingest_spectra_from_fits(p, lake, "2DFG_DR3", fmt="2df", norder=5)
        # Second ingest of same file should silently skip
        ingest_spectra_from_fits(
            p, lake, "2DFG_DR3",
            fmt="2df", norder=5, on_duplicate_source_id="skip",
        )

        acc = SpectrumAccessor(lake, "2DFG_DR3")
        sid = normalize_object_id("111111")
        # Tile should have exactly one row for this source
        import zarr
        import glob as _glob
        tiles = list((lake / "spectra" / "2DFG_DR3").rglob("*.zarr"))
        assert len(tiles) == 1
        root = zarr.open_group(str(tiles[0]), mode="r")
        assert root["source_id"].shape[0] == 1

    def test_spectrum_index_updated_in_catalog(self, tmp_path: Path) -> None:
        """After ingest, _spectrum_index must be patched in the Parquet catalog."""
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
        from data_lake.io.catalog import CatalogAccessor

        seqnum = 777777
        ra, dec = 5.0, -5.0
        p = tmp_path / f"{seqnum}.fits"
        _write_2df_spectrum(p, seqnum=seqnum, ra=ra, dec=dec)

        lake = tmp_path / "lake"

        # Build a catalog tile first with _spectrum_index = -1
        sid = normalize_object_id(str(seqnum))
        cat_dir = lake / "catalogs" / "2DFG_DR3" / "Norder=5" / "Dir=0"
        cat_dir.mkdir(parents=True, exist_ok=True)
        import pyarrow as pa, pyarrow.parquet as pq, healpy as hp
        nside = hp.order2nside(5)
        npix_val = int(hp.ang2pix(nside, ra, dec, nest=True, lonlat=True))
        cat_file = cat_dir / f"Npix={npix_val}.parquet"
        pq.write_table(
            pa.table({
                "source_id": pa.array([sid], type=pa.int64()),
                "ra": pa.array([ra]),
                "dec": pa.array([dec]),
                "serial": pa.array([seqnum], type=pa.int64()),
                "_healpix_norder5": pa.array([npix_val], type=pa.int64()),
                "_spectrum_index": pa.array([-1], type=pa.int64()),
            }),
            cat_file,
        )

        index_map = ingest_spectra_from_fits(
            p, lake, "2DFG_DR3",
            fmt="2df", norder=5, source_id_col="serial",
        )

        # Patch catalog explicitly (the CLI does this automatically)
        from data_lake.ingest.update_catalog_indices import update_index_column

        update_index_column(
            lake_root=lake,
            survey_name="2DFG_DR3",
            source_id_to_index=index_map,
            kind="spectrum",
            norder=5,
            source_id_col="serial",
        )

        # Re-read catalog tile
        refreshed = pq.read_table(cat_file)
        idx_col = refreshed["_spectrum_index"].to_pylist()
        assert idx_col[0] >= 0, f"_spectrum_index not updated; got {idx_col[0]}"
