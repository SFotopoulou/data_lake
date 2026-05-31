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
    spfile: str | None = None,
) -> str:
    """Write a minimal 2dFGRS-style FITS file. Returns the SPFILE label used."""
    if spfile is None:
        spfile = f"sgp{seqnum % 1000:03d}_010123_1z.fits"
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
    spec_hdu.header["SPFILE"] = spfile
    spec_hdu.header["OBSRA"] = ra
    spec_hdu.header["OBSDEC"] = dec
    spec_hdu.header["SNR"] = 12.0

    fits.HDUList([primary, spec_hdu]).writeto(path, overwrite=True)
    return spfile


def _write_2df_multi_spectrum(
    path: Path,
    *,
    seqnum: int,
    ra: float,
    dec: float,
    observations: list[dict],
    n_pix: int = 1024,
) -> list[str]:
    """Write a 2dF FITS with multiple SPECTRUM HDUs (one per observation)."""
    primary = fits.PrimaryHDU(np.zeros((49, 49), dtype=np.float32))
    primary.header["SEQNUM"] = seqnum
    primary.header["NAME"] = "TGS805Z023"
    primary.header["BJSEL"] = 19.0
    primary.header["RA"] = ra
    primary.header["DEC"] = dec

    hdus: list[fits.ImageHDU | fits.PrimaryHDU] = [primary]
    spfiles: list[str] = []
    for obs in observations:
        flux = np.ones(n_pix, dtype=np.float32) * float(obs.get("flux_scale", 10.0))
        variance = np.ones(n_pix, dtype=np.float32) * 4.0
        sky = np.zeros(n_pix, dtype=np.float32)
        data = np.stack([flux, variance, sky])
        spec_hdu = fits.ImageHDU(data, name="SPECTRUM")
        spec_hdu.header["NAXIS1"] = n_pix
        spec_hdu.header["NAXIS2"] = 3
        spec_hdu.header["CRVAL1"] = float(obs.get("crval1", 5756.0))
        spec_hdu.header["CRPIX1"] = 512.0
        spec_hdu.header["CDELT1"] = float(obs.get("cdelt1", 4.31))
        spec_hdu.header["Z"] = float(obs.get("z", 0.1))
        spec_hdu.header["QUALITY"] = 4
        spec_hdu.header["ABEMMA"] = 2
        spfile = str(obs["spfile"])
        spec_hdu.header["SPFILE"] = spfile
        spec_hdu.header["OBSRA"] = ra
        spec_hdu.header["OBSDEC"] = dec
        spec_hdu.header["SNR"] = float(obs.get("snr", 8.0))
        hdus.append(spec_hdu)
        spfiles.append(spfile)

    fits.HDUList(hdus).writeto(path, overwrite=True)
    return spfiles


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
    def test_finds_unnamed_spectrum_hdu(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _read_2df_spectrum

        n_pix = 64
        flux = np.ones(n_pix, dtype=np.float32)
        var = np.ones(n_pix, dtype=np.float32)
        sky = np.zeros(n_pix, dtype=np.float32)
        data = np.stack([flux, var, sky])
        primary = fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32))
        primary.header["SEQNUM"] = 5168
        primary.header["BJSEL"] = 17.0
        primary.header["RA"] = 1.0
        primary.header["DEC"] = -1.0
        spec = fits.ImageHDU(data)  # no HDU name
        spec.header["CRVAL1"] = 5000.0
        spec.header["CRPIX1"] = 1.0
        spec.header["CDELT1"] = 2.0
        p = tmp_path / "005168.fits"
        fits.HDUList([primary, spec]).writeto(p, overwrite=True)

        with fits.open(p, memmap=True) as hdul:
            records, _ = _read_2df_spectrum(hdul, p)
        assert len(records) == 1
        assert records[0].flux.shape == (n_pix,)

    def test_missing_spectrum_hdu_raises(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _read_2df_spectrum

        primary = fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32))
        primary.header["SEQNUM"] = 5168
        primary.header["BJSEL"] = 17.0
        p = tmp_path / "005168.fits"
        fits.HDUList([primary]).writeto(p, overwrite=True)

        with fits.open(p, memmap=True) as hdul:
            with pytest.raises(ValueError, match="no 2dF spectral extension|Found:"):
                _read_2df_spectrum(hdul, p)

    def test_stamp_only_fits_raises_clear_message(self, tmp_path: Path) -> None:
        """FDR tree can include 49×49 stamp images without a SPECTRUM extension."""
        from data_lake.ingest.fits_to_spectra_zarr import _read_2df_spectrum

        primary = fits.PrimaryHDU(np.zeros((49, 49), dtype=np.float32))
        primary.header["SEQNUM"] = 161216
        primary.header["BJSEL"] = 17.5
        primary.header["RA"] = 1.0
        primary.header["DEC"] = -1.0
        p = tmp_path / "161216.fits"
        fits.HDUList([primary]).writeto(p, overwrite=True)

        with fits.open(p, memmap=True) as hdul:
            with pytest.raises(ValueError, match="stamp-only FITS"):
                _read_2df_spectrum(hdul, p)

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

    def test_source_id_from_spfile(self, tmp_path: Path) -> None:
        from astropy.io import fits as afits
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_2df_spectrum

        p = tmp_path / "154714.fits"
        spfile = _write_2df_spectrum(p, spfile="sgp153_010123_1z.fits")
        with afits.open(str(p), memmap=True) as hdul:
            records, _ = _read_2df_spectrum(hdul, p)

        expected_id = normalize_object_id(spfile)
        assert records[0].source_id == expected_id

    def test_filename_stem_fallback_without_spfile(self, tmp_path: Path) -> None:
        from astropy.io import fits as afits
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_2df_spectrum

        p = tmp_path / "154714.fits"
        _write_2df_spectrum(p)
        with fits.open(p, mode="update") as hdul:
            del hdul[1].header["SPFILE"]
            hdul.flush()

        with afits.open(str(p), memmap=True) as hdul:
            records, _ = _read_2df_spectrum(hdul, p)

        assert records[0].source_id == normalize_object_id("154714")

    def test_multi_hdu_returns_one_record_per_spectrum(self, tmp_path: Path) -> None:
        from astropy.io import fits as afits
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_2df_spectrum

        p = tmp_path / "389442.fits"
        spfiles = _write_2df_multi_spectrum(
            p,
            seqnum=389442,
            ra=6.268,
            dec=-1.004,
            observations=[
                {"spfile": "sgp805_001203_2z.fits", "z": 0.120724, "crval1": 5849.6, "snr": 7.0},
                {"spfile": "sgp805_011009_2z.fits", "z": 0.119153, "crval1": 5826.3, "snr": 13.9},
            ],
        )
        with afits.open(str(p), memmap=True) as hdul:
            records, _ = _read_2df_spectrum(hdul, p)

        assert len(records) == 2
        ids = {r.source_id for r in records}
        assert ids == {normalize_object_id(s) for s in spfiles}
        assert records[0].meta["z"] == pytest.approx(0.120724, abs=1e-5)
        assert records[1].meta["z"] == pytest.approx(0.119153, abs=1e-5)
        assert records[0].meta["snr"] == pytest.approx(7.0)
        assert records[1].meta["snr"] == pytest.approx(13.9)
        assert not np.allclose(records[0].wavelength, records[1].wavelength)

    def test_duplicate_spfile_in_one_file_raises(self, tmp_path: Path) -> None:
        from astropy.io import fits as afits
        from data_lake.ingest.fits_to_spectra_zarr import _read_2df_spectrum

        p = tmp_path / "dup.fits"
        _write_2df_multi_spectrum(
            p,
            seqnum=1,
            ra=1.0,
            dec=-1.0,
            observations=[
                {"spfile": "same_obs.fits", "z": 0.1},
                {"spfile": "same_obs.fits", "z": 0.2},
            ],
        )
        with afits.open(str(p), memmap=True) as hdul:
            with pytest.raises(ValueError, match="duplicate 2dF source_id"):
                _read_2df_spectrum(hdul, p)

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
        spfile = _write_2df_spectrum(p, seqnum=154714, ra=0.497, dec=-0.466)

        lake = tmp_path / "lake"
        index_map = ingest_spectra_from_fits(
            p, lake, "2DFG_DR3",
            fmt="2df",
            norder=5,
            on_duplicate_source_id="skip",
        )

        expected_id = normalize_object_id(spfile)
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
        spfile = _write_2df_spectrum(p, seqnum=200001, ra=1.0, dec=-1.0)

        lake = tmp_path / "lake"
        index_map = ingest_spectra_from_fits(
            p, lake, "2DFG_DR3",
            # No explicit fmt — relies on auto-detection
            norder=5,
        )
        expected_id = normalize_object_id(spfile)
        assert expected_id in index_map

    def test_duplicate_skipped(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
        from data_lake.io.spectra import SpectrumAccessor

        p = tmp_path / "111111.fits"
        spfile = _write_2df_spectrum(p, seqnum=111111, ra=2.0, dec=-2.0)

        lake = tmp_path / "lake"
        ingest_spectra_from_fits(p, lake, "2DFG_DR3", fmt="2df", norder=5)
        # Second ingest of same file should silently skip
        ingest_spectra_from_fits(
            p, lake, "2DFG_DR3",
            fmt="2df", norder=5, on_duplicate_source_id="skip",
        )

        acc = SpectrumAccessor(lake, "2DFG_DR3")
        sid = normalize_object_id(spfile)
        # Tile should have exactly one row for this source
        import zarr
        import glob as _glob
        tiles = list((lake / "spectra" / "2DFG_DR3").rglob("*.zarr"))
        assert len(tiles) == 1
        root = zarr.open_group(str(tiles[0]), mode="r")
        from data_lake.ingest.zarr_ids import zarr_join_array

        assert zarr_join_array(root).shape[0] == 1

    def test_spectrum_index_updated_in_catalog(self, tmp_path: Path) -> None:
        """After ingest, _spectrum_index must be patched in the Parquet catalog."""
        from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits

        seqnum = 777777
        ra, dec = 5.0, -5.0
        spfile = "sgp777_010123_1z.fits"
        p = tmp_path / f"{seqnum}.fits"
        _write_2df_spectrum(p, seqnum=seqnum, ra=ra, dec=dec, spfile=spfile)

        lake = tmp_path / "lake"

        # Build a catalog tile first with _spectrum_index = -1
        sid = normalize_object_id(spfile)
        cat_dir = lake / "catalogs" / "2DFG_DR3" / "Norder=5" / "Dir=0"
        cat_dir.mkdir(parents=True, exist_ok=True)
        import pyarrow as pa, pyarrow.parquet as pq, healpy as hp
        nside = hp.order2nside(5)
        npix_val = int(hp.ang2pix(nside, ra, dec, nest=True, lonlat=True))
        cat_file = cat_dir / f"Npix={npix_val}.parquet"
        pq.write_table(
            pa.table({
                LAKE_JOIN_ID_COLUMN: pa.array([sid], type=pa.int64()),
                "ra": pa.array([ra]),
                "dec": pa.array([dec]),
                "serial": pa.array([seqnum], type=pa.int64()),
                "SPFILE": pa.array([spfile]),
                "_healpix_norder5": pa.array([npix_val], type=pa.int64()),
                "_spectrum_index": pa.array([-1], type=pa.int64()),
            }),
            cat_file,
        )

        index_map = ingest_spectra_from_fits(
            p, lake, "2DFG_DR3",
            fmt="2df", norder=5,
        )

        # Patch catalog explicitly (the CLI does this automatically)
        from data_lake.ingest.update_catalog_indices import update_index_column

        update_index_column(
            lake_root=lake,
            survey_name="2DFG_DR3",
            source_id_to_index=index_map,
            kind="spectrum",
            norder=5,
            link_id_col=None,
        )

        # Re-read catalog tile
        refreshed = pq.read_table(cat_file)
        idx_col = refreshed["_spectrum_index"].to_pylist()
        assert idx_col[0] >= 0, f"_spectrum_index not updated; got {idx_col[0]}"

    def test_multi_hdu_catalog_linkage(self, tmp_path: Path) -> None:
        """Two catalog rows with same serial but different SPFILE both link."""
        from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
        from data_lake.ingest.update_catalog_indices import update_index_column

        seqnum = 389442
        ra, dec = 6.268, -1.004
        spfiles = [
            "sgp805_001203_2z.fits",
            "sgp805_011009_2z.fits",
        ]
        p = tmp_path / f"{seqnum}.fits"
        _write_2df_multi_spectrum(
            p,
            seqnum=seqnum,
            ra=ra,
            dec=dec,
            observations=[
                {"spfile": spfiles[0], "z": 0.120724, "crval1": 5849.6},
                {"spfile": spfiles[1], "z": 0.119153, "crval1": 5826.3},
            ],
        )

        lake = tmp_path / "lake"
        import healpy as hp
        nside = hp.order2nside(5)
        npix_val = int(hp.ang2pix(nside, ra, dec, nest=True, lonlat=True))
        cat_dir = lake / "catalogs" / "2DFG_DR3" / "Norder=5" / "Dir=0"
        cat_dir.mkdir(parents=True, exist_ok=True)
        cat_file = cat_dir / f"Npix={npix_val}.parquet"
        sids = [normalize_object_id(s) for s in spfiles]
        pq.write_table(
            pa.table({
                LAKE_JOIN_ID_COLUMN: pa.array(sids, type=pa.int64()),
                "ra": pa.array([ra, ra]),
                "dec": pa.array([dec, dec]),
                "serial": pa.array([seqnum, seqnum], type=pa.int64()),
                "SPFILE": pa.array(spfiles),
                "_healpix_norder5": pa.array([npix_val, npix_val], type=pa.int64()),
                "_spectrum_index": pa.array([-1, -1], type=pa.int64()),
            }),
            cat_file,
        )

        index_map = ingest_spectra_from_fits(
            p, lake, "2DFG_DR3", fmt="2df", norder=5,
        )
        assert len(index_map) == 2

        update_index_column(
            lake_root=lake,
            survey_name="2DFG_DR3",
            source_id_to_index=index_map,
            kind="spectrum",
            norder=5,
            link_id_col=None,
        )

        refreshed = pq.read_table(cat_file)
        assert all(i >= 0 for i in refreshed["_spectrum_index"].to_pylist())

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[1] / "data" / "389442.fits").is_file(),
        reason="requires data/389442.fits",
    )
    def test_real_389442_fits_ingests_two_spectra(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits

        repo_root = Path(__file__).resolve().parents[1]
        p = repo_root / "data" / "389442.fits"
        lake = tmp_path / "lake"
        index_map = ingest_spectra_from_fits(
            p, lake, "2DFG_DR3", fmt="2df", norder=5,
        )
        expected = {
            normalize_object_id("sgp805_001203_2z.fits"),
            normalize_object_id("sgp805_011009_2z.fits"),
        }
        assert set(index_map) == expected
