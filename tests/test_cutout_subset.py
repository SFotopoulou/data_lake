"""Tests for CutoutAccessor.extract_subset_* and dl-extract-cutout-subset CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from data_lake.ingest.fits_to_zarr import ingest_cutouts_from_fits
from data_lake.io.cutouts import CutoutAccessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SURVEY = "test_cutout_survey"
NORDER = 3

# Source table: (sid, ra, dec, flux_value)
SOURCES = [
    (1001, 10.00, 20.00, 1.0),
    (1002, 10.01, 20.01, 2.0),
    (1003, 10.02, 20.02, 3.0),
    (2001, 60.00, -30.00, 4.0),  # different HEALPix tile
]


def _write_cutout_fits(path: Path, *, sid: int, ra: float, dec: float, flux: float) -> None:
    """One-band 8×8 stamp with TAN WCS and constant flux."""
    data = np.full((8, 8), flux, dtype=np.float32)
    hdu = fits.PrimaryHDU(data)
    h = hdu.header
    h["RA"] = ra
    h["DEC"] = dec
    h["OBJ_ID"] = sid
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRVAL1"] = ra
    h["CRVAL2"] = dec
    h["CRPIX1"] = 4.0
    h["CRPIX2"] = 4.0
    h["CD1_1"] = -1.0 / 3600.0
    h["CD1_2"] = 0.0
    h["CD2_1"] = 0.0
    h["CD2_2"] = 1.0 / 3600.0
    h["NAXIS1"] = 8
    h["NAXIS2"] = 8
    hdu.writeto(path, overwrite=True)


@pytest.fixture
def cutout_lake(tmp_path: Path) -> Path:
    """Ingest all SOURCES into a small lake; return lake root."""
    for sid, ra, dec, flux in SOURCES:
        f = tmp_path / f"cutout_{sid}.fits"
        _write_cutout_fits(f, sid=sid, ra=ra, dec=dec, flux=flux)
        ingest_cutouts_from_fits(
            f, tmp_path, SURVEY,
            ra_col="RA", dec_col="DEC",
            norder=NORDER,
            band_names=["r"],
        )
    return tmp_path


# ---------------------------------------------------------------------------
# extract_subset_to_zarr
# ---------------------------------------------------------------------------

class TestExtractSubsetToZarr:
    def test_round_trip_flux(self, cutout_lake: Path) -> None:
        import zarr

        acc = CutoutAccessor(cutout_lake, SURVEY, norder=NORDER)
        result = acc.extract_subset_to_zarr(
            [1001, 2001], cutout_lake / "sub.zarr", show_progress=False
        )
        assert result["n_written"] == 2
        assert result["missing_ids"] == []

        root = zarr.open_group(
            store=zarr.storage.LocalStore(str(cutout_lake / "sub.zarr")),
            mode="r", zarr_format=3,
        )
        sids = np.asarray(root["_source_id"][:])
        assert set(sids.tolist()) == {1001, 2001}

        id_to_row = result["id_to_row"]
        imgs = np.asarray(root["images"][:])
        # Our fixture: flux value == sid / 1000 relative… actually flux=1.0 for 1001
        flux_expected = {1001: 1.0, 2001: 4.0}
        for sid, expected in flux_expected.items():
            row = id_to_row[sid]
            np.testing.assert_allclose(imgs[row, 0, :, :], expected, rtol=1e-5)

    def test_wcs_preserved_in_zarr(self, cutout_lake: Path) -> None:
        import struct
        import zarr

        acc = CutoutAccessor(cutout_lake, SURVEY, norder=NORDER)
        result = acc.extract_subset_to_zarr(
            [1001], cutout_lake / "wcs_sub.zarr", show_progress=False
        )
        root = zarr.open_group(
            store=zarr.storage.LocalStore(str(cutout_lake / "wcs_sub.zarr")),
            mode="r", zarr_format=3,
        )
        row = result["id_to_row"][1001]
        raw = bytes(root["wcs"][row])
        vals = struct.unpack_from("8d2i", raw)
        crval1 = vals[0]  # first field
        crval2 = vals[1]
        assert abs(crval1 - 10.00) < 1e-6
        assert abs(crval2 - 20.00) < 1e-6

    def test_missing_skip(self, cutout_lake: Path) -> None:
        acc = CutoutAccessor(cutout_lake, SURVEY, norder=NORDER)
        result = acc.extract_subset_to_zarr(
            [1001, 9999], cutout_lake / "miss.zarr",
            missing="skip", show_progress=False,
        )
        assert result["n_written"] == 1
        assert 9999 in result["missing_ids"]

    def test_missing_error_raises(self, cutout_lake: Path) -> None:
        acc = CutoutAccessor(cutout_lake, SURVEY, norder=NORDER)
        with pytest.raises(KeyError, match="9999"):
            acc.extract_subset_to_zarr(
                [1001, 9999], cutout_lake / "err.zarr",
                missing="error", show_progress=False,
            )

    def test_overwrite_guard(self, cutout_lake: Path) -> None:
        acc = CutoutAccessor(cutout_lake, SURVEY, norder=NORDER)
        out = cutout_lake / "guard.zarr"
        acc.extract_subset_to_zarr([1001], out, show_progress=False)
        with pytest.raises(FileExistsError):
            acc.extract_subset_to_zarr([1001], out, show_progress=False)
        # overwrite=True succeeds
        acc.extract_subset_to_zarr([1001, 1002], out, overwrite=True, show_progress=False)

    def test_attrs_written(self, cutout_lake: Path) -> None:
        import zarr

        acc = CutoutAccessor(cutout_lake, SURVEY, norder=NORDER)
        acc.extract_subset_to_zarr([1001], cutout_lake / "attr.zarr", show_progress=False)
        root = zarr.open_group(
            store=zarr.storage.LocalStore(str(cutout_lake / "attr.zarr")),
            mode="r", zarr_format=3,
        )
        assert root.attrs["source_survey"] == SURVEY
        assert root.attrs["n_sources"] == 1
        assert root.attrs["n_bands"] == 1
        assert root.attrs["height"] == 8
        assert root.attrs["width"] == 8


# ---------------------------------------------------------------------------
# extract_subset_to_fits
# ---------------------------------------------------------------------------

class TestExtractSubsetToFits:
    def test_writes_fits_with_wcs(self, cutout_lake: Path) -> None:
        acc = CutoutAccessor(cutout_lake, SURVEY, norder=NORDER)
        out_dir = cutout_lake / "fits_out"
        result = acc.extract_subset_to_fits(
            [1001, 2001], out_dir, show_progress=False
        )
        assert result["n_written"] == 2
        for sid, expected_flux in [(1001, 1.0), (2001, 4.0)]:
            fpath = Path(result["id_to_path"][sid])
            assert fpath.is_file()
            with fits.open(fpath) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float32)
                hdr = hdul[0].header
                # Image shape (B, H, W) = (1, 8, 8); may be squeezed to (8,8)
                np.testing.assert_allclose(data.flat[0], expected_flux, rtol=1e-5)
                # WCS keywords must be present
                assert "CRVAL1" in hdr
                assert "CRVAL2" in hdr
                assert "CD1_1" in hdr
                assert "SOURCE_ID" in hdr
                assert int(hdr["SOURCE_ID"]) == sid

    def test_wcs_values_correct(self, cutout_lake: Path) -> None:
        """CRVAL1/2 in the output FITS must match the ingested RA/Dec."""
        acc = CutoutAccessor(cutout_lake, SURVEY, norder=NORDER)
        out_dir = cutout_lake / "wcs_fits"
        result = acc.extract_subset_to_fits([1001], out_dir, show_progress=False)
        fpath = Path(result["id_to_path"][1001])
        with fits.open(fpath) as hdul:
            hdr = hdul[0].header
        assert abs(hdr["CRVAL1"] - 10.00) < 1e-6
        assert abs(hdr["CRVAL2"] - 20.00) < 1e-6

    def test_custom_filename_template(self, cutout_lake: Path) -> None:
        acc = CutoutAccessor(cutout_lake, SURVEY, norder=NORDER)
        out_dir = cutout_lake / "tpl_fits"
        result = acc.extract_subset_to_fits(
            [1001], out_dir,
            filename_template="stamp_{source_id:020d}.fits",
            show_progress=False,
        )
        fpath = Path(result["id_to_path"][1001])
        assert fpath.name == "stamp_00000000000000001001.fits"


# ---------------------------------------------------------------------------
# extract_subset_to_hdf5
# ---------------------------------------------------------------------------

class TestExtractSubsetToHdf5:
    def test_hdf5_images_and_wcs(self, cutout_lake: Path) -> None:
        pytest.importorskip("h5py")
        import h5py

        acc = CutoutAccessor(cutout_lake, SURVEY, norder=NORDER)
        out = cutout_lake / "sub.h5"
        result = acc.extract_subset_to_hdf5([1001, 2001], out, show_progress=False)
        assert result["n_written"] == 2

        with h5py.File(str(out), "r") as hf:
            assert hf["images"].shape == (2, 1, 8, 8)
            assert "_source_id" in hf
            # WCS datasets
            for field in ["wcs_crval1", "wcs_crval2", "wcs_crpix1", "wcs_crpix2",
                          "wcs_cd1_1", "wcs_naxis1"]:
                assert field in hf, f"Missing WCS dataset: {field}"
            sids = hf["_source_id"][:]
            assert set(sids.tolist()) == {1001, 2001}

            id_to_row = result["id_to_row"]
            crval1 = hf["wcs_crval1"][:]
            assert abs(crval1[id_to_row[1001]] - 10.00) < 1e-6


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestExtractCutoutSubsetCli:
    def test_zarr_format(self, cutout_lake: Path) -> None:
        from click.testing import CliRunner
        from data_lake.export.cutout_subset import cli

        runner = CliRunner()
        out = cutout_lake / "cli_sub.zarr"
        ids_file = cutout_lake / "ids.txt"
        ids_file.write_text("1001\n1002\n")

        result = runner.invoke(cli, [
            "--survey", SURVEY,
            "--target-list", str(ids_file),
            "--target-id-col", "_source_id",
            "--format", "zarr",
            "--output", str(out),
            "--lake-root", str(cutout_lake),
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_fits_format(self, cutout_lake: Path) -> None:
        from click.testing import CliRunner
        from data_lake.export.cutout_subset import cli

        runner = CliRunner()
        out_dir = cutout_lake / "cli_fits"
        ids_file = cutout_lake / "ids2.txt"
        ids_file.write_text("1001\n")

        result = runner.invoke(cli, [
            "--survey", SURVEY,
            "--target-list", str(ids_file),
            "--format", "fits",
            "--output", str(out_dir),
            "--lake-root", str(cutout_lake),
        ])
        assert result.exit_code == 0, result.output
        assert any(out_dir.glob("cutout_*.fits"))
