"""Tests for SDSS/BOSS spec-*.fits spectrum ingest (_read_sdss_boss)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _write_sdss_spec_fits(
    path: Path,
    *,
    n_pix: int = 32,
    include_and_mask: bool = False,
    mask_name: str = "and_mask",
    specobjid: int | None = None,
    header_objid: int | None = 1234567890123456789,
) -> None:
    """Minimal BOSS-style spec file: primary header + COADD [+ SPALL]."""
    loglam = np.linspace(3.5, 3.6, n_pix)
    flux = np.ones(n_pix, dtype=np.float32) * 100.0
    ivar = np.ones(n_pix, dtype=np.float32) * 0.01
    cols = [
        fits.Column(name="loglam", format="D", array=loglam),
        fits.Column(name="flux", format="E", array=flux),
        fits.Column(name="ivar", format="E", array=ivar),
    ]
    if include_and_mask:
        cols.append(
            fits.Column(
                name=mask_name,
                format="I",
                array=np.zeros(n_pix, dtype=np.int16),
            )
        )
    coadd = fits.BinTableHDU.from_columns(cols, name="COADD")
    phdu = fits.PrimaryHDU()
    phdu.header["PLUG_RA"] = 120.0
    phdu.header["PLUG_DEC"] = 45.0
    if header_objid is not None:
        phdu.header["OBJID"] = header_objid
    phdu.header["Z"] = 0.1
    hdus: list = [phdu, coadd]
    if specobjid is not None:
        spall = fits.BinTableHDU.from_columns(
            [fits.Column(name="SPECOBJID", format="K", array=np.array([specobjid], dtype=np.uint64))],
            name="SPALL",
        )
        hdus.append(spall)
    fits.HDUList(hdus).writeto(path, overwrite=True)


class TestReadSdssBoss:
    def test_coadd_without_mask_column(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _read_sdss_boss

        path = tmp_path / "spec-test.fits"
        _write_sdss_spec_fits(path, include_and_mask=False)
        with fits.open(path) as hdul:
            records, wcs = _read_sdss_boss(hdul, link_id_col="OBJID")
        assert len(records) == 1
        assert len(records[0].flux) == 32
        assert records[0].mask.shape == (32,)
        assert np.all(records[0].mask == 0)

    def test_coadd_with_uppercase_and_mask(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _read_sdss_boss

        path = tmp_path / "spec-mask.fits"
        _write_sdss_spec_fits(path, include_and_mask=True, mask_name="AND_MASK")
        with fits.open(path) as hdul:
            records, _ = _read_sdss_boss(hdul, link_id_col="OBJID")
        assert len(records[0].mask) == 32

    def test_detect_format(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        path = tmp_path / "spec-test.fits"
        _write_sdss_spec_fits(path)
        assert _detect_format_from_path(path) == "sdss_boss"

    def test_specobjid_from_spall_not_primary_header(self, tmp_path: Path) -> None:
        """SPECOBJID is in SPALL (HDU 2) on real SDSS spec files, not HDU 0."""
        from data_lake.ingest.fits_to_spectra_zarr import _read_sdss_boss

        specobjid = 9223372435012999168
        path = tmp_path / "spec-spall.fits"
        _write_sdss_spec_fits(
            path,
            specobjid=specobjid,
            header_objid=None,
        )
        with fits.open(path) as hdul:
            records, _ = _read_sdss_boss(hdul, link_id_col="SPECOBJID")
        from data_lake.ingest.fits_to_parquet import normalize_object_id

        assert records[0].source_id == normalize_object_id(specobjid)


class TestSdssVariableLengthIngest:
    """Real spec files in data/ differ by one pixel (4628 vs 4627)."""

    @pytest.fixture
    def sdss_spec_paths(self) -> list[Path]:
        repo = Path(__file__).resolve().parents[1]
        paths = sorted((repo / "data").glob("spec-*.fits"))
        if len(paths) < 2:
            pytest.skip("Need two spec-*.fits files under data/")
        return paths

    def test_ingest_two_lengths_same_tile(self, tmp_path: Path, sdss_spec_paths: list[Path]) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
        from data_lake.io.spectra import SpectrumAccessor

        lake = tmp_path / "lake"
        survey = "sdss_test"
        for p in sdss_spec_paths:
            ingest_spectra_from_fits(
                p,
                lake,
                survey,
                link_id_col="SPECOBJID",
                norder=5,
            )

        acc = SpectrumAccessor(lake, survey)
        info_path = lake / "spectra" / survey / "spectrum_info.json"
        info = __import__("json").loads(info_path.read_text())
        assert info["wavelength_mode"] == "per_source"

        # Both files share HEALPix Npix=519 at norder 5
        tile = lake / "spectra" / survey / "Norder=5" / "Dir=0" / "Npix=519.zarr"
        assert tile.is_dir()
        import zarr

        root = zarr.open_group(str(tile), mode="r")
        assert root["flux"].shape[0] == 2
        tile_n_pix = root["flux"].shape[1]
        assert tile_n_pix >= 4627

        from data_lake.ingest.zarr_ids import zarr_join_array

        ids = set(int(x) for x in zarr_join_array(root)[:])
        assert len(ids) == 2

        for sid in ids:
            sp = acc.get_spectrum(sid)
            assert sp.flux.shape[0] == tile_n_pix
            assert sp.wavelength.shape[0] == tile_n_pix
            assert sp.flux.shape[0] >= 4627
