"""Tests for data_lake.cutouts.generate_fits (dl-generate-cutout-fits)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from data_lake.cutouts.generate_fits import generate_cutout_fits
from data_lake.ingest.fits_to_zarr import ingest_cutouts_from_fits


def _make_image_fits(path: Path, *, crval1: float, crval2: float, flux: float = 99.0) -> None:
    """Large-ish image with constant flux and a simple TAN WCS."""
    ny, nx = 400, 400
    data = np.full((ny, nx), flux, dtype=np.float32)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2, ny / 2]
    wcs.wcs.crval = [crval1, crval2]
    wcs.wcs.cd = [[-1.0 / 3600.0, 0.0], [0.0, 1.0 / 3600.0]]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdu = fits.PrimaryHDU(data=data, header=wcs.to_header())
    hdu.writeto(path, overwrite=True)


def test_generate_multiband_and_ingest(tmp_path: Path) -> None:
    r_img = tmp_path / "r.fits"
    i_img = tmp_path / "i.fits"
    z_img = tmp_path / "z.fits"
    # Centre of field ~ (150, 2.5) deg
    _make_image_fits(r_img, crval1=150.0, crval2=2.5, flux=10.0)
    _make_image_fits(i_img, crval1=150.0, crval2=2.5, flux=20.0)
    _make_image_fits(z_img, crval1=150.0, crval2=2.5, flux=30.0)

    cat = Table({
        "TARGETID": np.array([1001, 1002], dtype=np.int64),
        "TARGET_RA": np.array([150.0, 150.01]),
        "TARGET_DEC": np.array([2.5, 2.51]),
    })
    cat_path = tmp_path / "cat.ecsv"
    cat.write(cat_path, overwrite=True)

    out_dir = tmp_path / "stamps"
    result = generate_cutout_fits(
        cat_path,
        out_dir,
        [r_img, i_img, z_img],
        size_pix=32,
        id_col="TARGETID",
        ra_col="TARGET_RA",
        dec_col="TARGET_DEC",
        id_hdu_key="TARGETID",
        ra_hdu_key="TARGET_RA",
        dec_hdu_key="TARGET_DEC",
        band_names=["r", "i", "z"],
        show_progress=False,
    )
    assert result.n_written == 2
    f1 = out_dir / "cutout_1001.fits"
    assert f1.is_file()
    with fits.open(f1) as hdul:
        assert hdul[0].data.shape == (3, 32, 32)
        assert hdul[0].header["TARGETID"] == 1001
        # Band order: r=10, i=20, z=30 at centre
        centre = hdul[0].data[:, 16, 16]
        np.testing.assert_allclose(centre, [10.0, 20.0, 30.0], rtol=0, atol=1e-5)

    # Round-trip through lake ingest
    m = ingest_cutouts_from_fits(
        f1,
        tmp_path / "lake",
        "syn",
        ra_col="TARGET_RA",
        dec_col="TARGET_DEC",
        link_id_col="TARGETID",
        norder=5,
        band_names=["r", "i", "z"],
    )
    assert m[1001] == 0


def test_images_file_order(tmp_path: Path) -> None:
    paths = []
    for i, flux in enumerate([1.0, 2.0]):
        p = tmp_path / f"b{i}.fits"
        _make_image_fits(p, crval1=10.0, crval2=0.0, flux=flux)
        paths.append(p)
    (tmp_path / "bands.txt").write_text("\n".join(str(p) for p in paths) + "\n")

    cat = Table({"source_id": [1], "ra": [10.0], "dec": [0.0]})
    cat.write(tmp_path / "c.ecsv", overwrite=True)

    from data_lake.cutouts.generate_fits import _parse_image_paths

    ordered = _parse_image_paths(None, tmp_path / "bands.txt")
    assert ordered == paths


def test_missing_catalog_column(tmp_path: Path) -> None:
    cat = Table({"ra": [1.0], "dec": [2.0]})
    p = tmp_path / "c.ecsv"
    cat.write(p, overwrite=True)
    img = tmp_path / "x.fits"
    _make_image_fits(img, crval1=0.0, crval2=0.0)
    with pytest.raises(KeyError, match="source_id"):
        generate_cutout_fits(
            p, tmp_path / "out", [img], size_pix=8, show_progress=False,
        )
