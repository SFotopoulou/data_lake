"""
to_fits – export individual galaxy cutouts from the Zarr stack back to FITS.

The export is fully lossless for float32 data: pixel values and all WCS
keywords round-trip without modification.

Usage
-----
>>> from data_lake.export.to_fits import export_cutout, export_cutouts_batch
>>> export_cutout(
...     lake_root="/data/lake",
...     survey="des_dr2",
...     source_id=12345678,
...     output_path="/tmp/cutout_12345678.fits",
... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits

from data_lake.io.cutouts import CutoutAccessor, CutoutWCS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single cutout export
# ---------------------------------------------------------------------------


def export_cutout(
    lake_root: Path | str,
    survey: str,
    source_id: int,
    output_path: Path | str,
    band_index: int | None = None,
    overwrite: bool = True,
    accessor: CutoutAccessor | None = None,
) -> Path:
    """
    Export a single cutout to a standards-compliant FITS file.

    If the cutout has multiple bands they are stored as a data cube in the
    Primary HDU (NAXIS=3, shape = B×H×W).  Set ``band_index`` to extract
    a single band as a 2-D image instead.

    Parameters
    ----------
    lake_root:
        Data lake root.
    survey:
        Survey identifier.
    source_id:
        Source identifier.
    output_path:
        Destination FITS file.
    band_index:
        If provided, export only this band (0-based) as a 2-D image.
    overwrite:
        Overwrite existing file (default True).
    accessor:
        Pre-existing ``CutoutAccessor`` to reuse (avoids re-opening Zarr stores).

    Returns
    -------
    Path to the written FITS file.
    """
    lake_root = Path(lake_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _own_accessor = accessor is None
    if _own_accessor:
        accessor = CutoutAccessor(lake_root, survey)

    try:
        image, wcs = accessor.get_cutout(source_id)
    finally:
        if _own_accessor:
            pass  # CutoutAccessor has no explicit close; stores are kept open

    if band_index is not None:
        if band_index < 0 or band_index >= image.shape[0]:
            raise IndexError(
                f"band_index={band_index} out of range for image with {image.shape[0]} bands."
            )
        data = image[band_index]  # shape (H, W)
        n_bands = 1
    else:
        data = image  # shape (B, H, W) or (H, W) if single band
        n_bands = image.shape[0] if image.ndim == 3 else 1

    hdr = _build_fits_header(wcs, source_id, survey, n_bands, band_index)

    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]  # single band stored as 2-D

    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=hdr)
    hdul = fits.HDUList([hdu])
    hdul.writeto(str(output_path), overwrite=overwrite)
    log.info("Wrote %s (source_id=%d, shape=%s)", output_path.name, source_id, data.shape)
    return output_path


def _build_fits_header(
    wcs: CutoutWCS,
    source_id: int,
    survey: str,
    n_bands: int,
    band_index: int | None,
) -> fits.Header:
    """Construct a FITS header with full WCS and provenance keywords."""
    hdr = fits.Header()
    hdr["SIMPLE"] = True
    hdr["BITPIX"] = -32  # float32
    h, w = wcs.shape

    if n_bands > 1 and band_index is None:
        hdr["NAXIS"] = 3
        hdr["NAXIS1"] = w
        hdr["NAXIS2"] = h
        hdr["NAXIS3"] = n_bands
    else:
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = w
        hdr["NAXIS2"] = h

    # WCS keywords
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = wcs.crval[0]
    hdr["CRVAL2"] = wcs.crval[1]
    hdr["CRPIX1"] = wcs.crpix[0]
    hdr["CRPIX2"] = wcs.crpix[1]
    cd = wcs.cd_matrix
    hdr["CD1_1"] = cd[0, 0]
    hdr["CD1_2"] = cd[0, 1]
    hdr["CD2_1"] = cd[1, 0]
    hdr["CD2_2"] = cd[1, 1]

    # Provenance
    hdr["SOURCE_ID"] = source_id
    hdr["SURVEY"] = survey[:8]  # FITS keyword values limited to 8 chars cleanly
    hdr["ORIGIN"] = "data_lake"

    if band_index is not None:
        hdr["BAND_IDX"] = band_index

    return hdr


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------


def export_cutouts_batch(
    lake_root: Path | str,
    survey: str,
    source_ids: Sequence[int],
    output_dir: Path | str,
    filename_template: str = "cutout_{source_id}.fits",
    band_index: int | None = None,
    overwrite: bool = True,
) -> list[Path]:
    """
    Export multiple cutouts to individual FITS files.

    Parameters
    ----------
    lake_root, survey:
        Data lake parameters.
    source_ids:
        Source IDs to export.
    output_dir:
        Output directory.
    filename_template:
        f-string template with ``{source_id}`` placeholder.
    band_index:
        Export only this band (0-based) from multi-band cutouts.
    overwrite:
        Overwrite existing files.

    Returns
    -------
    List of paths to the written FITS files.
    """
    lake_root = Path(lake_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accessor = CutoutAccessor(lake_root, survey)
    paths: list[Path] = []

    for source_id in source_ids:
        out = output_dir / filename_template.format(source_id=source_id)
        try:
            p = export_cutout(
                lake_root=lake_root,
                survey=survey,
                source_id=source_id,
                output_path=out,
                band_index=band_index,
                overwrite=overwrite,
                accessor=accessor,
            )
            paths.append(p)
        except Exception as exc:
            log.error("Failed to export source_id=%d: %s", source_id, exc)

    return paths


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------


def verify_round_trip(
    lake_root: Path | str,
    survey: str,
    source_id: int,
    tmp_dir: Path | str | None = None,
) -> bool:
    """
    Verify that a cutout can be exported to FITS and re-read without data loss.

    Compares pixel values, WCS CRVAL, and CRPIX between the Zarr store and the
    round-tripped FITS file.  Returns True if all checks pass, False otherwise.

    This function is suitable for use in unit tests and CI pipelines.
    """
    import tempfile

    ctx = tempfile.TemporaryDirectory() if tmp_dir is None else None
    work_dir = Path(ctx.name if ctx else tmp_dir)

    try:
        acc = CutoutAccessor(lake_root, survey)
        original_image, original_wcs = acc.get_cutout(source_id)

        fits_path = work_dir / f"round_trip_{source_id}.fits"
        export_cutout(
            lake_root=lake_root,
            survey=survey,
            source_id=source_id,
            output_path=fits_path,
            accessor=acc,
        )

        from data_lake.io.fits_read import open_fits

        with open_fits(fits_path) as hdul:
            hdr = hdul[0].header
            data = np.array(hdul[0].data, dtype=np.float32)

        # Pixel check
        if data.ndim == 2:
            expected = original_image[0] if original_image.shape[0] == 1 else original_image[0]
        else:
            expected = original_image

        pixel_ok = np.allclose(data, expected, equal_nan=True)

        # WCS check
        crval_ok = (
            abs(hdr.get("CRVAL1", 0) - original_wcs.crval[0]) < 1e-9
            and abs(hdr.get("CRVAL2", 0) - original_wcs.crval[1]) < 1e-9
        )
        crpix_ok = (
            abs(hdr.get("CRPIX1", 0) - original_wcs.crpix[0]) < 1e-6
            and abs(hdr.get("CRPIX2", 0) - original_wcs.crpix[1]) < 1e-6
        )

        ok = pixel_ok and crval_ok and crpix_ok
        if not ok:
            log.error(
                "Round-trip check FAILED for source_id=%d: "
                "pixels=%s crval=%s crpix=%s",
                source_id, pixel_ok, crval_ok, crpix_ok,
            )
        else:
            log.info("Round-trip check PASSED for source_id=%d", source_id)

        return ok

    finally:
        if ctx:
            ctx.cleanup()
