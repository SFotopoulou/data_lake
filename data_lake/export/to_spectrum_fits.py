"""
to_spectrum_fits – export individual spectra from the Zarr stack back to FITS.

Output FITS layout
------------------
HDU 0  Primary  – flux array (float32), spectral WCS in header
HDU 1  BinTable – columns WAVELENGTH (float64), FLUX (float32),
                  IVAR (float32), MASK (uint8/16)
                  + table header with provenance keywords

The WCS in HDU 0 uses CTYPE1="WAVE-LOG" or "WAVE" as stored in the Zarr attrs.
If the survey stored ``wavelength_mode="shared"`` the WCS is fully encoded in
the header keywords; the WAVELENGTH column in HDU 1 is always materialised for
easy array access.

Usage
-----
>>> from data_lake.export.to_spectrum_fits import export_spectrum, verify_spectrum_round_trip
>>> export_spectrum(
...     lake_root="/data/lake",
...     survey="sdss_dr17",
...     source_id=1237654321098,
...     output_path="/tmp/spec_1237654321098.fits",
... )
>>> ok = verify_spectrum_round_trip("/data/lake", "sdss_dr17", 1237654321098)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits

from data_lake.io.spectra import Spectrum, SpectrumAccessor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single spectrum export
# ---------------------------------------------------------------------------


def export_spectrum(
    lake_root: Path | str,
    survey: str,
    source_id: int,
    output_path: Path | str,
    overwrite: bool = True,
    accessor: SpectrumAccessor | None = None,
) -> Path:
    """
    Export one spectrum to a standards-compliant FITS file.

    Parameters
    ----------
    lake_root:
        Data lake root.
    survey:
        Survey identifier.
    source_id:
        Source identifier.
    output_path:
        Destination FITS file path.
    overwrite:
        Overwrite existing file (default True).
    accessor:
        Pre-existing ``SpectrumAccessor`` to reuse (avoids reopening stores).

    Returns
    -------
    Path to the written FITS file.
    """
    lake_root = Path(lake_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    own_accessor = accessor is None
    if own_accessor:
        accessor = SpectrumAccessor(lake_root, survey)

    sp = accessor.get_spectrum(source_id)

    # ---- HDU 0: flux as a 1-D image with spectral WCS ----
    primary_header = _build_primary_header(sp, survey)
    primary_hdu = fits.PrimaryHDU(data=sp.flux.astype(np.float32), header=primary_header)

    # ---- HDU 1: BinTable with all arrays + wavelength ----
    table_hdu = _build_table_hdu(sp)

    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(str(output_path), overwrite=overwrite)
    log.info("Wrote %s (source_id=%d, n_pix=%d)", output_path.name, source_id, len(sp.flux))
    return output_path


def _build_primary_header(sp: Spectrum, survey: str) -> fits.Header:
    """Construct the Primary HDU header with full spectral WCS and provenance."""
    wcs = sp.wcs_attrs
    hdr = fits.Header()
    hdr["SIMPLE"] = True
    hdr["BITPIX"] = -32
    hdr["NAXIS"]  = 1
    hdr["NAXIS1"] = len(sp.flux)

    ctype = str(wcs.get("ctype", "WAVE")).upper()
    hdr["CTYPE1"] = ctype
    hdr["CRVAL1"] = float(wcs.get("crval", 0.0))
    hdr["CDELT1"] = float(wcs.get("cdelt", 1.0))
    hdr["CRPIX1"] = float(wcs.get("crpix", 1.0))
    hdr["CUNIT1"] = str(wcs.get("unit", "Angstrom"))
    hdr["DC-FLAG"] = 1 if "LOG" in ctype else 0

    # Provenance
    hdr["SOURCE_ID"] = sp.source_id
    hdr["SURVEY"]    = str(survey)[:8]
    hdr["ORIGIN"]    = "data_lake"

    # Per-source meta
    meta = sp.meta
    if meta.get("z") is not None:
        hdr["Z"]       = float(meta["z"])
    if meta.get("z_err") is not None:
        hdr["Z_ERR"]   = float(meta["z_err"])
    if meta.get("snr") is not None:
        hdr["SN_MED"]  = float(meta["snr"])
    if meta.get("exptime") is not None:
        hdr["EXPTIME"] = float(meta["exptime"])
    if meta.get("R") is not None:
        hdr["SPEC_RES"] = float(meta["R"])
    if meta.get("instr"):
        hdr["INSTRUME"] = str(meta["instr"])[:8]

    return hdr


def _build_table_hdu(sp: Spectrum) -> fits.BinTableHDU:
    """Construct HDU 1 BinTable with WAVELENGTH, FLUX, IVAR, MASK columns."""
    n_pix = len(sp.flux)
    wave_col = fits.Column(name="WAVELENGTH", format="D",  array=sp.wavelength.astype(np.float64),
                           unit="Angstrom")
    flux_col = fits.Column(name="FLUX",       format="E",  array=sp.flux.astype(np.float32))
    ivar_col = fits.Column(name="IVAR",       format="E",  array=sp.ivar.astype(np.float32))
    mask_col = fits.Column(name="MASK",       format="B",  array=sp.mask.astype(np.uint8))

    tbl = fits.BinTableHDU.from_columns([wave_col, flux_col, ivar_col, mask_col])
    tbl.header["EXTNAME"] = "SPECDATA"
    tbl.header["SOURCE_ID"] = sp.source_id
    tbl.header["NPIX"] = n_pix
    return tbl


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------


def export_spectra_batch(
    lake_root: Path | str,
    survey: str,
    source_ids: Sequence[int],
    output_dir: Path | str,
    filename_template: str = "spec_{source_id}.fits",
    overwrite: bool = True,
) -> list[Path]:
    """
    Export multiple spectra to individual FITS files.

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
    overwrite:
        Overwrite existing files.
    """
    lake_root = Path(lake_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accessor = SpectrumAccessor(lake_root, survey)
    paths: list[Path] = []

    for source_id in source_ids:
        out = output_dir / filename_template.format(source_id=source_id)
        try:
            p = export_spectrum(lake_root, survey, source_id, out,
                                overwrite=overwrite, accessor=accessor)
            paths.append(p)
        except Exception as exc:
            log.error("Failed to export source_id=%d: %s", source_id, exc)

    return paths


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------


def verify_spectrum_round_trip(
    lake_root: Path | str,
    survey: str,
    source_id: int,
    tmp_dir: Path | str | None = None,
) -> bool:
    """
    Verify that a spectrum can be exported to FITS and re-read without data loss.

    Checks:
    - Flux values match (allclose, float32 precision).
    - Wavelength values match (allclose, float64 precision).
    - IVAR and mask values match exactly.
    - CRVAL1, CDELT1 in header match the stored WCS attrs.

    Returns True if all checks pass, False otherwise.
    """
    import tempfile

    ctx = tempfile.TemporaryDirectory() if tmp_dir is None else None
    work_dir = Path(ctx.name if ctx else tmp_dir)

    try:
        accessor = SpectrumAccessor(lake_root, survey)
        original = accessor.get_spectrum(source_id)

        fits_path = work_dir / f"round_trip_{source_id}.fits"
        export_spectrum(lake_root, survey, source_id, fits_path, accessor=accessor)

        with fits.open(str(fits_path)) as hdul:
            hdr0  = hdul[0].header
            flux_rt = np.array(hdul[0].data, dtype=np.float32)
            tbl     = hdul["SPECDATA"].data
            wave_rt = np.array(tbl["WAVELENGTH"], dtype=np.float64)
            ivar_rt = np.array(tbl["IVAR"],       dtype=np.float32)
            mask_rt = np.array(tbl["MASK"],        dtype=np.uint8)

        flux_ok = np.allclose(flux_rt, original.flux, equal_nan=True, atol=1e-6)
        wave_ok = np.allclose(wave_rt, original.wavelength, rtol=1e-9)
        ivar_ok = np.allclose(ivar_rt, original.ivar, equal_nan=True, atol=1e-6)
        mask_ok = np.array_equal(mask_rt, original.mask)

        crval_ok = abs(hdr0.get("CRVAL1", 0) - float(original.wcs_attrs.get("crval", 0))) < 1e-9
        cdelt_ok = abs(hdr0.get("CDELT1", 0) - float(original.wcs_attrs.get("cdelt", 0))) < 1e-9

        ok = flux_ok and wave_ok and ivar_ok and mask_ok and crval_ok and cdelt_ok
        if not ok:
            log.error(
                "Round-trip FAILED for source_id=%d: "
                "flux=%s wave=%s ivar=%s mask=%s crval=%s cdelt=%s",
                source_id, flux_ok, wave_ok, ivar_ok, mask_ok, crval_ok, cdelt_ok,
            )
        else:
            log.info("Round-trip PASSED for source_id=%d", source_id)
        return ok

    finally:
        if ctx:
            ctx.cleanup()
