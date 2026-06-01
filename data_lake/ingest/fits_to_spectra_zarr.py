"""
fits_to_spectra_zarr – ingest 1-D spectra from FITS files into sharded Zarr v3 stacks.

Supported input formats
-----------------------
* **SDSS/BOSS** ``spec-*.fits``  – COADD binary table HDU (FLUX/IVAR/AND_MASK/LOGLAM).
  Uses raw ``astropy.io.fits``; no extra dependency required.
* **SDSS spPlate** ``spPlate-*.fits`` – 640 fiber spectra per plate (2-D flux HDUs +
  per-fiber metadata table).  Requires a specObj lookup (``--specobj-lookup`` or lake
  catalog) keyed by ``(survey, PLATE, MJD, FIBERID)`` → ``SPECOBJID``.
* **DESI**      ``coadd-*.fits`` – Uses ``desispec.io.read_spectra`` +
  ``desispec.coaddition.coadd_cameras`` for IVAR-weighted camera combination of
  the B/R/Z arms onto a single monotonic BRZ wavelength grid.
  Requires ``pip install 'data-lake[desi]'`` (``desispec>=0.62``).
* **6dFGS**     multi-extension target FITS – ingests every combined VR spectrum extension.
* **GAMA**      stacked AAOMEGA-2dF PRIMARY image ``(n_row, n_pix)`` with ``ROW1=Spectrum``,
  ``ROW2=Error`` (1-σ), optional sky rows; ``SPECID`` in the primary header.
* **Generic**   spectral WCS FITS – 1-D or multi-spectra image HDU with CTYPE1=WAVE*.

DESI note
---------
The DESI B/R/Z arms overlap in wavelength (~5772–5800 Å and ~7570–7770 Å).
Naive concatenation (the previous implementation) produced duplicated wavelength
regions and ignored IVAR-weighting at the overlaps. ``coadd_cameras`` handles
both correctly and is what ``desispec``'s own pipeline uses. If ``desispec``
is not installed and a DESI coadd file is encountered, an ``ImportError`` is
raised with an actionable install hint.

On-disk layout per tile
-----------------------
  <lake_root>/spectra/<survey>/Norder=<N>/Dir=<D>/Npix=<P>.zarr/
      flux/         (N_sources, N_pix)              float32  sharded
      ivar/         (N_sources, N_pix)              float32  sharded
      mask/         (N_sources, N_pix)              uint8    sharded (uint16 if mask_dtype='uint16')
      wavelength/   (N_pix,)                        float64  shared mode  (default)
                    (N_sources, N_pix)               float32  per-source mode
      resolution/   (N_sources, n_diag, N_pix)      float32  sharded  [opt-in, DESI only]
      _source_id/   (N_sources,)                    int64
      meta/         (N_sources,)                    void (structured: z, z_err, snr, exptime, R, instr)
  spectrum_info.json

Resolution matrix (DESI only, opt-in)
--------------------------------------
Pass ``--with-resolution`` (CLI) or ``with_resolution=True`` (Python API) to
store the DESI per-source banded resolution matrix alongside flux/ivar.  Each
row of ``resolution[i]`` is a ``(n_diag, N_pix)`` float32 array in
``scipy.sparse.dia_matrix`` diagonal storage.  The diagonal offsets and n_diag
are stored as Zarr group attributes.

Downstream use (requires ``scipy``):

    spec = accessor.get_spectrum(source_id)
    R = spec.resolution_operator()          # scipy.sparse.dia_matrix (N_pix, N_pix)
    model_obs = R @ template_resampled      # forward-model through the LSF
    chi2 = np.sum((spec.flux - model_obs) ** 2 * spec.ivar)

For batched template fitting (one sparse matmul over all templates):

    templates_obs = np.column_stack([
        np.interp(spec.wavelength, tpl.wave * (1 + z), tpl.flux, 0, 0)
        for tpl in template_library
    ])
    models = R @ templates_obs   # (N_pix, n_templates) in one call
    chi2s  = ((spec.flux[:, None] - models) ** 2 * spec.ivar[:, None]).sum(axis=0)

Storage cost: ~170–200 GB per 1 M coadded BRZ spectra after zstd-bitshuffle
compression (~3× the flux+ivar footprint).  Only enable when downstream science
requires template fitting, redshift refinement, or kinematic measurements.
Requires ``wavelength_mode="shared"``.

Returns
-------
``{source_id: spectrum_index}`` mapping so the catalog updater can write back
``_spectrum_index`` to the relevant Parquet tile files.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import zarr
import zarr.codecs
from astropy.io import fits

from data_lake.ingest.fits_to_parquet import (
    LAKE_JOIN_ID_COLUMN,
    assign_healpix,
    healpix_dir,
    is_valid_sky_position,
    object_id_from_fits_header,
    sky_from_fits_header,
)
from data_lake.ingest.zarr_ids import create_zarr_join_array, zarr_join_array

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional desispec import
# ---------------------------------------------------------------------------

def _import_desispec():
    """Return the desispec package, raising a helpful ImportError when absent."""
    try:
        import desispec.io          # noqa: F401
        import desispec.coaddition  # noqa: F401
        import desispec
        from data_lake.cli_utils import apply_parallel_worker_logging_after_heavy_imports

        apply_parallel_worker_logging_after_heavy_imports()
        return desispec
    except ImportError as exc:
        raise ImportError(
            "DESI coadd ingest requires desispec.\n"
            "Install with: pip install 'data-lake[desi]'"
        ) from exc

# ---------------------------------------------------------------------------
# Constants / schema
# ---------------------------------------------------------------------------

_CHUNKS_PER_SHARD = 512    # ~512 MB shards for typical 5000-px spectra
_ZSTD_LEVEL = 3
_DEFAULT_FLUX_DTYPE = np.float32
_DEFAULT_MASK_DTYPE = np.uint8

# Structured dtype for per-source scalar metadata stored inside the Zarr group.
# Canonical metadata lives in the Parquet catalog; this mirrors only the most
# commonly needed scalars so the tile is self-contained for offline use.
_META_DTYPE = np.dtype([
    ("z",        np.float32),   # spectroscopic redshift
    ("z_err",    np.float32),   # redshift error
    ("snr",      np.float32),   # median S/N per pixel
    ("exptime",  np.float32),   # total exposure time [s]
    ("R",        np.float32),   # spectral resolution R = λ/Δλ (representative)
    ("instr",    "S16"),        # instrument identifier (ASCII, <=16 chars)
])

# Bit-flag definitions stored in spectrum_info.json
_DEFAULT_MASK_BITS: dict[str, int] = {
    "NODATA":       0,
    "BADSKY":       1,
    "COSMICRAY":    2,
    "SATURATED":    3,
    "LOWFLAT":      4,
    "BADSKYSUBT":   5,
    "COMBINEREJ":   6,
    "BADFLUXFACTOR": 7,
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class SpectrumRecord:
    """A single 1-D spectrum extracted from a FITS file."""
    source_id: int
    ra: float
    dec: float
    flux: np.ndarray      # shape (N_pix,) float32
    ivar: np.ndarray      # shape (N_pix,) float32
    mask: np.ndarray      # shape (N_pix,) uint8
    wavelength: np.ndarray | None  # shape (N_pix,) float64; None if shared
    meta: dict[str, Any]


# ---------------------------------------------------------------------------
# Wavelength helpers
# ---------------------------------------------------------------------------


def _wavelength_from_wcs(header: fits.Header, n_pix: int) -> np.ndarray:
    """Reconstruct a wavelength array from FITS spectral WCS keywords."""
    crval = float(header.get("CRVAL1", 0.0))
    cdelt = float(header.get("CD1_1", header.get("CDELT1", 1.0)))
    crpix = float(header.get("CRPIX1", 1.0))
    pixel_indices = np.arange(n_pix, dtype=np.float64)
    wave = crval + cdelt * (pixel_indices - (crpix - 1))
    ctype = str(header.get("CTYPE1", "WAVE")).upper()
    if "LOG" in ctype or header.get("DC-FLAG", 0) == 1:
        wave = 10.0 ** wave
    return wave


def _wcs_attrs_from_header(header: fits.Header, n_pix: int) -> dict[str, Any]:
    """Extract WCS scalars to store as Zarr group attributes."""
    ctype = str(header.get("CTYPE1", "WAVE")).upper()
    is_log = "LOG" in ctype or header.get("DC-FLAG", 0) == 1
    return {
        "ctype": "WAVE-LOG" if is_log else "WAVE",
        "crval": float(header.get("CRVAL1", 0.0)),
        "cdelt": float(header.get("CD1_1", header.get("CDELT1", 1.0))),
        "crpix": float(header.get("CRPIX1", 1.0)),
        "unit": str(header.get("CUNIT1", "Angstrom")),
        "air_or_vacuum": "vacuum",
        "n_pix": int(n_pix),
    }


# ---------------------------------------------------------------------------
# Tile Zarr store
# ---------------------------------------------------------------------------


def _open_or_create_spectrum_tile(
    tile_path: Path,
    n_pix: int,
    wavelength_mode: str,
    mask_dtype: np.dtype,
    wcs_attrs: dict[str, Any],
    n_diag: int | None = None,
    resolution_offsets: np.ndarray | None = None,
) -> zarr.Group:
    """Open an existing tile Zarr group or create one with the correct layout.

    When ``n_diag`` is provided a sharded ``resolution`` array of shape
    ``(N_sources, n_diag, N_pix)`` is created alongside the standard arrays,
    and ``resolution_n_diag`` / ``resolution_offsets`` are stored as group attrs.
    """
    store = zarr.storage.LocalStore(str(tile_path))
    zarr_json = tile_path / "zarr.json"
    if tile_path.exists() and zarr_json.exists():
        return zarr.open_group(store=store, mode="a", zarr_format=3)

    root = zarr.open_group(store=store, mode="w", zarr_format=3)

    # Shard shape: one shard = CHUNKS_PER_SHARD rows
    shard_rows = _CHUNKS_PER_SHARD
    chunk_1d = (1, n_pix)
    shard_2d = (shard_rows, n_pix)

    compressors = zarr.codecs.BloscCodec(
        cname="zstd",
        clevel=_ZSTD_LEVEL,
        shuffle=zarr.codecs.BloscShuffle.bitshuffle,
    )

    def _sharded_array(name: str, dtype, fill=0, shape=None, chunks=None, shards=None):
        root.create_array(
            name,
            shape=shape if shape is not None else (0, n_pix),
            chunks=chunks if chunks is not None else chunk_1d,
            shards=shards if shards is not None else shard_2d,
            dtype=dtype,
            compressors=compressors,
            fill_value=fill,
        )

    _sharded_array("flux", np.float32, fill=np.nan)
    _sharded_array("ivar", np.float32, fill=0.0)
    _sharded_array("mask", mask_dtype, fill=0)

    if wavelength_mode == "shared":
        root.create_array(
            "wavelength",
            shape=(n_pix,),
            chunks=(n_pix,),
            dtype=np.float64,
            fill_value=0.0,
        )
    else:
        _sharded_array("wavelength", np.float32, fill=0.0)

    create_zarr_join_array(root, shape=(0,), chunks=(4096,), dtype=np.int64, fill_value=-1)
    root.create_array(
        "meta",
        shape=(0,),
        chunks=(1024,),
        dtype="|V" + str(_META_DTYPE.itemsize),
        fill_value=b"\x00" * _META_DTYPE.itemsize,
    )

    if n_diag is not None:
        # (N_sources, n_diag, N_pix) — one shard covers CHUNKS_PER_SHARD sources
        _sharded_array(
            "resolution",
            dtype=np.float32,
            fill=0.0,
            shape=(0, n_diag, n_pix),
            chunks=(1, n_diag, n_pix),
            shards=(shard_rows, n_diag, n_pix),
        )

    # Store WCS, mode, and (optional) resolution metadata as group attrs
    attrs: dict[str, Any] = {**wcs_attrs, "wavelength_mode": wavelength_mode}
    if n_diag is not None and resolution_offsets is not None:
        attrs["resolution_n_diag"] = int(n_diag)
        attrs["resolution_offsets"] = resolution_offsets.tolist()
    root.attrs.update(attrs)

    return root


def widen_spectrum_tile(
    tile_path: Path,
    new_n_pix: int,
    *,
    wavelength_mode: str,
    mask_dtype: np.dtype,
    wcs_attrs: dict[str, Any],
    n_diag: int | None = None,
    resolution_offsets: np.ndarray | None = None,
) -> zarr.Group:
    """Widen an existing spectrum tile to ``new_n_pix`` by padding existing rows.

    Writes a new tile to a sibling temp path, copies all arrays with right-padding
    on the pixel axis, then atomically replaces the original.  On any error the
    original tile is left untouched.

    Arrays widened:
    - ``flux``       → pad with ``NaN``
    - ``ivar``       → pad with ``0.0``
    - ``mask``       → pad with ``0``
    - ``wavelength`` (per_source) → pad rows with ``0.0``
    - ``wavelength`` (shared)     → extend 1-D vector with ``0.0``
    - ``resolution`` (if present) → pad pixel axis with ``0.0``
    - ``_source_id``, ``meta``    → copied unchanged
    """
    tmp_path = tile_path.parent / (tile_path.name + ".__widening__")
    backup_path = tile_path.parent / (tile_path.name + ".__widening_backup__")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    old_store = zarr.storage.LocalStore(str(tile_path))
    old_root = zarr.open_group(store=old_store, mode="r", zarr_format=3)
    old_n_pix = int(old_root["flux"].shape[1])
    n_rows = int(old_root["flux"].shape[0])

    if new_n_pix <= old_n_pix:
        return zarr.open_group(store=old_store, mode="a", zarr_format=3)

    pad_width = new_n_pix - old_n_pix
    log.info(
        "Widening spectrum tile %s: n_pix %d → %d (%d existing rows)",
        tile_path.name,
        old_n_pix,
        new_n_pix,
        n_rows,
    )

    new_root = _open_or_create_spectrum_tile(
        tmp_path,
        new_n_pix,
        wavelength_mode,
        mask_dtype,
        wcs_attrs,
        n_diag=n_diag,
        resolution_offsets=resolution_offsets,
    )

    _CHUNK = 256  # rows per migration batch

    def _copy_padded_2d(name: str, pad_val: float, dtype) -> None:
        src = old_root[name]
        dst = new_root[name]
        for start in range(0, n_rows, _CHUNK):
            end = min(start + _CHUNK, n_rows)
            chunk = np.asarray(src[start:end]).astype(dtype)
            padded = np.pad(chunk, ((0, 0), (0, pad_width)), constant_values=pad_val)
            dst.append(padded)

    _copy_padded_2d("flux", np.nan, np.float32)
    _copy_padded_2d("ivar", 0.0, np.float32)
    _copy_padded_2d("mask", 0, np.dtype(mask_dtype))

    if wavelength_mode == "per_source":
        _copy_padded_2d("wavelength", 0.0, np.float32)
    else:
        old_wave = np.asarray(old_root["wavelength"][:])
        new_wave = np.pad(old_wave, (0, pad_width), constant_values=0.0)
        new_root["wavelength"][:] = new_wave

    if n_rows > 0:
        new_root[LAKE_JOIN_ID_COLUMN].append(
            np.asarray(zarr_join_array(old_root)[:])
        )
        new_root["meta"].append(np.asarray(old_root["meta"][:]))

    if "resolution" in old_root and n_rows > 0:
        src_res = old_root["resolution"]
        dst_res = new_root["resolution"]
        for start in range(0, n_rows, _CHUNK):
            end = min(start + _CHUNK, n_rows)
            chunk = np.asarray(src_res[start:end]).astype(np.float32)
            # shape: (batch, n_diag, old_n_pix) → (batch, n_diag, new_n_pix)
            padded = np.pad(chunk, ((0, 0), (0, 0), (0, pad_width)), constant_values=0.0)
            dst_res.append(padded)

    # --- atomic swap ---
    os.rename(tile_path, backup_path)
    try:
        os.rename(tmp_path, tile_path)
    except Exception:
        os.rename(backup_path, tile_path)
        raise
    shutil.rmtree(backup_path, ignore_errors=True)

    new_store = zarr.storage.LocalStore(str(tile_path))
    return zarr.open_group(store=new_store, mode="a", zarr_format=3)


# ---------------------------------------------------------------------------
# Format-specific readers
# ---------------------------------------------------------------------------


def _fits_bintable_column(
    data: np.ndarray,
    *candidates: str,
    dtype: np.dtype | type | None = None,
    default: np.ndarray | None = None,
) -> np.ndarray:
    """Read a column from a FITS BINTABLE ``data`` recarray (not a dict).

    Tries each name case-insensitively (``AND_MASK`` vs ``and_mask``).  Returns
    *default* when no candidate exists and *default* is provided.
    """
    names = data.dtype.names
    if not names:
        if default is not None:
            return np.asarray(default)
        raise KeyError("FITS BINTABLE has no named columns")
    by_lower = {n.lower(): n for n in names}
    for cand in candidates:
        key = by_lower.get(cand.lower())
        if key is not None:
            out = np.asarray(data[key])
            if dtype is not None:
                out = out.astype(dtype, copy=False)
            return out
    if default is not None:
        return np.asarray(default)
    raise KeyError(
        f"None of {candidates!r} in FITS BINTABLE; available: {list(names)}"
    )


def _fits_header_keyword(header, name: str) -> object | None:
    """Return a primary-header keyword value (case-insensitive), or None."""
    target = name.upper()
    for key in header.keys():
        if key and str(key).upper() == target:
            return header[key]
    return None


def _sdss_spall_hdu(hdul: fits.HDUList) -> fits.BinTableHDU | None:
    """HDU 2 ``SPALL`` (one spAll/specObj row per spec file), if present."""
    for hdu in hdul:
        if (hdu.name or "").strip().upper() == "SPALL" and isinstance(hdu, fits.BinTableHDU):
            if hdu.data is not None and len(hdu.data) > 0:
                return hdu
    if len(hdul) > 2:
        hdu = hdul[2]
        if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None and len(hdu.data) > 0:
            return hdu
    return None


def _fits_bintable_scalar(data: np.ndarray, row_index: int, *candidates: str) -> object | None:
    """Scalar value from one row of a FITS BINTABLE ``data`` recarray."""
    names = data.dtype.names
    if not names:
        return None
    by_lower = {n.lower(): n for n in names}
    for cand in candidates:
        key = by_lower.get(cand.lower())
        if key is not None:
            return data[key][row_index]
    return None


def _sdss_source_id(hdul: fits.HDUList, link_id_col: str | None) -> int:
    """Resolve object ID for ``spec-PLATE-MJD-FIBER.fits`` (header + SPALL HDU).

    ``SPECOBJID`` and most spAll columns live in the **SPALL** BINTABLE (HDU 2),
    not in the primary header.  ``THING_ID`` is often duplicated on HDU 0.
    """
    from data_lake.ingest.fits_to_parquet import normalize_object_id

    phdr = hdul[0].header
    spall = _sdss_spall_hdu(hdul)

    candidates: list[str] = []
    if link_id_col:
        candidates.append(link_id_col)
    for name in (
        "SPECOBJID",
        "SPEC_OBJID",
        "OBJID",
        "THING_ID",
        "TARGETID",
        "SOURCE_ID",
    ):
        if name.lower() not in {c.lower() for c in candidates}:
            candidates.append(name)

    for cand in candidates:
        val = _fits_header_keyword(phdr, cand)
        if val is not None:
            return normalize_object_id(val)

    if spall is not None:
        for cand in candidates:
            val = _fits_bintable_scalar(spall.data, 0, cand)
            if val is not None:
                return normalize_object_id(val)

    hdr_keys = [k for k in phdr.keys() if k and not str(k).startswith("HISTORY")]
    spall_names = list(spall.data.dtype.names or ()) if spall is not None else []
    raise KeyError(
        "Could not resolve SDSS spectrum object ID"
        + (f" (requested {link_id_col!r})" if link_id_col else "")
        + ". Looked in the primary header and SPALL (HDU 2). "
        f"Header sample: {hdr_keys[:25]}{'…' if len(hdr_keys) > 25 else ''}"
        + (
            f"; SPALL columns: {spall_names[:25]}{'…' if len(spall_names) > 25 else ''}"
            if spall_names
            else "; SPALL missing or empty"
        )
    )


def _read_sdss_boss(
    hdul: fits.HDUList,
    *,
    link_id_col: str | None = None,
    ra_col: str = "RA",
    dec_col: str = "DEC",
) -> tuple[list[SpectrumRecord], dict]:
    """
    Read an SDSS/BOSS spec-*.fits file.

    The COADD extension (HDU 1) is a BINTABLE with columns:
      FLUX, IVAR, AND_MASK, LOGLAM  (one row = one pixel)
    Object metadata is in HDU 2 (spAll-style BINTABLE, one row).
    """
    records: list[SpectrumRecord] = []
    coadd_hdu = hdul["COADD"]
    data = coadd_hdu.data

    flux = _fits_bintable_column(data, "flux", dtype=np.float32)
    ivar = _fits_bintable_column(data, "ivar", dtype=np.float32)
    loglam = _fits_bintable_column(data, "loglam", dtype=np.float64)
    mask = _fits_bintable_column(
        data,
        "and_mask",
        "mask",
        "or_mask",
        dtype=np.uint8,
        default=np.zeros(len(flux), dtype=np.uint8),
    )
    wavelength = 10.0 ** loglam

    # Object-level header
    phdr = hdul[0].header
    ra, dec = sky_from_fits_header(phdr, ra_col, dec_col)
    if ra_col not in phdr and "PLUG_RA" in phdr:
        ra = float(phdr["PLUG_RA"])
    if dec_col not in phdr and "PLUG_DEC" in phdr:
        dec = float(phdr["PLUG_DEC"])
    source_id = _sdss_source_id(hdul, link_id_col)
    meta = {
        "z":       float(phdr.get("Z", 0.0)),
        "z_err":   float(phdr.get("Z_ERR", 0.0)),
        "snr":     float(phdr.get("SN_MEDIAN_ALL", 0.0)),
        "exptime": float(phdr.get("EXPTIME", 0.0)),
        "R":       float(phdr.get("SPEC_RES", 2000.0)),
        "instr":   "SDSS",
    }

    wcs_attrs = {
        "ctype": "WAVE-LOG",
        "crval": float(loglam[0]),
        "cdelt": float(loglam[1] - loglam[0]) if len(loglam) > 1 else 0.0,
        "crpix": 1.0,
        "unit": "Angstrom",
        "air_or_vacuum": "vacuum",
        "n_pix": len(flux),
    }

    records.append(SpectrumRecord(
        source_id=source_id, ra=ra, dec=dec,
        flux=flux, ivar=ivar, mask=mask,
        wavelength=wavelength,
        meta=meta,
    ))
    return records, wcs_attrs


def _spplate_fiber_table_hdu(hdul: fits.HDUList) -> fits.BinTableHDU | None:
    """Per-fiber metadata BINTABLE (typically HDU 5 PLUGMAP on spPlate files)."""
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None and len(hdu.data) > 0:
            names = hdu.data.dtype.names or ()
            if "FIBERID" in names or "fiberid" in {n.lower() for n in names}:
                return hdu
    return None


def _spplate_hdu_by_name(hdul: fits.HDUList, name: str) -> fits.ImageHDU | None:
    target = name.upper()
    for hdu in hdul:
        if (hdu.name or "").strip().upper() == target and hdu.data is not None:
            return hdu  # type: ignore[return-value]
    return None


def _spplate_2d_as_fiber_by_pix(data: np.ndarray, phdr: fits.Header) -> np.ndarray:
    """Ensure a 2-D spPlate image is ``(n_fiber, n_pix)`` (one spectrum per row)."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"spPlate 2-D HDU must be rank 2, got shape {arr.shape}")
    naxis1 = int(phdr.get("NAXIS1", 0) or 0)
    naxis2 = int(phdr.get("NAXIS2", 0) or 0)
    if naxis1 and naxis2:
        if arr.shape == (naxis2, naxis1):
            return arr
        if arr.shape == (naxis1, naxis2):
            return arr.T
    # Heuristic: spectral axis is the longer dimension (~3800 px).
    if arr.shape[0] > arr.shape[1] and arr.shape[1] in (500, 640, 1000):
        return arr.T
    return arr


def _spplate_flux_and_calib_hdus(
    hdul: fits.HDUList,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return ``(flux, ivar, and_mask, or_mask)`` each shaped ``(n_fiber, n_pix)``."""
    phdr = hdul[0].header
    flux_2d = _spplate_2d_as_fiber_by_pix(hdul[0].data, phdr)

    ivar_2d: np.ndarray | None = None
    sigma_2d: np.ndarray | None = None
    ivar_hdu = _spplate_hdu_by_name(hdul, "IVAR")
    if ivar_hdu is not None:
        ivar_2d = _spplate_2d_as_fiber_by_pix(ivar_hdu.data, ivar_hdu.header)
    elif len(hdul) > 1 and hdul[1].data is not None and getattr(hdul[1].data, "ndim", 0) == 2:
        h1 = hdul[1]
        arr = _spplate_2d_as_fiber_by_pix(h1.data, h1.header)
        if (h1.name or "").strip().upper() == "IVAR":
            ivar_2d = arr
        elif arr.shape == flux_2d.shape:
            sigma_2d = arr

    and_mask: np.ndarray | None = None
    or_mask: np.ndarray | None = None
    for name, slot in (("ANDMASK", "and"), ("ORMASK", "or")):
        hdu = _spplate_hdu_by_name(hdul, name)
        if hdu is None:
            continue
        arr = _spplate_2d_as_fiber_by_pix(hdu.data, hdu.header)
        if arr.shape != flux_2d.shape:
            log.warning(
                "spPlate %s shape %s != flux shape %s; skipping",
                name,
                arr.shape,
                flux_2d.shape,
            )
            continue
        if slot == "and":
            and_mask = arr.astype(np.int32, copy=False)
        else:
            or_mask = arr.astype(np.int32, copy=False)

    if and_mask is None and or_mask is None:
        for idx in (2, 3):
            if len(hdul) <= idx or hdul[idx].data is None:
                continue
            arr = hdul[idx].data
            if getattr(arr, "ndim", 0) != 2:
                continue
            arr2 = _spplate_2d_as_fiber_by_pix(arr, hdul[idx].header)
            if arr2.shape != flux_2d.shape:
                continue
            hname = (hdul[idx].name or "").upper()
            if "AND" in hname:
                and_mask = arr2.astype(np.int32, copy=False)
            elif "OR" in hname:
                or_mask = arr2.astype(np.int32, copy=False)
            elif and_mask is None:
                and_mask = arr2.astype(np.int32, copy=False)
            else:
                or_mask = arr2.astype(np.int32, copy=False)

    return flux_2d, ivar_2d, and_mask, or_mask


def _read_sdss_spplate(
    hdul: fits.HDUList,
    *,
    path: Path | None,
    fiber_to_specobjid: dict[int, int],
    ra_col: str = "RA",
    dec_col: str = "DEC",
    skip_unmatched: bool = True,
) -> tuple[list[SpectrumRecord], dict]:
    """
    Read an SDSS/BOSS ``spPlate-PLATE-MJD.fits`` file (up to 640 fibers).

    ``fiber_to_specobjid`` maps 1-based ``FIBERID`` → normalized ``SPECOBJID``.
    Fibers without a lookup entry are skipped when ``skip_unmatched`` is True.
    """
    from data_lake.ingest.sdss_specobj_lookup import spplate_plate_mjd_from_hdul

    plate, mjd = spplate_plate_mjd_from_hdul(hdul, path)
    phdr = hdul[0].header
    if "COEFF0" not in phdr or "COEFF1" not in phdr:
        raise KeyError("spPlate primary header missing COEFF0/COEFF1 wavelength calibration")
    coeff0 = float(phdr["COEFF0"])
    coeff1 = float(phdr["COEFF1"])

    flux_2d, ivar_2d, mask_and, mask_or = _spplate_flux_and_calib_hdus(hdul)
    n_fiber, n_pix = flux_2d.shape

    pix = np.arange(n_pix, dtype=np.float64)
    loglam = coeff0 + coeff1 * pix
    wavelength = (10.0 ** loglam).astype(np.float64)

    ftable = _spplate_fiber_table_hdu(hdul)
    if ftable is None:
        raise ValueError("spPlate: no per-fiber BINTABLE with FIBERID column found")

    fdata = ftable.data
    fiber_col = _fits_bintable_column(fdata, "fiberid", "FIBERID")
    ra_arr = _fits_bintable_column(fdata, ra_col.lower(), ra_col, "RA")
    dec_arr = _fits_bintable_column(fdata, dec_col.lower(), dec_col, "DEC")

    wcs_attrs = {
        "ctype": "WAVE-LOG",
        "crval": float(loglam[0]),
        "cdelt": float(coeff1),
        "crpix": 1.0,
        "unit": "Angstrom",
        "air_or_vacuum": "vacuum",
        "n_pix": int(n_pix),
    }

    records: list[SpectrumRecord] = []
    n_table = len(fdata)
    for row_i in range(min(n_fiber, n_table)):
        fiber_id = int(fiber_col[row_i])
        source_id = fiber_to_specobjid.get(fiber_id)
        if source_id is None:
            if skip_unmatched:
                continue
            raise KeyError(
                f"No SPECOBJID lookup for survey plate={plate} mjd={mjd} fiber={fiber_id}"
            )

        flux = np.asarray(flux_2d[row_i], dtype=np.float32)
        if not np.any(np.isfinite(flux)) or np.all(flux == 0):
            continue

        if ivar_2d is not None:
            ivar = np.asarray(ivar_2d[row_i], dtype=np.float32)
        else:
            ivar = np.ones(n_pix, dtype=np.float32)

        ra = float(ra_arr[row_i])
        dec = float(dec_arr[row_i])
        if not is_valid_sky_position(ra, dec):
            continue

        mask = np.zeros(n_pix, dtype=np.uint8)
        if mask_and is not None:
            m = np.asarray(mask_and[row_i], dtype=np.int32)
            mask = np.clip(m, 0, 255).astype(np.uint8)
        if mask_or is not None:
            m = np.asarray(mask_or[row_i], dtype=np.int32)
            mask = np.clip(mask | np.clip(m, 0, 255), 0, 255).astype(np.uint8)

        meta = {
            "z": 0.0,
            "z_err": 0.0,
            "snr": 0.0,
            "exptime": float(phdr.get("EXPTIME", 0.0)),
            "R": float(phdr.get("SPEC_RES", 2000.0)),
            "instr": "SDSS",
            "plate": plate,
            "mjd": mjd,
            "fiber": fiber_id,
        }
        records.append(SpectrumRecord(
            source_id=source_id,
            ra=ra,
            dec=dec,
            flux=flux,
            ivar=ivar,
            mask=mask,
            wavelength=wavelength.copy(),
            meta=meta,
        ))

    if not records:
        log.warning(
            "spPlate %s: no spectra after lookup (plate=%s mjd=%s; %d fibers in file)",
            path.name if path else "file",
            plate,
            mjd,
            n_table,
        )
    return records, wcs_attrs


def _desi_read_spectra_skip_hdus(*, with_resolution: bool) -> set[str]:
    """HDUs we do not need for flux/ivar/mask + fibermap ingest (smaller FITS read)."""
    skip = {"EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG"}
    if not with_resolution:
        skip.add("RESOLUTION")
    return skip


def _read_desi_with_desispec(
    path: Path,
    with_resolution: bool = False,
    link_id_col: str | None = None,
) -> tuple[list[SpectrumRecord], dict, list[np.ndarray] | None, np.ndarray | None]:
    """
    Read a DESI coadd-*.fits file using desispec.

    Uses ``desispec.io.read_spectra`` + ``desispec.coaddition.coadd_cameras`` for
    IVAR-weighted combination of the B/R/Z arms onto a single monotonic grid.
    This is the only scientifically-correct camera-coadd path; the previous
    approach (np.concatenate of arms) produced duplicated wavelength regions at
    the B/R and R/Z overlaps.

    Parameters
    ----------
    path:
        Path to the DESI coadd-*.fits file.
    with_resolution:
        When True, extract the banded resolution matrix ``(n_diag, N_pix)``
        per source from ``coadded.R["brz"]``.

    Returns
    -------
    (records, wcs_attrs, resolution_diags, resolution_offsets)
        ``resolution_diags`` is a list of (n_diag, N_pix) float32 arrays, one
        per source, or None when ``with_resolution=False``.
        ``resolution_offsets`` is a (n_diag,) int array or None.
    """
    desispec = _import_desispec()

    # ``single=True``: float32 from disk (matches Zarr).  ``skip_hdus`` avoids
    # reading EXP_FIBERMAP / SCORES / etc. — often a large fraction of coadd FITS.
    spectra = desispec.io.read_spectra(
        str(path),
        single=True,
        skip_hdus=_desi_read_spectra_skip_hdus(with_resolution=with_resolution),
    )
    coadded = desispec.coaddition.coadd_cameras(spectra)

    wave_all = np.asarray(coadded.wave["brz"], dtype=np.float64)
    flux_brz = np.asarray(coadded.flux["brz"], dtype=np.float32)
    ivar_brz = np.asarray(coadded.ivar["brz"], dtype=np.float32)
    mask_brz = np.asarray(coadded.mask["brz"], dtype=np.uint8)

    n_spec, n_pix = flux_brz.shape
    wcs_attrs = {
        "ctype": "WAVE",
        "crval": float(wave_all[0]),
        "cdelt": float(wave_all[1] - wave_all[0]) if n_pix > 1 else 0.0,
        "crpix": 1.0,
        "unit": "Angstrom",
        "air_or_vacuum": "vacuum",
        "n_pix": int(n_pix),
    }

    # Resolution matrix (opt-in)
    res_diags: list[np.ndarray] | None = None
    res_offsets: np.ndarray | None = None
    if with_resolution and coadded.R is not None and "brz" in coadded.R:
        res_diags = []
        for i in range(n_spec):
            r_obj = coadded.R["brz"][i]
            res_diags.append(np.asarray(r_obj.data, dtype=np.float32))
        # offsets are the same for every spectrum (grid-defined)
        res_offsets = np.asarray(coadded.R["brz"][0].offsets, dtype=np.int32)

    # Fibermap column names differ between pipeline releases; try in order.
    fmap = coadded.fibermap
    def _fmap_col(row, *names: str, default=0.0):
        for name in names:
            if name in fmap.colnames:
                return row[name]
        return default

    records: list[SpectrumRecord] = []
    for i in range(n_spec):
        row = fmap[i]
        ra  = float(_fmap_col(row, "TARGET_RA",  "RA_TARGET",  "FIBER_RA",  default=0.0))
        dec = float(_fmap_col(row, "TARGET_DEC", "DEC_TARGET", "FIBER_DEC", default=0.0))
        sid_key = link_id_col or "TARGETID"
        if sid_key not in fmap.colnames:
            raise KeyError(
                f"Fibermap column {sid_key!r} not found for object ID. "
                f"Available: {list(fmap.colnames)[:30]}"
            )
        from data_lake.ingest.fits_to_parquet import normalize_object_id

        source_id = normalize_object_id(row[sid_key])
        meta = {
            "z":       float(_fmap_col(row, "Z",    default=0.0)),
            "z_err":   float(_fmap_col(row, "ZERR", default=0.0)),
            "snr":     float(np.median(flux_brz[i] * np.sqrt(np.where(ivar_brz[i] > 0, ivar_brz[i], 0)))),
            "exptime": float(_fmap_col(row, "EXPTIME", default=0.0)),
            "R":       3000.0,
            "instr":   "DESI",
        }
        records.append(SpectrumRecord(
            source_id=source_id, ra=ra, dec=dec,
            flux=flux_brz[i], ivar=ivar_brz[i], mask=mask_brz[i],
            wavelength=wave_all,
            meta=meta,
        ))

    return records, wcs_attrs, res_diags, res_offsets


def _generic_extension_to_2d(
    data: np.ndarray,
    *,
    n_spec: int,
    n_pix: int,
    label: str,
) -> np.ndarray:
    """Coerce a flux/variance/ivar extension to ``(n_spec, n_pix)``."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != n_pix:
            raise ValueError(
                f"{label} length {arr.shape[0]} does not match flux length {n_pix}"
            )
        if n_spec != 1:
            raise ValueError(
                f"{label} is 1-D but flux HDU has {n_spec} spectra"
            )
        return arr[np.newaxis, :]
    if arr.ndim == 2:
        if arr.shape == (n_spec, n_pix):
            return arr
        if arr.shape == (n_pix, n_spec):
            return arr.T
        raise ValueError(
            f"{label} shape {arr.shape} does not match flux shape ({n_spec}, {n_pix})"
        )
    raise ValueError(f"{label} must be 1-D or 2-D, got shape {arr.shape}")


def _variance_to_ivar(variance: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(variance > 0.0, 1.0 / variance, 0.0).astype(np.float32)


def _load_generic_ivar_2d(
    hdul: fits.HDUList,
    *,
    flux_hdu_idx: int,
    n_spec: int,
    n_pix: int,
) -> np.ndarray | None:
    """
    Load per-pixel IVAR from a sibling extension, if present.

    Supports ``IVAR`` (used directly) and ``VARIANCE`` / ``VAR`` (converted to IVAR).
    """
    for i, hdu in enumerate(hdul):
        if i == flux_hdu_idx or hdu.data is None:
            continue
        name = (hdu.name or "").strip().upper()
        if name not in ("IVAR", "VARIANCE", "VAR"):
            continue
        arr_2d = _generic_extension_to_2d(
            hdu.data, n_spec=n_spec, n_pix=n_pix, label=name,
        )
        if name == "IVAR":
            return arr_2d.astype(np.float32)
        return _variance_to_ivar(arr_2d)
    return None


def _read_generic_1d(
    hdul: fits.HDUList,
    image_hdu: int = 0,
    *,
    link_id_col: str | None = None,
    ra_col: str = "RA",
    dec_col: str = "DEC",
) -> tuple[list[SpectrumRecord], dict]:
    """
    Read a generic 1-D FITS spectrum (spectral WCS in primary header).

    Handles both single-spectrum (1-D) and multi-spectrum (2-D) image HDUs.
    When a sibling ``VARIANCE`` (or ``VAR``) / ``IVAR`` extension is present, it
    is aligned to the flux HDU and converted to IVAR (variance → ``1/var``).
    """
    hdu = hdul[image_hdu]
    data = np.array(hdu.data, dtype=np.float64)
    header = hdu.header

    if data.ndim == 1:
        spectra_2d = data[np.newaxis, :]
    elif data.ndim == 2:
        spectra_2d = data
    else:
        raise ValueError(f"Unexpected data shape {data.shape} in HDU {image_hdu}")

    n_spec, n_pix = spectra_2d.shape
    wavelength = _wavelength_from_wcs(header, n_pix)
    wcs_attrs = _wcs_attrs_from_header(header, n_pix)

    base_id = object_id_from_fits_header(header, link_id_col, hdu_index=image_hdu)
    base_ra, base_dec = sky_from_fits_header(header, ra_col, dec_col)
    ivar_2d = _load_generic_ivar_2d(
        hdul, flux_hdu_idx=image_hdu, n_spec=n_spec, n_pix=n_pix,
    )

    records: list[SpectrumRecord] = []
    for i in range(n_spec):
        ra, dec = base_ra, base_dec
        source_id = base_id if n_spec == 1 else base_id + i
        if ivar_2d is not None:
            ivar = ivar_2d[i]
        else:
            ivar = np.ones(n_pix, dtype=np.float32)
        meta = {
            "z":       float(header.get("Z", 0.0)),
            "z_err":   float(header.get("Z_ERR", 0.0)),
            "snr":     0.0,
            "exptime": float(header.get("EXPTIME", 0.0)),
            "R":       float(header.get("SPEC_RES", 1000.0)),
            "instr":   str(header.get("INSTRUME", "UNKNOWN"))[:16],
        }
        records.append(SpectrumRecord(
            source_id=source_id + i, ra=ra, dec=dec,
            flux=spectra_2d[i].astype(np.float32),
            ivar=ivar,
            mask=np.zeros(n_pix, dtype=np.uint8),
            wavelength=wavelength,
            meta=meta,
        ))
    return records, wcs_attrs


def _wig_catalog_filename_key(path: Path) -> str:
    """Catalog match key for WiggleZ (e.g. ``wig225415.fits`` from ``wig225415.fits.gz``)."""
    name = path.name
    if name.lower().endswith(".fits.gz"):
        return name[: -len(".gz")]
    return name


def _wig_source_id_from_path(path: Path) -> int:
    """``source_id`` matching catalog rows keyed by spectrum filename."""
    from data_lake.ingest.fits_to_parquet import normalize_object_id

    return normalize_object_id(_wig_catalog_filename_key(path))


def _is_wig_spectrum_layout(hdul: fits.HDUList) -> bool:
    """True for 1-D flux + separate VARIANCE extension (WiggleZ-style)."""
    if len(hdul) < 2 or hdul[0].data is None:
        return False
    shape = np.asarray(hdul[0].data).shape
    if len(shape) != 1:
        return False
    hdu_names = [(h.name or "").strip().upper() for h in hdul]
    if not any(n in ("VARIANCE", "VAR") for n in hdu_names):
        return False
    ext = (hdul[0].header.get("EXTNAME") or hdul[0].name or "").strip().upper()
    return ext in ("SPECTRUM", "")


def _is_wig_hdul(hdul: fits.HDUList, path: Path) -> bool:
    """Return True if the file looks like a WiggleZ 1-D spectrum FITS."""
    if not path.stem.lower().startswith("wig"):
        return False
    return _is_wig_spectrum_layout(hdul)


def _read_wig_spectrum(
    hdul: fits.HDUList,
    source_path: Path,
) -> tuple[list[SpectrumRecord], dict]:
    """
    Read a WiggleZ 1-D spectrum (generic flux/variance layout).

    ``source_id`` is ``normalize_object_id(<basename>)`` so it matches the
    catalog column that stores the spectrum filename (e.g. ``wig225415.fits``).
    Sky position uses ``RA_OBJ`` / ``DEC_OBJ``.
    """
    from dataclasses import replace

    source_id = _wig_source_id_from_path(source_path)
    records, wcs_attrs = _read_generic_1d(
        hdul,
        ra_col="RA_OBJ",
        dec_col="DEC_OBJ",
    )
    out: list[SpectrumRecord] = []
    for rec in records:
        meta = dict(rec.meta)
        meta["instr"] = str(meta.get("instr", "WiggleZ"))[:16]
        out.append(replace(rec, source_id=source_id, meta=meta))
    return out, wcs_attrs


def _is_ozdes_stacked_layout(hdul: fits.HDUList) -> bool:
    """True when HDU 0/1/2 are stacked flux, variance, and bad-pixel mask."""
    if len(hdul) < 3:
        return False
    if hdul[0].data is None or hdul[1].data is None or hdul[2].data is None:
        return False
    flux_shape = np.asarray(hdul[0].data).shape
    if len(flux_shape) != 1:
        return False
    n_pix = int(flux_shape[0])
    if n_pix <= 0:
        return False
    names = [(h.name or "").strip().upper() for h in hdul[:3]]
    if names[1] not in ("VARIANCE", "VAR"):
        return False
    if names[2] not in ("BADPIX", "BAD_PIX", "MASK"):
        return False
    for idx in (1, 2):
        sh = np.asarray(hdul[idx].data).shape
        if sh != (n_pix,):
            return False
    return True


def _is_ozdes_hdul(hdul: fits.HDUList, path: Path) -> bool:
    """Return True if the file looks like an OzDES stacked 1-D spectrum FITS."""
    if not _is_ozdes_stacked_layout(hdul):
        return False
    if path.stem.lower().startswith("ozdes"):
        return True
    return "SOURCE" in hdul[0].header


def _read_ozdes_spectrum(
    hdul: fits.HDUList,
    source_path: Path,
) -> tuple[list[SpectrumRecord], dict]:
    """
    Read an OzDES stacked 1-D spectrum (PRIMARY + VARIANCE + BADPIX).

    Only the stacked HDUs 0–2 are ingested; per-epoch ``SPECTRUM_*`` extensions
    are ignored.  ``source_id`` is ``normalize_object_id(path.name)`` so it
    matches a catalog column that stores the spectrum filename
    (e.g. ``OzDES-DR2_00001.fits``).
    """
    from data_lake.ingest.fits_to_parquet import normalize_object_id

    if not _is_ozdes_stacked_layout(hdul):
        summary = _summarize_fits_hdus(hdul)
        raise ValueError(
            "OzDES stacked layout not found (expected HDU0 flux, HDU1 VARIANCE, "
            f"HDU2 BADPIX). Found: {summary}"
        )

    flux_hdu = hdul[0]
    header = flux_hdu.header
    flux = np.asarray(flux_hdu.data, dtype=np.float32)
    n_pix = int(flux.shape[0])

    variance = _generic_extension_to_2d(
        hdul[1].data, n_spec=1, n_pix=n_pix, label="VARIANCE",
    )[0]
    ivar = _variance_to_ivar(variance)

    bad = np.asarray(hdul[2].data, dtype=np.float64)
    bad = np.where(np.isfinite(bad), bad, 0.0)
    mask = (bad != 0).astype(np.uint8)

    source_id = normalize_object_id(source_path.name)
    ra, dec = sky_from_fits_header(header, "RA", "DEC")
    wavelength = _wavelength_from_wcs(header, n_pix)
    wcs_attrs = _wcs_attrs_from_header(header, n_pix)

    meta: dict[str, Any] = {
        "z": float(header.get("Z", 0.0)),
        "z_err": float(header.get("Z_ERR", 0.0)),
        "snr": 0.0,
        "exptime": float(header.get("EXPTIME", 0.0)),
        "R": float(header.get("SPEC_RES", 1000.0)),
        "instr": str(header.get("INSTRUME", "OzDES"))[:16],
    }
    record = SpectrumRecord(
        source_id=source_id,
        ra=ra,
        dec=dec,
        flux=flux,
        ivar=ivar,
        mask=mask,
        wavelength=wavelength,
        meta=meta,
    )
    return [record], wcs_attrs


def _gama_row_labels(header: fits.Header, n_rows: int) -> dict[str, int]:
    """Map normalised row labels (e.g. ``spectrum``) to 0-based row indices."""
    labels: dict[str, int] = {}
    for i in range(1, max(n_rows, 1) + 5):
        key = f"ROW{i}"
        if key not in header:
            continue
        labels[str(header[key]).strip().lower()] = i - 1
    return labels


def _is_gama_stacked_layout(hdul: fits.HDUList) -> bool:
    """True when PRIMARY is a GAMA-style stacked spectrum image (rows × pixels)."""
    if not hdul or hdul[0].data is None:
        return False
    arr = np.asarray(hdul[0].data)
    if arr.ndim != 2 or arr.shape[0] >= arr.shape[1] or arr.shape[0] < 2:
        return False
    header = hdul[0].header
    row1 = str(header.get("ROW1", "")).strip().upper()
    if row1 == "SPECTRUM":
        return True
    origin = str(header.get("ORIGIN", "")).strip().upper()
    ctype1 = str(header.get("CTYPE1", "")).strip().upper()
    return (
        origin == "GAMA"
        and "SPECID" in header
        and ctype1.startswith("WAVE")
    )


def _is_gama_hdul(hdul: fits.HDUList, path: Path) -> bool:
    """Return True if the file looks like a GAMA stacked 1-D spectrum FITS."""
    if not _is_gama_stacked_layout(hdul):
        return False
    if path.stem.upper().startswith(("G23_", "GAMA")):
        return True
    return str(hdul[0].header.get("ORIGIN", "")).strip().upper() == "GAMA"


def _read_gama_spectrum(
    hdul: fits.HDUList,
    source_path: Path,
    *,
    link_id_col: str = "SPECID",
    ra_col: str = "RA",
    dec_col: str = "DEC",
) -> tuple[list[SpectrumRecord], dict]:
    """
    Read a GAMA stacked 1-D spectrum from the PRIMARY image.

    Expected layout: 2-D array ``(n_row, n_pix)`` with ``ROW1=Spectrum``,
    ``ROW2=Error`` (1-σ noise), and optional calibration/sky rows.  Wavelength
    comes from the primary spectral WCS.  ``source_id`` defaults to
    ``normalize_object_id(SPECID)`` from the primary header.
    """
    if not _is_gama_stacked_layout(hdul):
        summary = _summarize_fits_hdus(hdul)
        raise ValueError(
            "GAMA stacked layout not found (expected PRIMARY (n_row, n_pix) with "
            f"ROW1=Spectrum and SPECID). Found: {summary}"
        )

    hdu = hdul[0]
    header = hdu.header
    arr = np.asarray(hdu.data, dtype=np.float64)
    n_rows, n_pix = int(arr.shape[0]), int(arr.shape[1])
    if n_pix <= 0:
        raise ValueError(f"GAMA spectrum has zero pixels in {source_path.name}")

    row_labels = _gama_row_labels(header, n_rows)
    flux_row = row_labels.get("spectrum", 0)
    if flux_row < 0 or flux_row >= n_rows:
        raise ValueError(f"GAMA flux row index {flux_row} out of range for shape {arr.shape}")

    flux = arr[flux_row].astype(np.float32)
    error_row = row_labels.get("error")
    if error_row is not None and 0 <= error_row < n_rows:
        sigma = arr[error_row]
        valid = np.isfinite(sigma) & (sigma > 0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ivar64 = np.where(valid, 1.0 / (sigma * sigma), 0.0)
        ivar = np.clip(ivar64, 0.0, np.finfo(np.float32).max).astype(np.float32)
    else:
        ivar = np.ones(n_pix, dtype=np.float32)

    mask = ((~np.isfinite(flux)) | (ivar <= 0.0)).astype(np.uint8)

    source_id = object_id_from_fits_header(header, link_id_col, hdu_index=0)
    ra, dec = sky_from_fits_header(header, ra_col, dec_col)
    wavelength = _wavelength_from_wcs(header, n_pix)
    wcs_attrs = _wcs_attrs_from_header(header, n_pix)

    sn_val = header.get("SN", header.get("SNR", 0.0))
    meta: dict[str, Any] = {
        "z": float(header.get("Z", 0.0)),
        "z_err": float(header.get("Z_ERR", 0.0)),
        "snr": float(sn_val) if sn_val not in ("", None) else 0.0,
        "exptime": float(header.get("T_EXP", header.get("EXPTIME", 0.0))),
        "R": float(header.get("SPEC_RES", 1000.0)),
        "instr": str(header.get("INSTRUME", "GAMA"))[:16],
    }
    record = SpectrumRecord(
        source_id=source_id,
        ra=ra,
        dec=dec,
        flux=flux,
        ivar=ivar,
        mask=mask,
        wavelength=wavelength,
        meta=meta,
    )
    return [record], wcs_attrs


def _is_zcosmos_container_hdu(hdu: fits.hdu.base.ExtensionHDU | fits.PrimaryHDU) -> bool:
    """True when an HDU is a zCOSMOS spectral container binary table."""
    if not isinstance(hdu, fits.BinTableHDU):
        return False
    cols = {c.upper() for c in hdu.columns.names}
    required = {"WAVE", "FLUX_REDUCED", "ERR"}
    return required.issubset(cols)


def _find_zcosmos_container_hdu(hdul: fits.HDUList) -> tuple[int, fits.BinTableHDU]:
    """Return the zCOSMOS spectral container table HDU."""
    for i, hdu in enumerate(hdul):
        if _is_zcosmos_container_hdu(hdu):
            return i, hdu
    summary = _summarize_fits_hdus(hdul)
    raise ValueError(
        "zCOSMOS spectral container not found (expected BinTable with columns "
        f"WAVE/FLUX_REDUCED/ERR). Found: {summary}"
    )


def _is_zcosmos_hdul(hdul: fits.HDUList, path: Path) -> bool:
    """Return True if this file looks like a zCOSMOS 1-D spectrum file."""
    if not path.name.lower().startswith("zcosmos"):
        return False
    try:
        _find_zcosmos_container_hdu(hdul)
    except ValueError:
        return False
    return True


def _read_zcosmos_spectrum(
    hdul: fits.HDUList,
    source_path: Path,
) -> tuple[list[SpectrumRecord], dict]:
    """
    Read a zCOSMOS 1-D spectrum from the spectral-container table.

    Uses filename-based source linking: ``source_id = normalize_object_id(path.name)``.
    """
    from data_lake.ingest.fits_to_parquet import normalize_object_id

    source_id = normalize_object_id(source_path.name)
    _, shdu = _find_zcosmos_container_hdu(hdul)
    row = shdu.data[0]

    wavelength = np.asarray(row["WAVE"], dtype=np.float64)
    flux = np.asarray(row["FLUX_REDUCED"], dtype=np.float32)
    err = np.asarray(row["ERR"], dtype=np.float64)

    if wavelength.ndim != 1 or flux.ndim != 1 or err.ndim != 1:
        raise ValueError(
            "zCOSMOS arrays must be 1-D "
            f"(got wave={wavelength.shape}, flux={flux.shape}, err={err.shape})"
        )
    if not (len(wavelength) == len(flux) == len(err)):
        raise ValueError(
            "zCOSMOS array lengths mismatch: "
            f"wave={len(wavelength)} flux={len(flux)} err={len(err)}"
        )
    n_pix = int(len(flux))
    finite_wave_flux = np.isfinite(wavelength) & np.isfinite(flux)
    valid_err = np.isfinite(err) & (err > 0.0)
    if np.any(valid_err):
        with np.errstate(divide="ignore", invalid="ignore"):
            ivar = np.where(valid_err, 1.0 / (err * err), 0.0).astype(np.float32)
        mask = np.where(finite_wave_flux & valid_err, 0, 1).astype(np.uint8)
    else:
        # Some zCOSMOS products carry placeholder ERR arrays (all zeros/non-finite).
        # Treat uncertainty as unavailable instead of masking the full spectrum.
        ivar = np.ones(n_pix, dtype=np.float32)
        mask = np.where(finite_wave_flux, 0, 1).astype(np.uint8)

    phdr = hdul[0].header
    shdr = shdu.header
    ra = float(shdr.get("RA", phdr.get("RA", 0.0)))
    dec = float(shdr.get("DEC", phdr.get("DEC", 0.0)))
    meta: dict[str, Any] = {
        "z": float(shdr.get("Z", phdr.get("Z", 0.0))),
        "z_err": float(shdr.get("Z_ERR", phdr.get("Z_ERR", 0.0))),
        "snr": 0.0,
        "exptime": float(shdr.get("EXPTIME", phdr.get("EXPTIME", 0.0))),
        "R": float(shdr.get("SPEC_RES", phdr.get("SPEC_RES", 1000.0))),
        "instr": str(shdr.get("INSTRUME", phdr.get("INSTRUME", "zCOSMOS")))[:16],
    }
    wcs_attrs = {
        "wcs_source": "explicit",
        "n_pix": n_pix,
    }
    return [SpectrumRecord(
        source_id=source_id,
        ra=ra,
        dec=dec,
        flux=flux,
        ivar=ivar,
        mask=mask,
        wavelength=wavelength,
        meta=meta,
    )], wcs_attrs


def _find_vandels_noise_hdu(hdul: fits.HDUList) -> tuple[int, fits.ImageHDU]:
    """Return the VANDELS 1-D noise estimate extension."""
    for i, hdu in enumerate(hdul):
        if (hdu.name or "").strip().upper() == "NOISE" and hdu.data is not None:
            return i, hdu  # type: ignore[return-value]
    summary = _summarize_fits_hdus(hdul)
    raise ValueError(
        "VANDELS NOISE extension not found (expected 1-D image HDU named NOISE). "
        f"Found: {summary}"
    )


def _is_vandels_stacked_layout(hdul: fits.HDUList) -> bool:
    """True when PRIMARY holds 1-D flux and a matching NOISE extension exists."""
    if not hdul or hdul[0].data is None:
        return False
    flux_shape = np.asarray(hdul[0].data).shape
    if len(flux_shape) != 1 or flux_shape[0] <= 0:
        return False
    n_pix = int(flux_shape[0])
    try:
        _, noise_hdu = _find_vandels_noise_hdu(hdul)
    except ValueError:
        return False
    return np.asarray(noise_hdu.data).shape == (n_pix,)


def _is_vandels_hdul(hdul: fits.HDUList, path: Path) -> bool:
    """Return True if the file looks like a VANDELS 1-D spectrum FITS."""
    if not _is_vandels_stacked_layout(hdul):
        return False
    if path.name.lower().startswith("sc_"):
        return True
    return "PND OBJID" in hdul[0].header


def _read_vandels_spectrum(
    hdul: fits.HDUList,
    source_path: Path,
) -> tuple[list[SpectrumRecord], dict]:
    """
    Read a VANDELS stacked 1-D spectrum (PRIMARY flux + NOISE extension).

    ``source_id`` is ``normalize_object_id(path.name)`` for catalog filename linkage.
    Sky position uses ``PND OBJRA`` / ``PND OBJDEC``; redshift from ``PND Z``.
    """
    if not _is_vandels_stacked_layout(hdul):
        summary = _summarize_fits_hdus(hdul)
        raise ValueError(
            "VANDELS stacked layout not found (expected PRIMARY 1-D flux + NOISE). "
            f"Found: {summary}"
        )

    source_id = _wig_source_id_from_path(source_path)
    flux_hdu = hdul[0]
    header = flux_hdu.header
    flux = np.asarray(flux_hdu.data, dtype=np.float32)
    n_pix = int(flux.shape[0])

    _, noise_hdu = _find_vandels_noise_hdu(hdul)
    noise = np.asarray(noise_hdu.data, dtype=np.float64)
    valid_noise = np.isfinite(noise) & (noise > 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ivar64 = np.where(valid_noise, 1.0 / (noise * noise), 0.0)
    ivar = np.clip(ivar64, 0.0, np.finfo(np.float32).max).astype(np.float32)

    finite = np.isfinite(flux) & np.isfinite(_wavelength_from_wcs(header, n_pix))
    mask = np.where(finite & valid_noise, 0, 1).astype(np.uint8)

    ra = float(header.get("PND OBJRA", header.get("RA", 0.0)))
    dec = float(header.get("PND OBJDEC", header.get("DEC", 0.0)))
    wavelength = _wavelength_from_wcs(header, n_pix)
    wcs_attrs = _wcs_attrs_from_header(header, n_pix)
    meta: dict[str, Any] = {
        "z": float(header.get("PND Z", header.get("Z", 0.0))),
        "z_err": float(header.get("PND ZERR", header.get("Z_ERR", 0.0))),
        "snr": 0.0,
        "exptime": float(header.get("EXPTIME", 0.0)),
        "R": float(header.get("SPEC_RES", 1000.0)),
        "instr": str(header.get("INSTRUME", "VIMOS"))[:16],
    }
    return [SpectrumRecord(
        source_id=source_id,
        ra=ra,
        dec=dec,
        flux=flux,
        ivar=ivar,
        mask=mask,
        wavelength=wavelength,
        meta=meta,
    )], wcs_attrs


def _is_vipers_table_hdu(hdu: fits.hdu.base.ExtensionHDU | fits.PrimaryHDU) -> bool:
    """True when an HDU is a VIPERS row-per-pixel spectral binary table."""
    if not isinstance(hdu, fits.BinTableHDU) or hdu.data is None or len(hdu.data) == 0:
        return False
    cols = {c.upper() for c in hdu.columns.names}
    return {"WAVES", "FLUXES", "NOISE", "MASK"}.issubset(cols)


def _find_vipers_table_hdu(hdul: fits.HDUList) -> tuple[int, fits.BinTableHDU]:
    """Return the VIPERS spectral binary table HDU."""
    for i, hdu in enumerate(hdul):
        if _is_vipers_table_hdu(hdu):
            return i, hdu
    summary = _summarize_fits_hdus(hdul)
    raise ValueError(
        "VIPERS spectral table not found (expected BinTable with columns "
        f"WAVES/FLUXES/NOISE/MASK). Found: {summary}"
    )


def _is_vipers_hdul(hdul: fits.HDUList, path: Path) -> bool:
    """Return True if the file looks like a VIPERS 1-D spectrum FITS."""
    try:
        _, thdu = _find_vipers_table_hdu(hdul)
    except ValueError:
        return False
    if path.name.lower().startswith("vipers"):
        return True
    return "ID" in thdu.header


def _read_vipers_spectrum(
    hdul: fits.HDUList,
    source_path: Path,
) -> tuple[list[SpectrumRecord], dict]:
    """
    Read a VIPERS 1-D spectrum from a row-per-pixel binary table.

    Columns ``WAVES``, ``FLUXES``, ``NOISE``, and ``MASK`` are stacked into 1-D
    arrays.  ``MASK`` values are stored as ingested (no remapping).  ``source_id``
    comes from the ``ID`` header keyword by default.
    """
    _, thdu = _find_vipers_table_hdu(hdul)
    rows = thdu.data
    wavelength = np.array([row["WAVES"] for row in rows], dtype=np.float64)
    flux = np.array([row["FLUXES"] for row in rows], dtype=np.float32)
    noise = np.array([row["NOISE"] for row in rows], dtype=np.float64)
    mask = np.array([row["MASK"] for row in rows], dtype=np.uint8)

    n_pix = int(len(flux))
    if not (len(wavelength) == len(noise) == len(mask) == n_pix):
        raise ValueError(
            "VIPERS column lengths mismatch: "
            f"wave={len(wavelength)} flux={n_pix} noise={len(noise)} mask={len(mask)}"
        )

    valid_noise = np.isfinite(noise) & (noise > 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ivar64 = np.where(valid_noise, 1.0 / (noise * noise), 0.0)
    ivar = np.clip(ivar64, 0.0, np.finfo(np.float32).max).astype(np.float32)

    header = thdu.header
    phdr = hdul[0].header
    source_id = object_id_from_fits_header(header, "ID", hdu_index=0)
    ra = float(header.get("RA", phdr.get("RA", 0.0)))
    dec = float(header.get("DEC", phdr.get("DEC", 0.0)))

    meta: dict[str, Any] = {
        "z": float(header.get("REDSHIFT", header.get("Z", 0.0))),
        "z_err": float(header.get("REDSHIFT_ERR", header.get("Z_ERR", 0.0))),
        "snr": 0.0,
        "exptime": float(header.get("EXPTIME", phdr.get("EXPTIME", 0.0))),
        "R": float(header.get("SPEC_RES", phdr.get("SPEC_RES", 1000.0))),
        "instr": str(header.get("INSTRUME", phdr.get("INSTRUME", "VIMOS")))[:16],
    }
    wcs_attrs = {
        "wcs_source": "explicit",
        "n_pix": n_pix,
    }
    return [SpectrumRecord(
        source_id=source_id,
        ra=ra,
        dec=dec,
        flux=flux,
        ivar=ivar,
        mask=mask,
        wavelength=wavelength,
        meta=meta,
    )], wcs_attrs


_VUDS_ID_KEY = "LAM CESAM VO IDENT"
_VUDS_RA_KEY = "LAM CESAM VO ALPHA"
_VUDS_DEC_KEY = "LAM CESAM VO DELTA"
_VUDS_Z_KEY = "LAM CESAM VO Z"


def _is_vuds_stacked_layout(hdul: fits.HDUList) -> bool:
    """True when PRIMARY holds a 1-D VUDS flux array with catalog ID metadata."""
    if not hdul or hdul[0].data is None:
        return False
    shape = np.asarray(hdul[0].data).shape
    if len(shape) != 1 or shape[0] <= 0:
        return False
    return _VUDS_ID_KEY in hdul[0].header


def _is_vuds_hdul(hdul: fits.HDUList, path: Path) -> bool:
    """Return True if the file looks like a VUDS 1-D spectrum FITS."""
    if not _is_vuds_stacked_layout(hdul):
        return False
    if path.name.lower().startswith("sc_"):
        return True
    return _VUDS_Z_KEY in hdul[0].header


def _read_vuds_spectrum(
    hdul: fits.HDUList,
    source_path: Path,
) -> tuple[list[SpectrumRecord], dict]:
    """
    Read a VUDS 1-D spectrum from PRIMARY (flux + spectral WCS).

    ``source_id`` comes from ``LAM CESAM VO IDENT`` by default.  Sky position
    and redshift use ``LAM CESAM VO ALPHA`` / ``DELTA`` / ``Z``.  No uncertainty
    or mask extensions are expected (IVAR defaults to 1, mask to 0).
    """
    if not _is_vuds_stacked_layout(hdul):
        summary = _summarize_fits_hdus(hdul)
        raise ValueError(
            "VUDS stacked layout not found (expected PRIMARY 1-D flux with "
            f"{_VUDS_ID_KEY!r}). Found: {summary}"
        )

    flux_hdu = hdul[0]
    header = flux_hdu.header
    flux = np.asarray(flux_hdu.data, dtype=np.float32)
    n_pix = int(flux.shape[0])

    sid_key = _VUDS_ID_KEY
    if sid_key not in header:
        source_id = object_id_from_fits_header(header, sid_key, hdu_index=0)
    else:
        from data_lake.ingest.fits_to_parquet import normalize_object_id

        raw_id = header[sid_key]
        if isinstance(raw_id, (float, np.floating)) and np.isfinite(raw_id) and raw_id == int(raw_id):
            source_id = normalize_object_id(int(raw_id))
        else:
            source_id = normalize_object_id(raw_id)
    ra = float(header.get(_VUDS_RA_KEY, header.get("RA", 0.0)))
    dec = float(header.get(_VUDS_DEC_KEY, header.get("DEC", 0.0)))
    wavelength = _wavelength_from_wcs(header, n_pix)
    wcs_attrs = _wcs_attrs_from_header(header, n_pix)

    meta: dict[str, Any] = {
        "z": float(header.get(_VUDS_Z_KEY, header.get("Z", 0.0))),
        "z_err": float(header.get("LAM CESAM VO ZERR", header.get("Z_ERR", 0.0))),
        "snr": 0.0,
        "exptime": float(header.get("EXPTIME", 0.0)),
        "R": float(header.get("SPEC_RES", 1000.0)),
        "instr": str(header.get("INSTRUME", header.get("ESO INS ID", "VIMOS")))[:16],
    }
    return [SpectrumRecord(
        source_id=source_id,
        ra=ra,
        dec=dec,
        flux=flux,
        ivar=np.ones(n_pix, dtype=np.float32),
        mask=np.zeros(n_pix, dtype=np.uint8),
        wavelength=wavelength,
        meta=meta,
    )], wcs_attrs


def _vvds_source_id_from_path(path: Path) -> int:
    """``source_id`` from ``sc_<ID>_...`` filename stem (matches catalog ``ID`` column)."""
    from data_lake.ingest.fits_to_parquet import normalize_object_id

    match = re.match(r"sc_(\d+)", path.stem, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"VVDS filename missing sc_<ID>_ prefix (expected catalog ID in name): {path.name}"
        )
    return normalize_object_id(match.group(1))


def _flatten_vvds_primary_flux(data: np.ndarray) -> np.ndarray:
    """Coerce VVDS PRIMARY data to a 1-D flux vector."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0].astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].astype(np.float32)
    raise ValueError(
        f"VVDS PRIMARY flux must be 1-D or single-row/column 2-D, got shape {arr.shape}"
    )


def _is_vvds_stacked_layout(hdul: fits.HDUList) -> bool:
    """True when PRIMARY holds a VVDS 1-D (or 1×N) flux array without VUDS metadata."""
    if not hdul or hdul[0].data is None:
        return False
    if _VUDS_ID_KEY in hdul[0].header:
        return False
    try:
        flux = _flatten_vvds_primary_flux(np.asarray(hdul[0].data))
    except ValueError:
        return False
    return flux.size > 0


def _is_vvds_hdul(hdul: fits.HDUList, path: Path) -> bool:
    """Return True if the file looks like a VVDS 1-D spectrum FITS."""
    if not path.name.lower().startswith("sc_"):
        return False
    return _is_vvds_stacked_layout(hdul)


def _read_vvds_spectrum(
    hdul: fits.HDUList,
    source_path: Path,
) -> tuple[list[SpectrumRecord], dict]:
    """
    Read a VVDS 1-D spectrum from PRIMARY (flux + spectral WCS).

    ``source_id`` is parsed from the filename ``sc_<ID>_...`` prefix.  Sky position
    uses ``RA`` / ``DEC``.  No uncertainty or mask extensions are expected.
    """
    if not _is_vvds_stacked_layout(hdul):
        summary = _summarize_fits_hdus(hdul)
        raise ValueError(
            "VVDS stacked layout not found (expected PRIMARY 1-D flux, not VUDS). "
            f"Found: {summary}"
        )

    flux_hdu = hdul[0]
    header = flux_hdu.header
    flux = _flatten_vvds_primary_flux(flux_hdu.data)
    n_pix = int(flux.shape[0])
    source_id = _vvds_source_id_from_path(source_path)
    ra = float(header.get("RA", 0.0))
    dec = float(header.get("DEC", 0.0))
    wavelength = _wavelength_from_wcs(header, n_pix)
    wcs_attrs = _wcs_attrs_from_header(header, n_pix)

    meta: dict[str, Any] = {
        "z": float(header.get("REDSHIFT", header.get("Z", 0.0))),
        "z_err": float(header.get("REDSHIFT_ERR", header.get("Z_ERR", 0.0))),
        "snr": 0.0,
        "exptime": float(header.get("EXPTIME", 0.0)),
        "R": float(header.get("SPEC_RES", 1000.0)),
        "instr": str(header.get("INSTRUME", header.get("ESO INS ID", "VIMOS")))[:16],
    }
    return [SpectrumRecord(
        source_id=source_id,
        ra=ra,
        dec=dec,
        flux=flux,
        ivar=np.ones(n_pix, dtype=np.float32),
        mask=np.zeros(n_pix, dtype=np.uint8),
        wavelength=wavelength,
        meta=meta,
    )], wcs_attrs


def _parse_2df_spectrum_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Parse 2dF-style ``(flux, variance, sky[, ...])`` image data."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D spectrum HDU, got shape {arr.shape}")
    if arr.shape[0] == 3:
        flux = arr[0]
        variance = arr[1]
        n_pix = int(arr.shape[1])
    elif arr.shape[1] == 3:
        flux = arr[:, 0]
        variance = arr[:, 1]
        n_pix = int(arr.shape[0])
    else:
        raise ValueError(
            f"expected (3, n_pix) or (n_pix, 3) spectrum layout, got {arr.shape}"
        )
    if n_pix <= 0:
        raise ValueError("spectrum HDU has zero pixels")
    return flux, variance, n_pix


def _is_2df_spectrum_hdu(hdu: fits.ImageHDU | fits.PrimaryHDU | fits.HDU) -> bool:
    """True when an HDU holds a 2dF-style 3-row spectrum image."""
    if hdu.data is None:
        return False
    try:
        arr = np.asarray(hdu.data)
    except Exception:
        return False
    if arr.ndim != 2 or 3 not in arr.shape:
        return False
    try:
        _parse_2df_spectrum_data(arr)
    except ValueError:
        return False
    return True


def _summarize_fits_hdus(hdul: fits.HDUList) -> str:
    """One-line summary of HDUs for error messages."""
    parts: list[str] = []
    for i, hdu in enumerate(hdul):
        shape = getattr(hdu.data, "shape", None)
        parts.append(f"HDU{i} {hdu.name!r} shape={shape}")
    return "; ".join(parts) if parts else "(empty HDU list)"


def _is_2df_stamp_only_hdul(hdul: fits.HDUList) -> bool:
    """True when the file looks like a 2dF stamp image without a 1-D spectrum."""
    if len(hdul) != 1:
        return False
    phdr = hdul[0].header
    if "SEQNUM" not in phdr and "BJSEL" not in phdr:
        return False
    data = hdul[0].data
    if data is None:
        return False
    shape = tuple(np.asarray(data).shape)
    return shape == (49, 49)


_2DF_DEFAULT_SOURCE_ID_COL = "SPFILE"


def _2df_filename_stem(source_path: Path) -> str:
    """Numeric/object stem from a 2dF FITS path (legacy link key)."""
    stem = source_path.stem
    for sfx in (".fits", ".fit"):
        if stem.lower().endswith(sfx):
            stem = stem[: -len(sfx)]
    return stem


def _iter_2df_spectrum_hdus(
    hdul: fits.HDUList,
) -> list[tuple[int, fits.ImageHDU | fits.PrimaryHDU]]:
    """Return all 2dF-style spectral image extensions, SPECTRUM-named first."""
    named: list[tuple[int, fits.ImageHDU | fits.PrimaryHDU]] = []
    other: list[tuple[int, fits.ImageHDU | fits.PrimaryHDU]] = []
    for i, hdu in enumerate(hdul):
        if not _is_2df_spectrum_hdu(hdu):
            continue
        if (hdu.name or "").strip().upper() == "SPECTRUM":
            named.append((i, hdu))  # type: ignore[arg-type]
        else:
            other.append((i, hdu))  # type: ignore[arg-type]
    hdus = named + other
    if hdus:
        return hdus
    summary = _summarize_fits_hdus(hdul)
    if _is_2df_stamp_only_hdul(hdul):
        raise ValueError(
            "2dF stamp-only FITS (49×49 PRIMARY image, no SPECTRUM extension). "
            "This is not a 1-D spectrum file; use the matching spectrum FITS for this "
            f"serial or remove it from the ingest list. Found: {summary}"
        )
    raise ValueError(
        "no 2dF spectral extension found (expected 2-D HDU with 3 rows: flux, variance, sky). "
        f"Found: {summary}"
    )


def _find_2df_spectrum_hdu(hdul: fits.HDUList) -> tuple[int, fits.ImageHDU | fits.PrimaryHDU]:
    """Locate the first 2dF spectral image extension (named SPECTRUM or heuristic)."""
    return _iter_2df_spectrum_hdus(hdul)[0]


def _2df_link_label_from_header(
    shdr: fits.Header,
    phdr: fits.Header,
    source_path: Path,
) -> tuple[str, bool]:
    """Resolve the catalog link label from a 2dF spectrum HDU header.

    Returns ``(label, used_fallback)`` where *used_fallback* is True when the
    filename stem was used because ``SPFILE`` was absent from headers.
    """
    key = _2DF_DEFAULT_SOURCE_ID_COL
    for hdr in (shdr, phdr):
        if key in hdr:
            label = str(hdr[key]).strip()
            if label:
                return label, False
    return _2df_filename_stem(source_path), True


def _2df_sky_from_headers(shdr: fits.Header, phdr: fits.Header) -> tuple[float, float]:
    """Per-observation sky position, falling back to PRIMARY ``RA``/``DEC``."""
    ra = shdr.get("OBSRA", phdr.get("RA", 0.0))
    dec = shdr.get("OBSDEC", phdr.get("DEC", 0.0))
    return float(ra), float(dec)


def _2df_spectra_wcs_differs(records: list[SpectrumRecord]) -> bool:
    """True when 2dF records in one file have incompatible wavelength grids."""
    if len(records) <= 1:
        return False
    ref = records[0].wavelength
    for rec in records[1:]:
        wave = rec.wavelength
        if wave is None or ref is None:
            return True
        if wave.shape != ref.shape or not np.allclose(wave, ref):
            return True
    return False


def _read_2df_spectrum(
    hdul: fits.HDUList,
    source_path: Path,
) -> tuple[list[SpectrumRecord], dict]:
    """Read a 2dFGRS 1-D spectrum FITS file (one row per SPECTRUM HDU).

    Expected layout:
    - HDU 0 (PRIMARY): object metadata (``SEQNUM``, ``NAME``, ``BJSEL``, ``RA``,
      ``DEC``).
    - HDU 1+ (SPECTRUM): 2-D image of shape ``(3, n_pix)`` where rows are
      ``[flux, variance, sky]``; per-observation metadata including ``SPFILE``,
      ``Z``, ``SNR``, ``OBSRA``/``OBSDEC``; spectral WCS in extension header
      (``CRVAL1``, ``CRPIX1``, ``CDELT1``).

    Source ID defaults to ``normalize_object_id(SPFILE)`` from each extension
    header (catalog link column).  When ``SPFILE`` is absent, falls back to the
    file basename stem with a warning.
    """
    from data_lake.ingest.fits_to_parquet import normalize_object_id

    phdr = hdul[0].header
    spec_hdus = _iter_2df_spectrum_hdus(hdul)
    records: list[SpectrumRecord] = []
    wcs_attrs: dict = {}
    seen_ids: dict[int, str] = {}
    used_stem_fallback = False

    for spec_hdu_idx, shdu in spec_hdus:
        if (shdu.name or "").strip().upper() != "SPECTRUM":
            log.warning(
                "2dF: unnamed spectral HDU in %s; using HDU %d (%r)",
                source_path.name,
                spec_hdu_idx,
                shdu.name,
            )

        shdr = shdu.header
        link_label, fallback = _2df_link_label_from_header(
            shdr, phdr, source_path,
        )
        if fallback:
            used_stem_fallback = True
        source_id = normalize_object_id(link_label)
        if source_id in seen_ids:
            raise ValueError(
                f"duplicate 2dF source_id in {source_path.name}: "
                f"SPFILE={link_label!r} and {seen_ids[source_id]!r} "
                f"both map to the same _source_id"
            )
        seen_ids[source_id] = link_label

        ra, dec = _2df_sky_from_headers(shdr, phdr)
        flux, variance, n_pix = _parse_2df_spectrum_data(np.asarray(shdu.data))
        flux = flux.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            ivar = np.where(variance > 0.0, 1.0 / variance, 0.0).astype(np.float32)
        mask = np.zeros(n_pix, dtype=np.uint8)

        wavelength = _wavelength_from_wcs(shdr, n_pix)
        if not wcs_attrs:
            wcs_attrs = _wcs_attrs_from_header(shdr, n_pix)

        snr_val = shdr.get("SNR", phdr.get("SNR", 0.0))
        meta: dict[str, Any] = {
            "z":       float(shdr.get("Z", phdr.get("Z", 0.0))),
            "z_err":   0.0,
            "snr":     float(snr_val) if snr_val not in ("", None) else 0.0,
            "exptime": float(shdr.get("EXPTIME", phdr.get("EXPTIME", 0.0))),
            "R":       float(shdr.get("SPEC_RES", 500.0)),
            "instr":   "2dFGRS",
        }

        records.append(SpectrumRecord(
            source_id=source_id,
            ra=ra,
            dec=dec,
            flux=flux,
            ivar=ivar,
            mask=mask,
            wavelength=wavelength,
            meta=meta,
        ))

    if used_stem_fallback:
        log.warning(
            "2dF: %s missing SPFILE in spectrum header(s); using filename stem %r "
            "as link key (catalog --link-id-col SPFILE at catalog ingest)",
            source_path.name,
            _2df_filename_stem(source_path),
        )

    return records, wcs_attrs


def _is_2df_hdul(hdul: fits.HDUList) -> bool:
    """Return True if the FITS HDU list looks like a 2dFGRS spectrum file."""
    phdr = hdul[0].header
    if "SEQNUM" not in phdr and "BJSEL" not in phdr:
        return False
    try:
        _find_2df_spectrum_hdu(hdul)
    except ValueError:
        return False
    return True


def _ensure_batch_rows_2d(batch: np.ndarray) -> np.ndarray:
    """Ensure flux/ivar/mask batch arrays are ``(n_rows, n_pix)`` for Zarr append."""
    arr = np.asarray(batch)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D batch, got shape {arr.shape}")
    return arr


def _ensure_batch_ids(batch_ids: np.ndarray) -> np.ndarray:
    """Ensure source_id batch is 1-D (Zarr append rejects 0-D scalars)."""
    ids = np.atleast_1d(np.asarray(batch_ids, dtype=np.int64))
    return ids


def _ensure_batch_meta(batch_meta: np.ndarray, n_rows: int) -> np.ndarray:
    """Ensure structured meta batch has shape ``(n_rows,)``."""
    meta = np.asarray(batch_meta)
    if meta.ndim == 0:
        return meta.reshape(1)
    if n_rows and meta.shape[0] != n_rows:
        return meta.reshape(n_rows)
    return meta



def _6df_hdu_spectrum_role(name: str) -> str | None:
    """Classify a 6dFGS extension name as ``v``, ``r``, or ``vr`` spectrum HDU."""
    n = (name or "").strip().upper()
    if n in ("VR", "VRSPEC", "VR_SPECTRUM", "SPECTRUM VR") or n.endswith(" VR"):
        return "vr"
    if n in ("V", "VSPEC", "SPECTRUM V") or (n.endswith(" V") and " VR" not in n):
        return "v"
    if n in ("R", "RSPEC", "SPECTRUM R") or (n.endswith(" R") and " VR" not in n):
        return "r"
    return None


def _6df_header_text_single(hdr: fits.Header, *keys: str) -> str | None:
    """Return the first non-empty FITS header string among *keys* on one header."""
    for key in keys:
        if key in hdr:
            text = str(hdr[key]).strip()
            if text:
                return text
    return None


def _6df_header_text(vhdr: fits.Header, phdr: fits.Header, *keys: str) -> str | None:
    """Return the first non-empty FITS header string among *keys* on VR/PRIMARY."""
    for hdr in (vhdr, phdr):
        found = _6df_header_text_single(hdr, *keys)
        if found:
            return found
    return None


def _6df_warn_triple_header_mismatch(
    source_path: Path,
    v_hdr: fits.Header,
    vr_hdr: fits.Header,
    *,
    v_idx: int,
    vr_idx: int,
) -> None:
    """Log when paired V/VR headers disagree on internal match keys."""
    for key in ("KBESTR", "Z"):
        if key not in v_hdr or key not in vr_hdr:
            continue
        v_val = v_hdr[key]
        vr_val = vr_hdr[key]
        if v_val != vr_val:
            log.warning(
                "6dF: %s V HDU %d and VR HDU %d disagree on %s (%r vs %r)",
                source_path.name,
                v_idx,
                vr_idx,
                key,
                v_val,
                vr_val,
            )


def _6df_link_label_from_triple(
    v_hdr: fits.Header,
    vr_hdr: fits.Header,
    phdr: fits.Header,
    source_path: Path,
) -> tuple[str, bool]:
    """Resolve the catalog link label for a paired 6dFGS V/VR spectrum block.

    Uses ``TARGET`` and ``NAME_V`` from the VR header and ``TITLE_V`` from the
    paired V header: ``target|name_v|title_v``.  Returns ``(label, used_fallback)``
    where *used_fallback* is True when the filename stem was used because
    ``TARGET`` was absent.
    """
    from data_lake.ingest.fits_to_parquet import composite_link_label

    target = _6df_header_text(vr_hdr, phdr, "TARGET", "TARGETNAME")
    name_v = _6df_header_text(vr_hdr, phdr, "NAME_V")
    title_v = _6df_header_text_single(v_hdr, "TITLE_V")

    parts: list[str] = []
    if target:
        parts.append(target)
    if name_v:
        parts.append(name_v)
    if title_v:
        parts.append(title_v)
    elif target or name_v:
        log.warning(
            "6dF: %s VR block missing TITLE_V on paired V header; link key omits "
            "title segment (catalog --link-id-col targetname,NAME_V,TITLE_V)",
            source_path.name,
        )

    if parts:
        return composite_link_label(*parts), target is None
    return source_path.stem, True


def _looks_like_sky_degrees(ra: float, dec: float) -> bool:
    """True when *ra*/*dec* plausibly represent equatorial degrees."""
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return False
    return abs(ra) <= 360.0 and abs(dec) <= 90.0


def _parse_fits_sexagesimal_sky(ra_val: object, dec_val: object) -> tuple[float, float] | None:
    """Parse HMS/DMS-style FITS sky strings (e.g. 6dF ``OBJCTRA``/``OBJCTDEC``)."""
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        ra_text = str(ra_val).strip()
        dec_text = str(dec_val).strip()
        if not ra_text or not dec_text:
            return None
        coord = SkyCoord(ra_text, dec_text, unit=(u.hourangle, u.deg))
        return float(coord.ra.deg), float(coord.dec.deg)
    except Exception:
        return None


def _6df_sky_from_headers(
    vhdr: fits.Header,
    phdr: fits.Header,
) -> tuple[float, float]:
    """Resolve 6dFGS sky position in degrees for spectrum HEALPix assignment.

    Production 6dF target files store per-observation coordinates as ``OBSRA`` /
    ``OBSDEC`` (degrees) on the VR extension.  PRIMARY headers often carry
    sexagesimal ``OBJCTRA``/``OBJCTDEC`` or image WCS ``CRVAL1``/``CRVAL2``
    instead of numeric ``RA``/``DEC``.
    """
    if "OBSRA" in vhdr and "OBSDEC" in vhdr:
        ra = float(vhdr["OBSRA"])
        dec = float(vhdr["OBSDEC"])
        if _looks_like_sky_degrees(ra, dec):
            return ra, dec

    for hdr in (vhdr, phdr):
        if "RA" in hdr and "DEC" in hdr:
            ra = float(hdr["RA"])
            dec = float(hdr["DEC"])
            if _looks_like_sky_degrees(ra, dec):
                return ra, dec

    if "CRVAL1" in phdr and "CRVAL2" in phdr:
        ra = float(phdr["CRVAL1"])
        dec = float(phdr["CRVAL2"])
        if _looks_like_sky_degrees(ra, dec):
            return ra, dec

    if "OBJCTRA" in phdr and "OBJCTDEC" in phdr:
        parsed = _parse_fits_sexagesimal_sky(phdr["OBJCTRA"], phdr["OBJCTDEC"])
        if parsed is not None:
            return parsed

    log.warning(
        "6dF: could not resolve sky position from VR/PRIMARY headers; using (0, 0)"
    )
    return 0.0, 0.0


def _is_6df_hdul(hdul: fits.HDUList) -> bool:
    """Return True if the FITS HDU list looks like a 6dFGS target file."""
    roles = {_6df_hdu_spectrum_role(h.name or "") for h in hdul}
    roles.discard(None)
    return "vr" in roles and ("v" in roles or "r" in roles)


def _iter_6df_vr_triples(
    hdul: fits.HDUList,
) -> list[tuple[fits.ImageHDU, fits.ImageHDU, fits.ImageHDU, tuple[int, int, int]]]:
    """Return V/R/VR spectral triples in HDU order (one VR row per triple).

    Production 6dF target files repeat ``(SPECTRUM V, SPECTRUM R, SPECTRUM VR)``
    after stamp image extensions.  Each VR HDU is paired with the immediately
    preceding V and R extensions in the same block.
    """
    triples: list[
        tuple[fits.ImageHDU, fits.ImageHDU, fits.ImageHDU, tuple[int, int, int]]
    ] = []
    n = len(hdul)
    i = 0
    while i + 2 < n:
        roles = [_6df_hdu_spectrum_role(hdul[j].name or "") for j in (i, i + 1, i + 2)]
        if roles == ["v", "r", "vr"]:
            v_hdu = hdul[i]  # type: ignore[assignment]
            r_hdu = hdul[i + 1]  # type: ignore[assignment]
            vr_hdu = hdul[i + 2]  # type: ignore[assignment]
            if v_hdu.data is not None and vr_hdu.data is not None:
                triples.append((v_hdu, r_hdu, vr_hdu, (i, i + 1, i + 2)))
            i += 3
            continue
        i += 1

    if triples:
        return triples

    # Legacy layout: single VR at index 7 with a preceding V extension.
    if len(hdul) > 7 and _6df_hdu_spectrum_role(hdul[7].name or "") == "vr":
        vr_hdu = hdul[7]  # type: ignore[assignment]
        if vr_hdu.data is not None:
            v_hdu = None
            r_hdu = hdul[6] if len(hdul) > 6 else hdul[0]
            for j in range(6, -1, -1):
                if _6df_hdu_spectrum_role(hdul[j].name or "") == "v" and hdul[j].data is not None:
                    v_hdu = hdul[j]  # type: ignore[assignment]
                    break
            if v_hdu is not None:
                log.warning(
                    "6dF: using legacy VR-at-index-7 layout; paired V at HDU %d",
                    hdul.index(v_hdu),
                )
                return [(v_hdu, r_hdu, vr_hdu, (hdul.index(v_hdu), 6, 7))]  # type: ignore[list-item]

    summary = _summarize_fits_hdus(hdul)
    raise ValueError(
        "6dFGS file has no V/R/VR spectral triple. "
        f"Found: {summary}"
    )


def _6df_spectra_wcs_differs(records: list[SpectrumRecord]) -> bool:
    """True when 6dF VR records in one file have incompatible wavelength grids."""
    if len(records) <= 1:
        return False
    ref = records[0].wavelength
    for rec in records[1:]:
        wave = rec.wavelength
        if wave is None or ref is None:
            return True
        if wave.shape != ref.shape or not np.allclose(wave, ref):
            return True
    return False


def _read_6df_vr_record(
    v_hdu: fits.ImageHDU,
    vr_hdu: fits.ImageHDU,
    phdr: fits.Header,
    source_path: Path,
    *,
    triple_indices: tuple[int, int, int] | None = None,
) -> SpectrumRecord:
    """Parse one 6dFGS VR extension (with paired V header) into a record."""
    from data_lake.ingest.fits_to_parquet import normalize_object_id

    v_hdr = v_hdu.header
    vhdr = vr_hdu.header
    if triple_indices is not None:
        _6df_warn_triple_header_mismatch(
            source_path,
            v_hdr,
            vhdr,
            v_idx=triple_indices[0],
            vr_idx=triple_indices[2],
        )

    ra, dec = _6df_sky_from_headers(vhdr, phdr)
    link_label, used_fallback = _6df_link_label_from_triple(
        v_hdr, vhdr, phdr, source_path,
    )
    if used_fallback:
        log.warning(
            "6dF: %s VR HDU %r missing TARGET; using filename stem %r as link key "
            "(catalog --link-id-col targetname,NAME_V,TITLE_V)",
            source_path.name,
            vr_hdu.name,
            source_path.stem,
        )
    source_id = normalize_object_id(link_label)
    data = np.asarray(vr_hdu.data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(
            f"6dFGS VR extension {vr_hdu.name!r} in {source_path.name} has shape "
            f"{data.shape}; expected 2-D"
        )

    if data.shape[0] in (3, 4):
        arr = data
    elif data.shape[1] in (3, 4):
        arr = data.T
    else:
        raise ValueError(
            f"6dFGS VR extension {vr_hdu.name!r} in {source_path.name} has shape "
            f"{data.shape}; expected (3|4, n_pix)"
        )

    n_pix = int(arr.shape[1])
    flux = arr[0].astype(np.float32)
    variance = arr[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ivar = np.where(variance > 0.0, 1.0 / variance, 0.0).astype(np.float32)
    mask = np.zeros(n_pix, dtype=np.uint8)

    if arr.shape[0] >= 4:
        explicit_wave = np.asarray(arr[3], dtype=np.float64)
        if np.all(np.isfinite(explicit_wave)) and np.all(np.diff(explicit_wave) > 0):
            wavelength = explicit_wave
        else:
            wavelength = _wavelength_from_wcs(vhdr, n_pix)
    else:
        wavelength = _wavelength_from_wcs(vhdr, n_pix)

    meta: dict[str, Any] = {
        "z": float(vhdr.get("Z", phdr.get("Z", 0.0))),
        "z_err": 0.0,
        "snr": 0.0,
        "exptime": float(vhdr.get("EXPTIME", phdr.get("EXPTIME", 0.0))),
        "R": float(vhdr.get("SPEC_RES", 1000.0)),
        "instr": "6dFGS",
    }
    return SpectrumRecord(
        source_id=source_id,
        ra=ra,
        dec=dec,
        flux=flux,
        ivar=ivar,
        mask=mask,
        wavelength=wavelength,
        meta=meta,
    )


def _read_6df_spectrum(
    hdul: fits.HDUList,
    source_path: Path,
) -> tuple[list[SpectrumRecord], dict]:
    """Read a 6dFGS FITS file, ingesting every combined VR extension.

    Each ``(SPECTRUM V, SPECTRUM R, SPECTRUM VR)`` block yields one spectrum
    row from the VR HDU.  ``source_id`` is built from VR ``TARGET`` and
    ``NAME_V`` plus ``TITLE_V`` on the paired V extension
    (``target|name_v|title_v``), matching catalog ingest with
    ``--link-id-col targetname,NAME_V,TITLE_V``.

    Sky coordinates are taken from each VR extension's ``OBSRA``/``OBSDEC``
    (degrees), with fallbacks to ``RA``/``DEC``, PRIMARY image WCS, or
    sexagesimal ``OBJCTRA``/``OBJCTDEC``.
    """
    phdr = hdul[0].header
    triples = _iter_6df_vr_triples(hdul)
    records: list[SpectrumRecord] = []
    wcs_attrs: dict = {}
    seen_ids: dict[int, str] = {}

    for v_hdu, _, vr_hdu, indices in triples:
        rec = _read_6df_vr_record(
            v_hdu, vr_hdu, phdr, source_path, triple_indices=indices,
        )
        if rec.source_id in seen_ids:
            raise ValueError(
                f"duplicate 6dF source_id in {source_path.name}: "
                f"VR HDU {vr_hdu.name!r} (index {indices[2]}) link label maps to "
                f"the same _source_id as {seen_ids[rec.source_id]!r}"
            )
        seen_ids[rec.source_id] = vr_hdu.name or f"HDU{indices[2]}"
        if not wcs_attrs:
            wcs_attrs = _wcs_attrs_from_header(vr_hdu.header, len(rec.flux))
        records.append(rec)

    return records, wcs_attrs


def _detect_format_from_path(path: Path) -> str:
    """
    Heuristically detect the FITS spectral format from HDU names only.

    Opens the file with ``lazy_load_hdus=True`` so no data are read,
    then closes it immediately.
    """
    stem = path.stem.lower()
    if stem.startswith("spplate-"):
        return "sdss_spplate"
    with fits.open(str(path), lazy_load_hdus=True) as hdul:
        names = [h.name.upper() for h in hdul]
        if "COADD" in names:
            return "sdss_boss"
        if any(arm + "_FLUX" in names for arm in ("B", "R", "Z")):
            return "desi_coadd"
        if _is_6df_hdul(hdul):
            return "6df"
        phdr = hdul[0].header
        naxis2 = int(phdr.get("NAXIS2", 0) or 0)
        if naxis2 in (640, 1000) and ("PLATEID" in phdr or "PLATE" in phdr) and "MJD" in phdr:
            return "sdss_spplate"
        if "COEFF0" in phdr and "COEFF1" in phdr and ("PLATEID" in phdr or "PLATE" in phdr):
            return "sdss_spplate"
        if _is_2df_hdul(hdul):
            return "2df"
        if _is_gama_hdul(hdul, path):
            return "gama"
        if _is_vipers_hdul(hdul, path):
            return "vipers"
        if _is_vuds_hdul(hdul, path):
            return "vuds"
        if _is_vandels_hdul(hdul, path):
            return "vandels"
        if _is_vvds_hdul(hdul, path):
            return "vvds"
        if _is_zcosmos_hdul(hdul, path):
            return "zcosmos"
        if _is_ozdes_hdul(hdul, path):
            return "ozdes"
        if stem.startswith("wig") and _is_wig_spectrum_layout(hdul):
            return "wig"
    return "generic"


def _filter_spectrum_tile_duplicates(
    tile_records: list[SpectrumRecord],
    existing_source_ids: set[int],
    on_duplicate: Literal["append", "error", "skip"],
) -> list[SpectrumRecord]:
    from data_lake.ingest.duplicate_policy import zarr_row_keep_mask

    if not tile_records:
        return tile_records
    sids = np.array([r.source_id for r in tile_records], dtype=np.int64)
    keep = zarr_row_keep_mask(sids, existing_source_ids, on_duplicate)
    return [r for r, k in zip(tile_records, keep.tolist()) if k]


# ---------------------------------------------------------------------------
# Core ingest
# ---------------------------------------------------------------------------


def _meta_to_bytes(meta: dict[str, Any]) -> bytes:
    arr = np.zeros(1, dtype=_META_DTYPE)
    arr["z"][0]       = meta.get("z",       0.0)
    arr["z_err"][0]   = meta.get("z_err",   0.0)
    arr["snr"][0]     = meta.get("snr",     0.0)
    arr["exptime"][0] = meta.get("exptime", 0.0)
    arr["R"][0]       = meta.get("R",       0.0)
    instr = str(meta.get("instr", ""))[:16].encode("ascii")
    arr["instr"][0]   = instr.ljust(16)[:16]
    return bytes(arr.view("|V" + str(_META_DTYPE.itemsize)))


def ingest_spectra_from_fits(
    source_path: Path | str,
    output_root: Path | str,
    survey_name: str,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    norder: int = 5,
    link_id_col: str | None = None,
    wavelength_mode: str = "shared",
    mask_dtype: np.dtype | type = _DEFAULT_MASK_DTYPE,
    fmt: str | None = None,
    n_pix_expected: int | None = None,
    on_length_mismatch: str = "error",
    with_resolution: bool = False,
    on_duplicate_source_id: Literal["append", "error", "skip"] = "skip",
    specobj_lookup: Path | str | None = None,
    specobj_lookup_survey: str | None = None,
    specobj_lookup_from_catalog: bool = False,
    specobj_lookup_from_plate: bool = False,
    specobj_id_layout: str = "auto",
) -> dict[int, int]:
    """
    Ingest 1-D spectra from a FITS file into HEALPix-partitioned Zarr v3 stacks.

    Parameters
    ----------
    source_path:
        Input FITS file (SDSS/BOSS spec-*.fits, DESI coadd-*.fits, or generic).
    output_root:
        Data lake root.
    survey_name:
        Survey identifier.
    ra_col / dec_col:
        Header keywords for sky coordinates (generic format; SDSS plug RA/Dec
        fallbacks when the named keys are absent).
    link_id_col:
        Header keyword or DESI fibermap column for object ID (e.g. ``TARGETID``).
        Must match the catalog ID column.  DESI coadds default to ``TARGETID``.
    norder:
        HEALPix partitioning order.
    wavelength_mode:
        ``"shared"`` (default) – one wavelength array stored per tile;
        ``"per_source"`` – wavelength stored as ``(N, N_pix)`` alongside flux.
    mask_dtype:
        Storage dtype for the mask array (``uint8`` or ``uint16``).
    fmt:
        Force format detection: ``"sdss_boss"``, ``"sdss_spplate"``, ``"desi_coadd"``,
        ``"2df"``, ``"6df"``, ``"gama"``, ``"wig"``, ``"ozdes"``, ``"zcosmos"``, ``"vandels"``, ``"vipers"``, ``"vuds"``, ``"vvds"``, ``"generic"``.
        Auto-detected from HDU names if ``None``.
    specobj_lookup:
        Parquet/CSV sidecar with ``survey``, ``PLATE``, ``MJD``, ``FIBERID``,
        ``SPECOBJID`` for spPlate ingest.  Mutually exclusive with
        ``specobj_lookup_from_catalog``.
    specobj_lookup_survey:
        When the sidecar has no survey column, use this string to scope rows
        (defaults to ``survey_name``).
    specobj_lookup_from_catalog:
        If True, join ``catalogs/<survey_name>/`` on plate/mjd/fiber and take
        IDs from ``link_id_col`` or the catalog's ID column (not required to
        be named ``specobjid``).
    n_pix_expected:
        If set, enforce that all spectra have this pixel count.
    on_length_mismatch:
        How to handle spectra with unexpected pixel count:
        ``"error"`` (default), ``"pad"``, ``"truncate"``.
    with_resolution:
        When True, store the DESI banded resolution matrix alongside flux/ivar.
        Requires ``wavelength_mode="shared"`` and DESI coadd input.
        Increases storage by ~3× (see README for cost estimates).
    on_duplicate_source_id:
        ``skip`` (default) drops incoming rows whose ``source_id`` is already in
        the tile.  ``error`` raises when an ID already exists.  ``append`` always
        appends (may duplicate rows if a file is ingested twice).

    Returns
    -------
    ``{source_id: spectrum_index_in_tile}`` for the catalog updater.
    """
    source_path = Path(source_path)
    output_root = Path(output_root)
    mask_dtype = np.dtype(mask_dtype)

    if with_resolution and wavelength_mode != "shared":
        raise ValueError(
            "--with-resolution requires wavelength_mode='shared' because the "
            "resolution matrix is defined relative to a fixed wavelength grid."
        )

    index_map: dict[int, int] = {}

    # --- Format detection and reading ---
    detected_fmt = fmt or _detect_format_from_path(source_path)

    res_diags: list[np.ndarray] | None = None
    res_offsets: np.ndarray | None = None

    if detected_fmt == "desi_coadd":
        records, wcs_attrs, res_diags, res_offsets = _read_desi_with_desispec(
            source_path,
            with_resolution=with_resolution,
            link_id_col=link_id_col,
        )
    else:
        if with_resolution:
            raise ValueError(
                "--with-resolution is only supported for DESI coadd files. "
                f"Detected format: {detected_fmt!r}"
            )
        with fits.open(str(source_path), memmap=True) as hdul:
            if detected_fmt == "sdss_boss":
                records, wcs_attrs = _read_sdss_boss(
                    hdul,
                    link_id_col=link_id_col,
                    ra_col=ra_col,
                    dec_col=dec_col,
                )
            elif detected_fmt == "sdss_spplate":
                n_lookup_modes = sum(
                    bool(x)
                    for x in (
                        specobj_lookup,
                        specobj_lookup_from_catalog,
                        specobj_lookup_from_plate,
                    )
                )
                if n_lookup_modes > 1:
                    raise ValueError(
                        "Pass only one of specobj_lookup=, specobj_lookup_from_catalog=True, "
                        "or specobj_lookup_from_plate=True"
                    )
                if n_lookup_modes == 0:
                    raise ValueError(
                        "sdss_spplate ingest requires specobj_lookup= (sidecar Parquet/CSV), "
                        "specobj_lookup_from_catalog=True (lake catalogs/<survey>/), or "
                        "specobj_lookup_from_plate=True (synthesize specObjID from header)."
                    )
                from data_lake.ingest.sdss_specobj_lookup import (
                    build_fiber_to_specobjid_map,
                    spplate_plate_mjd_from_hdul,
                )

                plate, mjd = spplate_plate_mjd_from_hdul(hdul, source_path)
                fiber_map = build_fiber_to_specobjid_map(
                    survey_name,
                    plate,
                    mjd,
                    lookup_path=specobj_lookup,
                    catalog_root=output_root if specobj_lookup_from_catalog else None,
                    lookup_survey=specobj_lookup_survey,
                    spplate_hdul=hdul,
                    lookup_from_plate=specobj_lookup_from_plate,
                    specobj_id_layout=specobj_id_layout,  # type: ignore[arg-type]
                    catalog_id_col=link_id_col,
                )
                records, wcs_attrs = _read_sdss_spplate(
                    hdul,
                    path=source_path,
                    fiber_to_specobjid=fiber_map,
                    ra_col=ra_col,
                    dec_col=dec_col,
                )
            elif detected_fmt == "2df":
                records, wcs_attrs = _read_2df_spectrum(hdul, source_path)
            elif detected_fmt == "6df":
                records, wcs_attrs = _read_6df_spectrum(hdul, source_path)
            elif detected_fmt == "wig":
                records, wcs_attrs = _read_wig_spectrum(hdul, source_path)
            elif detected_fmt == "gama":
                records, wcs_attrs = _read_gama_spectrum(
                    hdul,
                    source_path,
                    link_id_col=link_id_col or "SPECID",
                    ra_col=ra_col,
                    dec_col=dec_col,
                )
            elif detected_fmt == "ozdes":
                records, wcs_attrs = _read_ozdes_spectrum(hdul, source_path)
            elif detected_fmt == "zcosmos":
                records, wcs_attrs = _read_zcosmos_spectrum(hdul, source_path)
            elif detected_fmt == "vandels":
                records, wcs_attrs = _read_vandels_spectrum(hdul, source_path)
            elif detected_fmt == "vipers":
                records, wcs_attrs = _read_vipers_spectrum(hdul, source_path)
            elif detected_fmt == "vuds":
                records, wcs_attrs = _read_vuds_spectrum(hdul, source_path)
            elif detected_fmt == "vvds":
                records, wcs_attrs = _read_vvds_spectrum(hdul, source_path)
            else:
                records, wcs_attrs = _read_generic_1d(
                    hdul,
                    link_id_col=link_id_col,
                    ra_col=ra_col,
                    dec_col=dec_col,
                )

    if not records:
        log.warning("No spectra extracted from %s", source_path.name)
        return index_map

    # SDSS/BOSS spec files: per-object loglam grids and pixel counts differ slightly.
    wavelength_mode_effective = wavelength_mode
    length_policy = on_length_mismatch
    if detected_fmt in ("sdss_boss", "sdss_spplate"):
        if wavelength_mode == "shared":
            log.info(
                "SDSS (%s): using wavelength_mode='per_source' "
                "(per-object or per-plate loglam; tile storage pads to common n_pix).",
                detected_fmt,
            )
        wavelength_mode_effective = "per_source"
        if length_policy == "error":
            length_policy = "pad"
    elif detected_fmt == "2df" and (
        len(records) > 1 or _2df_spectra_wcs_differs(records)
    ):
        if wavelength_mode == "shared":
            log.info(
                "2dF: using wavelength_mode='per_source' "
                "(multiple SPECTRUM HDUs or differing WCS in %s).",
                source_path.name,
            )
        wavelength_mode_effective = "per_source"
    elif detected_fmt == "6df" and (
        len(records) > 1 or _6df_spectra_wcs_differs(records)
    ):
        if wavelength_mode == "shared":
            log.info(
                "6dF: using wavelength_mode='per_source' "
                "(multiple VR HDUs or differing WCS in %s).",
                source_path.name,
            )
        wavelength_mode_effective = "per_source"

    n_pix = max(len(r.flux) for r in records)
    if n_pix_expected is not None and n_pix != n_pix_expected:
        msg = (
            f"Spectrum length {n_pix} != expected {n_pix_expected} in {source_path.name}. "
            f"Pass on_length_mismatch='pad' or 'truncate' to suppress this error."
        )
        if length_policy == "error":
            raise ValueError(msg)
        log.warning(msg)
        records = _fix_length(records, n_pix_expected, length_policy)
        n_pix = n_pix_expected

    # Derive resolution dimensions once (same for all sources in a file)
    n_diag = int(res_diags[0].shape[0]) if res_diags is not None else None

    # Build a source_id → res_index mapping for the records list so we can
    # index into res_diags using the per-tile record positions.
    sid_to_res_idx: dict[int, int] = (
        {rec.source_id: i for i, rec in enumerate(records)}
        if res_diags is not None else {}
    )

    # Group records by HEALPix tile
    tile_groups: dict[int, list[SpectrumRecord]] = {}
    for rec in records:
        pix = int(assign_healpix(np.array([rec.ra]), np.array([rec.dec]), norder)[0])
        tile_groups.setdefault(pix, []).append(rec)

    for npix, tile_records in tile_groups.items():
        tile_dir = output_root / "spectra" / survey_name / healpix_dir(norder, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path = tile_dir / f"Npix={npix}.zarr"

        batch_max_pix = max(len(r.flux) for r in tile_records)
        tile_exists = tile_path.exists() and (tile_path / "zarr.json").exists()
        create_n_pix = batch_max_pix if not tile_exists else max(n_pix, batch_max_pix)

        root = _open_or_create_spectrum_tile(
            tile_path,
            create_n_pix,
            wavelength_mode_effective,
            mask_dtype,
            wcs_attrs,
            n_diag=n_diag,
            resolution_offsets=res_offsets,
        )
        tile_n_pix = int(root["flux"].shape[1])
        if root["flux"].ndim != 2:
            raise ValueError(
                f"Corrupt spectrum tile {tile_path}: flux array must be 2-D, "
                f"got shape {root['flux'].shape}"
            )
        tile_wavelength_mode = str(
            root.attrs.get("wavelength_mode", wavelength_mode_effective)
        )

        # Dynamic tile widening: when incoming batch has longer spectra than the
        # existing tile, widen (pad existing rows) rather than truncate new spectra.
        if tile_exists and length_policy == "pad" and batch_max_pix > tile_n_pix:
            root = widen_spectrum_tile(
                tile_path,
                batch_max_pix,
                wavelength_mode=tile_wavelength_mode,
                mask_dtype=mask_dtype,
                wcs_attrs=wcs_attrs,
                n_diag=n_diag,
                resolution_offsets=res_offsets,
            )
            tile_n_pix = int(root["flux"].shape[1])
            tile_wavelength_mode = str(
                root.attrs.get("wavelength_mode", wavelength_mode_effective)
            )

        n_existing = int(zarr_join_array(root).shape[0])
        existing: set[int] = set()
        if n_existing > 0:
            existing = set(np.asarray(zarr_join_array(root)[:]).tolist())

        tile_records = _filter_spectrum_tile_duplicates(
            tile_records, existing, on_duplicate_source_id,
        )
        if not tile_records:
            continue

        mismatched = [r for r in tile_records if len(r.flux) != tile_n_pix]
        if mismatched:
            if length_policy == "error":
                lengths = sorted({len(r.flux) for r in tile_records})
                raise ValueError(
                    f"Spectrum pixel lengths {lengths} disagree with tile "
                    f"Npix={npix} n_pix={tile_n_pix} in {source_path.name}. "
                    f"Use --on-length-mismatch pad or truncate."
                )
            log.info(
                "Aligning %d spectrum(s) to n_pix=%d for tile Npix=%s (%s; "
                "pad=extend short/truncate long)",
                len(mismatched),
                tile_n_pix,
                npix,
                length_policy,
            )
            tile_records = _fix_length(tile_records, tile_n_pix, length_policy)

        start_idx = root["flux"].shape[0]

        n_rows = len(tile_records)
        batch_flux = _ensure_batch_rows_2d(
            np.stack([r.flux for r in tile_records])
        ).astype(np.float32)
        batch_ivar = _ensure_batch_rows_2d(
            np.stack([r.ivar for r in tile_records])
        ).astype(np.float32)
        batch_mask = _ensure_batch_rows_2d(
            np.stack([r.mask for r in tile_records])
        ).astype(mask_dtype)
        batch_ids = _ensure_batch_ids(
            np.array([r.source_id for r in tile_records], dtype=np.int64)
        )
        batch_meta = _ensure_batch_meta(
            np.frombuffer(
                b"".join(_meta_to_bytes(r.meta) for r in tile_records),
                dtype="|V" + str(_META_DTYPE.itemsize),
            ),
            n_rows,
        )

        root["flux"].append(batch_flux)
        root["ivar"].append(batch_ivar)
        root["mask"].append(batch_mask)
        zarr_join_array(root).append(batch_ids)
        root["meta"].append(batch_meta)

        if tile_wavelength_mode == "per_source":
            batch_wave = _ensure_batch_rows_2d(np.stack([
                r.wavelength.astype(np.float32)
                if r.wavelength is not None
                else np.zeros(tile_n_pix, dtype=np.float32)
                for r in tile_records
            ]))
            root["wavelength"].append(batch_wave)
        elif start_idx == 0 and tile_records[0].wavelength is not None:
            # Write shared wavelength once (first time the tile is created)
            shared_wave = tile_records[0].wavelength.astype(np.float64)
            if len(shared_wave) != tile_n_pix:
                if length_policy == "error":
                    raise ValueError(
                        f"Shared wavelength length {len(shared_wave)} != tile "
                        f"n_pix {tile_n_pix}"
                    )
                shared_wave = (
                    shared_wave[:tile_n_pix]
                    if len(shared_wave) > tile_n_pix
                    else np.pad(shared_wave, (0, tile_n_pix - len(shared_wave)))
                )
            root["wavelength"][:] = shared_wave

        if res_diags is not None:
            batch_res = np.stack([
                res_diags[sid_to_res_idx[r.source_id]] for r in tile_records
            ]).astype(np.float32)
            root["resolution"].append(batch_res)

        for local_i, rec in enumerate(tile_records):
            index_map[rec.source_id] = start_idx + local_i

    has_resolution = res_diags is not None
    _write_spectrum_info(
        output_root / "spectra" / survey_name,
        survey_name, norder, n_pix,
        wavelength_mode_effective, str(mask_dtype), wcs_attrs,
        has_resolution=has_resolution,
        resolution_n_diag=n_diag,
        resolution_offsets=res_offsets.tolist() if res_offsets is not None else None,
        on_duplicate_source_id=on_duplicate_source_id,
    )

    log.info(
        "Ingested %d spectra from %s → %d tiles%s",
        len(records), source_path.name, len(tile_groups),
        " (with resolution)" if has_resolution else "",
    )
    return index_map


def _fix_length(
    records: list[SpectrumRecord],
    target_n_pix: int,
    mode: str,
) -> list[SpectrumRecord]:
    """Align spectrum length to ``target_n_pix`` for storage in an existing tile.

    ``pad`` (default for SDSS spPlate): pad shorter spectra with NaN flux; truncate
    longer spectra to the tile width (common when appending a longer spPlate to a
    tile created from shorter spec files).
    ``truncate``: truncate longer spectra only; error if shorter than target.
    """
    fixed: list[SpectrumRecord] = []
    n_pad = n_trunc = 0
    for r in records:
        n = len(r.flux)
        if n == target_n_pix:
            fixed.append(r)
            continue
        if n > target_n_pix:
            if mode == "error":
                raise ValueError(
                    f"Spectrum length {n} > tile n_pix {target_n_pix} for "
                    f"source_id={r.source_id}; use --on-length-mismatch pad or truncate."
                )
            n_trunc += 1
            fixed.append(
                SpectrumRecord(
                    source_id=r.source_id,
                    ra=r.ra,
                    dec=r.dec,
                    flux=r.flux[:target_n_pix].copy(),
                    ivar=r.ivar[:target_n_pix].copy(),
                    mask=r.mask[:target_n_pix].copy(),
                    wavelength=(
                        r.wavelength[:target_n_pix].copy()
                        if r.wavelength is not None
                        else None
                    ),
                    meta=r.meta,
                )
            )
            continue
        # n < target_n_pix
        if mode == "truncate":
            raise ValueError(
                f"Spectrum length {n} < tile n_pix {target_n_pix} for "
                f"source_id={r.source_id}; use --on-length-mismatch pad."
            )
        pad = target_n_pix - n
        n_pad += 1
        fixed.append(
            SpectrumRecord(
                source_id=r.source_id,
                ra=r.ra,
                dec=r.dec,
                flux=np.pad(r.flux, (0, pad), constant_values=np.nan),
                ivar=np.pad(r.ivar, (0, pad), constant_values=0.0),
                mask=np.pad(r.mask, (0, pad), constant_values=0),
                wavelength=(
                    np.pad(r.wavelength, (0, pad))
                    if r.wavelength is not None
                    else None
                ),
                meta=r.meta,
            )
        )
    if n_trunc:
        log.warning(
            "Truncated %d spectrum(s) to n_pix=%d (longer than existing tile width)",
            n_trunc,
            target_n_pix,
        )
    if n_pad:
        log.info(
            "Padded %d spectrum(s) to n_pix=%d (shorter than tile width)",
            n_pad,
            target_n_pix,
        )
    return fixed


def ingest_spectra_batch(
    source_paths: Sequence[Path | str],
    output_root: Path | str,
    survey_name: str,
    **kwargs,
) -> dict[int, int]:
    """Ingest multiple FITS files; returns merged source_id → spectrum_index map."""
    merged: dict[int, int] = {}
    for path in source_paths:
        merged.update(ingest_spectra_from_fits(path, output_root, survey_name, **kwargs))
    return merged


def _write_spectrum_info(
    spectra_survey_root: Path,
    survey_name: str,
    norder: int,
    n_pix: int,
    wavelength_mode: str,
    mask_dtype: str,
    wcs_attrs: dict,
    *,
    has_resolution: bool = False,
    resolution_n_diag: int | None = None,
    resolution_offsets: list[int] | None = None,
    on_duplicate_source_id: str = "skip",
) -> None:
    info = {
        "survey_name": survey_name,
        "hats_order": norder,
        "n_pix": n_pix,
        "wavelength_mode": wavelength_mode,
        "flux_dtype": "float32",
        "ivar_dtype": "float32",
        "mask_dtype": mask_dtype,
        "mask_bits": _DEFAULT_MASK_BITS,
        "on_duplicate_source_id": on_duplicate_source_id,
        "meta_fields": list(_META_DTYPE.names),
        "chunk_shape": [1, n_pix],
        "chunks_per_shard": _CHUNKS_PER_SHARD,
        "compression": "blosc-zstd-bitshuffle",
        "zarr_format": 3,
        "schema_version": "1",
        "wcs": wcs_attrs,
        "has_resolution": has_resolution,
        "resolution_n_diag": resolution_n_diag,
        "resolution_offsets": resolution_offsets,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    spectra_survey_root.mkdir(parents=True, exist_ok=True)
    with open(spectra_survey_root / "spectrum_info.json", "w") as fh:
        json.dump(info, fh, indent=2)
    try:
        from data_lake.schema_registry import write_spectra_schema_manifest

        write_spectra_schema_manifest(spectra_survey_root, survey_name)
    except Exception as exc:
        log.warning("Could not write spectra schema_manifest.json: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    from ..cli_utils import (
        config_option,
        configure_warning_filters,
        ingest_token_option,
        load_optional_config,
        pick,
        require_ingest_permission,
        require_output_root,
    )

    @click.command("dl-ingest-spectra")
    @click.argument("source_path", type=click.Path(exists=True, path_type=Path))
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @ingest_token_option
    @click.option("--survey", "survey_name", required=True, help="Short survey name.")
    @click.option("--ra-col", default="RA", show_default=True)
    @click.option("--dec-col", default="DEC", show_default=True)
    @click.option(
        "--link-id-col",
        default=None,
        help=(
            "Object ID for SDSS/DESI/generic/spPlate ingest: FITS header keyword "
            "(generic/SDSS) or fibermap column (DESI). Ignored by format-specific "
            "readers (2df, 6df, OzDES, …) which resolve IDs internally."
        ),
    )
    @click.option("--norder", default=None, type=int,
                  help="HEALPix order (overrides config; default 5).")
    @click.option("--wavelength-mode",
                  type=click.Choice(["shared", "per_source"]),
                  default=None,
                  help="Wavelength storage mode (overrides config; default 'shared').")
    @click.option("--mask-dtype",
                  type=click.Choice(["uint8", "uint16"]),
                  default=None,
                  help="Mask dtype (overrides config; default 'uint8').")
    @click.option("--fmt", default=None,
                  type=click.Choice(
                      [
                          "sdss_boss",
                          "sdss_spplate",
                          "desi_coadd",
                          "generic",
                          "2df",
                          "6df",
                          "gama",
                          "wig",
                          "ozdes",
                          "zcosmos",
                          "vandels",
                          "vipers",
                          "vuds",
                          "vvds",
                      ],
                  ),
                  help="Force input format (auto-detected by default).")
    @click.option(
        "--specobj-lookup",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Parquet/CSV: survey, PLATE, MJD, FIBERID, SPECOBJID (spPlate only).",
    )
    @click.option(
        "--specobj-lookup-from-catalog/--no-specobj-lookup-from-catalog",
        default=False,
        show_default=True,
        help="Resolve SPECOBJID from catalogs/<survey>/ instead of --specobj-lookup.",
    )
    @click.option(
        "--specobj-lookup-survey",
        default=None,
        help="Sidecar survey filter when the lookup file has no SURVEY column.",
    )
    @click.option(
        "--specobj-lookup-from-plate/--no-specobj-lookup-from-plate",
        default=False,
        show_default=True,
        help="Synthesize SPECOBJID from spPlate PLATE/MJD/FIBERID/RUN2D (no sidecar).",
    )
    @click.option(
        "--specobj-id-layout",
        type=click.Choice(["auto", "dr7", "dr8plus"], case_sensitive=False),
        default="auto",
        show_default=True,
        help="specObjID bit packing for --specobj-lookup-from-plate (DR7 vs DR8+).",
    )
    @click.option("--on-length-mismatch",
                  type=click.Choice(["error", "pad", "truncate"]),
                  default="error", show_default=True)
    @click.option(
        "--on-duplicate",
        type=click.Choice(["append", "error", "skip"]),
        default="skip",
        show_default=True,
        help="If a source_id already exists in a tile Zarr: skip (default), raise, or append.",
    )
    @click.option("--with-resolution/--no-with-resolution", default=None,
                  help=(
                      "Store the DESI banded resolution matrix (n_diag × N_pix per source). "
                      "Requires DESI coadd input and wavelength-mode=shared. "
                      "Overrides config; default false."
                  ))
    @click.option(
        "--update-catalog/--no-update-catalog", default=True, show_default=True,
        help="Patch _spectrum_index in the Parquet catalog after ingest "
             "(skipped silently if no catalog exists for this survey).",
    )
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        source_path: Path,
        output_root: Path | None,
        config_path: Path | None,
        ingest_token: str | None,
        survey_name: str,
        ra_col: str,
        dec_col: str,
        link_id_col: str | None,
        norder: int | None,
        wavelength_mode: str | None,
        mask_dtype: str | None,
        fmt: str | None,
        specobj_lookup: Path | None,
        specobj_lookup_from_catalog: bool,
        specobj_lookup_survey: str | None,
        specobj_lookup_from_plate: bool,
        specobj_id_layout: str,
        on_length_mismatch: str,
        on_duplicate: str,
        with_resolution: bool | None,
        update_catalog: bool,
        verbose: bool,
    ) -> None:
        """Ingest 1-D FITS spectra into sharded Zarr v3 stacks.

        OUTPUT_ROOT is optional when a lake config is available (via
        --config or $DATA_LAKE_CONFIG); in that case it defaults to
        ``<lake.root>/<paths.spectra>``.
        """
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        configure_warning_filters()
        cfg = load_optional_config(config_path)
        require_ingest_permission(cfg, ingest_token)
        resolved_output = require_output_root(output_root, cfg, kind="spectra")
        resolved_norder = pick(norder, cfg.partitioning.hats_order if cfg else None, 5)

        index_map = ingest_spectra_from_fits(
            source_path=source_path,
            output_root=resolved_output,
            survey_name=survey_name,
            ra_col=ra_col,
            dec_col=dec_col,
            link_id_col=link_id_col,
            norder=resolved_norder,
            wavelength_mode=pick(wavelength_mode,
                                 cfg.defaults.wavelength_mode if cfg else None,
                                 "shared"),
            mask_dtype=np.dtype(pick(mask_dtype,
                                     cfg.defaults.mask_dtype if cfg else None,
                                     "uint8")),
            fmt=fmt,
            on_length_mismatch=on_length_mismatch,
            on_duplicate_source_id=on_duplicate,  # type: ignore[arg-type]
            with_resolution=pick(with_resolution,
                                 cfg.defaults.with_resolution if cfg else None,
                                 False),
            specobj_lookup=specobj_lookup,
            specobj_lookup_from_catalog=specobj_lookup_from_catalog,
            specobj_lookup_survey=specobj_lookup_survey,
            specobj_lookup_from_plate=specobj_lookup_from_plate,
            specobj_id_layout=specobj_id_layout.lower(),
        )

        if update_catalog and index_map:
            try:
                from data_lake.ingest.update_catalog_indices import update_index_column
                n_modified = update_index_column(
                    lake_root=resolved_output,
                    survey_name=survey_name,
                    source_id_to_index=index_map,
                    kind="spectrum",
                    norder=resolved_norder,
                    link_id_col=None,
                )
                click.echo(f"Patched _spectrum_index in {n_modified} catalog tile(s).")
            except FileNotFoundError:
                log.info(
                    "No catalog found for survey %r — skipping _spectrum_index patch.",
                    survey_name,
                )

except ImportError:
    cli = None  # type: ignore[assignment]
