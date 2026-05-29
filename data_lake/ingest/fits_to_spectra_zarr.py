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
* **6dFGS**     multi-extension target FITS – ingests only the combined VR spectrum extension.
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
      source_id/    (N_sources,)                    int64
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
    assign_healpix,
    healpix_dir,
    object_id_from_fits_header,
    sky_from_fits_header,
)

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

    root.create_array("source_id", shape=(0,), chunks=(4096,), dtype=np.int64, fill_value=-1)
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
    - ``source_id``, ``meta``     → copied unchanged
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
        new_root["source_id"].append(np.asarray(old_root["source_id"][:]))
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


def _sdss_source_id(hdul: fits.HDUList, source_id_col: str | None) -> int:
    """Resolve object ID for ``spec-PLATE-MJD-FIBER.fits`` (header + SPALL HDU).

    ``SPECOBJID`` and most spAll columns live in the **SPALL** BINTABLE (HDU 2),
    not in the primary header.  ``THING_ID`` is often duplicated on HDU 0.
    """
    from data_lake.ingest.fits_to_parquet import normalize_object_id

    phdr = hdul[0].header
    spall = _sdss_spall_hdu(hdul)

    candidates: list[str] = []
    if source_id_col:
        candidates.append(source_id_col)
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
        + (f" (requested {source_id_col!r})" if source_id_col else "")
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
    source_id_col: str | None = None,
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
    source_id = _sdss_source_id(hdul, source_id_col)
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
            ra=float(ra_arr[row_i]),
            dec=float(dec_arr[row_i]),
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
    source_id_col: str | None = None,
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
        sid_key = source_id_col or "TARGETID"
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
    source_id_col: str | None = None,
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

    base_id = object_id_from_fits_header(header, source_id_col, hdu_index=image_hdu)
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


def _find_2df_spectrum_hdu(hdul: fits.HDUList) -> tuple[int, fits.ImageHDU | fits.PrimaryHDU]:
    """Locate the 2dF spectral image extension (named SPECTRUM or heuristic)."""
    for i, hdu in enumerate(hdul):
        if (hdu.name or "").strip().upper() == "SPECTRUM" and _is_2df_spectrum_hdu(hdu):
            return i, hdu  # type: ignore[return-value]
    for i, hdu in enumerate(hdul):
        if _is_2df_spectrum_hdu(hdu):
            return i, hdu  # type: ignore[return-value]
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


def _read_2df_spectrum(
    hdul: fits.HDUList,
    source_path: Path,
) -> tuple[list[SpectrumRecord], dict]:
    """Read a 2dFGRS 1-D spectrum FITS file.

    Expected layout:
    - HDU 0 (PRIMARY): sky position in header (``RA``, ``DEC``), ``SEQNUM``,
      ``NAME``, ``BJSEL``.
    - HDU 1 (SPECTRUM): 2-D image of shape ``(3, n_pix)`` where rows are
      ``[flux, variance, sky]``; spectral WCS in extension header
      (``CRVAL1``, ``CRPIX1``, ``CDELT1``).

    Source ID is derived from the file basename (numeric stem = ``serial``
    value in the catalog), normalised via :func:`normalize_object_id`.
    The ``serial`` numeric value is used directly when the stem is a pure
    integer, or hashed otherwise.
    """
    from data_lake.ingest.fits_to_parquet import normalize_object_id

    phdr = hdul[0].header
    ra = float(phdr.get("RA", 0.0))
    dec = float(phdr.get("DEC", 0.0))

    # Source ID from filename stem (matches catalog ``serial`` column)
    stem = source_path.stem
    # Remove any secondary extension (e.g. "154714.fits.gz" → "154714")
    for sfx in (".fits", ".fit"):
        if stem.lower().endswith(sfx):
            stem = stem[: -len(sfx)]
    source_id = normalize_object_id(stem)

    spec_hdu_idx, shdu = _find_2df_spectrum_hdu(hdul)
    if (shdu.name or "").strip().upper() != "SPECTRUM":
        log.warning(
            "2dF: no SPECTRUM HDU in %s; using HDU %d (%r)",
            source_path.name,
            spec_hdu_idx,
            shdu.name,
        )

    shdr = shdu.header
    flux, variance, n_pix = _parse_2df_spectrum_data(np.asarray(shdu.data))
    flux = flux.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ivar = np.where(variance > 0.0, 1.0 / variance, 0.0).astype(np.float32)
    mask = np.zeros(n_pix, dtype=np.uint8)

    wavelength = _wavelength_from_wcs(shdr, n_pix)
    wcs_attrs = _wcs_attrs_from_header(shdr, n_pix)

    meta: dict[str, Any] = {
        "z":       float(shdr.get("Z", phdr.get("Z", 0.0))),
        "z_err":   0.0,
        "snr":     0.0,
        "exptime": float(shdr.get("EXPTIME", phdr.get("EXPTIME", 0.0))),
        "R":       float(shdr.get("SPEC_RES", 500.0)),
        "instr":   "2dFGRS",
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


def _is_6df_hdul(hdul: fits.HDUList) -> bool:
    """Return True if the FITS HDU list looks like a 6dFGS target file."""
    names = [(h.name or "").strip().upper() for h in hdul]
    return "VR" in names and ("V" in names or "R" in names)


def _select_6df_vr_hdu(hdul: fits.HDUList) -> tuple[int, fits.ImageHDU]:
    """Pick the combined 6dFGS VR spectral extension."""
    for i, hdu in enumerate(hdul):
        name = (hdu.name or "").strip().upper()
        if name in ("VR", "VRSPEC", "VR_SPECTRUM") and hdu.data is not None:
            return i, hdu  # type: ignore[return-value]
    # 6dF docs: 8th extension (index 7) is typically combined/spliced VR.
    if len(hdul) > 7 and hdul[7].data is not None:
        return 7, hdul[7]  # type: ignore[return-value]
    raise ValueError("6dFGS file has no VR spectral extension")


def _read_6df_spectrum(
    hdul: fits.HDUList,
    source_path: Path,
) -> tuple[list[SpectrumRecord], dict]:
    """Read a 6dFGS FITS file, ingesting only the combined VR extension."""
    from data_lake.ingest.fits_to_parquet import normalize_object_id

    phdr = hdul[0].header
    ra = float(phdr.get("RA", 0.0))
    dec = float(phdr.get("DEC", 0.0))

    # Match catalog key by target filename stem (e.g. "g0001234-123456")
    source_id = normalize_object_id(source_path.stem)

    _, vr_hdu = _select_6df_vr_hdu(hdul)
    vhdr = vr_hdu.header
    data = np.asarray(vr_hdu.data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(
            f"6dFGS VR extension in {source_path.name} has shape {data.shape}; expected 2-D"
        )

    if data.shape[0] in (3, 4):
        arr = data
    elif data.shape[1] in (3, 4):
        arr = data.T
    else:
        raise ValueError(
            f"6dFGS VR extension in {source_path.name} has shape {data.shape}; expected (3|4, n_pix)"
        )

    n_pix = int(arr.shape[1])
    flux = arr[0].astype(np.float32)
    variance = arr[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ivar = np.where(variance > 0.0, 1.0 / variance, 0.0).astype(np.float32)
    mask = np.zeros(n_pix, dtype=np.uint8)

    # Some 6dF VR HDUs include an explicit wavelength row in addition to WCS.
    if arr.shape[0] >= 4:
        explicit_wave = np.asarray(arr[3], dtype=np.float64)
        if np.all(np.isfinite(explicit_wave)) and np.all(np.diff(explicit_wave) > 0):
            wavelength = explicit_wave
        else:
            wavelength = _wavelength_from_wcs(vhdr, n_pix)
    else:
        wavelength = _wavelength_from_wcs(vhdr, n_pix)

    wcs_attrs = _wcs_attrs_from_header(vhdr, n_pix)
    meta: dict[str, Any] = {
        "z": float(vhdr.get("Z", phdr.get("Z", 0.0))),
        "z_err": 0.0,
        "snr": 0.0,
        "exptime": float(vhdr.get("EXPTIME", phdr.get("EXPTIME", 0.0))),
        "R": float(vhdr.get("SPEC_RES", 1000.0)),
        "instr": "6dFGS",
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
    source_id_col: str | None = None,
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
    source_id_col:
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
        ``"2df"``, ``"6df"``, ``"generic"``.  Auto-detected from HDU names if ``None``.
    specobj_lookup:
        Parquet/CSV sidecar with ``survey``, ``PLATE``, ``MJD``, ``FIBERID``,
        ``SPECOBJID`` for spPlate ingest.  Mutually exclusive with
        ``specobj_lookup_from_catalog``.
    specobj_lookup_survey:
        When the sidecar has no survey column, use this string to scope rows
        (defaults to ``survey_name``).
    specobj_lookup_from_catalog:
        If True, join ``catalogs/<survey_name>/`` on plate/mjd/fiber and take
        IDs from ``source_id_col`` or the catalog's ID column (not required to
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
            source_id_col=source_id_col,
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
                    source_id_col=source_id_col,
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
                    catalog_id_col=source_id_col,
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
            else:
                records, wcs_attrs = _read_generic_1d(
                    hdul,
                    source_id_col=source_id_col,
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

        n_existing = int(root["source_id"].shape[0])
        existing: set[int] = set()
        if n_existing > 0:
            existing = set(np.asarray(root["source_id"][:]).tolist())

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
        root["source_id"].append(batch_ids)
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
        "--source-id-col",
        default=None,
        help="Object ID: FITS header keyword (generic/SDSS) or fibermap column (DESI).",
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
                  type=click.Choice(["sdss_boss", "sdss_spplate", "desi_coadd", "generic", "2df", "6df"]),
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
        source_id_col: str | None,
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
            source_id_col=source_id_col,
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

        sid_col = source_id_col or "SPECOBJID"
        if update_catalog and index_map:
            try:
                from data_lake.ingest.update_catalog_indices import update_index_column
                n_modified = update_index_column(
                    lake_root=resolved_output,
                    survey_name=survey_name,
                    source_id_to_index=index_map,
                    kind="spectrum",
                    norder=resolved_norder,
                    source_id_col=sid_col,
                )
                click.echo(f"Patched _spectrum_index in {n_modified} catalog tile(s).")
            except FileNotFoundError:
                log.info(
                    "No catalog found for survey %r — skipping _spectrum_index patch.",
                    survey_name,
                )

except ImportError:
    cli = None  # type: ignore[assignment]
