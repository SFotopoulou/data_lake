"""
fits_to_spectra_zarr – ingest 1-D spectra from FITS files into sharded Zarr v3 stacks.

Supported input formats
-----------------------
* **SDSS/BOSS** ``spec-*.fits``  – COADD binary table HDU (FLUX/IVAR/AND_MASK/LOGLAM).
  Uses raw ``astropy.io.fits``; no extra dependency required.
* **DESI**      ``coadd-*.fits`` – Uses ``desispec.io.read_spectra`` +
  ``desispec.coaddition.coadd_cameras`` for IVAR-weighted camera combination of
  the B/R/Z arms onto a single monotonic BRZ wavelength grid.
  Requires ``pip install 'data-lake[desi]'`` (``desispec>=0.62``).
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

    records: list[SpectrumRecord] = []
    for i in range(n_spec):
        ra, dec = base_ra, base_dec
        source_id = base_id if n_spec == 1 else base_id + i
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
            ivar=np.ones(n_pix, dtype=np.float32),
            mask=np.zeros(n_pix, dtype=np.uint8),
            wavelength=wavelength,
            meta=meta,
        ))
    return records, wcs_attrs


def _detect_format_from_path(path: Path) -> str:
    """
    Heuristically detect the FITS spectral format from HDU names only.

    Opens the file with ``lazy_load_hdus=True`` so no data are read,
    then closes it immediately.
    """
    with fits.open(str(path), lazy_load_hdus=True) as hdul:
        names = [h.name.upper() for h in hdul]
    if "COADD" in names:
        return "sdss_boss"
    if any(arm + "_FLUX" in names for arm in ("B", "R", "Z")):
        return "desi_coadd"
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
    on_duplicate_source_id: Literal["append", "error", "skip"] = "append",
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
        Force format detection: ``"sdss_boss"``, ``"desi_coadd"``, ``"generic"``.
        Auto-detected from HDU names if ``None``.
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
        ``append`` (default) may duplicate ``source_id`` rows if a file is
        ingested twice.  ``error`` raises when an ID already exists in the tile.
        ``skip`` drops only conflicting rows from the incoming batch.

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

    n_pix = len(records[0].flux)
    if n_pix_expected is not None and n_pix != n_pix_expected:
        msg = (
            f"Spectrum length {n_pix} != expected {n_pix_expected} in {source_path.name}. "
            f"Pass on_length_mismatch='pad' or 'truncate' to suppress this error."
        )
        if on_length_mismatch == "error":
            raise ValueError(msg)
        log.warning(msg)
        records = _fix_length(records, n_pix_expected, on_length_mismatch)
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

        root = _open_or_create_spectrum_tile(
            tile_path, n_pix, wavelength_mode, mask_dtype, wcs_attrs,
            n_diag=n_diag, resolution_offsets=res_offsets,
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

        start_idx = root["flux"].shape[0]

        batch_flux  = np.stack([r.flux  for r in tile_records]).astype(np.float32)
        batch_ivar  = np.stack([r.ivar  for r in tile_records]).astype(np.float32)
        batch_mask  = np.stack([r.mask  for r in tile_records]).astype(mask_dtype)
        batch_ids   = np.array([r.source_id for r in tile_records], dtype=np.int64)
        batch_meta  = np.frombuffer(
            b"".join(_meta_to_bytes(r.meta) for r in tile_records),
            dtype="|V" + str(_META_DTYPE.itemsize),
        )

        root["flux"].append(batch_flux)
        root["ivar"].append(batch_ivar)
        root["mask"].append(batch_mask)
        root["source_id"].append(batch_ids)
        root["meta"].append(batch_meta)

        if wavelength_mode == "per_source":
            batch_wave = np.stack([
                r.wavelength.astype(np.float32)
                if r.wavelength is not None
                else np.zeros(n_pix, dtype=np.float32)
                for r in tile_records
            ])
            root["wavelength"].append(batch_wave)
        elif start_idx == 0 and tile_records[0].wavelength is not None:
            # Write shared wavelength once (first time the tile is created)
            root["wavelength"][:] = tile_records[0].wavelength.astype(np.float64)

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
        wavelength_mode, str(mask_dtype), wcs_attrs,
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
    fixed = []
    for r in records:
        n = len(r.flux)
        if n == target_n_pix:
            fixed.append(r)
        elif mode == "truncate":
            fixed.append(SpectrumRecord(
                source_id=r.source_id, ra=r.ra, dec=r.dec,
                flux=r.flux[:target_n_pix],
                ivar=r.ivar[:target_n_pix],
                mask=r.mask[:target_n_pix],
                wavelength=r.wavelength[:target_n_pix] if r.wavelength is not None else None,
                meta=r.meta,
            ))
        else:  # pad
            pad = target_n_pix - n
            fixed.append(SpectrumRecord(
                source_id=r.source_id, ra=r.ra, dec=r.dec,
                flux=np.pad(r.flux, (0, pad), constant_values=np.nan),
                ivar=np.pad(r.ivar, (0, pad), constant_values=0.0),
                mask=np.pad(r.mask, (0, pad), constant_values=0),
                wavelength=np.pad(r.wavelength, (0, pad)) if r.wavelength is not None else None,
                meta=r.meta,
            ))
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
    on_duplicate_source_id: str = "append",
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
                  type=click.Choice(["sdss_boss", "desi_coadd", "generic"]),
                  help="Force input format (auto-detected by default).")
    @click.option("--on-length-mismatch",
                  type=click.Choice(["error", "pad", "truncate"]),
                  default="error", show_default=True)
    @click.option(
        "--on-duplicate",
        type=click.Choice(["append", "error", "skip"]),
        default="append",
        show_default=True,
        help="If a source_id already exists in a tile Zarr, append (default), raise, or skip.",
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
                    source_id_col=source_id_col,
                )
                click.echo(f"Patched _spectrum_index in {n_modified} catalog tile(s).")
            except FileNotFoundError:
                log.info(
                    "No catalog found for survey %r — skipping _spectrum_index patch.",
                    survey_name,
                )

except ImportError:
    cli = None  # type: ignore[assignment]
