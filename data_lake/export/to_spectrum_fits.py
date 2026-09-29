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

import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from astropy.io import fits

from data_lake.io.spectra import Spectrum, SpectrumAccessor

log = logging.getLogger(__name__)

DEFAULT_FITS_CHUNK_ROWS = 50_000
MAX_FITS_PARTS_SOFT = 100
MAX_FITS_PARTS_HARD = 10_000
# Leave headroom under 2**32 for FITS/CFITSIO 32-bit table size math.
_BINTABLE_SIZE_LIMIT_BYTES = 2**32 - 64 * 1024 * 1024


def catalog_fits_row_nbytes(n_pix: int, mask_dtype: np.dtype | type = np.uint8) -> int:
    """Bytes per SPECTRA BINTABLE row (TARGETID + Z + FLUX + IVAR + MASK)."""
    mask_item = np.dtype(mask_dtype).itemsize
    return 8 + 4 + int(n_pix) * (4 + 4 + mask_item)


def max_bintable_rows(n_pix: int, mask_dtype: np.dtype | type = np.uint8) -> int:
    """Max rows safe for a single Astropy vector-column BINTABLE write."""
    row = catalog_fits_row_nbytes(n_pix, mask_dtype)
    if row <= 0:
        raise ValueError("invalid n_pix / mask_dtype for catalog FITS row size")
    return max(1, _BINTABLE_SIZE_LIMIT_BYTES // row)


def _part_path(output_fits: Path, part: int) -> Path:
    return output_fits.with_name(f"{output_fits.stem}_part{part:05d}{output_fits.suffix}")


def _checkpoint_path(output_fits: Path) -> Path:
    return output_fits.with_name(f"{output_fits.stem}.extract_fits_checkpoint.json")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def validate_fits_part_count(
    n_written: int,
    chunk_rows: int,
    *,
    force_many_fits_parts: bool = False,
) -> int:
    """Return n_parts or raise if the chunking would create too many files."""
    if chunk_rows < 1:
        raise ValueError("--fits-chunk-rows / fits_chunk_rows must be >= 1")
    if n_written < 1:
        raise ValueError("n_written must be >= 1")
    n_parts = int(math.ceil(n_written / chunk_rows))
    if n_parts > MAX_FITS_PARTS_HARD:
        min_chunk = int(math.ceil(n_written / MAX_FITS_PARTS_HARD))
        raise ValueError(
            f"Catalog FITS would create {n_parts} temporary part files "
            f"(n_written={n_written}, chunk_rows={chunk_rows}), which exceeds the "
            f"hard limit of {MAX_FITS_PARTS_HARD}. Increase fits_chunk_rows to at "
            f"least {min_chunk}."
        )
    if n_parts > MAX_FITS_PARTS_SOFT and not force_many_fits_parts:
        min_chunk = int(math.ceil(n_written / MAX_FITS_PARTS_SOFT))
        raise ValueError(
            f"Catalog FITS would create {n_parts} temporary part files "
            f"(n_written={n_written}, chunk_rows={chunk_rows}). "
            f"Raise fits_chunk_rows to at least {min_chunk} (keeps parts ≤ "
            f"{MAX_FITS_PARTS_SOFT}), or pass force_many_fits_parts=True / "
            f"--force-many-fits-parts."
        )
    return n_parts


def format_catalog_fits_plan_message(
    *,
    n_written: int,
    chunk_rows: int,
    n_pix: int,
    output_fits: Path,
    mask_dtype: np.dtype | type = np.uint8,
) -> str:
    n_parts = int(math.ceil(n_written / chunk_rows))
    rows0 = min(chunk_rows, n_written)
    est_gib = rows0 * catalog_fits_row_nbytes(n_pix, mask_dtype) / (1024**3)
    return (
        "Catalog FITS extract plan:\n"
        f"  spectra to write:  {n_written}\n"
        f"  chunk rows:        {chunk_rows}\n"
        f"  temporary parts:   {n_parts}\n"
        f"  ~size per part:    {est_gib:.2f} GiB\n"
        f"  final output:      {output_fits}\n"
        "  (parts merged at end; intermediates deleted unless --keep-part-files)"
    )


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


def _wcs_header_from_attrs(wcs_attrs: dict[str, Any], n_pix: int) -> fits.Header:
    """Build a minimal spectral WCS header from lake WCS attrs."""
    hdr = fits.Header()
    hdr["SIMPLE"] = True
    hdr["BITPIX"] = -32
    hdr["NAXIS"] = 1
    hdr["NAXIS1"] = n_pix
    ctype = str(wcs_attrs.get("ctype", "WAVE")).upper()
    hdr["CTYPE1"] = ctype
    hdr["CRVAL1"] = float(wcs_attrs.get("crval", 0.0))
    hdr["CDELT1"] = float(wcs_attrs.get("cdelt", 1.0))
    hdr["CRPIX1"] = float(wcs_attrs.get("crpix", 1.0))
    hdr["CUNIT1"] = str(wcs_attrs.get("unit", "Angstrom"))
    hdr["DC-FLAG"] = 1 if "LOG" in ctype else 0
    return hdr


def write_spectra_catalog_fits(
    output_path: Path | str,
    *,
    source_id: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
    wavelength: np.ndarray,
    redshift: np.ndarray | None = None,
    wcs_attrs: dict[str, Any] | None = None,
    survey: str = "",
    overwrite: bool = True,
    table_name: str = "SPECTRA",
) -> Path:
    """Write many spectra into one multi-row FITS catalog file.

    Layout
    ------
    * HDU 0 ``PRIMARY`` – spectral WCS keywords (shared grid).
    * HDU 1 ``SPECTRA`` – BINTABLE: ``TARGETID``, ``Z``, ``FLUX``, ``IVAR``,
      ``MASK`` (fixed-length vector columns, one row per spectrum).
    * HDU 2 ``WAVELENGTH`` – shared 1-D wavelength grid (Angstrom).

    Parameters
    ----------
    source_id, flux, ivar, mask:
        Arrays with shape ``(N_spec,)`` or ``(N_spec, N_pix)`` for the 2-D fields.
    wavelength:
        Shared wavelength grid, shape ``(N_pix,)``.
    redshift:
        Per-source redshift; defaults to NaN if omitted.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass overwrite=True.")

    flux = np.asarray(flux, dtype=np.float32)
    ivar = np.asarray(ivar, dtype=np.float32)
    mask = np.asarray(mask)
    source_id = np.asarray(source_id, dtype=np.int64).ravel()
    wavelength = np.asarray(wavelength, dtype=np.float64).ravel()
    n_spec, n_pix = flux.shape
    if ivar.shape != (n_spec, n_pix) or mask.shape != (n_spec, n_pix):
        raise ValueError("flux, ivar, and mask must share shape (N_spec, N_pix)")
    if source_id.shape[0] != n_spec:
        raise ValueError("source_id length must match N_spec")

    max_rows = max_bintable_rows(n_pix, mask.dtype)
    if n_spec > max_rows:
        est_gib = n_spec * catalog_fits_row_nbytes(n_pix, mask.dtype) / (1024**3)
        raise ValueError(
            f"FITS vector BINTABLE would be ~{est_gib:.1f} GiB "
            f"({n_spec} rows × {catalog_fits_row_nbytes(n_pix, mask.dtype)} B/row), "
            f"which exceeds the ~4 GiB Astropy/CFITSIO table limit "
            f"(~{max_rows} rows max at n_pix={n_pix}). "
            "Use streamed/sharded catalog extract (fits_chunk_rows) or Zarr/HDF5."
        )

    if redshift is None:
        redshift = np.full(n_spec, np.nan, dtype=np.float32)
    else:
        redshift = np.asarray(redshift, dtype=np.float32).ravel()
        if redshift.shape[0] != n_spec:
            raise ValueError("redshift length must match N_spec")

    wcs = wcs_attrs or {}
    primary_hdr = _wcs_header_from_attrs(wcs, n_pix)
    primary_hdr["ORIGIN"] = "data_lake"
    primary_hdr["SURVEY"] = str(survey)[:8]
    primary_hdr["NSPEC"] = n_spec
    primary_hdr["NPIX"] = n_pix

    mask_fmt = "B" if mask.dtype == np.uint8 else "I"
    cols = [
        fits.Column("TARGETID", "K", array=source_id),
        fits.Column("Z", "E", array=redshift),
        fits.Column("FLUX", f"{n_pix}E", array=flux),
        fits.Column("IVAR", f"{n_pix}E", array=ivar),
        fits.Column("MASK", f"{n_pix}{mask_fmt}", array=mask),
    ]
    spectra_hdu = fits.BinTableHDU.from_columns(cols, name=table_name)
    wave_hdu = fits.ImageHDU(
        data=wavelength.astype(np.float64)[np.newaxis, :],
        name="WAVELENGTH",
    )
    wave_hdu.header["EXTNAME"] = "WAVELENGTH"
    wave_hdu.header["CUNIT1"] = "Angstrom"

    hdul = fits.HDUList([fits.PrimaryHDU(header=primary_hdr), spectra_hdu, wave_hdu])
    hdul.writeto(str(output_path), overwrite=overwrite)
    log.info("Wrote catalog FITS %s (%d spectra, %d pix)", output_path.name, n_spec, n_pix)
    return output_path


def merge_spectra_catalog_fits_parts(
    part_paths: Sequence[Path | str],
    output_fits: Path | str,
    *,
    overwrite: bool = True,
) -> Path:
    """Merge ``_partNNNNN.fits`` catalog shards into one file.

    Uses Astropy when the merged table fits under the vector-BINTABLE size
    limit; otherwise requires ``fitsio`` for append-based merge.
    """
    parts = [Path(p) for p in part_paths]
    if not parts:
        raise ValueError("part_paths is empty")
    for p in parts:
        if not p.is_file():
            raise FileNotFoundError(f"Missing catalog FITS part: {p}")

    output_fits = Path(output_fits)
    if output_fits.exists():
        if not overwrite:
            raise FileExistsError(f"{output_fits} already exists. Pass overwrite=True.")
        output_fits.unlink()

    if len(parts) == 1:
        shutil.copy2(parts[0], output_fits)
        log.info("Catalog FITS merge: single part → %s", output_fits)
        return output_fits

    # Inspect sizes from the first part.
    with fits.open(parts[0]) as hdul:
        n_pix = int(hdul[0].header.get("NPIX", hdul["WAVELENGTH"].data.shape[-1]))
        mask0 = np.asarray(hdul["SPECTRA"].data["MASK"])
        mask_dtype = mask0.dtype
        survey = str(hdul[0].header.get("SURVEY", ""))
        wcs_attrs = {
            "ctype": hdul[0].header.get("CTYPE1", "WAVE"),
            "crval": float(hdul[0].header.get("CRVAL1", 0.0)),
            "cdelt": float(hdul[0].header.get("CDELT1", 1.0)),
            "crpix": float(hdul[0].header.get("CRPIX1", 1.0)),
            "unit": hdul[0].header.get("CUNIT1", "Angstrom"),
        }
        wavelength = np.asarray(hdul["WAVELENGTH"].data, dtype=np.float64).ravel()

    total_rows = 0
    for p in parts:
        with fits.open(p) as hdul:
            total_rows += int(hdul["SPECTRA"].header.get("NAXIS2", 0))

    if total_rows <= max_bintable_rows(n_pix, mask_dtype):
        sid_chunks: list[np.ndarray] = []
        flux_chunks: list[np.ndarray] = []
        ivar_chunks: list[np.ndarray] = []
        mask_chunks: list[np.ndarray] = []
        z_chunks: list[np.ndarray] = []
        for p in parts:
            with fits.open(p) as hdul:
                data = hdul["SPECTRA"].data
                sid_chunks.append(np.asarray(data["TARGETID"], dtype=np.int64))
                flux_chunks.append(np.asarray(data["FLUX"], dtype=np.float32))
                ivar_chunks.append(np.asarray(data["IVAR"], dtype=np.float32))
                mask_chunks.append(np.asarray(data["MASK"]))
                z_chunks.append(np.asarray(data["Z"], dtype=np.float32))
        write_spectra_catalog_fits(
            output_fits,
            source_id=np.concatenate(sid_chunks),
            flux=np.concatenate(flux_chunks, axis=0),
            ivar=np.concatenate(ivar_chunks, axis=0),
            mask=np.concatenate(mask_chunks, axis=0),
            wavelength=wavelength,
            redshift=np.concatenate(z_chunks),
            wcs_attrs=wcs_attrs,
            survey=survey,
            overwrite=True,
        )
        log.info(
            "Merged %d catalog FITS parts → %s (%d spectra, Astropy)",
            len(parts), output_fits.name, total_rows,
        )
        return output_fits

    try:
        import fitsio
    except ImportError as exc:
        raise ImportError(
            "Merging catalog FITS parts above the ~4 GiB Astropy BINTABLE limit "
            "requires fitsio. Install with: uv sync --extra fitsio"
        ) from exc

    shutil.copy2(parts[0], output_fits)
    with fitsio.FITS(str(output_fits), "rw") as out_fits:
        spectra = out_fits["SPECTRA"]
        for part in parts[1:]:
            with fitsio.FITS(str(part), "r") as in_fits:
                data = in_fits["SPECTRA"].read()
            spectra.append(data)

    with fits.open(output_fits, mode="update") as hdul:
        n_spec = int(hdul["SPECTRA"].header.get("NAXIS2", 0))
        hdul[0].header["NSPEC"] = n_spec
        hdul.flush()

    log.info(
        "Merged %d catalog FITS parts → %s (%d spectra, fitsio)",
        len(parts), output_fits.name, n_spec,
    )
    return output_fits


def stream_spectra_catalog_fits(
    batch_iter: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    output_fits: Path | str,
    *,
    wavelength: np.ndarray,
    wcs_attrs: dict[str, Any] | None = None,
    survey: str = "",
    n_written: int,
    fits_chunk_rows: int = DEFAULT_FITS_CHUNK_ROWS,
    overwrite: bool = False,
    keep_part_files: bool = False,
    force_many_fits_parts: bool = False,
    confirm: bool | Callable[[str], bool] | None = None,
    mask_dtype: np.dtype | type = np.uint8,
) -> dict[str, Any]:
    """Stream spectrum batches into catalog FITS parts, then merge to ``output_fits``.

    ``batch_iter`` yields ``(source_id, flux, ivar, mask, redshift)`` arrays for
    each tile (or other) batch in write order.

    Parameters
    ----------
    confirm:
        ``True`` to proceed without prompting; ``False`` to abort; a callable
        receiving the plan message and returning bool; or ``None`` to require
        an explicit ``True``/callable when more than one part is needed.
    """
    output_fits = Path(output_fits)
    wavelength = np.asarray(wavelength, dtype=np.float64).ravel()
    n_pix = int(wavelength.shape[0])
    if fits_chunk_rows < 1:
        raise ValueError("fits_chunk_rows must be >= 1")

    n_parts = validate_fits_part_count(
        n_written,
        fits_chunk_rows,
        force_many_fits_parts=force_many_fits_parts,
    )
    plan_msg = format_catalog_fits_plan_message(
        n_written=n_written,
        chunk_rows=fits_chunk_rows,
        n_pix=n_pix,
        output_fits=output_fits,
        mask_dtype=mask_dtype,
    )
    log.info("\n%s", plan_msg)

    if confirm is False:
        raise RuntimeError("Catalog FITS extract cancelled (confirm=False).")
    if confirm is None and n_parts > 1:
        raise ValueError(
            "Catalog FITS extract would write multiple temporary part files and "
            "requires confirmation. Pass confirm=True, a confirm callback, or "
            "use the CLI prompt / --yes."
        )
    if callable(confirm):
        if not confirm(plan_msg):
            raise RuntimeError("Catalog FITS extract cancelled by user.")
    # confirm is True, or None with a single part → proceed

    ckpt_path = _checkpoint_path(output_fits)
    use_parts = n_parts > 1
    part_paths = (
        [_part_path(output_fits, i) for i in range(n_parts)]
        if use_parts
        else [output_fits]
    )

    if overwrite:
        if use_parts:
            for p in [_part_path(output_fits, i) for i in range(MAX_FITS_PARTS_HARD)]:
                if p.exists():
                    p.unlink()
        if ckpt_path.exists():
            ckpt_path.unlink()
        if output_fits.exists() and use_parts:
            output_fits.unlink()
        elif output_fits.exists() and overwrite and not use_parts:
            output_fits.unlink()

    ckpt: dict[str, Any] = {
        "chunk_rows": fits_chunk_rows,
        "n_written_total": n_written,
        "n_pix": n_pix,
        "part_nrows": [],
        "completed_parts": [],
        "merge_done": False,
    }
    if ckpt_path.exists() and not overwrite:
        try:
            loaded = json.loads(ckpt_path.read_text())
            if (
                int(loaded.get("chunk_rows", -1)) == fits_chunk_rows
                and int(loaded.get("n_written_total", -1)) == n_written
                and int(loaded.get("n_pix", -1)) == n_pix
            ):
                ckpt = loaded
        except Exception:
            log.warning("Ignoring unreadable checkpoint %s", ckpt_path)

    completed = {int(x) for x in ckpt.get("completed_parts", [])}
    part_nrows: list[int] = [int(x) for x in ckpt.get("part_nrows", [])]
    while len(part_nrows) < n_parts:
        part_nrows.append(0)

    # Validate / repair completed parts on disk.
    for part_i in sorted(completed):
        path = part_paths[part_i]
        if not path.is_file():
            completed.discard(part_i)
            part_nrows[part_i] = 0
            continue
        try:
            with fits.open(path) as hdul:
                n_on_disk = int(hdul["SPECTRA"].header.get("NAXIS2", 0))
            expected = (
                fits_chunk_rows
                if part_i < n_parts - 1
                else n_written - fits_chunk_rows * (n_parts - 1)
            )
            if n_on_disk != expected and n_on_disk != part_nrows[part_i]:
                # Prefer header count; if mismatch with expected full part, rewrite.
                if part_i < n_parts - 1 and n_on_disk != fits_chunk_rows:
                    path.unlink(missing_ok=True)
                    completed.discard(part_i)
                    part_nrows[part_i] = 0
                else:
                    part_nrows[part_i] = n_on_disk
        except Exception:
            path.unlink(missing_ok=True)
            completed.discard(part_i)
            part_nrows[part_i] = 0

    if ckpt.get("merge_done") and output_fits.is_file() and (
        not use_parts or len(completed) >= n_parts
    ):
        log.info("Catalog FITS already complete: %s", output_fits)
        return {
            "output": str(output_fits),
            "output_fits": str(output_fits),
            "n_written": n_written,
            "n_parts": n_parts,
            "fits_chunk_rows": fits_chunk_rows,
            "merged": True,
        }

    rows_to_skip = sum(
        part_nrows[i] for i in range(n_parts) if i in completed and part_nrows[i] > 0
    )
    if not rows_to_skip and completed:
        for part_i in range(n_parts):
            if part_i not in completed:
                continue
            if part_nrows[part_i] > 0:
                rows_to_skip += part_nrows[part_i]
            elif part_i < n_parts - 1:
                rows_to_skip += fits_chunk_rows

    buf_sid: list[np.ndarray] = []
    buf_flux: list[np.ndarray] = []
    buf_ivar: list[np.ndarray] = []
    buf_mask: list[np.ndarray] = []
    buf_z: list[np.ndarray] = []
    buf_rows = 0
    skipped = 0
    next_part = 0
    while next_part < n_parts and next_part in completed:
        next_part += 1

    def _concat_take(
        chunks: list[np.ndarray], n: int
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        flat = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        head, rest = flat[:n], flat[n:]
        return head, ([rest] if rest.shape[0] else [])

    def _flush(part_i: int) -> None:
        nonlocal buf_rows, next_part
        if buf_rows == 0:
            return
        if part_i < n_parts - 1:
            take = fits_chunk_rows
        else:
            take = buf_rows
        take = min(take, buf_rows)

        sid, buf_sid[:] = _concat_take(buf_sid, take)
        flux, buf_flux[:] = _concat_take(buf_flux, take)
        ivar, buf_ivar[:] = _concat_take(buf_ivar, take)
        mask, buf_mask[:] = _concat_take(buf_mask, take)
        z, buf_z[:] = _concat_take(buf_z, take)
        buf_rows -= take

        out_part = part_paths[part_i]
        write_spectra_catalog_fits(
            out_part,
            source_id=sid,
            flux=flux,
            ivar=ivar,
            mask=mask,
            wavelength=wavelength,
            redshift=z,
            wcs_attrs=wcs_attrs,
            survey=survey,
            overwrite=True,
        )
        part_nrows[part_i] = int(take)
        completed.add(part_i)
        ckpt["completed_parts"] = sorted(completed)
        ckpt["part_nrows"] = part_nrows
        ckpt["merge_done"] = False
        if use_parts:
            _atomic_write_json(ckpt_path, ckpt)
        log.info(
            "Wrote catalog FITS part %d/%d → %s (%d rows)",
            part_i + 1, n_parts, out_part.name, take,
        )
        next_part = part_i + 1

    for sid_b, flux_b, ivar_b, mask_b, z_b in batch_iter:
        sid_b = np.asarray(sid_b, dtype=np.int64).ravel()
        flux_b = np.asarray(flux_b, dtype=np.float32)
        ivar_b = np.asarray(ivar_b, dtype=np.float32)
        mask_b = np.asarray(mask_b)
        z_b = np.asarray(z_b, dtype=np.float32).ravel()
        n = int(sid_b.shape[0])
        if n == 0:
            continue

        if skipped < rows_to_skip:
            remain_skip = rows_to_skip - skipped
            if n <= remain_skip:
                skipped += n
                continue
            sid_b = sid_b[remain_skip:]
            flux_b = flux_b[remain_skip:]
            ivar_b = ivar_b[remain_skip:]
            mask_b = mask_b[remain_skip:]
            z_b = z_b[remain_skip:]
            skipped += remain_skip
            n = int(sid_b.shape[0])

        buf_sid.append(sid_b)
        buf_flux.append(flux_b)
        buf_ivar.append(ivar_b)
        buf_mask.append(mask_b)
        buf_z.append(z_b)
        buf_rows += n

        while next_part < n_parts - 1 and buf_rows >= fits_chunk_rows:
            _flush(next_part)

    if next_part < n_parts and buf_rows > 0:
        _flush(next_part)

    if len(completed) < n_parts:
        raise RuntimeError(
            f"Catalog FITS stream incomplete: finished {len(completed)}/{n_parts} parts "
            f"({buf_rows} rows left in buffer; expected n_written={n_written})."
        )

    if use_parts:
        try:
            merge_spectra_catalog_fits_parts(part_paths, output_fits, overwrite=True)
        except ImportError:
            ckpt["completed_parts"] = sorted(completed)
            ckpt["part_nrows"] = part_nrows
            ckpt["merge_done"] = False
            _atomic_write_json(ckpt_path, ckpt)
            raise

        ckpt["merge_done"] = True
        ckpt["completed_parts"] = sorted(completed)
        ckpt["part_nrows"] = part_nrows
        _atomic_write_json(ckpt_path, ckpt)

        if not keep_part_files:
            for p in part_paths:
                p.unlink(missing_ok=True)
            ckpt_path.unlink(missing_ok=True)
    else:
        # Single-file write already targeted output_fits.
        pass

    return {
        "output": str(output_fits),
        "output_fits": str(output_fits),
        "n_written": n_written,
        "n_parts": n_parts,
        "fits_chunk_rows": fits_chunk_rows,
        "merged": True,
        "kept_part_files": bool(keep_part_files and use_parts),
    }


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

        from data_lake.io.fits_read import open_fits

        with open_fits(fits_path) as hdul:
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
