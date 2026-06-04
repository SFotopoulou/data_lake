"""
spplate_parallel_ingest – vectorized SDSS spPlate batch ingest.

Architecture (mirrors desi_parallel_ingest)
-------------------------------------------
::

    file list (spPlate-*.fits)
            │
            ├──▶ ProcessPoolExecutor  (N workers)
            │       per file:
            │         astropy.io.fits memmap (flux/ivar/mask already 2D)
            │         vectorized keep-mask (sky validity + active flux)
            │         fiber-ID LUT → source_id array (no dict.get loop)
            │         assign_healpix → group by tile
            │         return numpy TileBatch payloads
            │
            └──▶ main-process writer  (shared with DESI path)
                      ingest_spectra_parallel from desi_parallel_ingest

Key differences from DESI path
-------------------------------
* No desispec dependency — raw astropy.io.fits only.
* Shared log-lambda wavelength grid (COEFF0/COEFF1) → ``WorkerResult.wavelength``
  so tiles use ``wavelength_mode=shared`` (same as DESI), saving per-fiber copies.
* Fiber-ID lookup is a numpy LUT of size ``max_fiberid + 1``, not a Python dict.
* All array filtering (sky validity, active flux, matched ID) is done with
  boolean mask operations on (n_fiber, n_pix) slabs before any Python iteration.

Legacy lookup modes (--specobj-lookup / --specobj-lookup-from-plate)
--------------------------------------------------------------------
Sidecar and plate-synthesis modes are supported in the worker.
``--specobj-lookup-from-catalog`` (tile scan) is intentionally excluded:
use ``dl-ingest-spectra-from-list`` with the existing slow path if needed.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import numpy as np

from data_lake.ingest.fits_to_parquet import (
    assign_healpix,
    valid_sky_position_mask,
)
from data_lake.ingest.fits_to_spectra_zarr import _META_DTYPE
from data_lake.ingest.desi_parallel_ingest import (
    TileBatch,
    WorkerResult,
    ZarrDuplicateMode,
    _canonical_fits_path,
    ingest_spectra_parallel,
)
from data_lake.ingest.sdss_specobj_lookup import (
    build_fiber_to_source_id_from_triplet,
    spplate_plate_mjd_from_hdul,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-worker decode config (picklable)
# ---------------------------------------------------------------------------


@dataclass
class SpplateBatchConfig:
    """Decode configuration for one spPlate batch run (picklable for workers)."""

    norder: int = 5
    ra_col: str = "RA"
    dec_col: str = "DEC"
    # ID source — exactly one of:
    #   triplet_hash=True (default), lookup_path set, or lookup_from_plate=True
    triplet_hash: bool = True
    lookup_path: str | None = None          # sidecar Parquet/CSV
    lookup_survey: str | None = None        # survey filter for sidecar
    lookup_from_plate: bool = False         # CAS specObjID synthesis
    specobj_id_layout: str = "auto"         # "auto" | "dr7" | "dr8plus"


# ---------------------------------------------------------------------------
# Vectorized worker
# ---------------------------------------------------------------------------


_LUT_SENTINEL = np.iinfo(np.int64).min  # marks unmapped fiber slots in LUT


def _build_fiber_lut(
    fiber_ids: np.ndarray,
    source_ids: np.ndarray,
) -> np.ndarray:
    """Return int64 LUT indexed by fiber_id; _LUT_SENTINEL for unmapped fibers.

    *fiber_ids* and *source_ids* must be 1-D with the same length.
    Max fiber_id determines array size (~1001 for BOSS plates).
    Source IDs can be any int64 (including negative values from BLAKE2b hashes),
    so _LUT_SENTINEL = INT64_MIN is used as the unmapped-slot marker.
    """
    max_fid = int(fiber_ids.max()) + 1 if fiber_ids.size else 0
    lut = np.full(max_fid, _LUT_SENTINEL, dtype=np.int64)
    for fid, sid in zip(fiber_ids.tolist(), source_ids.tolist()):
        fi = int(fid)
        if fi < max_fid:
            lut[fi] = int(sid)
    return lut


def _decode_spplate_to_worker_result(
    path: Path,
    config: SpplateBatchConfig,
) -> WorkerResult:
    """Decode one spPlate FITS file into per-tile TileBatch payloads.

    Returns a :class:`~data_lake.ingest.desi_parallel_ingest.WorkerResult` with
    ``wavelength`` set to the shared log-lambda grid and ``TileBatch`` entries
    without ``wavelength_rows`` (shared-wavelength mode).
    """
    from astropy.io import fits

    from data_lake.ingest.fits_to_spectra_zarr import (
        _fits_bintable_column,
        _spplate_fiber_table_hdu,
        _spplate_flux_and_calib_hdus,
    )

    t0 = time.perf_counter()

    with fits.open(str(path), memmap=True) as hdul:
        plate, mjd = spplate_plate_mjd_from_hdul(hdul, path)
        phdr = hdul[0].header

        if "COEFF0" not in phdr or "COEFF1" not in phdr:
            return WorkerResult(
                path=str(path), ok=False,
                error="spPlate primary header missing COEFF0/COEFF1",
                elapsed_s=time.perf_counter() - t0,
            )

        coeff0 = float(phdr["COEFF0"])
        coeff1 = float(phdr["COEFF1"])

        flux_2d, ivar_2d, mask_and, mask_or = _spplate_flux_and_calib_hdus(hdul)
        n_fiber, n_pix = flux_2d.shape

        pix_idx = np.arange(n_pix, dtype=np.float64)
        loglam = coeff0 + coeff1 * pix_idx
        wavelength = (10.0 ** loglam).astype(np.float64)

        ftable = _spplate_fiber_table_hdu(hdul)
        if ftable is None:
            return WorkerResult(
                path=str(path), ok=False,
                error="spPlate: no per-fiber BINTABLE with FIBERID column",
                elapsed_s=time.perf_counter() - t0,
            )

        fdata = ftable.data
        n_table = len(fdata)
        n_rows = min(n_fiber, n_table)

        fiber_col = _fits_bintable_column(fdata, "fiberid", "FIBERID")
        ra_col_name = _fits_bintable_column_name(fdata, config.ra_col.lower(), config.ra_col, "RA")
        dec_col_name = _fits_bintable_column_name(fdata, config.dec_col.lower(), config.dec_col, "DEC")
        ra_arr = np.asarray(fdata[ra_col_name][:n_rows], dtype=np.float64)
        dec_arr = np.asarray(fdata[dec_col_name][:n_rows], dtype=np.float64)
        fids = np.asarray(fiber_col[:n_rows], dtype=np.int32)

        # --- Build fiber → source_id LUT ---
        if config.triplet_hash:
            unique_fids = np.unique(fids)
            fid_to_sid = build_fiber_to_source_id_from_triplet(plate, mjd, unique_fids.tolist())
            lut_max = max((int(k) for k in fid_to_sid), default=-1) + 2
            lut = np.full(lut_max, _LUT_SENTINEL, dtype=np.int64)
            for fid, sid in fid_to_sid.items():
                fi = int(fid)
                if fi < lut_max:
                    lut[fi] = int(sid)
        elif config.lookup_from_plate:
            from data_lake.ingest.sdss_specobj_lookup import (
                build_fiber_to_specobjid_from_spplate,
            )
            fiber_map = build_fiber_to_specobjid_from_spplate(
                hdul,
                path,
                fiber_ids=fids.tolist(),
                specobj_id_layout=config.specobj_id_layout,  # type: ignore[arg-type]
            )
            lut_max = max((int(k) for k in fiber_map), default=-1) + 2
            lut = np.full(lut_max, _LUT_SENTINEL, dtype=np.int64)
            for fid, sid in fiber_map.items():
                fi = int(fid)
                if fi < lut_max:
                    lut[fi] = int(sid)
        elif config.lookup_path is not None:
            from data_lake.ingest.sdss_specobj_lookup import build_fiber_to_specobjid_map
            fiber_map = build_fiber_to_specobjid_map(
                "",
                plate,
                mjd,
                lookup_path=config.lookup_path,
                lookup_survey=config.lookup_survey,
            )
            lut_max = max((int(k) for k in fiber_map), default=-1) + 2
            lut = np.full(lut_max, _LUT_SENTINEL, dtype=np.int64)
            for fid, sid in fiber_map.items():
                fi = int(fid)
                if fi < lut_max:
                    lut[fi] = int(sid)
        else:
            return WorkerResult(
                path=str(path), ok=False,
                error=(
                    "No ID lookup mode set. Use triplet_hash=True (default), "
                    "lookup_from_plate=True, or provide lookup_path."
                ),
                elapsed_s=time.perf_counter() - t0,
            )

        # --- Vectorized keep mask ---
        # Clip fiber IDs to lut range then lookup.
        # Valid source IDs can be any int64 (including negative BLAKE2b hashes);
        # _LUT_SENTINEL = INT64_MIN marks genuinely unmapped fibers.
        fids_clipped = np.clip(fids, 0, len(lut) - 1)
        sid_arr = lut[fids_clipped]
        matched = sid_arr != _LUT_SENTINEL

        sky_valid = valid_sky_position_mask(ra_arr, dec_arr)

        flux_rows = flux_2d[:n_rows]
        active = np.any(np.isfinite(flux_rows) & (flux_rows != 0.0), axis=1)

        keep = matched & sky_valid & active

        n_keep = int(keep.sum())
        if n_keep == 0:
            log.warning(
                "spPlate %s: no active/matched fibers (plate=%d mjd=%d; %d rows)",
                path.name, plate, mjd, n_rows,
            )
            return WorkerResult(
                path=str(path), ok=True,
                wavelength=wavelength,
                wcs_attrs={
                    "ctype": "WAVE-LOG",
                    "crval": float(loglam[0]),
                    "cdelt": float(coeff1),
                    "crpix": 1.0,
                    "unit": "Angstrom",
                    "air_or_vacuum": "vacuum",
                    "n_pix": int(n_pix),
                },
                n_pix=int(n_pix),
                n_spectra=0,
                elapsed_s=time.perf_counter() - t0,
            )

        flux_keep = flux_rows[keep].astype(np.float32, copy=False)
        if ivar_2d is not None:
            ivar_keep = ivar_2d[:n_rows][keep].astype(np.float32, copy=False)
        else:
            ivar_keep = np.ones((n_keep, n_pix), dtype=np.float32)

        # Combined mask
        mask_keep = np.zeros((n_keep, n_pix), dtype=np.uint8)
        if mask_and is not None:
            m = np.clip(mask_and[:n_rows][keep], 0, 255).astype(np.uint8)
            mask_keep = m
        if mask_or is not None:
            m = np.clip(mask_or[:n_rows][keep], 0, 255).astype(np.uint8)
            mask_keep = np.clip(mask_keep | m, 0, 255).astype(np.uint8)

        ra_keep = ra_arr[keep]
        dec_keep = dec_arr[keep]
        sids_keep = sid_arr[keep]
        fids_keep = fids[keep]

        # --- Build meta in bulk (one structured array, no per-row dict) ---
        exptime = float(phdr.get("EXPTIME", 0.0))
        spec_res = float(phdr.get("SPEC_RES", 2000.0))
        src_file = path.name[:128].encode("ascii", errors="replace")
        ra_key_b = ra_col_name[:32].encode("ascii", errors="replace")
        dec_key_b = dec_col_name[:32].encode("ascii", errors="replace")

        meta_arr = np.zeros(n_keep, dtype=_META_DTYPE)
        meta_arr["exptime"] = exptime
        meta_arr["R"] = spec_res
        meta_arr["instr"] = b"SDSS".ljust(16)[:16]
        meta_arr["ra_key"] = ra_key_b.ljust(32)[:32]
        meta_arr["dec_key"] = dec_key_b.ljust(32)[:32]
        meta_arr["ra"] = ra_keep.astype(np.float32)
        meta_arr["dec"] = dec_keep.astype(np.float32)
        meta_arr["source_file"] = src_file.ljust(128)[:128]
        meta_bytes_total = bytes(meta_arr.view(f"|V{_META_DTYPE.itemsize}"))

        # --- HEALPix grouping ---
        npix_arr = assign_healpix(ra_keep, dec_keep, config.norder)
        sort_idx = np.argsort(npix_arr, kind="stable")
        npix_sorted = npix_arr[sort_idx]

        unique_pix, group_starts = np.unique(npix_sorted, return_index=True)
        group_starts = np.append(group_starts, n_keep)

        itemsize = _META_DTYPE.itemsize
        batches: list[TileBatch] = []
        for g, pix in enumerate(unique_pix):
            s, e = int(group_starts[g]), int(group_starts[g + 1])
            idx = sort_idx[s:e]
            meta_chunk = b"".join(
                meta_bytes_total[int(i) * itemsize: int(i) * itemsize + itemsize]
                for i in idx.tolist()
            )
            batches.append(TileBatch(
                npix=int(pix),
                flux=flux_keep[idx],
                ivar=ivar_keep[idx],
                mask=mask_keep[idx],
                source_ids=sids_keep[idx].astype(np.int64),
                meta_bytes=meta_chunk,
            ))

    wcs_attrs = {
        "ctype": "WAVE-LOG",
        "crval": float(loglam[0]),
        "cdelt": float(coeff1),
        "crpix": 1.0,
        "unit": "Angstrom",
        "air_or_vacuum": "vacuum",
        "n_pix": int(n_pix),
    }

    log.debug(
        "spPlate %s: plate=%d mjd=%d %d/%d fibers → %d tile(s) in %.2fs",
        path.name, plate, mjd, n_keep, n_rows, len(batches),
        time.perf_counter() - t0,
    )

    return WorkerResult(
        path=str(path),
        ok=True,
        batches=batches,
        wavelength=wavelength,
        wcs_attrs=wcs_attrs,
        n_pix=int(n_pix),
        n_spectra=n_keep,
        elapsed_s=time.perf_counter() - t0,
    )


def _fits_bintable_column_name(fdata, *candidates: str) -> str:
    """Return the first matching column name from fdata (case-insensitive)."""
    names_lower = {n.lower(): n for n in (fdata.dtype.names or ())}
    for cand in candidates:
        found = names_lower.get(cand.lower())
        if found is not None:
            return found
    return candidates[-1]  # fallback to last candidate as default name


def decode_one_spplate(path_str: str, config: SpplateBatchConfig) -> WorkerResult:
    """Worker entry point: decode one spPlate file into TileBatch payloads."""
    from data_lake.cli_utils import apply_parallel_worker_logging_after_heavy_imports

    apply_parallel_worker_logging_after_heavy_imports()
    return _decode_spplate_to_worker_result(Path(path_str), config)


def decode_one_spplate_safe(path_str: str, config: SpplateBatchConfig) -> WorkerResult:
    """Picklable wrapper: never raises, returns a failure-marked result instead."""
    try:
        return decode_one_spplate(path_str, config)
    except Exception as exc:
        return WorkerResult(
            path=path_str,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            tb=traceback.format_exc(),
        )


class _SpplateDecoder:
    """Picklable callable adapter: ``(path_str, norder)`` → WorkerResult.

    ``ingest_spectra_parallel`` passes ``(path, norder)`` to the decoder; the
    config already carries the correct ``norder`` so the argument is ignored.
    Using a class (rather than a closure) ensures pickling works correctly with
    ``ProcessPoolExecutor``.
    """

    def __init__(self, config: SpplateBatchConfig) -> None:
        self.config = config

    def __call__(self, path_str: str, _norder: int) -> WorkerResult:
        return decode_one_spplate_safe(path_str, self.config)


# ---------------------------------------------------------------------------
# Public batch ingest entry point
# ---------------------------------------------------------------------------


def ingest_spplate_files_parallel(
    file_paths: Sequence[Path | str],
    output_root: Path | str,
    survey_name: str,
    *,
    n_workers: int,
    norder: int = 5,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    triplet_hash: bool = True,
    lookup_path: Path | str | None = None,
    lookup_survey: str | None = None,
    lookup_from_plate: bool = False,
    specobj_id_layout: str = "auto",
    checkpoint_path: Path | str | None = None,
    failures_log: Path | str | None = None,
    worker_log_file: Path | str | None = None,
    worker_verbose: bool = False,
    show_progress: bool = True,
    on_duplicate_source_id: ZarrDuplicateMode = "skip",
    max_in_flight: int | None = None,
    max_open_tiles: int = 64,
    track_index_map: bool = False,
    executor_factory: Callable | None = None,
    on_length_mismatch: str = "pad",
) -> dict:
    """Ingest many spPlate FITS files in parallel using a vectorized decoder.

    Uses the shared DESI parallel writer from
    :func:`~data_lake.ingest.desi_parallel_ingest.ingest_spectra_parallel`.
    Catalog must be ingested with ``--link-id-col PLATE,MJD,FIBERID`` to match
    the default triplet-hash IDs.

    Parameters
    ----------
    file_paths:
        ``spPlate-PLATE-MJD.fits`` paths to ingest.
    output_root / survey_name:
        Lake root and survey identifier.
    n_workers:
        Decoder process count; typically ``cpu_count - 1``.
    norder:
        HEALPix partitioning order (default 5).
    ra_col / dec_col:
        Plugmap column names for sky coordinates (default ``RA`` / ``DEC``).
    triplet_hash:
        When True (default), derive ``_source_id`` from the composite
        ``PLATE|MJD|FIBERID`` hash.  Mutually exclusive with ``lookup_path``
        and ``lookup_from_plate``.
    lookup_path:
        Sidecar Parquet/CSV with PLATE, MJD, FIBERID, and a source ID column.
    lookup_survey:
        Survey filter when the sidecar has a SURVEY column.
    lookup_from_plate:
        Synthesize CAS specObjID from header PLATE/MJD/FIBERID/RUN2D.
    specobj_id_layout:
        ``"auto"`` | ``"dr7"`` | ``"dr8plus"`` (only used with
        ``lookup_from_plate``).
    checkpoint_path / failures_log / worker_log_file:
        Same meaning as in ``ingest_spectra_parallel``.
    on_duplicate_source_id:
        ``"skip"`` | ``"append"`` | ``"error"``.
    max_in_flight / max_open_tiles / track_index_map:
        Writer tuning — see ``ingest_spectra_parallel``.
    on_length_mismatch:
        ``pad`` (default), ``truncate``, or ``error`` when pixel count differs
        across plates or from an existing tile (same as sequential spPlate ingest).

    Returns
    -------
    dict
        Same keys as :func:`~data_lake.ingest.desi_parallel_ingest.ingest_spectra_parallel`.
    """
    _n_modes = sum([triplet_hash, lookup_path is not None, lookup_from_plate])
    if _n_modes > 1:
        raise ValueError(
            "Pass only one of triplet_hash=True, lookup_path=, or lookup_from_plate=True."
        )
    if _n_modes == 0:
        raise ValueError(
            "No ID lookup mode: set triplet_hash=True (default), provide lookup_path=, "
            "or set lookup_from_plate=True."
        )

    config = SpplateBatchConfig(
        norder=norder,
        ra_col=ra_col,
        dec_col=dec_col,
        triplet_hash=triplet_hash,
        lookup_path=str(lookup_path) if lookup_path else None,
        lookup_survey=lookup_survey,
        lookup_from_plate=lookup_from_plate,
        specobj_id_layout=specobj_id_layout,
    )

    return ingest_spectra_parallel(
        file_paths=file_paths,
        output_root=output_root,
        survey_name=survey_name,
        n_workers=n_workers,
        norder=norder,
        checkpoint_path=checkpoint_path,
        failures_log=failures_log,
        show_progress=show_progress,
        decoder=_SpplateDecoder(config),
        executor_factory=executor_factory,
        worker_log_file=worker_log_file,
        worker_verbose=worker_verbose,
        on_duplicate_source_id=on_duplicate_source_id,
        max_in_flight=max_in_flight,
        max_open_tiles=max_open_tiles,
        track_index_map=track_index_map,
        length_policy=on_length_mismatch,
        wavelength_mode="shared",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


try:
    import click

    from data_lake.cli_utils import (
        config_option,
        configure_warning_filters,
        ingest_token_option,
        load_optional_config,
        pick,
        require_ingest_permission,
        require_output_root,
    )
    from data_lake.ingest.checkpoint_sidecars import paths_from_file_list_file as _paths_from_file_list_file
    from data_lake.ingest.desi_parallel_ingest import _configure_file_logging

    @click.command("dl-ingest-spectra-batch-spplate")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @ingest_token_option
    @click.option("--survey", "survey_name", required=True, help="Short survey name.")
    @click.option(
        "--file-list",
        "file_list",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help=(
            "Plain-text file with one spPlate FITS path per line "
            "(mutually exclusive with --spplate-root)."
        ),
    )
    @click.option(
        "--spplate-root",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=None,
        help="Directory to recursively search for spPlate files "
             "(mutually exclusive with --file-list).",
    )
    @click.option(
        "--spplate-glob",
        default="spPlate-*.fits",
        show_default=True,
        help="Glob pattern under --spplate-root.",
    )
    @click.option(
        "--n-workers",
        required=True,
        type=int,
        help="Number of decoder worker processes (typically cpu_count - 1).",
    )
    @click.option(
        "--max-in-flight",
        "max_in_flight",
        default=None,
        type=int,
        help="Max plate decodes queued ahead of the writer (default: n_workers + 2).",
    )
    @click.option(
        "--max-open-tiles",
        "max_open_tiles",
        default=64,
        show_default=True,
        type=int,
        help="Max HEALPix tile Zarr groups open in the writer; 0 = unlimited.",
    )
    @click.option(
        "--norder",
        default=None,
        type=int,
        help="HEALPix order (overrides config; default 5).",
    )
    @click.option(
        "--ra-col",
        default="RA",
        show_default=True,
        help="Plugmap column for RA (degrees).",
    )
    @click.option(
        "--dec-col",
        default="DEC",
        show_default=True,
        help="Plugmap column for Dec (degrees).",
    )
    @click.option(
        "--specobj-lookup",
        "lookup_path",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help=(
            "Sidecar Parquet/CSV with PLATE, MJD, FIBERID, and a source ID column "
            "(replaces the default triplet hash)."
        ),
    )
    @click.option(
        "--specobj-lookup-survey",
        "lookup_survey",
        default=None,
        help="Survey filter for the sidecar (when the sidecar has a SURVEY column).",
    )
    @click.option(
        "--specobj-lookup-from-plate",
        "lookup_from_plate",
        is_flag=True,
        default=False,
        help="Synthesize CAS specObjID from header PLATE/MJD/FIBERID/RUN2D "
             "(replaces the default triplet hash).",
    )
    @click.option(
        "--specobj-id-layout",
        type=click.Choice(["auto", "dr7", "dr8plus"], case_sensitive=False),
        default="auto",
        show_default=True,
        help="specObjID layout for --specobj-lookup-from-plate.",
    )
    @click.option(
        "--on-length-mismatch",
        type=click.Choice(["error", "pad", "truncate"]),
        default="pad",
        show_default=True,
        help="When pixel count differs across plates or from a tile: pad (default), "
             "truncate, or error.",
    )
    @click.option(
        "--on-duplicate",
        type=click.Choice(["append", "error", "skip"]),
        default="skip",
        show_default=True,
        help="If source_id already exists in a tile Zarr: skip (default), raise, or append.",
    )
    @click.option(
        "--checkpoint",
        "checkpoint_path",
        type=click.Path(path_type=Path),
        default=None,
        help=(
            "JSON checkpoint of completed files.  "
            "Default: <output>/spectra/<survey>/.ingest_checkpoint.json"
        ),
    )
    @click.option(
        "--failures-log",
        "failures_log",
        type=click.Path(path_type=Path),
        default=None,
        help=(
            "JSONL log for per-file failures.  "
            "Default: <output>/spectra/<survey>/.ingest_failures.jsonl"
        ),
    )
    @click.option(
        "--log-file",
        "log_file_path",
        type=click.Path(path_type=Path),
        default=None,
        help=(
            "File to receive INFO/DEBUG logs (terminal stays clean for the "
            "progress bar).  "
            "Default: <output>/spectra/<survey>/.ingest.log"
        ),
    )
    @click.option(
        "--update-catalog/--no-update-catalog",
        default=True,
        show_default=True,
        help="Patch _spectrum_index in the Parquet catalog after ingest.",
    )
    @click.option(
        "-v",
        "--verbose",
        is_flag=True,
        help="Use DEBUG level in the log file (no terminal effect).",
    )
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        ingest_token: str | None,
        survey_name: str,
        file_list: Path | None,
        spplate_root: Path | None,
        spplate_glob: str,
        n_workers: int,
        max_in_flight: int | None,
        max_open_tiles: int,
        norder: int | None,
        ra_col: str,
        dec_col: str,
        lookup_path: Path | None,
        lookup_survey: str | None,
        lookup_from_plate: bool,
        specobj_id_layout: str,
        on_length_mismatch: str,
        on_duplicate: str,
        checkpoint_path: Path | None,
        failures_log: Path | None,
        log_file_path: Path | None,
        update_catalog: bool,
        verbose: bool,
    ) -> None:
        """Parallel vectorized ingest of SDSS spPlate FITS files.

        Decode is vectorized (no per-fiber Python loop); all fibers in each
        plate file are processed with NumPy array operations.  A single
        main-thread writer appends to HEALPix Zarr tiles (same architecture
        as dl-ingest-spectra-batch-desi-coadds).

        Default ID mode: composite PLATE|MJD|FIBERID hash (catalog must use
        --link-id-col PLATE,MJD,FIBERID).  For native CAS specObjIDs use
        --specobj-lookup or --specobj-lookup-from-plate.

        OUTPUT_ROOT is optional when a lake config is available (via --config
        or $DATA_LAKE_CONFIG).
        """
        if (file_list is None) == (spplate_root is None):
            raise click.UsageError("Pass exactly one of --file-list or --spplate-root.")
        n_lookup_modes = sum([
            lookup_path is not None,
            lookup_from_plate,
        ])
        if n_lookup_modes > 1:
            raise click.UsageError(
                "Pass only one of --specobj-lookup or --specobj-lookup-from-plate."
            )
        if n_workers < 1:
            raise click.UsageError("--n-workers must be >= 1")
        if max_in_flight is not None and max_in_flight < 1:
            raise click.UsageError("--max-in-flight must be >= 1")
        if max_open_tiles < 0:
            raise click.UsageError("--max-open-tiles must be >= 0")

        cfg = load_optional_config(config_path)
        require_ingest_permission(cfg, ingest_token)
        resolved_output = require_output_root(output_root, cfg, kind="spectra")
        resolved_norder = pick(norder, cfg.partitioning.hats_order if cfg else None, 5)

        if file_list is not None:
            paths: list[Path] = _paths_from_file_list_file(file_list)
        else:
            assert spplate_root is not None
            paths = sorted(spplate_root.rglob(spplate_glob))
        if not paths:
            raise click.UsageError(
                "No FITS files found.  Check --file-list or --spplate-root/--spplate-glob."
            )

        survey_root = resolved_output / "spectra" / survey_name
        if checkpoint_path is None:
            checkpoint_path = survey_root / ".ingest_checkpoint.json"
        if failures_log is None:
            failures_log = survey_root / ".ingest_failures.jsonl"
        if log_file_path is None:
            log_file_path = survey_root / ".ingest.log"

        _configure_file_logging(log_file_path, verbose=verbose)
        configure_warning_filters()

        id_mode = (
            "triplet hash PLATE|MJD|FIBERID"
            if n_lookup_modes == 0
            else ("sidecar lookup" if lookup_path is not None else "plate synthesis")
        )
        click.echo(
            f"Ingesting {len(paths)} spPlate file(s) into survey={survey_name!r} "
            f"with {n_workers} workers.  ID mode: {id_mode}.  "
            f"Logs → {log_file_path}"
        )

        result = ingest_spplate_files_parallel(
            file_paths=paths,
            output_root=resolved_output,
            survey_name=survey_name,
            n_workers=n_workers,
            norder=resolved_norder,
            ra_col=ra_col,
            dec_col=dec_col,
            triplet_hash=(n_lookup_modes == 0),
            lookup_path=lookup_path,
            lookup_survey=lookup_survey,
            lookup_from_plate=lookup_from_plate,
            specobj_id_layout=specobj_id_layout,
            checkpoint_path=checkpoint_path,
            failures_log=failures_log,
            worker_log_file=log_file_path,
            worker_verbose=verbose,
            on_duplicate_source_id=on_duplicate,  # type: ignore[arg-type]
            on_length_mismatch=on_length_mismatch,
            max_in_flight=max_in_flight,
            max_open_tiles=max_open_tiles,
        )

        click.echo(
            f"\nDone: {result['n_files_succeeded']}/{result['n_files_processed']} files "
            f"succeeded ({result['n_files_failed']} failed, "
            f"{result['n_files_skipped']} skipped from checkpoint).  "
            f"Wrote {result['n_spectra']} spectra to {result['n_tiles']} tile(s) "
            f"in {result['elapsed_s']:.1f}s."
        )
        if result["n_files_failed"]:
            click.echo(f"See failures log: {failures_log}")

        if update_catalog:
            try:
                from data_lake.ingest.update_catalog_indices import (
                    update_index_column_from_zarr_tiles,
                )
                n_modified = update_index_column_from_zarr_tiles(
                    lake_root=resolved_output,
                    survey_name=survey_name,
                    kind="spectrum",
                    norder=resolved_norder,
                )
                click.echo(f"Patched _spectrum_index in {n_modified} catalog tile(s).")
            except FileNotFoundError as exc:
                log.info("Skipping _spectrum_index patch (%s).", exc)

except ImportError:
    cli = None  # type: ignore[assignment]
