"""
Dry-run smoke test for the DESI ingest pipeline.

Ingests a small set of DESI coadd files into a temporary output directory,
once without --with-resolution and once with it, then reads back a handful
of sources and verifies invariants.  Logs timing per stage so you can
extrapolate to a full run.

Run it from anywhere - it uses temp directories that are cleaned up automatically.

    python scripts/dry_run_desi_ingest.py path/to/coadd1.fits path/to/coadd2.fits ...

Or with the defaults:

    python scripts/dry_run_desi_ingest.py
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

DEFAULT_FILES = [
    "/data/DESI/fits/coadd-cmx-other-2152.fits",
    "/data/DESI/fits/coadd-cmx-other-2153.fits",
    "/data/DESI/fits/coadd-cmx-other-2154.fits",
]

log = logging.getLogger("dry_run")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(name)-22s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n //= 1024
    return f"{n:.1f} PB"


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def run_one_pass(
    files: list[Path],
    output_root: Path,
    *,
    with_resolution: bool,
    survey: str,
) -> dict:
    """Run a single ingest pass and return stage timings + tile stats."""
    from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits

    log.info("=" * 70)
    log.info(
        "PASS: %s, %d files -> %s",
        "with --with-resolution" if with_resolution else "no resolution",
        len(files),
        output_root,
    )
    log.info("=" * 70)

    timings: list[tuple[str, float]] = []
    total_sources = 0
    total_index_map: dict[int, int] = {}

    for i, fits_path in enumerate(files):
        log.info("[file %d/%d] ingesting %s", i + 1, len(files), fits_path.name)
        t0 = time.perf_counter()
        index_map = ingest_spectra_from_fits(
            source_path=fits_path,
            output_root=output_root,
            survey_name=survey,
            with_resolution=with_resolution,
            wavelength_mode="shared",
        )
        elapsed = time.perf_counter() - t0
        n_new = len(index_map)
        total_index_map.update(index_map)
        total_sources += n_new
        timings.append((fits_path.name, elapsed))
        log.info(
            "[file %d/%d] done in %5.2fs  (%d sources, %5.2f sources/s)",
            i + 1, len(files), elapsed, n_new, n_new / max(elapsed, 1e-6),
        )

    total_time = sum(t for _, t in timings)
    on_disk = _dir_size(output_root)
    log.info(
        "PASS total: %5.2fs, %d sources, %s on disk (%5.2f kB/source)",
        total_time, total_sources, _human_bytes(on_disk),
        on_disk / max(total_sources, 1) / 1024,
    )

    return {
        "timings": timings,
        "total_time": total_time,
        "total_sources": total_sources,
        "on_disk_bytes": on_disk,
        "index_map": total_index_map,
    }


def verify_readback(output_root: Path, survey: str, index_map: dict, with_resolution: bool) -> None:
    """Read back a few sources and check invariants."""
    from data_lake.io.spectra import SpectrumAccessor

    log.info("VERIFY: reading back from %s", output_root)
    acc = SpectrumAccessor(output_root, survey)

    # Sample up to 5 source IDs
    sample_ids = list(index_map.keys())[: min(5, len(index_map))]
    log.info("Sampling %d source IDs: %s", len(sample_ids), sample_ids)

    for sid in sample_ids:
        spec = acc.get_spectrum(sid)
        n_pix = spec.flux.shape[0]
        log.info(
            "  TID %d:  N_pix=%d  flux range=[%6.2f, %6.2f]  good_pix=%d  z=%.4f",
            sid, n_pix, float(spec.flux.min()), float(spec.flux.max()),
            int(spec.good.sum()), float(spec.meta.get("z", 0.0)),
        )
        assert spec.wavelength.shape[0] == n_pix, "wavelength length mismatch"
        assert spec.ivar.shape[0] == n_pix, "ivar length mismatch"
        assert spec.mask.shape[0] == n_pix, "mask length mismatch"
        assert np.isfinite(spec.flux).any(), "flux is all NaN/Inf"

        if with_resolution:
            assert spec.resolution is not None, "resolution missing"
            assert spec.resolution_offsets is not None, "offsets missing"
            R = spec.resolution_operator()
            row_sums = np.asarray(R.sum(axis=1)).ravel()
            half = len(spec.resolution_offsets) // 2
            interior = row_sums[half:n_pix - half]
            mean_rs = float(interior.mean())
            log.info(
                "    resolution: n_diag=%d  interior row-sum mean=%.4f (target ~1.0)",
                R.shape[0] and spec.resolution.shape[0], mean_rs,
            )
            assert 0.98 < mean_rs < 1.02, f"interior row sums off: {mean_rs:.4f}"
        else:
            assert spec.resolution is None, "resolution unexpectedly present"

    log.info("VERIFY: all invariants OK for %d sampled sources", len(sample_ids))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="DESI coadd FITS files")
    parser.add_argument("--survey", default="desi_dryrun")
    parser.add_argument(
        "--keep-output",
        type=Path,
        default=None,
        help="If given, write to this directory instead of a temp dir (and keep it)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    file_paths = [Path(p) for p in (args.files or DEFAULT_FILES)]
    missing = [p for p in file_paths if not p.exists()]
    if missing:
        log.error("Missing input files: %s", missing)
        return 2

    # Quick desispec availability check up-front
    from data_lake.ingest.fits_to_spectra_zarr import _import_desispec
    desispec = _import_desispec()
    log.info("desispec %s available", desispec.__version__)

    if args.keep_output:
        args.keep_output.mkdir(parents=True, exist_ok=True)
        tmp_root = args.keep_output
        cleanup = False
    else:
        tmp_root = Path(tempfile.mkdtemp(prefix="data_lake_dryrun_"))
        cleanup = True

    log.info("Working in %s", tmp_root)

    try:
        # ------ Pass 1: no resolution matrix ------
        no_res_root = tmp_root / "no_resolution"
        pass1 = run_one_pass(file_paths, no_res_root, with_resolution=False, survey=args.survey)
        verify_readback(no_res_root, args.survey, pass1["index_map"], with_resolution=False)

        # ------ Pass 2: with resolution matrix ------
        res_root = tmp_root / "with_resolution"
        pass2 = run_one_pass(file_paths, res_root, with_resolution=True, survey=args.survey)
        verify_readback(res_root, args.survey, pass2["index_map"], with_resolution=True)

        # ------ Summary ------
        log.info("=" * 70)
        log.info("SUMMARY")
        log.info("=" * 70)
        log.info(
            "Without resolution: %5.2fs total, %s on disk (%5.2f kB/source)",
            pass1["total_time"], _human_bytes(pass1["on_disk_bytes"]),
            pass1["on_disk_bytes"] / max(pass1["total_sources"], 1) / 1024,
        )
        log.info(
            "With resolution:    %5.2fs total, %s on disk (%5.2f kB/source)",
            pass2["total_time"], _human_bytes(pass2["on_disk_bytes"]),
            pass2["on_disk_bytes"] / max(pass2["total_sources"], 1) / 1024,
        )
        ratio = pass2["on_disk_bytes"] / max(pass1["on_disk_bytes"], 1)
        log.info("Resolution-matrix storage overhead: %.1fx", ratio)
        log.info(
            "Time overhead from --with-resolution: %.1fx",
            pass2["total_time"] / max(pass1["total_time"], 1e-6),
        )

        # Extrapolation for 10k-file run
        n_files_target = 10_000
        avg_per_file_no_res = pass1["total_time"] / len(file_paths)
        avg_per_file_with_res = pass2["total_time"] / len(file_paths)
        log.info(
            "Extrapolation to %d files: ~%.1f h (no res) / ~%.1f h (with res), single-process",
            n_files_target,
            avg_per_file_no_res * n_files_target / 3600,
            avg_per_file_with_res * n_files_target / 3600,
        )

    finally:
        if cleanup:
            log.info("Cleaning up %s", tmp_root)
            shutil.rmtree(tmp_root, ignore_errors=True)
        else:
            log.info("Output kept at %s", tmp_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
