#!/usr/bin/env python3
"""
Profile DESI parallel spectra ingest bottlenecks.

The batch CLI uses a process pool (workers) plus a single-thread Zarr writer in
the parent. Profile those paths separately — one cProfile of the parent mostly
shows ``wait()`` and pickle IPC, not ``read_spectra`` inside workers.

Usage
-----
Worker path (read + coadd + HEALPix pack for one coadd)::

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python scripts/profile_desi_batch_ingest.py worker /data/DESI/fits/coadd-main-dark-10000.fits

Several files (serial, to compare per-file cost)::

    python scripts/profile_desi_batch_ingest.py worker --count 5 /path/to/list.txt

Parent + writer on a tiny parallel run (2 files, thread pool = same writer code)::

    python scripts/profile_desi_batch_ingest.py batch \\
        --config $DATA_LAKE_CONFIG --survey DESI_DR1 \\
        --file-list /path/to/two_coadds.txt

Inspect a saved profile::

    python scripts/profile_desi_batch_ingest.py report worker.prof --lines 40

Live job (install py-spy: pip install py-spy)::

    PARENT=$(pgrep -f 'dl-ingest-spectra-batch-desi-coadds' | head -1)
    py-spy top --pid "$PARENT" --subprocesses
    py-spy record -o ingest.svg --pid "$PARENT" --subprocesses --duration 120

Pair with disk metrics::

    iostat -xz 5 /dev/sdc /dev/nvme2n1
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import tempfile
from pathlib import Path


def _pin_blas_threads() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _cmd_worker(args: argparse.Namespace) -> int:
    from data_lake.ingest.desi_parallel_ingest import _decode_one_coadd

    paths: list[str] = []
    if args.file_list:
        paths = [
            ln.strip()
            for ln in Path(args.file_list).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    for p in args.fits_paths:
        paths.append(str(Path(p).resolve()))
    if args.count is not None:
        paths = paths[: args.count]
    if not paths:
        print("No FITS paths given.", file=sys.stderr)
        return 2

    prof_path = Path(args.output)

    def _run() -> None:
        for path_str in paths:
            res = _decode_one_coadd(path_str, args.norder)
            if not res.ok:
                print(f"FAIL {path_str}: {res.error}", file=sys.stderr)
            else:
                print(
                    f"OK {Path(path_str).name}: {res.n_spectra} spectra, "
                    f"{len(res.batches)} tile batch(es), {res.elapsed_s:.2f}s"
                )

    prof = cProfile.Profile()
    prof.enable()
    _run()
    prof.disable()
    prof.dump_stats(str(prof_path))
    print(f"Wrote {prof_path}")
    _print_stats(prof_path, args.lines, args.sort)
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from data_lake.ingest.checkpoint_sidecars import paths_from_file_list_file
    from data_lake.ingest.desi_parallel_ingest import (
        _decode_one_coadd_safe,
        ingest_spectra_parallel,
    )

    if args.config:
        os.environ["DATA_LAKE_CONFIG"] = str(Path(args.config).resolve())
    from data_lake.cli_utils import load_optional_config, require_output_root

    cfg = load_optional_config(Path(args.config) if args.config else None)
    out = require_output_root(
        Path(args.output_root) if args.output_root else None,
        cfg,
        kind="spectra",
    )
    paths = paths_from_file_list_file(Path(args.file_list))
    if args.count is not None:
        paths = paths[: args.count]
    if not paths:
        print("Empty file list.", file=sys.stderr)
        return 2

    survey = args.survey
    prof_path = Path(args.profile_output)
    tmp = Path(tempfile.mkdtemp(prefix="dl_profile_"))
    survey_root = tmp / "spectra" / survey

    def _run() -> None:
        ingest_spectra_parallel(
            file_paths=paths,
            output_root=tmp,
            survey_name=survey,
            n_workers=args.n_workers,
            norder=args.norder,
            checkpoint_path=survey_root / ".ingest_checkpoint.json",
            failures_log=survey_root / ".ingest_failures.jsonl",
            show_progress=False,
            decoder=_decode_one_coadd_safe,
            executor_factory=lambda nw: ThreadPoolExecutor(max_workers=nw),
            max_in_flight=args.max_in_flight,
            max_open_tiles=args.max_open_tiles,
            on_duplicate_source_id="append",
        )

    prof = cProfile.Profile()
    prof.enable()
    _run()
    prof.disable()
    prof.dump_stats(str(prof_path))
    print(f"Wrote {prof_path} (temp lake data under {tmp})")
    _print_stats(prof_path, args.lines, args.sort)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    _print_stats(Path(args.profile), args.lines, args.sort)
    return 0


def _print_stats(prof_path: Path, lines: int, sort: str = "cumulative") -> None:
    stats = pstats.Stats(str(prof_path))
    stats.strip_dirs().sort_stats(sort)
    stats.print_stats(lines)


def main() -> int:
    _pin_blas_threads()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_worker = sub.add_parser(
        "worker",
        help="Profile _decode_one_coadd (desispec read + coadd + numpy pack)",
    )
    p_worker.add_argument(
        "fits_paths",
        nargs="*",
        help="One or more coadd FITS paths",
    )
    p_worker.add_argument(
        "--file-list",
        type=Path,
        help="Text file with one FITS path per line (used if no fits_paths)",
    )
    p_worker.add_argument("--count", type=int, default=None, help="Max files to run")
    p_worker.add_argument("--norder", type=int, default=5)
    p_worker.add_argument("-o", "--output", default="worker.prof")
    p_worker.add_argument("--lines", type=int, default=30)
    p_worker.add_argument(
        "--sort",
        choices=("cumulative", "time", "calls"),
        default="cumulative",
    )
    p_worker.set_defaults(func=_cmd_worker)

    p_batch = sub.add_parser(
        "batch",
        help="Profile parent writer path (ThreadPool decoders, real Zarr writes)",
    )
    p_batch.add_argument("--config", type=Path, default=None)
    p_batch.add_argument("--output-root", type=Path, default=None)
    p_batch.add_argument("--survey", required=True)
    p_batch.add_argument("--file-list", type=Path, required=True)
    p_batch.add_argument("--count", type=int, default=2)
    p_batch.add_argument("--n-workers", type=int, default=2)
    p_batch.add_argument("--max-in-flight", type=int, default=None)
    p_batch.add_argument("--max-open-tiles", type=int, default=64)
    p_batch.add_argument("--norder", type=int, default=5)
    p_batch.add_argument("-o", "--profile-output", default="batch.prof")
    p_batch.add_argument("--lines", type=int, default=30)
    p_batch.add_argument(
        "--sort",
        choices=("cumulative", "time", "calls"),
        default="cumulative",
    )
    p_batch.set_defaults(func=_cmd_batch)

    p_report = sub.add_parser("report", help="Print pstats summary from a .prof file")
    p_report.add_argument("profile", type=Path)
    p_report.add_argument("--lines", type=int, default=40)
    p_report.add_argument(
        "--sort",
        choices=("cumulative", "time", "calls"),
        default="cumulative",
    )
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
