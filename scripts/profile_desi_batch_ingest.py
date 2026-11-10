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

import cProfile
import os
import pstats
import sys
import tempfile
from pathlib import Path

import click


def _pin_blas_threads() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _print_stats(prof_path: Path, lines: int, sort: str = "cumulative") -> None:
    stats = pstats.Stats(str(prof_path))
    stats.strip_dirs().sort_stats(sort)
    stats.print_stats(lines)


def _cmd_worker(
    fits_paths: tuple[str, ...],
    file_list: Path | None,
    count: int | None,
    norder: int,
    output: str,
    lines: int,
    sort: str,
) -> int:
    from data_lake.ingest.desi_parallel_ingest import _decode_one_coadd

    paths: list[str] = []
    if file_list:
        paths = [
            ln.strip()
            for ln in file_list.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    for p in fits_paths:
        paths.append(str(Path(p).resolve()))
    if count is not None:
        paths = paths[:count]
    if not paths:
        print("No FITS paths given.", file=sys.stderr)
        return 2

    prof_path = Path(output)

    def _run() -> None:
        for path_str in paths:
            res = _decode_one_coadd(path_str, norder)
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
    _print_stats(prof_path, lines, sort)
    return 0


def _cmd_batch(
    config: Path | None,
    output_root: Path | None,
    survey: str,
    file_list: Path,
    count: int | None,
    n_workers: int,
    max_in_flight: int | None,
    max_open_tiles: int,
    norder: int,
    profile_output: str,
    lines: int,
    sort: str,
) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from data_lake.ingest.checkpoint_sidecars import paths_from_file_list_file
    from data_lake.ingest.desi_parallel_ingest import (
        _decode_one_coadd_safe,
        ingest_spectra_parallel,
    )

    if config:
        os.environ["DATA_LAKE_CONFIG"] = str(config.resolve())
    from data_lake.cli_utils import load_optional_config, require_output_root

    cfg = load_optional_config(config)
    out = require_output_root(
        output_root,
        cfg,
        kind="spectra",
    )
    paths = paths_from_file_list_file(file_list)
    if count is not None:
        paths = paths[:count]
    if not paths:
        print("Empty file list.", file=sys.stderr)
        return 2

    prof_path = Path(profile_output)
    tmp = Path(tempfile.mkdtemp(prefix="dl_profile_"))
    survey_root = tmp / "spectra" / survey

    def _run() -> None:
        ingest_spectra_parallel(
            file_paths=paths,
            output_root=tmp,
            survey_name=survey,
            n_workers=n_workers,
            norder=norder,
            checkpoint_path=survey_root / ".ingest_checkpoint.json",
            failures_log=survey_root / ".ingest_failures.jsonl",
            show_progress=False,
            decoder=_decode_one_coadd_safe,
            executor_factory=lambda nw: ThreadPoolExecutor(max_workers=nw),
            max_in_flight=max_in_flight,
            max_open_tiles=max_open_tiles,
            on_duplicate_source_id="append",
        )

    prof = cProfile.Profile()
    prof.enable()
    _run()
    prof.disable()
    prof.dump_stats(str(prof_path))
    print(f"Wrote {prof_path} (temp lake data under {tmp})")
    _print_stats(prof_path, lines, sort)
    _ = out
    return 0


def _cmd_report(profile: Path, lines: int, sort: str) -> int:
    _print_stats(profile, lines, sort)
    return 0


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, help=__doc__)
def cli() -> None:
    _pin_blas_threads()


@cli.command("worker", help="Profile _decode_one_coadd (desispec read + coadd + numpy pack)")
@click.argument("fits_paths", nargs=-1)
@click.option(
    "--file-list",
    type=click.Path(path_type=Path),
    help="Text file with one FITS path per line (used if no fits_paths)",
)
@click.option("--count", type=int, default=None, help="Max files to run")
@click.option("--norder", type=int, default=5, show_default=True)
@click.option("-o", "--output", default="worker.prof", show_default=True)
@click.option("--lines", type=int, default=30, show_default=True)
@click.option(
    "--sort",
    type=click.Choice(("cumulative", "time", "calls")),
    default="cumulative",
    show_default=True,
)
def worker_cmd(
    fits_paths: tuple[str, ...],
    file_list: Path | None,
    count: int | None,
    norder: int,
    output: str,
    lines: int,
    sort: str,
) -> int:
    return _cmd_worker(fits_paths, file_list, count, norder, output, lines, sort)


@cli.command(
    "batch",
    help="Profile parent writer path (ThreadPool decoders, real Zarr writes)",
)
@click.option("--config", type=click.Path(path_type=Path), default=None)
@click.option("--output-root", type=click.Path(path_type=Path), default=None)
@click.option("--survey", required=True)
@click.option("--file-list", type=click.Path(path_type=Path), required=True)
@click.option("--count", type=int, default=2, show_default=True)
@click.option("--n-workers", type=int, default=2, show_default=True)
@click.option("--max-in-flight", type=int, default=None)
@click.option("--max-open-tiles", type=int, default=64, show_default=True)
@click.option("--norder", type=int, default=5, show_default=True)
@click.option("-o", "--profile-output", default="batch.prof", show_default=True)
@click.option("--lines", type=int, default=30, show_default=True)
@click.option(
    "--sort",
    type=click.Choice(("cumulative", "time", "calls")),
    default="cumulative",
    show_default=True,
)
def batch_cmd(
    config: Path | None,
    output_root: Path | None,
    survey: str,
    file_list: Path,
    count: int | None,
    n_workers: int,
    max_in_flight: int | None,
    max_open_tiles: int,
    norder: int,
    profile_output: str,
    lines: int,
    sort: str,
) -> int:
    return _cmd_batch(
        config,
        output_root,
        survey,
        file_list,
        count,
        n_workers,
        max_in_flight,
        max_open_tiles,
        norder,
        profile_output,
        lines,
        sort,
    )


@cli.command("report", help="Print pstats summary from a .prof file")
@click.argument("profile", type=click.Path(path_type=Path))
@click.option("--lines", type=int, default=40, show_default=True)
@click.option(
    "--sort",
    type=click.Choice(("cumulative", "time", "calls")),
    default="cumulative",
    show_default=True,
)
def report_cmd(profile: Path, lines: int, sort: str) -> int:
    return _cmd_report(profile, lines, sort)


def main() -> int:
    return cli.main(standalone_mode=False)  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())
