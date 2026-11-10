#!/usr/bin/env python3
"""
Phase 0: characterize DESI coadd FITS for exposure multiplicity.

Quick scan (FIBERMAP only — suitable for 10k+ files)::

    python scripts/characterize_desi_coadds.py --file-list coadds.txt

Match ingest's read_spectra row set on a sample::

    python scripts/characterize_desi_coadds.py --file-list coadds.txt \\
        --mode desispec --max-files 20

Recursive directory::

    python scripts/characterize_desi_coadds.py --coadd-root /data/DESI/fits \\
        --coadd-glob 'coadd-*.fits' --sample 200
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import click

# Allow running without install
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_lake.ingest.desi_coadd_characterize import (  # noqa: E402
    CoaddExposureReport,
    characterize_coadd_paths,
)


def _collect_paths(
    *,
    paths_arg: tuple[Path, ...],
    file_list: Path | None,
    coadd_root: Path | None,
    coadd_glob: str,
    sample: int | None,
    seed: int,
    max_files: int | None,
) -> list[Path]:
    if file_list is not None:
        root = file_list.parent
        lines = [
            ln.strip()
            for ln in file_list.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        paths = [(root / ln if not Path(ln).is_absolute() else Path(ln)) for ln in lines]
    elif coadd_root is not None:
        paths = sorted(coadd_root.rglob(coadd_glob))
    else:
        paths = [Path(p) for p in paths_arg]
    paths = [p.expanduser().resolve() for p in paths if p.expanduser().exists()]
    if sample is not None and len(paths) > sample:
        rng = random.Random(seed)
        paths = sorted(rng.sample(paths, sample))
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def _report_line(r: CoaddExposureReport) -> str:
    flag = "NEEDS_EXPCOADD" if r.needs_exposure_coadd else "ok"
    exp = (
        f" exp_fibermap={r.n_exp_fibermap_rows}"
        if r.n_exp_fibermap_rows is not None
        else ""
    )
    return (
        f"{flag:14} rows={r.n_fibermap_rows:5} unique={r.n_unique_targetid:5} "
        f"max_per_target={r.max_rows_per_targetid}{exp}  {r.path}"
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]}, help=__doc__)
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
@click.option("--file-list", type=click.Path(path_type=Path), help="One path per line")
@click.option("--coadd-root", type=click.Path(path_type=Path), help="Search root for coadd files")
@click.option("--coadd-glob", default="coadd-*.fits", show_default=True)
@click.option(
    "--mode",
    type=click.Choice(("quick", "desispec")),
    default="quick",
    show_default=True,
    help="quick=FIBERMAP only; desispec=read_spectra like ingest",
)
@click.option("--sample", type=int, help="Random subsample of paths")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--max-files", type=int, help="Cap number of files processed")
@click.option("--json-out", type=click.Path(path_type=Path), help="Write per-file reports + summary as JSON")
@click.option(
    "--only-needs-coadd",
    is_flag=True,
    help="Print only files where max rows per TARGETID > 1",
)
def main(
    paths: tuple[Path, ...],
    file_list: Path | None,
    coadd_root: Path | None,
    coadd_glob: str,
    mode: str,
    sample: int | None,
    seed: int,
    max_files: int | None,
    json_out: Path | None,
    only_needs_coadd: bool,
) -> int:
    collected = _collect_paths(
        paths_arg=paths,
        file_list=file_list,
        coadd_root=coadd_root,
        coadd_glob=coadd_glob,
        sample=sample,
        seed=seed,
        max_files=max_files,
    )
    if not collected:
        print("No input files found.", file=sys.stderr)
        return 1

    reports, summary = characterize_coadd_paths(collected, mode=mode)

    print(f"Mode: {mode}  Files: {summary.n_files}")
    print(
        f"  single row/target: {summary.n_files_single_row_per_target}  "
        f"needs exposure coadd: {summary.n_files_needing_exposure_coadd}  "
        f"max rows/target (any file): {summary.max_rows_per_targetid_overall}"
    )
    print(f"  pooled histogram (rows per TARGETID -> #targets): {summary.pooled_histogram}")

    for r in reports:
        if only_needs_coadd and not r.needs_exposure_coadd:
            continue
        print(_report_line(r))

    if summary.files_needing_coadd:
        print("\nExample files needing exposure coadd (up to 50):")
        for p in summary.files_needing_coadd:
            print(f"  {p}")

    if json_out is not None:
        payload = {
            "mode": mode,
            "summary": {
                "n_files": summary.n_files,
                "n_files_needing_exposure_coadd": summary.n_files_needing_exposure_coadd,
                "n_files_single_row_per_target": summary.n_files_single_row_per_target,
                "max_rows_per_targetid_overall": summary.max_rows_per_targetid_overall,
                "pooled_histogram": summary.pooled_histogram,
            },
            "reports": [
                {
                    "path": r.path,
                    "n_fibermap_rows": r.n_fibermap_rows,
                    "n_unique_targetid": r.n_unique_targetid,
                    "max_rows_per_targetid": r.max_rows_per_targetid,
                    "n_targets_multi_row": r.n_targets_multi_row,
                    "needs_exposure_coadd": r.needs_exposure_coadd,
                    "n_exp_fibermap_rows": r.n_exp_fibermap_rows,
                    "rows_per_targetid_histogram": r.rows_per_targetid_histogram,
                }
                for r in reports
            ],
        }
        json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {json_out}")

    # Exit 2 if any file needs exposure coadd (useful in shell scripts).
    return 2 if summary.n_files_needing_exposure_coadd else 0


if __name__ == "__main__":
    raise SystemExit(main())
