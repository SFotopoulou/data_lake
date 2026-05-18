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

import argparse
import json
import random
import sys
from pathlib import Path

# Allow running without install
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_lake.ingest.desi_coadd_characterize import (  # noqa: E402
    CoaddExposureReport,
    characterize_coadd_paths,
)


def _collect_paths(args: argparse.Namespace) -> list[Path]:
    if args.file_list is not None:
        root = args.file_list.parent
        lines = [
            ln.strip()
            for ln in args.file_list.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        paths = [(root / ln if not Path(ln).is_absolute() else Path(ln)) for ln in lines]
    elif args.coadd_root is not None:
        paths = sorted(args.coadd_root.rglob(args.coadd_glob))
    else:
        paths = [Path(p) for p in args.paths]
    paths = [p.expanduser().resolve() for p in paths if p.expanduser().exists()]
    if args.sample is not None and len(paths) > args.sample:
        rng = random.Random(args.seed)
        paths = sorted(rng.sample(paths, args.sample))
    if args.max_files is not None:
        paths = paths[: args.max_files]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Coadd FITS paths")
    parser.add_argument("--file-list", type=Path, help="One path per line")
    parser.add_argument("--coadd-root", type=Path, help="Search root for coadd files")
    parser.add_argument("--coadd-glob", default="coadd-*.fits")
    parser.add_argument(
        "--mode",
        choices=("quick", "desispec"),
        default="quick",
        help="quick=FIBERMAP only; desispec=read_spectra like ingest",
    )
    parser.add_argument("--sample", type=int, help="Random subsample of paths")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-files", type=int, help="Cap number of files processed")
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Write per-file reports + summary as JSON",
    )
    parser.add_argument(
        "--only-needs-coadd",
        action="store_true",
        help="Print only files where max rows per TARGETID > 1",
    )
    args = parser.parse_args()

    paths = _collect_paths(args)
    if not paths:
        print("No input files found.", file=sys.stderr)
        return 1

    reports, summary = characterize_coadd_paths(paths, mode=args.mode)

    print(f"Mode: {args.mode}  Files: {summary.n_files}")
    print(
        f"  single row/target: {summary.n_files_single_row_per_target}  "
        f"needs exposure coadd: {summary.n_files_needing_exposure_coadd}  "
        f"max rows/target (any file): {summary.max_rows_per_targetid_overall}"
    )
    print(f"  pooled histogram (rows per TARGETID -> #targets): {summary.pooled_histogram}")

    for r in reports:
        if args.only_needs_coadd and not r.needs_exposure_coadd:
            continue
        print(_report_line(r))

    if summary.files_needing_coadd:
        print("\nExample files needing exposure coadd (up to 50):")
        for p in summary.files_needing_coadd:
            print(f"  {p}")

    if args.json_out:
        payload = {
            "mode": args.mode,
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
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json_out}")

    # Exit 2 if any file needs exposure coadd (useful in shell scripts).
    return 2 if summary.n_files_needing_exposure_coadd else 0


if __name__ == "__main__":
    raise SystemExit(main())
