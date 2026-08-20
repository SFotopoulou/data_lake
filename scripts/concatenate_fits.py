#!/usr/bin/env python3
"""Concatenate row-normal catalog FITS tables from a folder into one file.

Designed for surveys shipped as many small BINTABLE shards (e.g. Legacy bricks,
Gaia run folders) before ``dl-ingest-catalog`` / parallel batch ingest.

Packed-vector FITS (``NAXIS2=1`` with long vector columns, e.g. STILTS colfits)
are rejected — convert those to row-normal FITS or Parquet first.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from astropy.io import fits
from astropy.table import Table, vstack
from astropy.utils.exceptions import AstropyWarning
from tqdm import tqdm

# Quiet noisy FITS header / unit warnings during bulk shard reads.
warnings.filterwarnings("ignore", category=AstropyWarning)

from data_lake.ingest.fits_to_parquet import (
    _bintable_hdu_index,
    _is_packed_vector_bintable,
    is_catalog_fits_path,
)
from data_lake.io.fits_read import FitsReadPolicy, resolve_memmap


def _collect_fits_paths(
    input_dir: Path,
    *,
    pattern: str,
    recursive: bool,
) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    globber = input_dir.rglob if recursive else input_dir.glob
    paths = sorted(p for p in globber(pattern) if p.is_file() and is_catalog_fits_path(p))
    if not paths:
        raise FileNotFoundError(
            f"No catalog FITS files matching {pattern!r} under {input_dir}"
        )
    return paths


def _read_table_hdu(path: Path, *, hdu: int | str | None, memmap: str) -> tuple[Table, int]:
    policy = FitsReadPolicy(memmap=memmap)  # type: ignore[arg-type]
    with fits.open(
        path,
        memmap=resolve_memmap(path, policy),
        lazy_load_hdus=True,
    ) as hdul:
        idx = _bintable_hdu_index(hdul) if hdu is None else hdul.index_of(hdu)
        hdu_obj = hdul[idx]
        if _is_packed_vector_bintable(hdu_obj):
            raise ValueError(
                f"{path}: packed-vector BINTABLE (NAXIS2=1); "
                "re-export as row-normal FITS before concatenation."
            )
        table = Table.read(hdul, hdu=idx, format="fits", memmap=resolve_memmap(path, policy))
    return table, idx


def concatenate_fits_tables(
    paths: list[Path],
    *,
    hdu: int | str | None = None,
    join_type: str = "exact",
    memmap: str = "auto",
    show_progress: bool = True,
) -> Table:
    if not paths:
        raise ValueError("No input files to concatenate.")

    tables: list[Table] = []
    ref_names: tuple[str, ...] | None = None
    ref_hdu: int | None = None

    iterator = tqdm(paths, desc="read FITS", unit="file", disable=not show_progress)
    for path in iterator:
        table, hdu_idx = _read_table_hdu(path, hdu=hdu, memmap=memmap)
        if ref_hdu is None:
            ref_hdu = hdu_idx
        elif hdu_idx != ref_hdu:
            raise ValueError(
                f"{path}: table HDU index {hdu_idx} differs from first file ({ref_hdu}). "
                "Pass --hdu to select a consistent extension."
            )

        names = tuple(table.colnames)
        if ref_names is None:
            ref_names = names
        elif names != ref_names:
            raise ValueError(
                f"{path}: column names differ from the first file.\n"
                f"  first: {list(ref_names)}\n"
                f"  this:  {list(names)}"
            )
        tables.append(table)

    return vstack(tables, join_type=join_type)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing FITS table shards to concatenate.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output FITS path.",
    )
    parser.add_argument(
        "--pattern",
        default="*.fits",
        help="Glob pattern for input files relative to input_dir (default: %(default)s).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively.",
    )
    parser.add_argument(
        "--hdu",
        default=None,
        help="Table HDU index or name (default: first BINTABLE in each file).",
    )
    parser.add_argument(
        "--join",
        dest="join_type",
        choices=("exact", "outer"),
        default="exact",
        help="How to align columns when stacking tables (default: %(default)s).",
    )
    parser.add_argument(
        "--fits-memmap",
        choices=("auto", "on", "off"),
        default="auto",
        help="FITS read policy, same semantics as dl-ingest-catalog (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output file if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List input files and exit without writing output.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    args = parser.parse_args()

    hdu: int | str | None
    if args.hdu is None:
        hdu = None
    else:
        try:
            hdu = int(args.hdu)
        except ValueError:
            hdu = args.hdu

    try:
        paths = _collect_fits_paths(
            args.input_dir,
            pattern=args.pattern,
            recursive=args.recursive,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Found {len(paths)} FITS file(s) under {args.input_dir}", file=sys.stderr)
    if args.dry_run:
        for path in paths:
            print(path)
        return 0

    if args.output.exists() and not args.overwrite:
        print(f"Refusing to overwrite existing file: {args.output}", file=sys.stderr)
        print("Pass --overwrite to replace it.", file=sys.stderr)
        return 1

    try:
        merged = concatenate_fits_tables(
            paths,
            hdu=hdu,
            join_type=args.join_type,
            memmap=args.fits_memmap,
            show_progress=not args.no_progress,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.write(str(args.output), format="fits", overwrite=True)
    print(
        f"Wrote {len(merged)} row(s) from {len(paths)} file(s) to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
