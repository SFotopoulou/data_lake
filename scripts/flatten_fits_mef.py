#!/usr/bin/env python3
"""Flatten row-aligned multi-extension FITS catalogs to Parquet.

Joins table HDUs horizontally with the same rule as ``dl-ingest-catalog-mef``:
shared columns from the first HDU, remaining columns appended. Typical
FastSpec layout is METADATA + SPECPHOT + FASTSPEC.

Not a FastSpec reader — any MEF whose table extensions share columns and the
same row count works. Vector columns (e.g. length-5 COEFF) are kept as
fixed-size lists.

Examples
--------
    python scripts/flatten_fits_mef.py data/spectroscopy/fastspec-iron-cmx-other.fits

    python scripts/flatten_fits_mef.py catalog.fits -o catalog.parquet

    python scripts/flatten_fits_mef.py --file-list files.txt --output-dir ./flat \\
        --hdu METADATA --hdu FASTSPEC --key-cols TARGETID,SURVEY,PROGRAM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

from data_lake.ingest.checkpoint_sidecars import paths_from_file_list_file
from data_lake.ingest.fits_to_parquet import (
    joined_fits_hdu_spec,
    read_joined_catalog_hdus,
)
from data_lake.io.fits_read import catalog_fits_read_policy


def parquet_path_for(source: Path, *, output: Path | None, output_dir: Path | None) -> Path:
    """Resolve the Parquet path for one input FITS file."""
    if output_dir is not None:
        return output_dir / f"{source.stem}.parquet"
    if output is not None:
        return output
    return source.with_suffix(".parquet")


def flatten_fits_mef(
    source: Path,
    dest: Path,
    *,
    extensions: list[str] | None = None,
    key_cols: list[str] | None = None,
    fits_memmap: str = "auto",
    overwrite: bool = False,
) -> tuple[int, int, tuple[str, ...]]:
    """Join table HDUs in *source* and write Parquet to *dest*.

    Returns ``(n_rows, n_cols, extension_labels)``.
    """
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {dest} (pass --overwrite)."
        )
    policy = catalog_fits_read_policy(fits_memmap, None)
    table, labels = read_joined_catalog_hdus(
        source,
        extensions=extensions,
        key_cols=key_cols,
        fits_read_policy=policy,
    )
    meta = dict(table.schema.metadata or {})
    meta[b"data_lake.fits_hdu"] = joined_fits_hdu_spec(labels).encode()
    meta[b"data_lake.source"] = str(source).encode()
    table = table.replace_schema_metadata(meta)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dest, compression="zstd")
    return table.num_rows, table.num_columns, labels


def _collect_sources(paths: list[Path], file_list: Path | None) -> list[Path]:
    if paths and file_list is not None:
        raise ValueError("Pass FITS paths or --file-list, not both.")
    if file_list is not None:
        sources = paths_from_file_list_file(file_list)
    else:
        sources = list(paths)
    if not sources:
        raise ValueError("Provide at least one FITS path or --file-list.")
    missing = [p for p in sources if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Not a file: {missing[0]}")
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="Input MEF FITS file(s).",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        default=None,
        help="Text file, one FITS path per line (instead of positional paths).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (single input only). Default: <stem>.parquet "
             "next to the FITS file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for one Parquet per input (required for more than one file "
             "unless you want each <stem>.parquet next to its FITS).",
    )
    parser.add_argument(
        "--hdu",
        action="append",
        dest="hdu_names",
        metavar="NAME",
        help="Table extension to join (index, EXTNAME, or EXTNAME,EXTVER). "
             "Repeatable; omit to join every non-empty table HDU.",
    )
    parser.add_argument(
        "--key-cols",
        default=None,
        help="Comma-separated shared columns that must match row by row "
             "(default: every column name present on all selected HDUs).",
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
        help="Replace output file(s) if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned input → output paths and exit.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar on multi-file runs.",
    )
    args = parser.parse_args(argv)

    try:
        sources = _collect_sources(args.sources, args.file_list)
    except (ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.output is not None and args.output_dir is not None:
        print("Pass -o or --output-dir, not both.", file=sys.stderr)
        return 1
    if args.output is not None and len(sources) > 1:
        print(
            " -o is a single Parquet file; use --output-dir for more than one input.",
            file=sys.stderr,
        )
        return 1
    if args.hdu_names is not None and len(args.hdu_names) < 2:
        print("Pass --hdu at least twice, or omit it to join every table HDU.", file=sys.stderr)
        return 1

    keys = (
        [c.strip() for c in args.key_cols.split(",") if c.strip()]
        if args.key_cols else None
    )
    extensions = list(args.hdu_names) if args.hdu_names else None

    planned = [
        (src, parquet_path_for(src, output=args.output, output_dir=args.output_dir))
        for src in sources
    ]
    if args.dry_run:
        for src, dest in planned:
            print(f"{src} → {dest}")
        return 0

    iterator = planned
    if len(planned) > 1 and not args.no_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(planned, desc="flatten MEF", unit="file")
        except ImportError:
            pass

    n_ok = 0
    for src, dest in iterator:
        try:
            n_rows, n_cols, labels = flatten_fits_mef(
                src,
                dest,
                extensions=extensions,
                key_cols=keys,
                fits_memmap=args.fits_memmap,
                overwrite=args.overwrite,
            )
        except (OSError, ValueError, FileExistsError) as exc:
            print(f"ERROR: {src}: {exc}", file=sys.stderr)
            return 1
        n_ok += 1
        if len(planned) == 1:
            print(
                f"Joined {src.name} [{'+'.join(labels)}] → "
                f"{n_rows} rows × {n_cols} columns in {dest}",
                file=sys.stderr,
            )

    if len(planned) > 1:
        print(f"Wrote {n_ok} Parquet file(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
