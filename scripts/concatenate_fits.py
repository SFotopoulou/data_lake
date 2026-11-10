#!/usr/bin/env python3
"""Concatenate row-normal catalog FITS tables from a folder into one file.

Designed for surveys shipped as many small BINTABLE shards (e.g. Legacy bricks,
Gaia run folders) before ``dl-ingest-catalog`` / parallel batch ingest.

Packed-vector FITS (``NAXIS2=1`` with long vector columns, e.g. STILTS colfits)
are rejected — convert those to row-normal FITS or Parquet first.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import click
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


@click.command(context_settings={"help_option_names": ["-h", "--help"]}, help=__doc__)
@click.argument("input_dir", type=click.Path(path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output FITS path.",
)
@click.option(
    "--pattern",
    default="*.fits",
    show_default=True,
    help="Glob pattern for input files relative to input_dir.",
)
@click.option("--recursive", is_flag=True, help="Search subdirectories recursively.")
@click.option(
    "--hdu",
    default=None,
    help="Table HDU index or name (default: first BINTABLE in each file).",
)
@click.option(
    "--join",
    "join_type",
    type=click.Choice(("exact", "outer")),
    default="exact",
    show_default=True,
    help="How to align columns when stacking tables.",
)
@click.option(
    "--fits-memmap",
    type=click.Choice(("auto", "on", "off")),
    default="auto",
    show_default=True,
    help="FITS read policy, same semantics as dl-ingest-catalog.",
)
@click.option("--overwrite", is_flag=True, help="Replace output file if it already exists.")
@click.option("--dry-run", is_flag=True, help="List input files and exit without writing output.")
@click.option("--no-progress", is_flag=True, help="Disable tqdm progress bar.")
def main(
    input_dir: Path,
    output: Path,
    pattern: str,
    recursive: bool,
    hdu: str | None,
    join_type: str,
    fits_memmap: str,
    overwrite: bool,
    dry_run: bool,
    no_progress: bool,
) -> int:
    hdu_resolved: int | str | None
    if hdu is None:
        hdu_resolved = None
    else:
        try:
            hdu_resolved = int(hdu)
        except ValueError:
            hdu_resolved = hdu

    try:
        paths = _collect_fits_paths(
            input_dir,
            pattern=pattern,
            recursive=recursive,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Found {len(paths)} FITS file(s) under {input_dir}", file=sys.stderr)
    if dry_run:
        for path in paths:
            print(path)
        return 0

    if output.exists() and not overwrite:
        print(f"Refusing to overwrite existing file: {output}", file=sys.stderr)
        print("Pass --overwrite to replace it.", file=sys.stderr)
        return 1

    try:
        merged = concatenate_fits_tables(
            paths,
            hdu=hdu_resolved,
            join_type=join_type,
            memmap=fits_memmap,
            show_progress=not no_progress,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    merged.write(str(output), format="fits", overwrite=True)
    print(
        f"Wrote {len(merged)} row(s) from {len(paths)} file(s) to {output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
