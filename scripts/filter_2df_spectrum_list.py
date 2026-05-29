#!/usr/bin/env python3
"""Keep only 2dF FITS paths that contain a ingestable 1-D spectrum HDU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from astropy.io import fits
from tqdm import tqdm

from data_lake.ingest.fits_to_spectra_zarr import _is_2df_spectrum_hdu


def _has_2df_spectrum(path: Path) -> bool:
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            if (hdu.name or "").strip().upper() == "SPECTRUM" and _is_2df_spectrum_hdu(hdu):
                return True
        return any(_is_2df_spectrum_hdu(hdu) for hdu in hdul)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_list", type=Path, help="Input paths, one per line")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output paths that pass the spectrum HDU check",
    )
    parser.add_argument(
        "--rejects",
        type=Path,
        default=None,
        help="Optional path listing rejected inputs",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    args = parser.parse_args()

    entries: list[str] = []
    for line in args.file_list.read_text().splitlines():
        path_str = line.strip()
        if path_str and not path_str.startswith("#"):
            entries.append(path_str)

    kept: list[str] = []
    rejected: list[str] = []
    for path_str in tqdm(entries, desc="filter 2df", unit="file", disable=args.no_progress):
        p = Path(path_str)
        if not p.is_file():
            rejected.append(path_str)
            continue
        try:
            ok = _has_2df_spectrum(p)
        except OSError:
            rejected.append(path_str)
            ok = False
        if ok:
            kept.append(path_str)
        else:
            rejected.append(path_str)

    args.output.write_text("\n".join(kept) + ("\n" if kept else ""))
    if args.rejects is not None:
        args.rejects.write_text("\n".join(rejected) + ("\n" if rejected else ""))

    print(f"kept {len(kept)} / {len(kept) + len(rejected)}", file=sys.stderr)
    if rejected:
        print(f"rejected {len(rejected)} (see --rejects)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
