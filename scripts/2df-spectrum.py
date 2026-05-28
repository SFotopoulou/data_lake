#!/usr/bin/env python3
"""
Python port of 2df-spectrum.f (SPECFITS).

Lists object spectrum, variance array, and sky spectrum from 2dFGRS FITS files
into ASCII text files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from astropy.io import fits


SPEC_SIZE = 1024
DEFAULT_DATABASE_PREFIX = Path("/2dFGRS_Database")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List object spectrum, variance array, and sky spectrum from 2dFGRS "
            "FITS files into ASCII text files."
        )
    )
    parser.add_argument(
        "name",
        help=(
            "Either a FITS filename or a file containing a list of FITS filenames "
            "(with optional extension numbers when xtn > 0)."
        ),
    )
    parser.add_argument(
        "xtn",
        type=int,
        help="0 for all extensions, >0 for one extension (or list-provided extensions).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate PNG plots for each extracted spectrum.",
    )
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_DATABASE_PREFIX),
        help=(
            "Base directory to prepend when FITS paths are not found directly "
            "(default: /2dFGRS_Database)."
        ),
    )
    return parser.parse_args()


def is_fits_name(path_like: str) -> bool:
    suffix = Path(path_like).suffix.lower()
    return suffix in {".fit", ".fits"}


def parse_file_list(path: Path, xtn: int) -> List[Tuple[str, int]]:
    files: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if xtn == 0:
                files.append((line, 0))
            else:
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(
                        f"Invalid list line for xtn>0 (expected '<file> <ext>'): {line}"
                    )
                files.append((parts[0], int(parts[-1])))
    return files


def build_inputs(name: str, xtn: int) -> List[Tuple[str, int]]:
    if is_fits_name(name):
        return [(name, xtn)]
    list_path = Path(name)
    if not list_path.exists():
        raise FileNotFoundError(f"Cannot open list file named {name}")
    return parse_file_list(list_path, xtn)


def extract_components(data: np.ndarray, naxis1: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("FITS extension data is not 2D as required")

    if arr.shape == (3, naxis1):
        spec, vari, sky = arr[0], arr[1], arr[2]
    elif arr.shape == (naxis1, 3):
        spec, vari, sky = arr[:, 0], arr[:, 1], arr[:, 2]
    else:
        flat = arr.reshape(-1)
        if flat.size < 3 * naxis1:
            raise ValueError("FITS extension data does not contain expected samples")
        spec = flat[0:naxis1]
        vari = flat[naxis1 : 2 * naxis1]
        sky = flat[2 * naxis1 : 3 * naxis1]

    return np.asarray(spec, dtype=float), np.asarray(vari, dtype=float), np.asarray(sky, dtype=float)


def output_name(ff: str, xtn: int) -> str:
    lower = ff.lower()
    idx = lower.rfind(".fits")
    if idx == -1:
        stem = Path(ff).stem
    else:
        stem = ff[:idx]
    off = stem[-6:] if len(stem) >= 6 else stem
    return f"{off}_{xtn}.txt"


def printspec(
    fff: str,
    ff: str,
    seqnum: int,
    xtn: int,
    name: str,
    crval1: float,
    crpix1: float,
    cdelt1: float,
    spec: np.ndarray,
    var: np.ndarray,
    sky: np.ndarray,
    bjsel: float,
    z: float,
    quality: int,
    abemma: int,
) -> None:
    outfile = output_name(ff, xtn)
    outpath = Path(outfile)

    with outpath.open("w", encoding="utf-8") as out:
        out.write(f"# FILE: {fff}\n")
        out.write("#\n")
        out.write(f"# SEQNUM = {seqnum:06d}  EXTNUM = {xtn:d}  NAME = {name[:10]:<10}\n")
        out.write(
            f"# BJSEL = {bjsel:5.2f}  Z ={z:9.6f}  QUALITY = {quality:d}  ABEMMA = {abemma:d}\n"
        )
        out.write("#\n")
        out.write("#  Pixel  Wavlength    Object  Variance       Sky\n")
        out.write("# Number  Angstroms    Counts    Counts    Counts\n")

        for i in range(1, len(spec) + 1):
            lam = crval1 + cdelt1 * (float(i) - crpix1)
            outspec = max(float(spec[i - 1]), 0.0)
            outvar = max(float(var[i - 1]), 0.0)
            outsky = max(float(sky[i - 1]), 0.0)
            out.write(f"{i:8d} {lam:10.2f}{outspec:10.2f}{outvar:10.2f}{outsky:10.2f}\n")

    print(f"SEQNUM={seqnum:06d} EXTNUM={xtn:d} NAME={name.strip()} --> {outfile}")
    return outfile


def plot_spectrum(
    outfile: str,
    seqnum: int,
    xtn: int,
    name: str,
    wavelengths: np.ndarray,
    spec: np.ndarray,
    var: np.ndarray,
    sky: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib is not installed; skipping plot output")
        return

    plotfile = str(Path(outfile).with_suffix(".png"))
    clipped_spec = np.maximum(spec, 0.0)
    clipped_var = np.maximum(var, 0.0)
    clipped_sky = np.maximum(sky, 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wavelengths, clipped_spec, label="Object", linewidth=1.0)
    ax.plot(wavelengths, clipped_var, label="Variance", linewidth=1.0, alpha=0.8)
    ax.plot(wavelengths, clipped_sky, label="Sky", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Counts")
    ax.set_title(f"SEQNUM {seqnum:06d} EXT {xtn} {name.strip()}")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plotfile, dpi=150)
    plt.close(fig)
    print(f"SEQNUM={seqnum:06d} EXTNUM={xtn:d} NAME={name.strip()} --> {plotfile}")


def iter_extensions(requested_xtn: int, num_spec: int) -> Iterable[int]:
    for i in range(1, num_spec + 1):
        if requested_xtn == 0 or requested_xtn == i:
            yield i


def resolve_fits_path(ff: str, base_dir: Path) -> Path:
    candidate = Path(ff)
    if candidate.exists():
        return candidate
    prefixed = base_dir / ff
    if prefixed.exists():
        return prefixed
    return prefixed


def process_one_file(ff: str, requested_xtn: int, base_dir: Path, plot: bool = False) -> None:
    fits_path = resolve_fits_path(ff, base_dir)
    fff = str(fits_path)
    try:
        hdul = fits.open(fff)
    except OSError:
        print(
            f"ERROR: cannot open FITS file named {fff} "
            f"(also checked direct path: {Path(ff)})"
        )
        return

    with hdul:
        primary = hdul[0].header
        try:
            seqnum = int(primary["SEQNUM"])
            name = str(primary["NAME"])
            bjsel = float(primary["BJSEL"])
        except Exception:
            print("ERROR: failed to read APM keywords from primary image")
            return

        num_spec = len(hdul) - 1
        if num_spec <= 0:
            return

        for extnum in iter_extensions(requested_xtn, num_spec):
            try:
                hdu = hdul[extnum]
            except Exception:
                print(f"ERROR: failed to select spectral extension {extnum}")
                continue

            hdr = hdu.header
            try:
                sp_naxis1 = int(hdr["NAXIS1"])
                sp_naxis2 = int(hdr["NAXIS2"])
            except Exception:
                print("ERROR: failed to read spectrum dimension keywords")
                continue

            if sp_naxis2 != 3:
                print("ERROR: FITS file has invalid structure")
                continue
            if sp_naxis1 != SPEC_SIZE:
                print("ERROR: spectral dimension is not 1024")
                continue

            try:
                cdelt1 = float(hdr["CDELT1"])
                crval1 = float(hdr["CRVAL1"])
                crpix1 = float(hdr["CRPIX1"])
            except Exception:
                print("ERROR: failed to get wavelength scale keywords")
                continue

            try:
                z = float(hdr["Z"])
                quality = int(hdr["QUALITY"])
                abemma = int(hdr["ABEMMA"])
            except Exception:
                print("ERROR: failed to get spectral keywords")
                continue

            data = hdu.data
            if data is None:
                print("ERROR: failed to get object spectrum")
                continue

            try:
                spec, vari, sky = extract_components(data, sp_naxis1)
            except Exception:
                print("ERROR: failed to get object spectrum")
                continue

            outfile = printspec(
                fff=fff,
                ff=ff,
                seqnum=seqnum,
                xtn=extnum,
                name=name,
                crval1=crval1,
                crpix1=crpix1,
                cdelt1=cdelt1,
                spec=spec,
                var=vari,
                sky=sky,
                bjsel=bjsel,
                z=z,
                quality=quality,
                abemma=abemma,
            )
            if plot:
                wavelengths = crval1 + cdelt1 * (np.arange(1, len(spec) + 1, dtype=float) - crpix1)
                plot_spectrum(
                    outfile=outfile,
                    seqnum=seqnum,
                    xtn=extnum,
                    name=name,
                    wavelengths=wavelengths,
                    spec=spec,
                    var=vari,
                    sky=sky,
                )


def main() -> int:
    args = parse_args()
    try:
        inputs = build_inputs(args.name, args.xtn)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    for ff, xtn in inputs:
        process_one_file(ff, xtn, base_dir=Path(args.base_dir), plot=args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
