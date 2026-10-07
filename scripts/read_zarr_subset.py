#!/usr/bin/env python3
"""
read_zarr_subset.py
-------------------
Read a spectra subset produced by dl-extract-spectra-subset --format zarr
and export a FITS catalog + individual spectra suitable for TOPCAT validation.

The script resolves the original identifier columns (e.g. TARGETID, SURVEY,
PROGRAM) from the lake's Parquet catalog by joining on _source_id — the same
way redshift is resolved during extraction.  No data_lake install is needed.

Usage
-----
    python read_zarr_subset.py subset.zarr                     # summary only
    python read_zarr_subset.py subset.zarr --catalog out.fits  # TOPCAT catalog
    python read_zarr_subset.py subset.zarr --catalog out.fits \
        --plot-row 0                                           # plot spectrum 0

Requirements:  zarr, numpy, astropy, pyarrow
               (no desispec, no data_lake install needed)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr


# ---------------------------------------------------------------------------
# Zarr layout → FITS/DESI column mapping
# ---------------------------------------------------------------------------
#
#  Zarr array         FITS / DESI equivalent        Notes
#  -----------------  ----------------------------  -------------------------
#  flux               FLUX                          float32, shape (N, N_pix)
#  ivar               IVAR                          float32, shape (N, N_pix)
#  mask               MASK                          uint8,   shape (N, N_pix)
#  wavelength         WAVELENGTH                    float64, shape (N_pix,)
#                                                   Ångström, vacuum, shared grid
#  _source_id         —                             int64 hash of TARGETID|SURVEY|PROGRAM
#                                                   NOT the raw TARGETID integer
#  redshift           Z                             float32, shape (N,)
#                                                   from catalog if available
#
#  Group attributes (out_root.attrs):
#    source_survey          survey name (e.g. "DESI_DR1")
#    source_lake_root       absolute path to the lake root
#    n_sources              rows written
#    n_pix                  wavelength pixels per spectrum
#    wavelength_mode        "shared" (one grid) | "per_source" (N × N_pix)
#    extract_created_utc    ISO timestamp
#    schema_version         "1"
#
#  catalog_info.json  link_id_mode examples:
#    "column:TARGETID"              → single identifier column
#    "composite:TARGETID,SURVEY,PROGRAM"  → composite; all parts in Parquet
# ---------------------------------------------------------------------------


def open_zarr(path: str | Path) -> zarr.Group:
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: {p} not found")
    return zarr.open_group(store=zarr.storage.LocalStore(str(p)), mode="r", zarr_format=3)


# ---------------------------------------------------------------------------
# Catalog helpers: resolve original ID columns from the lake Parquet
# ---------------------------------------------------------------------------

def _catalog_root(lake_root: Path, survey: str) -> Path | None:
    """Return the catalog root dir for *survey*, or None if not found."""
    for sub in ("catalogs", "products"):
        p = lake_root / sub / survey
        if p.is_dir():
            return p
    return None


def _link_id_columns(catalog_info: dict[str, Any]) -> list[str]:
    """Parse ``link_id_mode`` → list of original column names.

    Examples
    --------
    "column:TARGETID"              → ["TARGETID"]
    "composite:TARGETID,SURVEY,PROGRAM" → ["TARGETID", "SURVEY", "PROGRAM"]
    "sequential" / "label:..."    → []   (no original column to recover)
    """
    mode: str = catalog_info.get("link_id_mode", "")
    if mode.startswith("column:") or mode.startswith("composite:") or mode.startswith("label:"):
        spec = mode.split(":", 1)[1]
        return [c.strip() for c in spec.split(",") if c.strip()]
    return []


def _lookup_id_columns(
    catalog_root: Path,
    norder: int,
    source_ids: np.ndarray,
    id_cols: list[str],
) -> dict[str, np.ndarray]:
    """Read Parquet tiles and return a {col_name → array} mapping aligned to *source_ids*.

    Only tiles that contain at least one requested _source_id are read.
    Uses pyarrow for zero-copy column reads.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    sid_set = set(source_ids.tolist())
    # Build output arrays filled with sentinel values
    n = len(source_ids)
    sid_to_row: dict[int, int] = {int(sid): i for i, sid in enumerate(source_ids)}

    # Determine column dtypes from the first tile
    result: dict[str, list] = {c: [None] * n for c in id_cols}

    glob_root = catalog_root / f"Norder={norder}"
    tiles = list(glob_root.rglob("Npix=*.parquet"))
    if not tiles:
        print(f"  WARNING: no Parquet tiles found under {glob_root}", file=sys.stderr)
        return {}

    read_cols = ["_source_id"] + id_cols
    found = 0
    for tile in tiles:
        try:
            tbl = pq.read_table(str(tile), columns=read_cols)
        except Exception as exc:
            print(f"  WARNING: could not read {tile.name}: {exc}", file=sys.stderr)
            continue

        tile_sids = tbl.column("_source_id").to_pylist()
        # Fast check: does this tile overlap with what we need?
        if not sid_set.intersection(tile_sids):
            continue

        for i, sid in enumerate(tile_sids):
            row = sid_to_row.get(int(sid))
            if row is None:
                continue
            for col in id_cols:
                result[col][row] = tbl.column(col)[i].as_py()
            found += 1
            if found == n:
                break  # all IDs resolved — stop scanning tiles

        if found == n:
            break

    if found < n:
        print(f"  NOTE: {n - found} source_ids not found in catalog tiles "
              f"(they may have no catalog entry).", file=sys.stderr)

    # Convert to numpy arrays; choose dtype based on values
    out: dict[str, np.ndarray] = {}
    for col, vals in result.items():
        sample = next((v for v in vals if v is not None), None)
        if isinstance(sample, (int, np.integer)):
            arr = np.array([v if v is not None else -1 for v in vals], dtype=np.int64)
        elif isinstance(sample, float):
            arr = np.array([v if v is not None else np.nan for v in vals], dtype=np.float64)
        else:
            # String / bytes → object array (astropy Table handles it fine)
            arr = np.array([str(v) if v is not None else "" for v in vals], dtype=object)
        out[col] = arr
    return out


def resolve_id_columns_from_lake(
    attrs: dict[str, Any],
    source_ids: np.ndarray,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Try to resolve original identifier columns for *source_ids* from the lake.

    Returns (id_col_names, {col → array}).  Empty dict if the lake is
    unreachable or the survey has no catalog.
    """
    lake_root_str = attrs.get("source_lake_root", "")
    survey = attrs.get("source_survey", "")
    if not lake_root_str or not survey:
        return [], {}

    lake_root = Path(lake_root_str)
    cat_root = _catalog_root(lake_root, survey)
    if cat_root is None:
        print(f"  NOTE: catalog not found at {lake_root}/catalogs/{survey} "
              f"— skipping ID column resolution.", file=sys.stderr)
        return [], {}

    info_path = cat_root / "catalog_info.json"
    if not info_path.exists():
        return [], {}
    catalog_info = json.loads(info_path.read_text())

    id_cols = _link_id_columns(catalog_info)
    if not id_cols:
        return [], {}

    norder = int(catalog_info.get("hats_order", 5))
    print(f"  Resolving {id_cols} from {cat_root.name} catalog (norder={norder}) …",
          file=sys.stderr)
    col_arrays = _lookup_id_columns(cat_root, norder, source_ids, id_cols)
    return id_cols, col_arrays


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(root: zarr.Group) -> None:
    attrs = dict(root.attrs)
    print("=" * 60)
    print("  Zarr subset summary")
    print("=" * 60)
    for k, v in attrs.items():
        print(f"  {k:<28} {v}")
    print()
    print("Arrays:")
    for name in sorted(root.array_keys()):
        arr = root[name]
        print(f"  {name:<20} shape={arr.shape}  dtype={arr.dtype}")
    print()

    wave = np.asarray(root["wavelength"])
    if wave.ndim == 1:
        print(f"Wavelength grid:  {wave[0]:.2f} – {wave[-1]:.2f} Å  "
              f"({len(wave)} pixels,  Δλ ≈ {float(wave[1]-wave[0]):.3f} Å/pix)")
    print()

    n = min(10, root["flux"].shape[0])
    flux_sample = np.asarray(root["flux"][:n])
    mask_sample = np.asarray(root["mask"][:n])
    good = mask_sample == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        snr_sample = np.where(good, flux_sample * np.sqrt(np.asarray(root["ivar"][:n])), np.nan)
    snr_per = np.where(good.any(axis=1), np.nanmedian(snr_sample, axis=1), np.nan)
    print(f"Median S/N (first {n} spectra, good pixels): "
          f"min={np.nanmin(snr_per):.1f}  median={np.nanmedian(snr_per):.1f}  "
          f"max={np.nanmax(snr_per):.1f}")


# ---------------------------------------------------------------------------
# Catalog export
# ---------------------------------------------------------------------------

def export_catalog(root: zarr.Group, out_path: Path) -> None:
    """Write a FITS BINTABLE with one row per spectrum for TOPCAT.

    Columns
    -------
    Original identifier columns (e.g. TARGETID, SURVEY, PROGRAM) are resolved
    from the lake's Parquet catalog by joining on _source_id — the same
    mechanism used to pull redshift during extraction.  If the lake is
    unreachable, only _source_id is written.
    """
    from astropy.table import Table

    attrs = dict(root.attrs)
    n    = root["flux"].shape[0]
    wave = np.asarray(root["wavelength"])
    flux = np.asarray(root["flux"])
    ivar = np.asarray(root["ivar"])
    mask = np.asarray(root["mask"])
    source_ids = np.asarray(root["_source_id"]).astype(np.int64)

    # Per-spectrum scalar summaries
    good = mask == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        snr_arr = np.where(ivar > 0, flux * np.sqrt(ivar), np.nan)

    median_snr  = np.where(good.any(axis=1),
                           np.nanmedian(np.where(good, snr_arr, np.nan), axis=1),
                           np.nan).astype(np.float32)
    median_flux = np.where(good.any(axis=1),
                           np.nanmedian(np.where(good, flux, np.nan), axis=1),
                           np.nan).astype(np.float32)
    n_good_pix  = good.sum(axis=1).astype(np.int32)

    # Resolve original identifier columns from the lake catalog (same path as redshift)
    id_col_names, id_col_arrays = resolve_id_columns_from_lake(attrs, source_ids)

    # Build output table: original ID cols first, then _source_id, then scalars
    cols: dict[str, Any] = {}
    for col in id_col_names:
        cols[col] = id_col_arrays[col]
    cols["_source_id"]  = source_ids
    cols["redshift"]    = np.asarray(root["redshift"]).astype(np.float32)
    cols["median_snr"]  = median_snr
    cols["median_flux"] = median_flux
    cols["n_good_pix"]  = n_good_pix
    cols["wave_min_A"]  = np.full(n, float(wave[0]),  dtype=np.float32)
    cols["wave_max_A"]  = np.full(n, float(wave[-1]), dtype=np.float32)

    tbl = Table(cols)
    tbl.meta["SURVEY"]   = str(attrs.get("source_survey", ""))
    tbl.meta["NPIX"]     = int(attrs.get("n_pix", len(wave)))
    tbl.meta["WAVEMODE"] = str(attrs.get("wavelength_mode", "shared"))
    tbl.meta["CREATED"]  = str(attrs.get("extract_created_utc", ""))
    if id_col_names:
        tbl.meta["LINKID"] = ",".join(id_col_names)

    tbl.write(str(out_path), format="fits", overwrite=True)
    print(f"Wrote TOPCAT catalog ({n} rows) → {out_path}")
    print(f"  Columns: {list(cols)}")
    if id_col_names:
        print(f"  Cross-match in TOPCAT: join your input FITS on "
              f"{id_col_names[0] if len(id_col_names) == 1 else str(id_col_names)}.")
    else:
        print("  Cross-match in TOPCAT: join on _source_id (lake not accessible).")


def plot_spectrum(root: zarr.Group, row: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required for --plot-row. Install it with: pip install matplotlib")

    wave = np.asarray(root["wavelength"])
    if wave.ndim == 2:
        wave = wave[row]

    flux = np.asarray(root["flux"][row])
    ivar = np.asarray(root["ivar"][row])
    mask = np.asarray(root["mask"][row])

    good = (mask == 0) & (ivar > 0)
    sigma = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan)
    z = float(np.asarray(root["redshift"])[row])
    sid = int(np.asarray(root["_source_id"])[row])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(wave[good], flux[good], where="mid", lw=0.8, color="steelblue", label="flux (good px)")
    ax.fill_between(wave[good],
                    flux[good] - sigma[good], flux[good] + sigma[good],
                    alpha=0.25, color="steelblue")
    if (~good).any():
        ax.scatter(wave[~good], flux[~good], s=4, color="tomato", alpha=0.5, label="masked")

    ax.set_xlabel("Observed wavelength (Å)")
    ax.set_ylabel("Flux")
    ax.set_title(f"Row {row}  |  _source_id={sid}  |  z={z:.4f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zarr_path", help="Path to the .zarr directory")
    ap.add_argument("--catalog", metavar="OUT.fits",
                    help="Export a per-spectrum scalar summary FITS for TOPCAT")
    ap.add_argument("--plot-row", metavar="N", type=int,
                    help="Plot spectrum at row N (0-based)")
    args = ap.parse_args()

    root = open_zarr(args.zarr_path)
    print_summary(root)

    if args.catalog:
        export_catalog(root, Path(args.catalog))

    if args.plot_row is not None:
        n = root["flux"].shape[0]
        if args.plot_row >= n:
            sys.exit(f"ERROR: --plot-row {args.plot_row} is out of range (0–{n-1})")
        plot_spectrum(root, args.plot_row)


if __name__ == "__main__":
    main()
