#!/usr/bin/env python3
"""
read_zarr_subset.py
-------------------
Read a spectra OR cutout subset produced by dl-extract-*-subset --format zarr
and export a FITS catalog suitable for TOPCAT validation.

The script auto-detects the modality from the Zarr group layout:
  • spectra  → arrays: flux, ivar, mask, wavelength, _source_id, redshift
  • cutouts  → arrays: images, wcs, _source_id

Identifier columns (e.g. TARGETID, SURVEY, PROGRAM) are read from the subset
itself when embedded at extract time (``link_id_columns`` attr).  Older
extracts fall back to joining the lake Parquet on ``_source_id``.  No
``data_lake`` install is needed to run this script.

Usage
-----
    python read_zarr_subset.py subset.zarr                         # summary only
    python read_zarr_subset.py subset.zarr --catalog out.parquet   # preferred
    python read_zarr_subset.py subset.zarr --catalog out.fits      # also OK
    python read_zarr_subset.py subset.zarr --plot-row 0
    python read_zarr_subset.py cutouts.zarr --show-cutout 0

Requirements:  zarr, numpy, pyarrow
               (astropy only for .fits catalog; matplotlib for plots)
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
#  SPECTRA layout (dl-extract-spectra-subset --format zarr)
#  ---------------------------------------------------------
#  Zarr array         FITS / DESI equivalent        Notes
#  flux               FLUX                          float32, (N, N_pix)
#  ivar               IVAR                          float32, (N, N_pix)
#  mask               MASK                          uint8,   (N, N_pix)
#  wavelength         WAVELENGTH                    float64, (N_pix,) Å vacuum
#  _source_id         —                             int64 hash (not raw TARGETID)
#  redshift           Z                             float32, (N,)
#
#  CUTOUT layout (dl-extract-cutout-subset --format zarr)
#  -------------------------------------------------------
#  images             image data                    float32, (N, B, H, W)
#  wcs                WCS structured bytes          (N,) — decode for CRVAL/CRPIX/CD
#  _source_id         —                             int64 hash (not raw TARGETID)
#
#  Group attributes (both modalities)
#  ------------------------------------
#    source_survey          survey name
#    source_lake_root       absolute path to the lake root
#    n_sources              rows written
#    extract_created_utc    ISO timestamp
#    schema_version         "1"
#
#  catalog_info.json  link_id_mode examples (for ID column resolution)
#    "column:TARGETID"                    → single column
#    "composite:TARGETID,SURVEY,PROGRAM"  → all parts stored in Parquet
# ---------------------------------------------------------------------------


def _detect_modality(root: zarr.Group) -> str:
    """Return 'spectra' or 'cutout' based on which arrays are present."""
    keys = set(root.array_keys())
    if "images" in keys:
        return "cutout"
    if "flux" in keys:
        return "spectra"
    raise ValueError(
        f"Cannot determine modality from arrays: {sorted(keys)}. "
        "Expected 'flux' (spectra) or 'images' (cutouts)."
    )


def _parse_wcs_raw(raw) -> dict[str, Any]:
    """Decode a WCS structured-bytes entry into a plain dict."""
    import struct

    # _WCS_DTYPE: crval1,crval2,crpix1,crpix2,cd1_1,cd1_2,cd2_1,cd2_2  (8×float64)
    #             naxis1, naxis2 (2×int32)
    fields = ["crval1", "crval2", "crpix1", "crpix2",
              "cd1_1", "cd1_2", "cd2_1", "cd2_2"]
    raw_bytes = bytes(raw)
    vals = struct.unpack_from("8d2i", raw_bytes)
    result = dict(zip(fields + ["naxis1", "naxis2"], vals))
    return result


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


def resolve_id_columns_from_zarr(
    root: zarr.Group,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Read embedded link-id columns from the subset itself (preferred).

    Written by ``dl-extract-*-subset`` when a catalog was available at extract
    time (attrs ``link_id_columns`` + peer arrays).
    """
    attrs = dict(root.attrs)
    cols = attrs.get("link_id_columns")
    if not cols:
        return [], {}
    # attrs may store lists as numpy arrays / JSON-ish
    if isinstance(cols, np.ndarray):
        cols = cols.tolist()
    cols = [str(c) for c in cols]
    available = set(root.array_keys())
    present = [c for c in cols if c in available]
    if not present:
        return [], {}
    if len(present) < len(cols):
        print(
            f"  NOTE: subset attrs list {cols} but only {present} arrays present.",
            file=sys.stderr,
        )
    print(
        f"  Using embedded link-id columns from subset: {present}"
        + (f"  (mode={attrs.get('link_id_mode')})" if attrs.get("link_id_mode") else ""),
        file=sys.stderr,
    )

    def _as_column(name: str) -> np.ndarray:
        arr = np.asarray(root[name][:])
        # Embedded strings are UTF-8 bytes (|S) for Zarr-v3 stability
        if arr.dtype.kind == "S":
            return np.asarray([b.decode("utf-8") for b in arr.tolist()], dtype=object)
        return arr

    return present, {c: _as_column(c) for c in present}


def resolve_id_columns_from_lake(
    attrs: dict[str, Any],
    source_ids: np.ndarray,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Fallback: resolve original identifier columns from the lake Parquet.

    Only used for older extracts that lack embedded ``link_id_columns``.
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
    print(f"  Resolving {id_cols} from lake catalog {cat_root.name} "
          f"(norder={norder}; legacy extract without embedded IDs) …",
          file=sys.stderr)
    col_arrays = _lookup_id_columns(cat_root, norder, source_ids, id_cols)
    return id_cols, col_arrays


def resolve_id_columns(
    root: zarr.Group,
    source_ids: np.ndarray,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Prefer embedded subset columns; fall back to lake join if needed."""
    cols, arrays = resolve_id_columns_from_zarr(root)
    if cols:
        return cols, arrays
    return resolve_id_columns_from_lake(dict(root.attrs), source_ids)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(root: zarr.Group) -> None:
    attrs = dict(root.attrs)
    modality = _detect_modality(root)
    print("=" * 60)
    print(f"  Zarr subset summary  [{modality}]")
    print("=" * 60)
    for k, v in attrs.items():
        print(f"  {k:<28} {v}")
    print()
    print("Arrays:")
    for name in sorted(root.array_keys()):
        arr = root[name]
        print(f"  {name:<20} shape={arr.shape}  dtype={arr.dtype}")
    print()

    if modality == "spectra":
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
            snr_sample = np.where(
                good, flux_sample * np.sqrt(np.asarray(root["ivar"][:n])), np.nan
            )
        snr_per = np.where(good.any(axis=1), np.nanmedian(snr_sample, axis=1), np.nan)
        print(f"Median S/N (first {n} spectra, good pixels): "
              f"min={np.nanmin(snr_per):.1f}  median={np.nanmedian(snr_per):.1f}  "
              f"max={np.nanmax(snr_per):.1f}")

    else:  # cutout
        _, n_bands, h, w = root["images"].shape
        band_names = list(attrs.get("band_names", []))
        print(f"Image shape:  {n_bands} band(s) × {h} × {w} px")
        if band_names:
            print(f"Band names:   {band_names}")
        n = min(5, root["images"].shape[0])
        imgs = np.asarray(root["images"][:n])
        print(f"Flux range (first {n} cutouts):  "
              f"min={float(np.nanmin(imgs)):.3g}  max={float(np.nanmax(imgs)):.3g}  "
              f"median={float(np.nanmedian(imgs)):.3g}")


# ---------------------------------------------------------------------------
# Catalog export (Parquet preferred; FITS optional)
# ---------------------------------------------------------------------------

_BATCH = 512  # rows per chunk when computing flux stats (keeps RAM bounded)


def _write_table(cols: dict[str, Any], out_path: Path, meta: dict[str, Any]) -> None:
    """Write *cols* as Parquet (default) or FITS, chosen by suffix."""
    out_path = Path(out_path)
    suffix = out_path.suffix.lower()

    if suffix in {".fits", ".fit"}:
        from astropy.table import Table

        tbl = Table(cols)
        for k, v in meta.items():
            if v is not None and v != "":
                tbl.meta[k] = v
        tbl.write(str(out_path), format="fits", overwrite=True)
    else:
        # Default / .parquet / .pq / no suffix → Parquet
        import pyarrow as pa
        import pyarrow.parquet as pq

        if suffix not in {".parquet", ".pq"}:
            out_path = out_path.with_suffix(".parquet")
        arrays = {}
        for name, values in cols.items():
            arr = np.asarray(values)
            if arr.dtype == object:
                arrays[name] = pa.array(arr.tolist(), type=pa.string())
            else:
                arrays[name] = pa.array(arr)
        table = pa.table(arrays)
        # Store light metadata as Parquet key/value strings
        kv = {str(k).encode(): str(v).encode() for k, v in meta.items() if v not in (None, "")}
        table = table.replace_schema_metadata(kv)
        pq.write_table(table, str(out_path), compression="zstd")

    print(f"Wrote catalog ({len(next(iter(cols.values())))} rows) → {out_path}")
    print(f"  Columns: {list(cols)}")


def _spectrum_flux_stats(root: zarr.Group, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream flux/ivar/mask in batches → (median_snr, median_flux, n_good_pix)."""
    median_snr = np.empty(n, dtype=np.float32)
    median_flux = np.empty(n, dtype=np.float32)
    n_good_pix = np.empty(n, dtype=np.int32)

    for start in range(0, n, _BATCH):
        stop = min(start + _BATCH, n)
        flux = np.asarray(root["flux"][start:stop])
        ivar = np.asarray(root["ivar"][start:stop])
        mask = np.asarray(root["mask"][start:stop])
        good = mask == 0
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = np.where(ivar > 0, flux * np.sqrt(ivar), np.nan)
        any_good = good.any(axis=1)
        median_snr[start:stop] = np.where(
            any_good, np.nanmedian(np.where(good, snr, np.nan), axis=1), np.nan
        ).astype(np.float32)
        median_flux[start:stop] = np.where(
            any_good, np.nanmedian(np.where(good, flux, np.nan), axis=1), np.nan
        ).astype(np.float32)
        n_good_pix[start:stop] = good.sum(axis=1).astype(np.int32)
    return median_snr, median_flux, n_good_pix


def _cutout_flux_stats(root: zarr.Group, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Stream images in batches → (median_flux, max_flux). Never loads full cube."""
    median_flux = np.empty(n, dtype=np.float32)
    max_flux = np.empty(n, dtype=np.float32)

    for start in range(0, n, _BATCH):
        stop = min(start + _BATCH, n)
        imgs = np.asarray(root["images"][start:stop])  # (k, B, H, W)
        flat = imgs.reshape(stop - start, -1)
        median_flux[start:stop] = np.nanmedian(flat, axis=1).astype(np.float32)
        max_flux[start:stop] = np.nanmax(flat, axis=1).astype(np.float32)
    return median_flux, max_flux


def export_catalog(root: zarr.Group, out_path: Path) -> None:
    """Write a per-spectrum scalar catalog (Parquet or FITS)."""
    attrs = dict(root.attrs)
    n = root["flux"].shape[0]
    wave = np.asarray(root["wavelength"])
    if wave.ndim == 2:
        wave_min = float(np.nanmin(wave[:, 0]))
        wave_max = float(np.nanmax(wave[:, -1]))
    else:
        wave_min, wave_max = float(wave[0]), float(wave[-1])

    source_ids = np.asarray(root["_source_id"]).astype(np.int64)
    median_snr, median_flux, n_good_pix = _spectrum_flux_stats(root, n)
    id_col_names, id_col_arrays = resolve_id_columns(root, source_ids)

    cols: dict[str, Any] = {}
    for col in id_col_names:
        cols[col] = id_col_arrays[col]
    cols["_source_id"] = source_ids
    cols["redshift"] = np.asarray(root["redshift"]).astype(np.float32)
    cols["median_snr"] = median_snr
    cols["median_flux"] = median_flux
    cols["n_good_pix"] = n_good_pix
    cols["wave_min_A"] = np.full(n, wave_min, dtype=np.float32)
    cols["wave_max_A"] = np.full(n, wave_max, dtype=np.float32)

    meta = {
        "SURVEY": str(attrs.get("source_survey", "")),
        "NPIX": int(attrs.get("n_pix", len(wave) if wave.ndim == 1 else wave.shape[-1])),
        "WAVEMODE": str(attrs.get("wavelength_mode", "shared")),
        "CREATED": str(attrs.get("extract_created_utc", "")),
        "LINKID": ",".join(id_col_names) if id_col_names else "",
    }
    _write_table(cols, out_path, meta)
    if id_col_names:
        print(f"  Cross-match: join on "
              f"{id_col_names[0] if len(id_col_names) == 1 else id_col_names}.")
    else:
        print("  Cross-match: join on _source_id (lake not accessible).")


def export_cutout_catalog(root: zarr.Group, out_path: Path) -> None:
    """Write a per-cutout scalar catalog (WCS centre + streamed flux stats)."""
    attrs = dict(root.attrs)
    n = root["images"].shape[0]
    source_ids = np.asarray(root["_source_id"]).astype(np.int64)

    # Decode WCS — small structured array, fine to loop
    crval1 = np.empty(n, dtype=np.float64)
    crval2 = np.empty(n, dtype=np.float64)
    cd1_1 = np.empty(n, dtype=np.float64)
    naxis1 = np.empty(n, dtype=np.int32)
    naxis2 = np.empty(n, dtype=np.int32)
    for i in range(n):
        p = _parse_wcs_raw(root["wcs"][i])
        crval1[i] = p["crval1"]
        crval2[i] = p["crval2"]
        cd1_1[i] = p["cd1_1"]
        naxis1[i] = p["naxis1"]
        naxis2[i] = p["naxis2"]

    pix_scale_arcsec = (np.abs(cd1_1) * 3600.0).astype(np.float32)
    median_flux, max_flux = _cutout_flux_stats(root, n)
    id_col_names, id_col_arrays = resolve_id_columns(root, source_ids)

    cols: dict[str, Any] = {}
    for col in id_col_names:
        cols[col] = id_col_arrays[col]
    cols["_source_id"] = source_ids
    cols["ra"] = crval1
    cols["dec"] = crval2
    cols["pix_scale_arcsec"] = pix_scale_arcsec
    cols["stamp_width_px"] = naxis1
    cols["stamp_height_px"] = naxis2
    cols["median_flux"] = median_flux
    cols["max_flux"] = max_flux

    meta = {
        "SURVEY": str(attrs.get("source_survey", "")),
        "NBANDS": int(attrs.get("n_bands", root["images"].shape[1])),
        "CREATED": str(attrs.get("extract_created_utc", "")),
        "LINKID": ",".join(id_col_names) if id_col_names else "",
    }
    _write_table(cols, out_path, meta)
    if id_col_names:
        print(f"  Cross-match: join on "
              f"{id_col_names[0] if len(id_col_names) == 1 else id_col_names}"
              f" or ra/dec (1-arcsec cone).")


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


def show_cutout(root: zarr.Group, row: int) -> None:
    """Display a cutout stamp with WCS overlay using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        sys.exit("matplotlib is required for --show-cutout. pip install matplotlib")

    n = root["images"].shape[0]
    if row >= n:
        sys.exit(f"ERROR: --show-cutout {row} out of range (0–{n-1})")

    attrs = dict(root.attrs)
    band_names: list[str] = list(attrs.get("band_names", []))
    img = np.asarray(root["images"][row])  # (B, H, W)
    sid = int(np.asarray(root["_source_id"])[row])
    wcs_params = _parse_wcs_raw(root["wcs"][row])

    n_bands = img.shape[0]
    fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4), squeeze=False)
    axes = axes[0]

    for b, ax in enumerate(axes):
        plane = img[b]
        vmin, vmax = float(np.nanpercentile(plane, 1)), float(np.nanpercentile(plane, 99))
        ax.imshow(plane, origin="lower", cmap="gray",
                  norm=Normalize(vmin=vmin, vmax=vmax))
        label = band_names[b] if b < len(band_names) else f"band {b}"
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        # Mark reference pixel
        ax.scatter([wcs_params["crpix1"] - 1], [wcs_params["crpix2"] - 1],
                   s=40, c="red", marker="+", linewidths=1)

    ra, dec = wcs_params["crval1"], wcs_params["crval2"]
    pix_as = abs(wcs_params["cd1_1"]) * 3600.0
    fig.suptitle(
        f"Row {row}  |  _source_id={sid}\n"
        f"RA={ra:.5f}  Dec={dec:.5f}  pix={pix_as:.3f} arcsec",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zarr_path", help="Path to the .zarr directory")
    ap.add_argument(
        "--catalog", metavar="OUT.parquet",
        help=(
            "Export a per-row scalar catalog for TOPCAT. "
            "Prefer .parquet (memory-safe; TOPCAT opens Parquet natively). "
            "Use .fits only for small subsets. "
            "Spectra: flux stats + redshift; cutouts: WCS centre + flux stats."
        ),
    )
    ap.add_argument("--plot-row", metavar="N", type=int,
                    help="[spectra] Plot spectrum at row N (0-based).")
    ap.add_argument("--show-cutout", metavar="N", type=int,
                    help="[cutouts] Display cutout stamp at row N (0-based).")
    args = ap.parse_args()

    root = open_zarr(args.zarr_path)
    modality = _detect_modality(root)
    print_summary(root)

    if args.catalog:
        if modality == "spectra":
            export_catalog(root, Path(args.catalog))
        else:
            export_cutout_catalog(root, Path(args.catalog))

    if args.plot_row is not None:
        if modality != "spectra":
            sys.exit("ERROR: --plot-row is for spectra Zarr. Use --show-cutout for cutouts.")
        n = root["flux"].shape[0]
        if args.plot_row >= n:
            sys.exit(f"ERROR: --plot-row {args.plot_row} out of range (0–{n-1})")
        plot_spectrum(root, args.plot_row)

    if args.show_cutout is not None:
        if modality != "cutout":
            sys.exit("ERROR: --show-cutout is for cutout Zarr. Use --plot-row for spectra.")
        show_cutout(root, args.show_cutout)


if __name__ == "__main__":
    main()
