"""
fits_to_zarr – extract galaxy image cutouts from FITS files into sharded Zarr v3 stacks.

Each HEALPix tile is stored as a separate Zarr group:

  <root>/cutouts/<survey>/Norder=<N>/Npix=<pix>.zarr/
      images/    shape=(N_sources, N_bands, H, W)  float32, sharded
      source_id/ shape=(N_sources,)                 int64
      wcs/       shape=(N_sources,)                 structured array with WCS scalars

After ingest, update the survey Parquet catalog with ``_cutout_index`` (tile-local
row index) using :func:`data_lake.ingest.update_catalog_indices.update_index_column`.
Fast random access uses that index together with the tile's ``source_id`` array;
there is **no** separate ``index.parquet`` sidecar on disk.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import zarr
import zarr.codecs
from astropy.io import fits
from astropy.wcs import WCS

from data_lake.ingest.fits_to_parquet import assign_healpix, healpix_dir

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHUNKS_PER_SHARD = 1024   # target ~760 MB shards for medium cutouts
_ZSTD_LEVEL = 3
_DEFAULT_DTYPE = np.float32

# WCS structured dtype stored per source
_WCS_DTYPE = np.dtype([
    ("crval1", np.float64),
    ("crval2", np.float64),
    ("crpix1", np.float64),
    ("crpix2", np.float64),
    ("cd1_1", np.float64),
    ("cd1_2", np.float64),
    ("cd2_1", np.float64),
    ("cd2_2", np.float64),
    ("naxis1", np.int32),
    ("naxis2", np.int32),
])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CutoutRecord:
    """Single cutout extracted from a FITS file."""
    source_id: int
    ra: float
    dec: float
    image: np.ndarray        # shape (N_bands, H, W)
    wcs_params: dict[str, Any]


# ---------------------------------------------------------------------------
# WCS helpers
# ---------------------------------------------------------------------------


def _extract_wcs_params(header: fits.Header) -> dict[str, Any]:
    """Extract the minimal WCS parameters needed for a round-trip to FITS."""
    wcs = WCS(header, naxis=2)
    cd = wcs.pixel_scale_matrix  # 2×2 CD matrix

    crval = wcs.wcs.crval if wcs.wcs.crval is not None else [0.0, 0.0]
    crpix = wcs.wcs.crpix if wcs.wcs.crpix is not None else [1.0, 1.0]

    return {
        "crval1": float(crval[0]),
        "crval2": float(crval[1]),
        "crpix1": float(crpix[0]),
        "crpix2": float(crpix[1]),
        "cd1_1": float(cd[0, 0]),
        "cd1_2": float(cd[0, 1]),
        "cd2_1": float(cd[1, 0]),
        "cd2_2": float(cd[1, 1]),
        "naxis1": int(header.get("NAXIS1", 0)),
        "naxis2": int(header.get("NAXIS2", 0)),
    }


def _wcs_params_to_structured(params: dict[str, Any]) -> np.ndarray:
    row = np.zeros(1, dtype=_WCS_DTYPE)
    for k in _WCS_DTYPE.names:
        row[k][0] = params.get(k, 0)
    return row


# ---------------------------------------------------------------------------
# Zarr store helpers
# ---------------------------------------------------------------------------


def _open_or_create_tile_store(
    tile_path: Path,
    n_bands: int,
    height: int,
    width: int,
    dtype: np.dtype = _DEFAULT_DTYPE,
) -> zarr.Group:
    """
    Open an existing Zarr v3 tile group or create one with the correct shape.

    Arrays are created with unlimited first axis (``N_sources``) using Zarr's
    append capability; chunks = ``(1, n_bands, height, width)``.
    """
    store = zarr.storage.LocalStore(str(tile_path))
    if tile_path.exists() and (tile_path / "zarr.json").exists():
        return zarr.open_group(store=store, mode="a", zarr_format=3)

    root = zarr.open_group(store=store, mode="w", zarr_format=3)

    blosc = zarr.codecs.BloscCodec(
        cname="zstd",
        clevel=_ZSTD_LEVEL,
        shuffle=zarr.codecs.BloscShuffle.bitshuffle,
    )
    shard_n = _CHUNKS_PER_SHARD
    chunk_img = (1, n_bands, height, width)
    shard_img = (shard_n, n_bands, height, width)

    root.create_array(
        "images",
        shape=(0, n_bands, height, width),
        chunks=chunk_img,
        shards=shard_img,
        dtype=dtype,
        compressors=blosc,
        fill_value=np.nan,
    )
    root.create_array("source_id", shape=(0,), chunks=(4096,), dtype=np.int64, fill_value=-1)

    # WCS stored as 1-D array of structured dtype; Zarr stores it as uint8 bytes
    root.create_array(
        "wcs",
        shape=(0,),
        chunks=(512,),
        dtype="|V" + str(_WCS_DTYPE.itemsize),
        fill_value=b"\x00" * _WCS_DTYPE.itemsize,
    )

    return root


def _filter_tile_records_duplicates(
    tile_records: list[CutoutRecord],
    existing_source_ids: set[int],
    on_duplicate: Literal["append", "error", "skip"],
) -> list[CutoutRecord]:
    """Enforce duplicate policy for one tile's incoming batch vs Zarr contents."""
    seen_batch: set[int] = set()
    for r in tile_records:
        if r.source_id in seen_batch:
            raise ValueError(
                f"Duplicate source_id {r.source_id} within a single ingest batch for one tile"
            )
        seen_batch.add(r.source_id)

    if on_duplicate == "append":
        return tile_records
    if on_duplicate == "error":
        for r in tile_records:
            if r.source_id in existing_source_ids:
                raise ValueError(
                    f"source_id {r.source_id} already exists in this tile's Zarr; "
                    f"use on_duplicate_source_id='skip' or 'append'."
                )
        return tile_records
    return [r for r in tile_records if r.source_id not in existing_source_ids]


# ---------------------------------------------------------------------------
# Core ingest
# ---------------------------------------------------------------------------


def ingest_cutouts_from_fits(
    source_path: Path | str,
    output_root: Path | str,
    survey_name: str,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    image_hdu_index: int = 0,
    band_axis: int | None = None,
    band_names: list[str] | None = None,
    norder: int = 5,
    dtype: np.dtype | type = _DEFAULT_DTYPE,
    on_duplicate_source_id: Literal["append", "error", "skip"] = "append",
) -> dict[int, int]:
    """
    Ingest cutout images from a FITS file (one HDU = one source or one MEF
    with multiple extensions = multiple bands per source).

    Parameters
    ----------
    source_path:
        FITS file containing cutout images.
    output_root:
        Data lake root.
    survey_name:
        Survey identifier.
    ra_col / dec_col:
        Header keywords that store RA/Dec of the cutout centre.
    image_hdu_index:
        Primary HDU index (0-based).  For multi-band MEF files, set
        ``band_axis`` to the extension axis.
    band_axis:
        If the image data has a band axis (e.g. shape = [B,H,W]), specify
        which axis index it is.  ``None`` means a 2-D (single-band) image.
    band_names:
        Optional list of band names for Zarr attributes.
    norder:
        HEALPix partitioning order.
    dtype:
        Storage dtype (float32 default).
    on_duplicate_source_id:
        ``append`` (default) always appends new rows (may duplicate ``source_id``
        in a tile if re-run).  ``error`` raises if any incoming ``source_id`` is
        already present.  ``skip`` drops conflicting rows and only appends new IDs.

    Returns
    -------
    dict mapping ``source_id → cutout_index`` for this file.
    """
    source_path = Path(source_path)
    output_root = Path(output_root)
    dtype = np.dtype(dtype)

    index_map: dict[int, int] = {}

    with fits.open(str(source_path), memmap=True) as hdul:
        records = _extract_records_from_hdul(hdul, ra_col, dec_col, image_hdu_index, band_axis, dtype)

    if not records:
        log.warning("No records extracted from %s", source_path.name)
        return index_map

    # Group by HEALPix tile
    tile_groups: dict[int, list[CutoutRecord]] = {}
    for rec in records:
        pix = int(assign_healpix(np.array([rec.ra]), np.array([rec.dec]), norder)[0])
        tile_groups.setdefault(pix, []).append(rec)

    n_bands = records[0].image.shape[0]
    h, w = records[0].image.shape[1], records[0].image.shape[2]

    for npix, tile_records in tile_groups.items():
        tile_dir = output_root / "cutouts" / survey_name / healpix_dir(norder, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path = tile_dir / f"Npix={npix}.zarr"

        root = _open_or_create_tile_store(tile_path, n_bands, h, w, dtype)
        images_arr = root["images"]
        sid_arr = root["source_id"]
        wcs_arr = root["wcs"]

        n_existing = int(sid_arr.shape[0])
        existing: set[int] = set()
        if n_existing > 0:
            existing = set(np.asarray(sid_arr[:]).tolist())

        tile_records = _filter_tile_records_duplicates(
            tile_records, existing, on_duplicate_source_id,
        )
        if not tile_records:
            continue

        start_idx = images_arr.shape[0]

        batch_images = np.stack([r.image for r in tile_records], axis=0).astype(dtype)
        batch_ids = np.array([r.source_id for r in tile_records], dtype=np.int64)
        batch_wcs_raw = np.concatenate(
            [_wcs_params_to_structured(r.wcs_params) for r in tile_records], axis=0
        )
        # Store WCS as raw bytes
        batch_wcs_bytes = batch_wcs_raw.view("|V" + str(_WCS_DTYPE.itemsize))

        images_arr.append(batch_images)
        sid_arr.append(batch_ids)
        wcs_arr.append(batch_wcs_bytes)

        for local_i, rec in enumerate(tile_records):
            index_map[rec.source_id] = start_idx + local_i

        # Store band names as attribute if provided
        if band_names and "band_names" not in root.attrs:
            root.attrs["band_names"] = band_names

    # Write cutout_info.json at survey root
    cutout_root = output_root / "cutouts" / survey_name
    _write_cutout_info(
        cutout_root,
        survey_name,
        norder,
        n_bands,
        h,
        w,
        band_names or [],
        dtype=dtype,
        on_duplicate_source_id=on_duplicate_source_id,
    )

    return index_map


def _extract_records_from_hdul(
    hdul: fits.HDUList,
    ra_col: str,
    dec_col: str,
    image_hdu_index: int,
    band_axis: int | None,
    dtype: np.dtype,
) -> list[CutoutRecord]:
    """Extract CutoutRecord list from an open HDUList."""
    records: list[CutoutRecord] = []

    for hdu_idx, hdu in enumerate(hdul):
        if not hasattr(hdu, "data") or hdu.data is None:
            continue
        if image_hdu_index != 0 and hdu_idx != image_hdu_index:
            continue

        header = hdu.header
        ra = float(header.get(ra_col, header.get("RA_TARG", header.get("CRVAL1", 0.0))))
        dec = float(header.get(dec_col, header.get("DEC_TARG", header.get("CRVAL2", 0.0))))
        source_id = int(header.get("SOURCE_ID", header.get("OBJ_ID", hdu_idx)))

        data = np.array(hdu.data, dtype=np.float64)

        if data.ndim == 2:
            # Single band: add band axis
            image = data[np.newaxis, :, :].astype(dtype)
        elif data.ndim == 3:
            if band_axis is None:
                band_axis = 0
            image = np.moveaxis(data, band_axis, 0).astype(dtype)
        else:
            log.warning("Skipping HDU %d: unsupported data shape %s", hdu_idx, data.shape)
            continue

        wcs_params = _extract_wcs_params(header)
        records.append(CutoutRecord(
            source_id=source_id,
            ra=ra,
            dec=dec,
            image=image,
            wcs_params=wcs_params,
        ))

    return records


def ingest_cutouts_batch(
    source_paths: Sequence[Path | str],
    output_root: Path | str,
    survey_name: str,
    **kwargs: Any,
) -> dict[int, int]:
    """Ingest multiple FITS files; returns merged source_id → cutout_index map."""
    merged: dict[int, int] = {}
    for path in source_paths:
        idx_map = ingest_cutouts_from_fits(
            source_path=path,
            output_root=output_root,
            survey_name=survey_name,
            **kwargs,
        )
        merged.update(idx_map)
    return merged


def _write_cutout_info(
    cutout_root: Path,
    survey_name: str,
    norder: int,
    n_bands: int,
    height: int,
    width: int,
    band_names: list[str],
    *,
    dtype: np.dtype,
    on_duplicate_source_id: str,
) -> None:
    info = {
        "survey_name": survey_name,
        "hats_order": norder,
        "n_bands": n_bands,
        "height": height,
        "width": width,
        "band_names": band_names,
        "dtype": np.dtype(dtype).name,
        "on_duplicate_source_id": on_duplicate_source_id,
        "chunk_shape": [1, n_bands, height, width],
        "chunks_per_shard": _CHUNKS_PER_SHARD,
        "compression": "blosc-zstd-bitshuffle",
        "zarr_format": 3,
        "schema_version": "1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    cutout_root.mkdir(parents=True, exist_ok=True)
    with open(cutout_root / "cutout_info.json", "w") as fh:
        json.dump(info, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    from ..cli_utils import (
        config_option,
        configure_warning_filters,
        load_optional_config,
        pick,
        require_output_root,
    )

    @click.command("dl-ingest-cutouts")
    @click.argument("source_path", type=click.Path(exists=True, path_type=Path))
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option("--survey", "survey_name", required=True)
    @click.option("--ra-col", default="RA", show_default=True)
    @click.option("--dec-col", default="DEC", show_default=True)
    @click.option("--image-hdu", "image_hdu_index", default=0, type=int, show_default=True)
    @click.option("--band-axis", default=None, type=int)
    @click.option("--norder", default=None, type=int,
                  help="HEALPix order (overrides config; default 5).")
    @click.option(
        "--band-names",
        default=None,
        help="Comma-separated band names (stored on the Zarr group as band_names).",
    )
    @click.option(
        "--dtype",
        default="float32",
        show_default=True,
        help="NumPy dtype name for stored image arrays.",
    )
    @click.option(
        "--on-duplicate",
        type=click.Choice(["append", "error", "skip"]),
        default="append",
        show_default=True,
        help="If source_id already exists in a tile, append (default), raise, or skip rows.",
    )
    @click.option(
        "--update-catalog/--no-update-catalog", default=True, show_default=True,
        help="Patch _cutout_index in the Parquet catalog after ingest "
             "(skipped silently if no catalog exists for this survey).",
    )
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        source_path: Path,
        output_root: Path | None,
        config_path: Path | None,
        survey_name: str,
        ra_col: str,
        dec_col: str,
        image_hdu_index: int,
        band_axis: int | None,
        norder: int | None,
        band_names: str | None,
        dtype: str,
        on_duplicate: str,
        update_catalog: bool,
        verbose: bool,
    ) -> None:
        """Ingest FITS cutouts into sharded Zarr v3 stacks.

        OUTPUT_ROOT is optional when a lake config is available
        (via --config or $DATA_LAKE_CONFIG); in that case it defaults to
        ``<lake.root>/<paths.cutouts>``.
        """
        import numpy as np

        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        configure_warning_filters()
        cfg = load_optional_config(config_path)
        resolved_output = require_output_root(output_root, cfg, kind="cutouts")
        resolved_norder = pick(norder, cfg.partitioning.hats_order if cfg else None, 5)

        bn = [x.strip() for x in band_names.split(",")] if band_names else None

        index_map = ingest_cutouts_from_fits(
            source_path=source_path,
            output_root=resolved_output,
            survey_name=survey_name,
            ra_col=ra_col,
            dec_col=dec_col,
            image_hdu_index=image_hdu_index,
            band_axis=band_axis,
            band_names=bn,
            norder=resolved_norder,
            dtype=np.dtype(dtype),
            on_duplicate_source_id=on_duplicate,  # type: ignore[arg-type]
        )

        if update_catalog and index_map:
            try:
                from data_lake.ingest.update_catalog_indices import update_index_column
                n_modified = update_index_column(
                    lake_root=resolved_output,
                    survey_name=survey_name,
                    source_id_to_index=index_map,
                    kind="cutout",
                    norder=resolved_norder,
                )
                click.echo(f"Patched _cutout_index in {n_modified} catalog tile(s).")
            except FileNotFoundError:
                log.info(
                    "No catalog found for survey %r — skipping _cutout_index patch.",
                    survey_name,
                )

except ImportError:
    cli = None  # type: ignore[assignment]
