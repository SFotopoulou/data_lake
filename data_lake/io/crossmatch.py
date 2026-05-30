"""
crossmatch – build and query a precomputed HATS association catalog.

The crossmatch catalog links source_ids from two surveys using a sky-based
nearest-neighbour match.  The result is stored as HATS-partitioned Parquet
under::

    <lake_root>/catalogs/crossmatch/<surveyA>_x_<surveyB>/
        Norder=<N>/Dir=<D>/Npix=<P>.parquet
        catalog_info.json

Each Parquet row contains::

    source_id_a  int64  – source_id from surveyA
    source_id_b  int64  – source_id from surveyB
    sep_arcsec   float32 – angular separation in arcseconds
    _healpix_norder<N>  int64 – tile of surveyA source (determines partition)

Usage
-----
>>> from data_lake.io.crossmatch import build_crossmatch, CrossmatchAccessor
>>> build_crossmatch(
...     lake_root="/data/lake",
...     survey_a="des_dr2",
...     survey_b="kids_dr4",
...     radius_arcsec=1.0,
... )
>>> xm = CrossmatchAccessor("/data/lake", "des_dr2", "kids_dr4")
>>> matches = xm.get_matches(source_id_a=12345678)
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import healpy as hp
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.io.catalog import CatalogAccessor, ReturnFormat
from data_lake.ingest.fits_to_parquet import healpix_dir, _ZSTD_LEVEL

log = logging.getLogger(__name__)

_NPIX_FROM_PATH = re.compile(r"Npix=(\d+)\.parquet$", re.IGNORECASE)


@dataclass(frozen=True)
class CrossmatchResult:
    """Summary statistics from :func:`build_crossmatch`."""

    survey_a: str
    survey_b: str
    crossmatch_name: str
    output_root: Path
    n_tiles_written: int
    n_match_rows: int
    radius_arcsec: float
    norder: int
    elapsed_s: float
    n_workers: int = 1


@dataclass(frozen=True)
class CrossmatchSurveySettings:
    """Per-survey sky columns and HEALPix order."""

    ra_col: str
    dec_col: str
    norder: int


@dataclass(frozen=True)
class CrossmatchSettings:
    """Resolved cross-match configuration for both surveys."""

    survey_a: CrossmatchSurveySettings
    survey_b: CrossmatchSurveySettings


@dataclass(frozen=True)
class CrossmatchTileConfig:
    """Pickle-friendly config for one survey-A tile (parallel workers)."""

    lake_root: str
    survey_a: str
    survey_b: str
    npix_a: int
    norder_a: int
    norder_b: int
    ra_col_a: str
    dec_col_a: str
    ra_col_b: str
    dec_col_b: str
    radius_arcsec: float
    out_root: str


@dataclass(frozen=True)
class CrossmatchTileResult:
    npix_a: int
    n_match_rows: int
    n_tiles_written: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def crossmatch_name(survey_a: str, survey_b: str) -> str:
    return f"{survey_a}_x_{survey_b}"


def crossmatch_root(lake_root: Path | str, survey_a: str, survey_b: str) -> Path:
    return Path(lake_root) / "catalogs" / "crossmatch" / crossmatch_name(survey_a, survey_b)


def _sky_columns_from_catalog_info(catalog_root: Path) -> tuple[str, str, int]:
    info_path = catalog_root / "catalog_info.json"
    if not info_path.is_file():
        return "ra", "dec", 5
    with open(info_path) as fh:
        info = json.load(fh)
    return (
        str(info.get("ra_column", "ra")),
        str(info.get("dec_column", "dec")),
        int(info.get("hats_order", 5)),
    )


def resolve_crossmatch_settings(
    lake_root: Path | str,
    survey_a: str,
    survey_b: str,
    *,
    ra_col_a: str | None = None,
    dec_col_a: str | None = None,
    norder_a: int | None = None,
    ra_col_b: str | None = None,
    dec_col_b: str | None = None,
    norder_b: int | None = None,
    ra_col: str | None = None,
    dec_col: str | None = None,
    norder: int | None = None,
) -> CrossmatchSettings:
    """Resolve per-survey RA/Dec columns and HEALPix order from catalog_info.json.

    Legacy ``ra_col`` / ``dec_col`` / ``norder`` apply to survey A only.
    """
    root = Path(lake_root)
    ra_a, dec_a, order_a = _sky_columns_from_catalog_info(root / "catalogs" / survey_a)
    ra_b, dec_b, order_b = _sky_columns_from_catalog_info(root / "catalogs" / survey_b)

    settings = CrossmatchSettings(
        survey_a=CrossmatchSurveySettings(
            ra_col=ra_col_a or ra_col or ra_a,
            dec_col=dec_col_a or dec_col or dec_a,
            norder=norder_a if norder_a is not None else (norder if norder is not None else order_a),
        ),
        survey_b=CrossmatchSurveySettings(
            ra_col=ra_col_b or ra_b,
            dec_col=dec_col_b or dec_b,
            norder=norder_b if norder_b is not None else order_b,
        ),
    )
    if settings.survey_a.norder != settings.survey_b.norder:
        log.info(
            "Cross-match %s (Norder=%d) × %s (Norder=%d); output partitioned at survey-A order.",
            survey_a,
            settings.survey_a.norder,
            survey_b,
            settings.survey_b.norder,
        )
    if settings.survey_a.ra_col != settings.survey_b.ra_col or settings.survey_a.dec_col != settings.survey_b.dec_col:
        log.info(
            "Sky columns: %s (%r, %r) × %s (%r, %r)",
            survey_a,
            settings.survey_a.ra_col,
            settings.survey_a.dec_col,
            survey_b,
            settings.survey_b.ra_col,
            settings.survey_b.dec_col,
        )
    return settings


def resolve_crossmatch_sky_columns(
    lake_root: Path | str,
    survey_a: str,
    survey_b: str,
    *,
    ra_col: str | None = None,
    dec_col: str | None = None,
    norder: int | None = None,
) -> tuple[str, str, int]:
    """Legacy helper returning survey-A settings only."""
    s = resolve_crossmatch_settings(
        lake_root,
        survey_a,
        survey_b,
        ra_col=ra_col,
        dec_col=dec_col,
        norder=norder,
    )
    return s.survey_a.ra_col, s.survey_a.dec_col, s.survey_a.norder


def iter_populated_tile_npixels(catalog_root: Path, *, norder: int | None = None) -> list[int]:
    """Return sorted HEALPix pixel indices that have ``Npix=*.parquet`` tiles."""
    npixels: set[int] = set()
    if norder is not None:
        search_roots = [catalog_root / f"Norder={norder}"]
    else:
        search_roots = [p for p in catalog_root.iterdir() if p.is_dir() and p.name.startswith("Norder=")]
        if not search_roots:
            search_roots = [catalog_root]

    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("Npix=*.parquet"):
            match = _NPIX_FROM_PATH.search(path.name)
            if match:
                npixels.add(int(match.group(1)))
    return sorted(npixels)


def survey_b_pixels_for_tile(
    nside_a: int,
    npix_a: int,
    radius_rad: float,
    *,
    nside_b: int | None = None,
) -> list[int]:
    """Return survey-B HEALPix pixels to load when matching survey-A tile *npix_a*.

    Pixel indices are at **survey-B** resolution (``nside_b``).  The search region
    is derived from the geometry of pixel A at ``nside_a`` (boundary vertices,
    edge neighbours, match radius) — not from source centroids.

    When ``nside_b != nside_a``, sky coverage is mapped to the finer/coarser B grid
    via :func:`healpy.query_disc` at ``nside_b``.
    """
    nside_b = nside_b or nside_a
    npix_a = int(npix_a)
    pixels: set[int] = set()

    theta, phi = hp.pix2ang(nside_a, npix_a, nest=True)
    vec_center = hp.ang2vec(theta, phi)
    verts = hp.boundaries(nside_a, npix_a, nest=True)

    if radius_rad > 0.0:
        for k in range(verts.shape[1]):
            for p in hp.query_disc(nside_b, verts[:, k], radius_rad, nest=True, inclusive=True):
                pixels.add(int(p))

    cosines = np.clip(verts.T @ vec_center, -1.0, 1.0)
    pixel_ext_rad = float(np.arccos(cosines.min()))
    search_rad = pixel_ext_rad + radius_rad
    for p in hp.query_disc(nside_b, vec_center, search_rad, nest=True, inclusive=True):
        pixels.add(int(p))

    if nside_a == nside_b:
        pixels.add(npix_a)
        for n in hp.get_all_neighbours(nside_a, npix_a, nest=True):
            if n >= 0:
                pixels.add(int(n))
    else:
        npix_b_centre = int(hp.ang2pix(nside_b, theta, phi, nest=True))
        pixels.add(npix_b_centre)
        for n in hp.get_all_neighbours(nside_b, npix_b_centre, nest=True):
            if n >= 0:
                pixels.add(int(n))

    return sorted(pixels)


def tile_search_cone(
    nside_a: int,
    npix_a: int,
    radius_rad: float,
) -> tuple[float, float, float]:
    """Return ``(ra_deg, dec_deg, search_radius_deg)`` covering survey-A tile *npix_a*.

    The search disc spans the pixel extent (centre to farthest vertex) plus the
    match radius, so every source in tile A and every survey-B candidate within
    ``radius_rad`` of any A source is included.
    """
    theta, phi = hp.pix2ang(nside_a, int(npix_a), nest=True)
    ra_deg = float(np.degrees(phi))
    dec_deg = float(90.0 - np.degrees(theta))

    search_deg = float(np.degrees(_tile_search_radius_rad(nside_a, npix_a, radius_rad)))
    return ra_deg, dec_deg, search_deg


def healpix_pixels_covering_tile(
    nside_a: int,
    npix_a: int,
    radius_rad: float,
    *,
    nside_b: int,
) -> list[int]:
    """Survey-B HEALPix pixels overlapping survey-A tile *npix_a* (incl. match radius)."""
    theta, phi = hp.pix2ang(nside_a, int(npix_a), nest=True)
    vec = hp.ang2vec(theta, phi)
    search_rad = _tile_search_radius_rad(nside_a, npix_a, radius_rad)
    return sorted(int(p) for p in hp.query_disc(
        nside_b, vec, search_rad, nest=True, inclusive=True,
    ))


def _source_bbox_deg(
    ra: np.ndarray,
    dec: np.ndarray,
    margin_deg: float,
) -> tuple[float, float, float, float]:
    """RA/Dec bounding box (degrees) around sources with angular margin."""
    dec_min = float(np.min(dec) - margin_deg)
    dec_max = float(np.max(dec) + margin_deg)
    cos_dec = max(float(np.cos(np.radians(np.mean(dec)))), 1e-6)
    ra_pad = margin_deg / cos_dec
    ra_min = float(np.min(ra) - ra_pad)
    ra_max = float(np.max(ra) + ra_pad)
    if ra_max - ra_min >= 360.0:
        return 0.0, 360.0, dec_min, dec_max
    return ra_min, ra_max, dec_min, dec_max


def _tile_search_radius_rad(nside_a: int, npix_a: int, radius_rad: float) -> float:
    """Angular radius (rad) covering survey-A pixel extent plus match radius."""
    theta, phi = hp.pix2ang(nside_a, int(npix_a), nest=True)
    vec = hp.ang2vec(theta, phi)
    verts = hp.boundaries(nside_a, int(npix_a), nest=True)
    cosines = np.clip(verts.T @ vec, -1.0, 1.0)
    return float(np.arccos(cosines.min()) + radius_rad)


def filter_survey_a_tiles_overlapping_survey_b(
    npix_a_list: Sequence[int],
    *,
    nside_a: int,
    nside_b: int,
    b_populated: set[int],
    radius_rad: float,
) -> list[int]:
    """Return survey-A tile indices that may contain matches in survey B.

    A tile is kept when the same search disc used for survey-B cone queries
    (pixel extent + match radius) intersects at least one populated survey-B
    HEALPix pixel.  Works across different ``nside_a`` / ``nside_b``.
    """
    if not npix_a_list or not b_populated:
        return []

    npix_a_set = {int(p) for p in npix_a_list}
    b_pop = {int(p) for p in b_populated}

    if len(b_pop) <= len(npix_a_set):
        keep: set[int] = set()
        for npix_b in b_pop:
            theta, phi = hp.pix2ang(nside_b, npix_b, nest=True)
            vec = hp.ang2vec(theta, phi)
            verts = hp.boundaries(nside_b, npix_b, nest=True)
            cosines = np.clip(verts.T @ vec, -1.0, 1.0)
            search_rad = float(np.arccos(cosines.min()) + radius_rad)
            for npix_a in hp.query_disc(
                nside_a, vec, search_rad, nest=True, inclusive=True,
            ):
                if int(npix_a) in npix_a_set:
                    keep.add(int(npix_a))
        return sorted(keep)

    keep_list: list[int] = []
    for npix_a in sorted(npix_a_set):
        tiles_b = healpix_pixels_covering_tile(
            nside_a, npix_a, radius_rad, nside_b=nside_b,
        )
        if b_pop.intersection(tiles_b):
            keep_list.append(npix_a)
    return keep_list


def _catalog_ids_to_int64(values) -> np.ndarray:
    """Coerce catalog ID column values to int64 for cross-match output."""
    from data_lake.ingest.fits_to_parquet import normalize_object_id

    return np.asarray([normalize_object_id(v) for v in values], dtype=np.int64)


def _crossmatch_one_tile(
    *,
    npix_a: int,
    norder_a: int,
    norder_b: int,
    ra_col_a: str,
    dec_col_a: str,
    ra_col_b: str,
    dec_col_b: str,
    radius_deg: float,
    radius_rad: float,
    out_root: Path,
    acc_a: CatalogAccessor,
    acc_b: CatalogAccessor,
) -> int:
    """Match one survey-A tile; write Parquet. Returns number of match rows."""
    nside_a = hp.order2nside(norder_a)
    nside_b = hp.order2nside(norder_b)
    hp_col = f"_healpix_norder{norder_a}"
    id_col_a = acc_a.source_id_column
    id_col_b = acc_b.source_id_column
    cols_a = [id_col_a, ra_col_a, dec_col_a]
    cols_b = [id_col_b, ra_col_b, dec_col_b]

    out_dir = out_root / healpix_dir(norder_a, npix_a)
    out_file = out_dir / f"Npix={npix_a}.parquet"

    df_a = acc_a.sources_in_tile(npix_a, columns=cols_a, fmt="polars")
    if df_a.is_empty():
        return 0

    ra_a = df_a[ra_col_a].to_numpy().astype(np.float64)
    dec_a = df_a[dec_col_a].to_numpy().astype(np.float64)
    ids_a = _catalog_ids_to_int64(df_a[id_col_a].to_list())

    b_pixels = healpix_pixels_covering_tile(
        nside_a, npix_a, radius_rad, nside_b=nside_b,
    )
    ra_min, ra_max, dec_min, dec_max = _source_bbox_deg(ra_a, dec_a, radius_deg)
    df_b = acc_b.sources_in_healpix_pixels(
        b_pixels,
        columns=cols_b,
        fmt="polars",
        ra_col=ra_col_b,
        dec_col=dec_col_b,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=dec_min,
        dec_max=dec_max,
    )
    if df_b.is_empty():
        return 0
    ra_b = df_b[ra_col_b].to_numpy().astype(np.float64)
    dec_b = df_b[dec_col_b].to_numpy().astype(np.float64)
    ids_b = _catalog_ids_to_int64(df_b[id_col_b].to_list())

    matched_a, matched_b, sep = _match_sky(
        ra_a, dec_a, ids_a, ra_b, dec_b, ids_b, radius_deg
    )
    if matched_a.size == 0:
        return 0

    table = pa.table({
        "source_id_a": pa.array(matched_a, type=pa.int64()),
        "source_id_b": pa.array(matched_b, type=pa.int64()),
        "sep_arcsec": pa.array((sep * 3600.0).astype(np.float32), type=pa.float32()),
        hp_col: pa.array(np.full(len(matched_a), npix_a, dtype=np.int64), type=pa.int64()),
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        str(out_file),
        compression="zstd",
        compression_level=_ZSTD_LEVEL,
        write_statistics=True,
    )
    return table.num_rows


def _crossmatch_tile_worker(config: CrossmatchTileConfig) -> CrossmatchTileResult:
    from data_lake.cli_utils import apply_parallel_worker_logging_after_heavy_imports

    apply_parallel_worker_logging_after_heavy_imports()
    try:
        lake_root = Path(config.lake_root)
        out_root = Path(config.out_root)
        radius_deg = config.radius_arcsec / 3600.0
        radius_rad = np.radians(radius_deg)

        with CatalogAccessor(lake_root, config.survey_a, norder=config.norder_a) as acc_a, \
             CatalogAccessor(lake_root, config.survey_b, norder=config.norder_b) as acc_b:
            n_rows = _crossmatch_one_tile(
                npix_a=config.npix_a,
                norder_a=config.norder_a,
                norder_b=config.norder_b,
                ra_col_a=config.ra_col_a,
                dec_col_a=config.dec_col_a,
                ra_col_b=config.ra_col_b,
                dec_col_b=config.dec_col_b,
                radius_deg=radius_deg,
                radius_rad=radius_rad,
                out_root=out_root,
                acc_a=acc_a,
                acc_b=acc_b,
            )
        return CrossmatchTileResult(
            config.npix_a,
            n_rows,
            1 if n_rows > 0 else 0,
        )
    except Exception as exc:
        log.exception("Cross-match tile %d failed", config.npix_a)
        return CrossmatchTileResult(config.npix_a, 0, 0, str(exc))


def _pending_crossmatch_tiles(
    tile_npixels: list[int],
    out_root: Path,
    norder: int,
    overwrite: bool,
) -> tuple[list[int], int, int]:
    """Split into tiles to compute vs already-written (resume). Returns (pending, n_rows, n_written)."""
    pending: list[int] = []
    n_match_rows = 0
    n_tiles_written = 0
    for npix_a in tile_npixels:
        out_file = out_root / healpix_dir(norder, npix_a) / f"Npix={npix_a}.parquet"
        if out_file.exists() and not overwrite:
            try:
                n_match_rows += pq.read_metadata(str(out_file)).num_rows
                n_tiles_written += 1
            except Exception:
                pass
            continue
        pending.append(npix_a)
    return pending, n_match_rows, n_tiles_written


def build_crossmatch(
    lake_root: Path | str,
    survey_a: str,
    survey_b: str,
    radius_arcsec: float = 1.0,
    norder: int | None = None,
    ra_col: str | None = None,
    dec_col: str | None = None,
    overwrite: bool = False,
    *,
    norder_a: int | None = None,
    norder_b: int | None = None,
    ra_col_a: str | None = None,
    dec_col_a: str | None = None,
    ra_col_b: str | None = None,
    dec_col_b: str | None = None,
    populated_tiles_only: bool = True,
    show_progress: bool = False,
    n_workers: int = 1,
) -> CrossmatchResult:
    """
    Build a precomputed cross-match between two surveys.

    Uses tile-by-tile nearest-neighbour matching.  Survey-A tiles are limited to
    those overlapping the survey-B populated footprint (see
    :func:`filter_survey_a_tiles_overlapping_survey_b`).  Survey-B candidates
    per tile are read from the matching survey-B HEALPix tile Parquet files only,
    with an RA/Dec bounding-box filter around survey-A sources.

    Parameters
    ----------
    lake_root:
        Data lake root directory.
    survey_a / survey_b:
        Survey names to cross-match.  surveyA defines the partition (its tile
        pixel is used to assign the output row to a Parquet partition).
    radius_arcsec:
        Maximum matching radius in arcseconds.
    norder:
        HEALPix order for survey A (legacy alias for ``norder_a``).
    ra_col / dec_col:
        Sky columns for survey A (legacy aliases for ``ra_col_a`` / ``dec_col_a``).
    norder_a / norder_b:
        Per-survey HEALPix order (default: each catalog's ``catalog_info.json``).
    ra_col_a / dec_col_a / ra_col_b / dec_col_b:
        Per-survey sky column overrides.
    overwrite:
        If False (default), skip existing output tiles.
    populated_tiles_only:
        When True (default), iterate only HEALPix tiles present in survey A
        instead of the full ``12 * 4**norder`` pixel range.  Tiles with no
        overlap with survey-B populated footprint are always skipped.
    show_progress:
        Show a tqdm progress bar over survey-A tiles when available.
    n_workers:
        Parallel worker processes for disjoint survey-A tiles (default 1).

    Returns
    -------
    CrossmatchResult
        Tile and row counts plus output location.
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")

    lake_root = Path(lake_root)
    settings = resolve_crossmatch_settings(
        lake_root,
        survey_a,
        survey_b,
        ra_col_a=ra_col_a,
        dec_col_a=dec_col_a,
        norder_a=norder_a if norder_a is not None else norder,
        ra_col_b=ra_col_b,
        dec_col_b=dec_col_b,
        norder_b=norder_b,
        ra_col=ra_col,
        dec_col=dec_col,
        norder=norder,
    )
    norder_a = settings.survey_a.norder
    norder_b = settings.survey_b.norder

    xm_name = crossmatch_name(survey_a, survey_b)
    out_root = crossmatch_root(lake_root, survey_a, survey_b)

    catalog_root_a = lake_root / "catalogs" / survey_a
    catalog_root_b = lake_root / "catalogs" / survey_b
    radius_rad = np.radians(radius_arcsec / 3600.0)
    nside_a = hp.order2nside(norder_a)
    nside_b = hp.order2nside(norder_b)

    if populated_tiles_only:
        tile_npixels = iter_populated_tile_npixels(catalog_root_a, norder=norder_a)
        log.info(
            "Cross-match %s × %s: %d populated tile(s) in survey A at Norder=%d",
            survey_a,
            survey_b,
            len(tile_npixels),
            norder_a,
        )
    else:
        tile_npixels = list(range(hp.nside2npix(nside_a)))
        log.info(
            "Cross-match %s × %s: scanning all %d HEALPix pixels at Norder=%d",
            survey_a,
            survey_b,
            len(tile_npixels),
            norder_a,
        )

    b_populated = set(iter_populated_tile_npixels(catalog_root_b, norder=norder_b))
    n_a_before = len(tile_npixels)
    tile_npixels = filter_survey_a_tiles_overlapping_survey_b(
        tile_npixels,
        nside_a=nside_a,
        nside_b=nside_b,
        b_populated=b_populated,
        radius_rad=radius_rad,
    )
    if n_a_before != len(tile_npixels):
        log.info(
            "Sky overlap: %d / %d survey-A tile(s) overlap survey-B footprint "
            "(%d populated B tile(s) at Norder=%d)",
            len(tile_npixels),
            n_a_before,
            len(b_populated),
            norder_b,
        )
    if not tile_npixels:
        log.warning(
            "Cross-match %s × %s: no survey-A tiles overlap survey-B populated footprint",
            survey_a,
            survey_b,
        )

    pending, n_match_rows, n_tiles_written = _pending_crossmatch_tiles(
        tile_npixels, out_root, norder_a, overwrite,
    )
    if n_workers > 1:
        log.info(
            "Cross-match %s × %s: %d tile(s) to compute with %d workers (%d skipped)",
            survey_a,
            survey_b,
            len(pending),
            n_workers,
            len(tile_npixels) - len(pending),
        )

    t0 = time.perf_counter()
    failures: list[str] = []

    if n_workers == 1:
        radius_deg = radius_arcsec / 3600.0
        iterator: Iterable[int] = pending
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(pending, unit="tile", desc=f"{survey_a}×{survey_b}")
            except ImportError:
                pass

        with CatalogAccessor(lake_root, survey_a, norder=norder_a) as acc_a, \
             CatalogAccessor(lake_root, survey_b, norder=norder_b) as acc_b:
            log.info(
                "ID columns: %s (%r) × %s (%r)",
                survey_a,
                acc_a.source_id_column,
                survey_b,
                acc_b.source_id_column,
            )
            for npix_a in iterator:
                n_rows = _crossmatch_one_tile(
                    npix_a=npix_a,
                    norder_a=norder_a,
                    norder_b=norder_b,
                    ra_col_a=settings.survey_a.ra_col,
                    dec_col_a=settings.survey_a.dec_col,
                    ra_col_b=settings.survey_b.ra_col,
                    dec_col_b=settings.survey_b.dec_col,
                    radius_deg=radius_deg,
                    radius_rad=radius_rad,
                    out_root=out_root,
                    acc_a=acc_a,
                    acc_b=acc_b,
                )
                if n_rows > 0:
                    n_tiles_written += 1
                    n_match_rows += n_rows
    else:
        from data_lake.cli_utils import init_parallel_ingest_subprocess

        configs = [
            CrossmatchTileConfig(
                lake_root=str(lake_root),
                survey_a=survey_a,
                survey_b=survey_b,
                npix_a=npix_a,
                norder_a=norder_a,
                norder_b=norder_b,
                ra_col_a=settings.survey_a.ra_col,
                dec_col_a=settings.survey_a.dec_col,
                ra_col_b=settings.survey_b.ra_col,
                dec_col_b=settings.survey_b.dec_col,
                radius_arcsec=radius_arcsec,
                out_root=str(out_root),
            )
            for npix_a in pending
        ]

        pbar = None
        if show_progress:
            try:
                from tqdm.auto import tqdm

                pbar = tqdm(total=len(configs), unit="tile", desc=f"{survey_a}×{survey_b}")
            except ImportError:
                pass

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_parallel_ingest_subprocess,
        ) as pool:
            futures = [pool.submit(_crossmatch_tile_worker, cfg) for cfg in configs]
            for fut in as_completed(futures):
                res = fut.result()
                if pbar is not None:
                    pbar.update(1)
                if res.error:
                    failures.append(f"Npix={res.npix_a}: {res.error}")
                    continue
                n_match_rows += res.n_match_rows
                n_tiles_written += res.n_tiles_written

        if pbar is not None:
            pbar.close()

    elapsed = time.perf_counter() - t0
    if failures:
        log.error("%d tile(s) failed:\n%s", len(failures), "\n".join(failures[:20]))

    log.info(
        "Cross-match %s × %s: %d match row(s) in %d tile(s) in %.1f s",
        survey_a,
        survey_b,
        n_match_rows,
        n_tiles_written,
        elapsed,
    )

    _write_xm_info(
        out_root,
        xm_name,
        survey_a,
        survey_b,
        settings,
        radius_arcsec,
    )
    return CrossmatchResult(
        survey_a=survey_a,
        survey_b=survey_b,
        crossmatch_name=xm_name,
        output_root=out_root,
        n_tiles_written=n_tiles_written,
        n_match_rows=n_match_rows,
        radius_arcsec=radius_arcsec,
        norder=norder_a,
        elapsed_s=elapsed,
        n_workers=n_workers,
    )


def _match_sky(
    ra_a: np.ndarray,
    dec_a: np.ndarray,
    ids_a: np.ndarray,
    ra_b: np.ndarray,
    dec_b: np.ndarray,
    ids_b: np.ndarray,
    radius_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Nearest-neighbour sky match using astropy.

    Returns (ids_a_matched, ids_b_matched, separations_deg).
    Only keeps pairs within ``radius_deg``.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coords_a = SkyCoord(ra=ra_a * u.deg, dec=dec_a * u.deg)
    coords_b = SkyCoord(ra=ra_b * u.deg, dec=dec_b * u.deg)

    idx_b, sep2d, _ = coords_a.match_to_catalog_sky(coords_b)
    sep_deg = sep2d.deg

    mask = sep_deg <= radius_deg
    return ids_a[mask], ids_b[idx_b[mask]], sep_deg[mask]


def _write_xm_info(
    out_root: Path,
    xm_name: str,
    survey_a: str,
    survey_b: str,
    settings: CrossmatchSettings,
    radius_arcsec: float,
) -> None:
    info = {
        "catalog_name": xm_name,
        "catalog_type": "association",
        "survey_a": survey_a,
        "survey_b": survey_b,
        "match_radius_arcsec": radius_arcsec,
        "hats_order": settings.survey_a.norder,
        "survey_a_norder": settings.survey_a.norder,
        "survey_b_norder": settings.survey_b.norder,
        "survey_a_ra_column": settings.survey_a.ra_col,
        "survey_a_dec_column": settings.survey_a.dec_col,
        "survey_b_ra_column": settings.survey_b.ra_col,
        "survey_b_dec_column": settings.survey_b.dec_col,
        "schema_version": "1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "catalog_info.json", "w") as fh:
        json.dump(info, fh, indent=2)


# ---------------------------------------------------------------------------
# Accessor
# ---------------------------------------------------------------------------


class CrossmatchAccessor:
    """
    Query interface for a precomputed cross-match catalog.

    Parameters
    ----------
    lake_root:
        Data lake root.
    survey_a / survey_b:
        The two surveys that were cross-matched.
    norder:
        HEALPix order (read from catalog_info.json if not provided).
    """

    def __init__(
        self,
        lake_root: Path | str,
        survey_a: str,
        survey_b: str,
        norder: int | None = None,
    ) -> None:
        import duckdb
        self.lake_root = Path(lake_root)
        self.survey_a = survey_a
        self.survey_b = survey_b
        xm_name = crossmatch_name(survey_a, survey_b)
        self._xm_root = crossmatch_root(lake_root, survey_a, survey_b)

        info_path = self._xm_root / "catalog_info.json"
        self._info: dict = {}
        if info_path.exists():
            with open(info_path) as fh:
                self._info = json.load(fh)

        self.norder: int = norder if norder is not None else int(self._info.get("hats_order", 5))
        self._con = duckdb.connect(":memory:")
        glob = str(self._xm_root / f"Norder={self.norder}" / "**" / "*.parquet")
        self._con.execute(
            f"CREATE OR REPLACE VIEW xmatch AS SELECT * FROM parquet_scan('{glob}')"
        )

    def get_matches(
        self,
        source_id_a: int | None = None,
        source_id_b: int | None = None,
        max_sep_arcsec: float | None = None,
        fmt: ReturnFormat = "polars",
    ):
        """
        Query cross-match rows.

        At least one of ``source_id_a`` or ``source_id_b`` must be supplied.
        """
        if source_id_a is None and source_id_b is None:
            raise ValueError("Provide at least one of source_id_a or source_id_b.")

        conditions = []
        if source_id_a is not None:
            conditions.append(f"source_id_a = {source_id_a}")
        if source_id_b is not None:
            conditions.append(f"source_id_b = {source_id_b}")
        if max_sep_arcsec is not None:
            conditions.append(f"sep_arcsec <= {max_sep_arcsec}")

        where = " AND ".join(conditions)
        sql = f"SELECT * FROM xmatch WHERE {where}"
        result = self._con.execute(sql).arrow()
        if isinstance(result, pa.RecordBatchReader):
            result = result.read_all()
        return CatalogAccessor._convert(result, fmt)

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    from data_lake.cli_utils import config_option, load_optional_config, require_output_root

    @click.command("dl-crossmatch")
    @click.argument("survey_a")
    @click.argument("survey_b")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option(
        "--radius-arcsec",
        default=1.0,
        show_default=True,
        type=float,
        help="Maximum match radius in arcseconds (nearest neighbour within radius).",
    )
    @click.option(
        "--norder",
        "norder_a",
        type=int,
        default=None,
        help="Survey-A HEALPix order (default: catalog_info.json).",
    )
    @click.option("--norder-b", type=int, default=None, help="Survey-B HEALPix order.")
    @click.option("--ra-col", "ra_col_a", default=None, help="Survey-A RA column override.")
    @click.option("--dec-col", "dec_col_a", default=None, help="Survey-A Dec column override.")
    @click.option("--ra-col-b", default=None, help="Survey-B RA column override.")
    @click.option("--dec-col-b", default=None, help="Survey-B Dec column override.")
    @click.option("--overwrite", is_flag=True, help="Rebuild cross-match tiles that already exist.")
    @click.option(
        "--all-tiles",
        is_flag=True,
        help="Scan every HEALPix pixel (slow); default is survey-A populated tiles only.",
    )
    @click.option("--progress", "show_progress", is_flag=True, help="Show tile progress bar.")
    @click.option(
        "--n-workers",
        default=1,
        show_default=True,
        type=int,
        help="Parallel worker processes (one survey-A tile per task).",
    )
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        survey_a: str,
        survey_b: str,
        output_root: Path | None,
        config_path: Path | None,
        radius_arcsec: float,
        norder_a: int | None,
        norder_b: int | None,
        ra_col_a: str | None,
        dec_col_a: str | None,
        ra_col_b: str | None,
        dec_col_b: str | None,
        overwrite: bool,
        all_tiles: bool,
        show_progress: bool,
        n_workers: int,
        verbose: bool,
    ) -> None:
        """Build an in-lake positional cross-match between two ingested catalogs.

        Output: ``catalogs/crossmatch/<survey_a>_x_<survey_b>/`` (HATS Parquet).
        Each survey uses its own RA/Dec columns and Norder from ``catalog_info.json``
        unless overridden.  Survey A defines the output partition key.
        """
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        cfg = load_optional_config(config_path)
        lake = require_output_root(output_root, cfg, kind="catalogs")

        for name, label in ((survey_a, "survey A"), (survey_b, "survey B")):
            cat_root = lake / "catalogs" / name
            if not cat_root.is_dir():
                raise click.ClickException(f"{label} catalog not found: {cat_root}")

        if n_workers < 1:
            raise click.ClickException("--n-workers must be >= 1")

        result = build_crossmatch(
            lake,
            survey_a,
            survey_b,
            radius_arcsec=radius_arcsec,
            norder_a=norder_a,
            norder_b=norder_b,
            ra_col_a=ra_col_a,
            dec_col_a=dec_col_a,
            ra_col_b=ra_col_b,
            dec_col_b=dec_col_b,
            overwrite=overwrite,
            populated_tiles_only=not all_tiles,
            show_progress=show_progress,
            n_workers=n_workers,
        )
        click.echo(
            f"Cross-match {result.crossmatch_name}: "
            f"{result.n_match_rows:,} match row(s) in {result.n_tiles_written:,} tile(s) "
            f"→ {result.output_root} ({result.elapsed_s:.1f} s, {result.n_workers} process worker(s)"
            + (
                "; DuckDB may use additional CPU threads for parquet I/O"
                if result.n_workers == 1
                else ""
            )
            + ")"
        )

except ImportError:
    cli = None  # type: ignore[misc, assignment]
