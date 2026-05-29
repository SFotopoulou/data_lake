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
from typing import Iterable

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
class CrossmatchTileConfig:
    """Pickle-friendly config for one survey-A tile (parallel workers)."""

    lake_root: str
    survey_a: str
    survey_b: str
    npix_a: int
    norder: int
    ra_col: str
    dec_col: str
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


def resolve_crossmatch_sky_columns(
    lake_root: Path | str,
    survey_a: str,
    survey_b: str,
    *,
    ra_col: str | None = None,
    dec_col: str | None = None,
    norder: int | None = None,
) -> tuple[str, str, int]:
    """Resolve shared RA/Dec column names and HEALPix order for a cross-match."""
    root = Path(lake_root)
    ra_a, dec_a, order_a = _sky_columns_from_catalog_info(root / "catalogs" / survey_a)
    ra_b, dec_b, order_b = _sky_columns_from_catalog_info(root / "catalogs" / survey_b)

    ra = ra_col or ra_a
    dec = dec_col or dec_a
    order = norder if norder is not None else order_a

    if ra_col is None and ra_a != ra_b:
        log.warning(
            "Survey %s uses ra_column=%r but %s uses %r; using %r for both.",
            survey_a,
            ra_a,
            survey_b,
            ra_b,
            ra,
        )
    if dec_col is None and dec_a != dec_b:
        log.warning(
            "Survey %s uses dec_column=%r but %s uses %r; using %r for both.",
            survey_a,
            dec_a,
            survey_b,
            dec_b,
            dec,
        )
    if norder is None and order_a != order_b:
        log.warning(
            "Surveys use different hats_order (%d vs %d); using %d (survey A). "
            "Re-ingest at a common order if matches look wrong.",
            order_a,
            order_b,
            order,
        )
    return ra, dec, order


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
    nside: int,
    npix_a: int,
    radius_rad: float,
) -> list[int]:
    """Return survey-B HEALPix pixels to load when matching survey-A tile *npix_a*.

    Uses the **geometry of pixel A** (boundary vertices and edge neighbours),
    not the centroid of sources in the tile:

    * *npix_a* itself
    * immediate edge neighbours via :func:`healpy.get_all_neighbours`
    * pixels within ``radius_rad`` of each boundary vertex (edge + radius)
    * pixels within ``pixel_angular_extent + radius_rad`` of the pixel centre
      (covers interior points near edges for large match radii)
    """
    npix_a = int(npix_a)
    pixels: set[int] = {npix_a}

    for n in hp.get_all_neighbours(nside, npix_a, nest=True):
        if n >= 0:
            pixels.add(int(n))

    theta, phi = hp.pix2ang(nside, npix_a, nest=True)
    vec_center = hp.ang2vec(theta, phi)
    verts = hp.boundaries(nside, npix_a, nest=True)

    if radius_rad > 0.0:
        for k in range(verts.shape[1]):
            for p in hp.query_disc(nside, verts[:, k], radius_rad, nest=True, inclusive=True):
                pixels.add(int(p))

    cosines = np.clip(verts.T @ vec_center, -1.0, 1.0)
    pixel_ext_rad = float(np.arccos(cosines.min()))
    search_rad = pixel_ext_rad + radius_rad
    for p in hp.query_disc(nside, vec_center, search_rad, nest=True, inclusive=True):
        pixels.add(int(p))

    return sorted(pixels)


def _crossmatch_one_tile(
    *,
    lake_root: Path | str,
    survey_a: str,
    survey_b: str,
    npix_a: int,
    norder: int,
    ra_col: str,
    dec_col: str,
    radius_deg: float,
    radius_rad: float,
    out_root: Path,
    acc_a: CatalogAccessor,
    acc_b: CatalogAccessor,
) -> int:
    """Match one survey-A tile; write Parquet. Returns number of match rows."""
    nside = hp.order2nside(norder)
    hp_col = f"_healpix_norder{norder}"
    cols_a = ["source_id", ra_col, dec_col]
    cols_b = ["source_id", ra_col, dec_col]

    out_dir = out_root / healpix_dir(norder, npix_a)
    out_file = out_dir / f"Npix={npix_a}.parquet"

    df_a = acc_a.sources_in_tile(npix_a, columns=cols_a, fmt="polars")
    if df_a.is_empty():
        return 0

    ra_a = df_a[ra_col].to_numpy().astype(np.float64)
    dec_a = df_a[dec_col].to_numpy().astype(np.float64)
    ids_a = df_a["source_id"].to_numpy().astype(np.int64)

    neighbour_pixels = survey_b_pixels_for_tile(nside, npix_a, radius_rad)

    frames_b = []
    for npix_b in neighbour_pixels:
        df_b_tile = acc_b.sources_in_tile(npix_b, columns=cols_b, fmt="polars")
        if not df_b_tile.is_empty():
            frames_b.append(df_b_tile)

    if not frames_b:
        return 0

    df_b = pl.concat(frames_b).unique(subset=["source_id"], keep="first")
    ra_b = df_b[ra_col].to_numpy().astype(np.float64)
    dec_b = df_b[dec_col].to_numpy().astype(np.float64)
    ids_b = df_b["source_id"].to_numpy().astype(np.int64)

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

        with CatalogAccessor(lake_root, config.survey_a, norder=config.norder) as acc_a, \
             CatalogAccessor(lake_root, config.survey_b, norder=config.norder) as acc_b:
            n_rows = _crossmatch_one_tile(
                lake_root=lake_root,
                survey_a=config.survey_a,
                survey_b=config.survey_b,
                npix_a=config.npix_a,
                norder=config.norder,
                ra_col=config.ra_col,
                dec_col=config.dec_col,
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
    populated_tiles_only: bool = True,
    show_progress: bool = False,
    n_workers: int = 1,
) -> CrossmatchResult:
    """
    Build a precomputed cross-match between two surveys.

    Uses tile-by-tile nearest-neighbour matching.  Survey-B tiles are chosen from
    the HEALPix **geometry** of each survey-A pixel (edge neighbours + boundary
    vertices + match radius) via :func:`survey_b_pixels_for_tile`.

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
        HEALPix partitioning order.
    ra_col / dec_col:
        Column names for sky coordinates in both surveys.
    overwrite:
        If False (default), skip existing output tiles.
    populated_tiles_only:
        When True (default), iterate only HEALPix tiles present in survey A
        instead of the full ``12 * 4**norder`` pixel range.
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
    ra_col, dec_col, norder = resolve_crossmatch_sky_columns(
        lake_root,
        survey_a,
        survey_b,
        ra_col=ra_col,
        dec_col=dec_col,
        norder=norder,
    )

    xm_name = crossmatch_name(survey_a, survey_b)
    out_root = crossmatch_root(lake_root, survey_a, survey_b)

    catalog_root_a = lake_root / "catalogs" / survey_a
    if populated_tiles_only:
        tile_npixels = iter_populated_tile_npixels(catalog_root_a, norder=norder)
        log.info(
            "Cross-match %s × %s: %d populated tile(s) in survey A at Norder=%d",
            survey_a,
            survey_b,
            len(tile_npixels),
            norder,
        )
    else:
        nside = hp.order2nside(norder)
        tile_npixels = list(range(hp.nside2npix(nside)))
        log.info(
            "Cross-match %s × %s: scanning all %d HEALPix pixels at Norder=%d",
            survey_a,
            survey_b,
            len(tile_npixels),
            norder,
        )

    pending, n_match_rows, n_tiles_written = _pending_crossmatch_tiles(
        tile_npixels, out_root, norder, overwrite,
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
        radius_rad = np.radians(radius_deg)
        iterator: Iterable[int] = pending
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(pending, unit="tile", desc=f"{survey_a}×{survey_b}")
            except ImportError:
                pass

        with CatalogAccessor(lake_root, survey_a, norder=norder) as acc_a, \
             CatalogAccessor(lake_root, survey_b, norder=norder) as acc_b:
            for npix_a in iterator:
                n_rows = _crossmatch_one_tile(
                    lake_root=lake_root,
                    survey_a=survey_a,
                    survey_b=survey_b,
                    npix_a=npix_a,
                    norder=norder,
                    ra_col=ra_col,
                    dec_col=dec_col,
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
                norder=norder,
                ra_col=ra_col,
                dec_col=dec_col,
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

    _write_xm_info(out_root, xm_name, survey_a, survey_b, norder, radius_arcsec)
    return CrossmatchResult(
        survey_a=survey_a,
        survey_b=survey_b,
        crossmatch_name=xm_name,
        output_root=out_root,
        n_tiles_written=n_tiles_written,
        n_match_rows=n_match_rows,
        radius_arcsec=radius_arcsec,
        norder=norder,
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
    norder: int,
    radius_arcsec: float,
) -> None:
    info = {
        "catalog_name": xm_name,
        "catalog_type": "association",
        "survey_a": survey_a,
        "survey_b": survey_b,
        "match_radius_arcsec": radius_arcsec,
        "hats_order": norder,
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
        type=int,
        default=None,
        help="HEALPix order (default: survey A catalog_info.json hats_order).",
    )
    @click.option("--ra-col", default=None, help="RA column override (default: catalog_info).")
    @click.option("--dec-col", default=None, help="Dec column override (default: catalog_info).")
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
        norder: int | None,
        ra_col: str | None,
        dec_col: str | None,
        overwrite: bool,
        all_tiles: bool,
        show_progress: bool,
        n_workers: int,
        verbose: bool,
    ) -> None:
        """Build an in-lake positional cross-match between two ingested catalogs.

        Output: ``catalogs/crossmatch/<survey_a>_x_<survey_b>/`` (HATS Parquet).
        Survey A defines the partition key. Matches are nearest-neighbour within
        ``--radius-arcsec``. Re-run without ``--overwrite`` to resume (skips
        existing tiles).
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
            norder=norder,
            ra_col=ra_col,
            dec_col=dec_col,
            overwrite=overwrite,
            populated_tiles_only=not all_tiles,
            show_progress=show_progress,
            n_workers=n_workers,
        )
        click.echo(
            f"Cross-match {result.crossmatch_name}: "
            f"{result.n_match_rows:,} match row(s) in {result.n_tiles_written:,} tile(s) "
            f"→ {result.output_root} ({result.elapsed_s:.1f} s, {result.n_workers} worker(s))"
        )

except ImportError:
    cli = None  # type: ignore[misc, assignment]
