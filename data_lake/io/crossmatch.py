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
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.io.catalog import CatalogAccessor
from data_lake.ingest.fits_to_parquet import healpix_dir, _HATS_DIR_STRIDE, _ZSTD_LEVEL

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_crossmatch(
    lake_root: Path | str,
    survey_a: str,
    survey_b: str,
    radius_arcsec: float = 1.0,
    norder: int = 5,
    ra_col: str = "ra",
    dec_col: str = "dec",
    overwrite: bool = False,
) -> None:
    """
    Build a precomputed cross-match between two surveys.

    Uses tile-by-tile nearest-neighbour matching with a search radius that
    includes a buffer of neighbouring tiles to handle sources near tile borders.

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
    """
    lake_root = Path(lake_root)
    radius_deg = radius_arcsec / 3600.0
    radius_rad = np.radians(radius_deg)

    xm_name = f"{survey_a}_x_{survey_b}"
    out_root = lake_root / "catalogs" / "crossmatch" / xm_name

    with CatalogAccessor(lake_root, survey_a, norder=norder) as acc_a, \
         CatalogAccessor(lake_root, survey_b, norder=norder) as acc_b:

        nside = hp.order2nside(norder)
        n_pix = hp.nside2npix(nside)
        hp_col = f"_healpix_norder{norder}"

        t0 = time.perf_counter()
        written = 0

        for npix_a in range(n_pix):
            out_dir = out_root / healpix_dir(norder, npix_a)
            out_file = out_dir / f"Npix={npix_a}.parquet"
            if out_file.exists() and not overwrite:
                continue

            # Fetch surveyA sources in this tile
            df_a = acc_a.sources_in_tile(
                npix_a,
                columns=["source_id", ra_col, dec_col],
                fmt="pandas",
            )
            if df_a.empty:
                continue

            ra_a = df_a[ra_col].values.astype(np.float64)
            dec_a = df_a[dec_col].values.astype(np.float64)
            ids_a = df_a["source_id"].values.astype(np.int64)

            # Neighbouring tiles for surveyB to cover border effects
            vec_center = hp.ang2vec(
                np.radians(90.0 - dec_a.mean()), np.radians(ra_a.mean())
            )
            neighbour_pixels = hp.query_disc(
                nside, vec_center, radius_rad * 10 + hp.nside2resol(nside),
                nest=True, inclusive=True,
            ).tolist()
            neighbour_pixels = list(set([npix_a] + neighbour_pixels))

            # Collect surveyB sources from neighbouring tiles
            frames_b = []
            for npix_b in neighbour_pixels:
                df_b_tile = acc_b.sources_in_tile(
                    npix_b,
                    columns=["source_id", ra_col, dec_col],
                    fmt="pandas",
                )
                if not df_b_tile.empty:
                    frames_b.append(df_b_tile)

            if not frames_b:
                continue

            import pandas as pd
            df_b = pd.concat(frames_b, ignore_index=True).drop_duplicates("source_id")
            ra_b = df_b[ra_col].values.astype(np.float64)
            dec_b = df_b[dec_col].values.astype(np.float64)
            ids_b = df_b["source_id"].values.astype(np.int64)

            # Sky match
            matched_a, matched_b, sep = _match_sky(
                ra_a, dec_a, ids_a, ra_b, dec_b, ids_b, radius_deg
            )

            if matched_a.size == 0:
                continue

            # Assign tile partition = surveyA pixel
            pix_col = np.full(len(matched_a), npix_a, dtype=np.int64)

            table = pa.table({
                "source_id_a": pa.array(matched_a, type=pa.int64()),
                "source_id_b": pa.array(matched_b, type=pa.int64()),
                "sep_arcsec": pa.array((sep * 3600.0).astype(np.float32), type=pa.float32()),
                hp_col: pa.array(pix_col, type=pa.int64()),
            })

            out_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                table,
                str(out_file),
                compression="zstd",
                compression_level=_ZSTD_LEVEL,
                write_statistics=True,
            )
            written += 1

        elapsed = time.perf_counter() - t0
        log.info("Cross-match %s × %s: wrote %d tiles in %.1f s", survey_a, survey_b, written, elapsed)

    _write_xm_info(out_root, xm_name, survey_a, survey_b, norder, radius_arcsec)


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
        xm_name = f"{survey_a}_x_{survey_b}"
        self._xm_root = self.lake_root / "catalogs" / "crossmatch" / xm_name

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
        fmt: str = "pandas",
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
        from data_lake.io.catalog import CatalogAccessor
        return CatalogAccessor._convert(result, fmt)  # type: ignore[arg-type]

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
