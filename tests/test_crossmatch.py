"""Tests for in-lake catalog cross-match."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, assign_healpix, healpix_dir
from data_lake.io.crossmatch import (
    CrossmatchAccessor,
    CrossmatchTileConfig,
    build_crossmatch,
    crossmatch_root,
    export_crossmatch_flat,
    iter_populated_tile_npixels,
    load_crossmatch_table,
    resolve_crossmatch_settings,
    resolve_crossmatch_sky_columns,
    filter_survey_a_tiles_overlapping_survey_b,
    healpix_pixels_covering_tile,
    survey_b_pixels_for_tile,
    tile_search_cone,
    _crossmatch_tile_worker,
)


def _write_catalog_tile(
    lake: Path,
    survey: str,
    *,
    norder: int,
    npix: int,
    source_ids: list[int],
    ra: list[float],
    dec: list[float],
    ra_col: str = "ra",
    dec_col: str = "dec",
    id_col: str = LAKE_JOIN_ID_COLUMN,
    source_id_mode: str | None = None,
) -> None:
    tile_dir = lake / "catalogs" / survey / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    hp_col = f"_healpix_norder{norder}"
    cols: dict = {
        LAKE_JOIN_ID_COLUMN: pa.array(source_ids, type=pa.int64()),
        ra_col: pa.array(ra, type=pa.float64()),
        dec_col: pa.array(dec, type=pa.float64()),
        hp_col: pa.array([npix] * len(source_ids), type=pa.int64()),
        "_cutout_index": pa.array([-1] * len(source_ids), type=pa.int64()),
        "_spectrum_index": pa.array([-1] * len(source_ids), type=pa.int64()),
    }
    if id_col != LAKE_JOIN_ID_COLUMN:
        cols[id_col] = pa.array(source_ids, type=pa.int64())
    pq.write_table(
        pa.table(cols),
        tile_dir / f"Npix={npix}.parquet",
    )
    mode = source_id_mode or (
        f"column:{id_col}" if id_col != LAKE_JOIN_ID_COLUMN else "sequential"
    )
    info = {
        "hats_order": norder,
        "ra_column": ra_col,
        "dec_column": dec_col,
        "source_id_mode": mode,
        "source_id_column": LAKE_JOIN_ID_COLUMN,
        "total_rows": len(source_ids),
        "total_columns": len(cols),
    }
    if id_col != LAKE_JOIN_ID_COLUMN:
        info["native_id_column"] = id_col
    (lake / "catalogs" / survey / "catalog_info.json").write_text(json.dumps(info))


class TestCrossmatchHelpers:
    def test_iter_populated_tiles(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_catalog_tile(
            lake, "A", norder=5, npix=100,
            source_ids=[1], ra=[10.0], dec=[0.0],
        )
        _write_catalog_tile(
            lake, "A", norder=5, npix=200,
            source_ids=[2], ra=[11.0], dec=[0.1],
        )
        root = lake / "catalogs" / "A"
        assert iter_populated_tile_npixels(root, norder=5) == [100, 200]

    def test_resolve_per_survey_settings(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_catalog_tile(
            lake, "A", norder=5, npix=1, source_ids=[1], ra=[0.0], dec=[0.0],
            ra_col="RA", dec_col="DEC",
        )
        _write_catalog_tile(
            lake, "B", norder=6, npix=1, source_ids=[2], ra=[0.0], dec=[0.0],
            ra_col="RAJ2000", dec_col="DEJ2000",
        )
        settings = resolve_crossmatch_settings(lake, "A", "B")
        assert settings.survey_a.ra_col == "RA"
        assert settings.survey_b.ra_col == "RAJ2000"
        assert settings.survey_a.norder == 5
        assert settings.survey_b.norder == 6

    def test_legacy_resolve_sky_columns(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_catalog_tile(lake, "A", norder=5, npix=1, source_ids=[1], ra=[0.0], dec=[0.0])
        _write_catalog_tile(lake, "B", norder=5, npix=1, source_ids=[2], ra=[0.0], dec=[0.0])
        ra, dec, order = resolve_crossmatch_sky_columns(lake, "A", "B")
        assert (ra, dec, order) == ("ra", "dec", 5)

    def test_survey_b_pixels_use_edges(self) -> None:
        import healpy as hp

        norder = 5
        nside = hp.order2nside(norder)
        npix = int(assign_healpix(np.array([120.0]), np.array([45.0]), norder)[0])
        radius_rad = np.radians(1.0 / 3600.0)  # 1 arcsec

        pixels = survey_b_pixels_for_tile(nside, npix, radius_rad)
        assert npix in pixels

        neighbours = [int(n) for n in hp.get_all_neighbours(nside, npix, nest=True) if n >= 0]
        for n in neighbours:
            assert n in pixels

        assert len(pixels) >= len(neighbours) + 1

    def test_survey_b_pixels_different_norder(self) -> None:
        import healpy as hp

        ra, dec = 120.0, 45.0
        norder_a, norder_b = 5, 6
        nside_a = hp.order2nside(norder_a)
        nside_b = hp.order2nside(norder_b)
        npix_a = int(assign_healpix(np.array([ra]), np.array([dec]), norder_a)[0])
        npix_b = int(assign_healpix(np.array([ra]), np.array([dec]), norder_b)[0])
        assert npix_a != npix_b

        pixels = survey_b_pixels_for_tile(
            nside_a, npix_a, np.radians(1.0 / 3600.0), nside_b=nside_b,
        )
        assert npix_b in pixels

    def test_tile_search_cone_covers_pixel(self) -> None:
        import healpy as hp

        nside = hp.order2nside(6)
        npix = 42
        radius_rad = np.radians(0.5 / 3600.0)
        ra_c, dec_c, search_deg = tile_search_cone(nside, npix, radius_rad)

        theta, phi = hp.pix2ang(nside, npix, nest=True)
        vec_center = hp.ang2vec(theta, phi)
        verts = hp.boundaries(nside, npix, nest=True)
        cosines = np.clip(verts.T @ vec_center, -1.0, 1.0)
        pixel_ext_deg = np.degrees(np.arccos(cosines.min()))

        assert search_deg >= pixel_ext_deg
        assert abs(ra_c - np.degrees(phi)) < 1e-9
        assert abs(dec_c - (90.0 - np.degrees(theta))) < 1e-9

    def test_filter_a_tiles_by_b_footprint(self, tmp_path: Path) -> None:
        import healpy as hp

        lake = tmp_path / "lake"
        norder = 5
        nside = hp.order2nside(norder)
        npix_overlap = int(assign_healpix(np.array([120.0]), np.array([45.0]), norder)[0])
        npix_far = int(assign_healpix(np.array([10.0]), np.array([0.0]), norder)[0])

        _write_catalog_tile(
            lake, "A", norder=norder, npix=npix_overlap,
            source_ids=[1], ra=[120.0], dec=[45.0],
        )
        _write_catalog_tile(
            lake, "A", norder=norder, npix=npix_far,
            source_ids=[2], ra=[10.0], dec=[0.0],
        )
        _write_catalog_tile(
            lake, "B", norder=norder, npix=npix_overlap,
            source_ids=[100], ra=[120.0], dec=[45.0],
        )

        b_pop = set(iter_populated_tile_npixels(lake / "catalogs" / "B", norder=norder))
        kept = filter_survey_a_tiles_overlapping_survey_b(
            [npix_overlap, npix_far],
            nside_a=nside,
            nside_b=nside,
            b_populated=b_pop,
            radius_rad=np.radians(1.0 / 3600.0),
        )
        assert kept == [npix_overlap]

    def test_healpix_pixels_covering_tile(self) -> None:
        import healpy as hp

        norder_a, norder_b = 6, 4
        nside_a = hp.order2nside(norder_a)
        nside_b = hp.order2nside(norder_b)
        npix_a = int(assign_healpix(np.array([120.0]), np.array([45.0]), norder_a)[0])
        pixels = healpix_pixels_covering_tile(
            nside_a, npix_a, np.radians(0.5 / 3600.0), nside_b=nside_b,
        )
        npix_b = int(assign_healpix(np.array([120.0]), np.array([45.0]), norder_b)[0])
        assert npix_b in pixels
        assert len(pixels) < hp.nside2npix(nside_b) // 10


class TestBuildCrossmatch:
    def test_nearest_match_within_radius(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])

        _write_catalog_tile(
            lake, "SURVEY_A", norder=norder, npix=npix,
            source_ids=[1001], ra=[ra], dec=[dec],
        )
        _write_catalog_tile(
            lake, "SURVEY_B", norder=norder, npix=npix,
            source_ids=[2001], ra=[ra + 0.0001], dec=[dec + 0.0001],
        )

        result = build_crossmatch(
            lake,
            "SURVEY_A",
            "SURVEY_B",
            radius_arcsec=2.0,
            show_progress=False,
        )
        assert result.n_match_rows == 1
        assert crossmatch_root(lake, "SURVEY_A", "SURVEY_B").is_dir()

        with CrossmatchAccessor(lake, "SURVEY_A", "SURVEY_B") as xm:
            matches = xm.get_matches(source_id_a=1001, fmt="polars")
        assert matches.height == 1
        row = matches.row(0, named=True)
        assert row["source_id_b"] == 2001
        assert row["sep_arcsec"] < 2.0

    def test_mixed_columns_and_norder(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        ra, dec = 120.0, 45.0
        norder_a, norder_b = 5, 6
        npix_a = int(assign_healpix(np.array([ra]), np.array([dec]), norder_a)[0])
        npix_b = int(assign_healpix(np.array([ra]), np.array([dec]), norder_b)[0])

        _write_catalog_tile(
            lake, "SURVEY_A", norder=norder_a, npix=npix_a,
            source_ids=[1001], ra=[ra], dec=[dec],
            ra_col="RA", dec_col="DEC",
        )
        _write_catalog_tile(
            lake, "SURVEY_B", norder=norder_b, npix=npix_b,
            source_ids=[2001], ra=[ra + 0.0001], dec=[dec + 0.0001],
            ra_col="RAJ2000", dec_col="DEJ2000",
        )

        result = build_crossmatch(
            lake,
            "SURVEY_A",
            "SURVEY_B",
            radius_arcsec=2.0,
            show_progress=False,
        )
        assert result.n_match_rows == 1

    def test_native_id_column_names(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])

        _write_catalog_tile(
            lake, "EUCLID", norder=norder, npix=npix,
            source_ids=[1001], ra=[ra], dec=[dec],
            ra_col="right_ascension", dec_col="declination",
            id_col="object_id",
        )
        _write_catalog_tile(
            lake, "SDSS", norder=norder, npix=npix,
            source_ids=[2001], ra=[ra + 0.0001], dec=[dec + 0.0001],
            ra_col="PLUG_RA", dec_col="PLUG_DEC",
            id_col="TARGETID",
        )

        result = build_crossmatch(
            lake, "EUCLID", "SDSS", radius_arcsec=2.0, show_progress=False,
        )
        assert result.n_match_rows == 1

    def test_coarse_b_tile_decoys_outside_cone(self, tmp_path: Path) -> None:
        """B at coarse Norder: distant sources in the same tile must not affect matching."""
        lake = tmp_path / "lake"
        ra, dec = 120.0, 45.0
        norder_a, norder_b = 6, 4
        npix_a = int(assign_healpix(np.array([ra]), np.array([dec]), norder_a)[0])
        npix_b = int(assign_healpix(np.array([ra]), np.array([dec]), norder_b)[0])

        _write_catalog_tile(
            lake, "SURVEY_A", norder=norder_a, npix=npix_a,
            source_ids=[1001], ra=[ra], dec=[dec],
        )
        # Same coarse B tile holds a match and a far decoy (would dominate a full-tile load).
        _write_catalog_tile(
            lake, "SURVEY_B", norder=norder_b, npix=npix_b,
            source_ids=[2001, 9999],
            ra=[ra + 0.0001, 50.0],
            dec=[dec + 0.0001, 0.0],
        )

        result = build_crossmatch(
            lake, "SURVEY_A", "SURVEY_B", radius_arcsec=2.0, show_progress=False,
        )
        assert result.n_match_rows == 1

    def test_disjoint_footprints_skip_a_tiles(self, tmp_path: Path) -> None:
        """Survey-A tiles outside survey-B footprint are not processed."""
        lake = tmp_path / "lake"
        norder = 5
        npix_a = int(assign_healpix(np.array([10.0]), np.array([0.0]), norder)[0])
        npix_b = int(assign_healpix(np.array([120.0]), np.array([45.0]), norder)[0])

        _write_catalog_tile(
            lake, "A", norder=norder, npix=npix_a,
            source_ids=[1], ra=[10.0], dec=[0.0],
        )
        _write_catalog_tile(
            lake, "B", norder=norder, npix=npix_b,
            source_ids=[2], ra=[120.0], dec=[45.0],
        )

        result = build_crossmatch(lake, "A", "B", radius_arcsec=2.0, show_progress=False)
        assert result.n_match_rows == 0
        assert result.n_tiles_written == 0

    def test_no_match_beyond_radius(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        npix_a = int(assign_healpix(np.array([10.0]), np.array([0.0]), norder)[0])
        npix_b = int(assign_healpix(np.array([50.0]), np.array([0.0]), norder)[0])

        _write_catalog_tile(
            lake, "A", norder=norder, npix=npix_a,
            source_ids=[1], ra=[10.0], dec=[0.0],
        )
        _write_catalog_tile(
            lake, "B", norder=norder, npix=npix_b,
            source_ids=[2], ra=[50.0], dec=[0.0],
        )

        result = build_crossmatch(lake, "A", "B", radius_arcsec=1.0)
        assert result.n_match_rows == 0

    def test_tile_worker(self, tmp_path: Path) -> None:
        """Exercise the parallel worker entry point without ProcessPoolExecutor."""
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])

        _write_catalog_tile(
            lake, "SURVEY_A", norder=norder, npix=npix,
            source_ids=[1001], ra=[ra], dec=[dec],
        )
        _write_catalog_tile(
            lake, "SURVEY_B", norder=norder, npix=npix,
            source_ids=[2001], ra=[ra + 0.0001], dec=[dec + 0.0001],
        )

        out_root = crossmatch_root(lake, "SURVEY_A", "SURVEY_B")
        cfg = CrossmatchTileConfig(
            lake_root=str(lake),
            survey_a="SURVEY_A",
            survey_b="SURVEY_B",
            npix_a=npix,
            norder_a=norder,
            norder_b=norder,
            ra_col_a="ra",
            dec_col_a="dec",
            ra_col_b="ra",
            dec_col_b="dec",
            radius_arcsec=2.0,
            out_root=str(out_root),
        )
        res = _crossmatch_tile_worker(cfg)
        assert res.error is None
        assert res.n_match_rows == 1
        assert res.n_tiles_written == 1


class TestCrossmatchNativeIdColumn:
    def test_crossmatch_uses_native_id_not_missing_source_id(self, tmp_path: Path) -> None:
        """Metadata may say sequential/source_id while tiles store survey-native ``id``."""
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])

        _write_catalog_tile(
            lake, "SURVEY_A", norder=norder, npix=npix,
            source_ids=[1001], ra=[ra], dec=[dec],
            id_col="id",
            source_id_mode="sequential",
        )
        _write_catalog_tile(
            lake, "SURVEY_B", norder=norder, npix=npix,
            source_ids=[2001], ra=[ra + 0.0001], dec=[dec + 0.0001],
            id_col="id",
            source_id_mode="column:id",
        )

        result = build_crossmatch(
            lake, "SURVEY_A", "SURVEY_B", radius_arcsec=2.0, show_progress=False,
        )
        assert result.n_match_rows == 1


class TestCrossmatchExport:
    def test_export_parquet_and_fits(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])

        _write_catalog_tile(
            lake, "SURVEY_A", norder=norder, npix=npix,
            source_ids=[1001], ra=[ra], dec=[dec],
        )
        _write_catalog_tile(
            lake, "SURVEY_B", norder=norder, npix=npix,
            source_ids=[2001], ra=[ra + 0.0001], dec=[dec + 0.0001],
        )

        parquet_out = tmp_path / "matches.parquet"
        fits_out = tmp_path / "matches.fits"
        result = build_crossmatch(
            lake,
            "SURVEY_A",
            "SURVEY_B",
            radius_arcsec=2.0,
            show_progress=False,
            export_parquet=parquet_out,
            export_fits=fits_out,
        )
        assert result.n_match_rows == 1
        assert result.export_parquet == parquet_out
        assert result.export_fits == fits_out
        assert parquet_out.is_file()
        assert fits_out.is_file()

        table = load_crossmatch_table(crossmatch_root(lake, "SURVEY_A", "SURVEY_B"))
        assert table.num_rows == 1

        assert export_crossmatch_flat(
            crossmatch_root(lake, "SURVEY_A", "SURVEY_B"),
            tmp_path / "reexport.parquet",
        ) == 1
