"""Tests for in-lake catalog cross-match."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, assign_healpix, healpix_dir
from data_lake.io.crossmatch import (
    CROSSMATCH_HEALPIX_NPIX_B,
    CrossmatchAccessor,
    CrossmatchTileConfig,
    build_crossmatch,
    build_column_crossmatch,
    column_crossmatch_name,
    crossmatch_root,
    export_crossmatch_flat,
    find_column_crossmatch_roots,
    iter_populated_tile_npixels,
    load_crossmatch_table,
    parse_column_crossmatch_dirname,
    resolve_column_crossmatch_root,
    resolve_crossmatch_settings,
    resolve_crossmatch_sky_columns,
    filter_survey_a_tiles_overlapping_survey_b,
    healpix_pixels_covering_tile,
    survey_b_pixels_for_tile,
    tile_search_cone,
    _crossmatch_tile_worker,
)
from data_lake.io.crossmatch_matchers import rapids_available


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
    link_id_mode: str | None = None,
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
    mode = link_id_mode or (
        f"column:{id_col}" if id_col != LAKE_JOIN_ID_COLUMN else "sequential"
    )
    info = {
        "hats_order": norder,
        "ra_column": ra_col,
        "dec_column": dec_col,
        "link_id_mode": mode,
        "link_id_column": LAKE_JOIN_ID_COLUMN,
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

    def test_resolve_sky_columns(self, tmp_path: Path) -> None:
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
        assert crossmatch_root(lake, "SURVEY_A", "SURVEY_B", 2.0).is_dir()

        with CrossmatchAccessor(lake, "SURVEY_A", "SURVEY_B") as xm:
            matches = xm.get_matches(source_id_a=1001, fmt="polars")
        assert matches.height == 1
        row = matches.row(0, named=True)
        assert row["source_id_b"] == 2001
        assert row["sep_arcsec"] < 2.0
        assert CROSSMATCH_HEALPIX_NPIX_B in matches.columns

    def test_crossmatch_writes_healpix_npix_b(self, tmp_path: Path) -> None:
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
            source_ids=[2001], ra=[ra + 0.00001], dec=[dec + 0.00001],
        )
        build_crossmatch(lake, "SURVEY_A", "SURVEY_B", radius_arcsec=2.0)
        xm_path = (
            crossmatch_root(lake, "SURVEY_A", "SURVEY_B", 2.0)
            / healpix_dir(norder, npix)
            / f"Npix={npix}.parquet"
        )
        table = pq.read_table(str(xm_path))
        assert CROSSMATCH_HEALPIX_NPIX_B in table.column_names
        assert table.column(CROSSMATCH_HEALPIX_NPIX_B)[0].as_py() == npix

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

        out_root = crossmatch_root(lake, "SURVEY_A", "SURVEY_B", 2.0)
        cfg = CrossmatchTileConfig(
            lake_root=str(lake),
            survey_a="SURVEY_A",
            survey_b="SURVEY_B",
            npix_a_list=(npix,),
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
            link_id_mode="sequential",
        )
        _write_catalog_tile(
            lake, "SURVEY_B", norder=norder, npix=npix,
            source_ids=[2001], ra=[ra + 0.0001], dec=[dec + 0.0001],
            id_col="id",
            link_id_mode="column:id",
        )

        result = build_crossmatch(
            lake, "SURVEY_A", "SURVEY_B", radius_arcsec=2.0, show_progress=False,
        )
        assert result.n_match_rows == 1


@pytest.mark.gpu
class TestBuildCrossmatchRapids:
    def test_rapids_backend_end_to_end(self, tmp_path: Path) -> None:
        if not rapids_available():
            pytest.skip("cuML/CuPy not installed (uv sync --extra rapids)")

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
            match_backend="rapids",
            gpu_id=0,
            n_workers=1,
        )
        assert result.n_match_rows == 1
        assert result.match_backend == "rapids"

        info = json.loads(
            (crossmatch_root(lake, "SURVEY_A", "SURVEY_B", 2.0) / "crossmatch_info.json").read_text()
        )
        assert info["match_backend"] == "rapids"
        assert info["gpu_id"] == 0
        assert info["total_rows"] == 1
        assert info["n_match_rows"] == 1
        assert info.get("match_mode", "sky") == "sky"


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

        table = load_crossmatch_table(crossmatch_root(lake, "SURVEY_A", "SURVEY_B", 2.0))
        assert table.num_rows == 1

        assert export_crossmatch_flat(
            crossmatch_root(lake, "SURVEY_A", "SURVEY_B", 2.0),
            tmp_path / "reexport.parquet",
        ) == 1


class TestCrossmatchPaddedColumnNames:
    """Crossmatch must work when survey-B tiles have padded FITS TTYPE column names."""

    def test_crossmatch_with_padded_survey_b(self, tmp_path: Path) -> None:
        """Survey B written with ' ra' / ' dec' should still match via resolve_column."""
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])

        # Survey A: clean column names
        _write_catalog_tile(
            lake, "SURVEY_A", norder=norder, npix=npix,
            source_ids=[1], ra=[ra], dec=[dec],
        )
        # Survey B: padded column names (' ra', ' dec') as produced by some FITS writers
        tile_dir = lake / "catalogs" / "SURVEY_B" / healpix_dir(norder, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        hp_col = f"_healpix_norder{norder}"
        padded_tile = pa.table({
            LAKE_JOIN_ID_COLUMN: pa.array([101], type=pa.int64()),
            " ra": pa.array([ra + 0.00001], type=pa.float64()),
            " dec": pa.array([dec + 0.00001], type=pa.float64()),
            hp_col: pa.array([npix], type=pa.int64()),
            "_cutout_index": pa.array([-1], type=pa.int64()),
            "_spectrum_index": pa.array([-1], type=pa.int64()),
        })
        pq.write_table(padded_tile, tile_dir / f"Npix={npix}.parquet")
        (lake / "catalogs" / "SURVEY_B" / "catalog_info.json").write_text(
            json.dumps({
                "hats_order": norder,
                "ra_column": "ra",   # logical name (stripped) in catalog_info
                "dec_column": "dec",
                "link_id_mode": "sequential",
                "link_id_column": LAKE_JOIN_ID_COLUMN,
                "total_rows": 1,
                "total_columns": len(padded_tile.schema),
            })
        )

        result = build_crossmatch(
            lake, "SURVEY_A", "SURVEY_B",
            radius_arcsec=10.0,
        )
        assert result.n_match_rows >= 1


class TestCrossmatchNamingAndReuse:
    def test_radius_in_tree_name(self, tmp_path: Path) -> None:
        from data_lake.io.crossmatch import (
            crossmatch_name,
            format_match_radius,
            parse_crossmatch_dirname,
        )

        assert crossmatch_name("A", "B", 1.0) == "A_x_B__r1.0"
        assert crossmatch_name("A", "B", 0.5) == "A_x_B__r0.5"
        assert format_match_radius(2.0) == "2.0"
        assert parse_crossmatch_dirname("EUCLID_x_DESI_DR1__r1.5") == (
            "EUCLID", "DESI_DR1", 1.5,
        )
        assert parse_crossmatch_dirname("not_a_match") is None

    def test_different_radii_are_distinct_trees(self, tmp_path: Path) -> None:
        from data_lake.io.crossmatch import find_crossmatch_roots

        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(lake, "A", norder=norder, npix=npix,
                            source_ids=[1], ra=[ra], dec=[dec])
        _write_catalog_tile(lake, "B", norder=norder, npix=npix,
                            source_ids=[2], ra=[ra + 0.0001], dec=[dec + 0.0001])

        build_crossmatch(lake, "A", "B", radius_arcsec=1.0)
        build_crossmatch(lake, "A", "B", radius_arcsec=2.0)

        found = find_crossmatch_roots(lake, "A", "B")
        radii = sorted(r for _, _, r, _ in found)
        assert radii == [1.0, 2.0]
        assert crossmatch_root(lake, "A", "B", 1.0).is_dir()
        assert crossmatch_root(lake, "A", "B", 2.0).is_dir()

    def test_reuse_rejects_norder_mismatch(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(lake, "A", norder=norder, npix=npix,
                            source_ids=[1], ra=[ra], dec=[dec])
        _write_catalog_tile(lake, "B", norder=norder, npix=npix,
                            source_ids=[2], ra=[ra + 0.0001], dec=[dec + 0.0001])
        build_crossmatch(lake, "A", "B", radius_arcsec=1.0)

        # Tamper with the sidecar to simulate a tree built at a different B order.
        info_path = crossmatch_root(lake, "A", "B", 1.0) / "crossmatch_info.json"
        info = json.loads(info_path.read_text())
        info["survey_b_norder"] = norder + 1
        info_path.write_text(json.dumps(info))

        with pytest.raises(ValueError, match="different"):
            build_crossmatch(lake, "A", "B", radius_arcsec=1.0)
        # overwrite bypasses the guard
        build_crossmatch(lake, "A", "B", radius_arcsec=1.0, overwrite=True)

    def test_accessor_autodiscovers_single_radius_setup(self, tmp_path: Path) -> None:
        pass

    def _two_partner_lake(self, tmp_path: Path):
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(lake, "EUCLID", norder=norder, npix=npix,
                            source_ids=[1], ra=[ra], dec=[dec])
        _write_catalog_tile(lake, "DESI_DR1", norder=norder, npix=npix,
                            source_ids=[2], ra=[ra + 0.0001], dec=[dec + 0.0001])
        _write_catalog_tile(lake, "ALLWISE", norder=norder, npix=npix,
                            source_ids=[3], ra=[ra + 0.0002], dec=[dec + 0.0002])
        return lake, ra, dec, npix

    def test_execute_crossmatch_plan(self, tmp_path: Path) -> None:
        from data_lake.io.crossmatch import execute_crossmatch_plan

        lake, ra, dec, npix = self._two_partner_lake(tmp_path)
        plan = {
            "base_catalog": "EUCLID",
            "partners": [
                {"survey": "DESI_DR1", "radius_arcsec": 1.0},
                {"survey": "ALLWISE", "radius_arcsec": 2.0},
            ],
        }
        results = execute_crossmatch_plan(lake, plan)
        assert len(results) == 2
        assert crossmatch_root(lake, "EUCLID", "DESI_DR1", 1.0).is_dir()
        assert crossmatch_root(lake, "EUCLID", "ALLWISE", 2.0).is_dir()

    def test_plan_region_restriction(self, tmp_path: Path) -> None:
        from data_lake.discovery.region import Region
        from data_lake.io.crossmatch import execute_crossmatch_plan

        lake, ra, dec, npix = self._two_partner_lake(tmp_path)
        plan = {"base_catalog": "EUCLID",
                "partners": [{"survey": "DESI_DR1", "radius_arcsec": 1.0}]}

        # Region on the opposite side -> no base tiles -> no matches.
        empty_region = Region.cone(300.0, -45.0, 30.0)
        results = execute_crossmatch_plan(lake, plan, region=empty_region)
        assert results[0].n_match_rows == 0

        # Region covering the tile -> matches.
        good_region = Region.cone(ra, dec, 60.0)
        results = execute_crossmatch_plan(lake, plan, region=good_region, overwrite=True)
        assert results[0].n_match_rows == 1

    def test_cli_from_area(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.discovery.areas import make_area, save_area
        from data_lake.discovery.region import Region
        from data_lake.io.crossmatch import cli

        lake, ra, dec, npix = self._two_partner_lake(tmp_path)
        area = make_area(
            "Field1",
            Region.cone(ra, dec, 60.0),
            crossmatch_plan={
                "base_catalog": "EUCLID",
                "partners": [{"survey": "DESI_DR1", "radius_arcsec": 1.0}],
            },
        )
        save_area(lake, area)

        result = CliRunner().invoke(cli, [str(lake), "--from-area", "Field1"])
        assert result.exit_code == 0, result.output
        assert "EUCLID_x_DESI_DR1__r1.0" in result.output
        assert crossmatch_root(lake, "EUCLID", "DESI_DR1", 1.0).is_dir()

    def test_accessor_autodiscovers_single_radius(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(lake, "A", norder=norder, npix=npix,
                            source_ids=[1], ra=[ra], dec=[dec])
        _write_catalog_tile(lake, "B", norder=norder, npix=npix,
                            source_ids=[2], ra=[ra + 0.0001], dec=[dec + 0.0001])
        build_crossmatch(lake, "A", "B", radius_arcsec=1.0)
        build_crossmatch(lake, "A", "B", radius_arcsec=2.0)

        # Ambiguous without radius
        with pytest.raises(ValueError, match="Multiple"):
            CrossmatchAccessor(lake, "A", "B")
        # Explicit radius resolves
        with CrossmatchAccessor(lake, "A", "B", radius_arcsec=1.0) as xm:
            assert xm.get_matches(source_id_a=1, fmt="polars").height == 1


# ---------------------------------------------------------------------------
# Column-equality crossmatch tests
# ---------------------------------------------------------------------------


def _write_catalog_tile_with_col(
    lake: Path,
    survey: str,
    *,
    norder: int,
    npix: int,
    source_ids: list[int],
    ra: list[float],
    dec: list[float],
    extra_col: str,
    extra_values: list,
) -> None:
    """Write a catalog tile with an additional match column."""
    tile_dir = lake / "catalogs" / survey / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    hp_col = f"_healpix_norder{norder}"
    table = pa.table({
        "_source_id": pa.array(source_ids, type=pa.int64()),
        "ra": pa.array(ra, type=pa.float64()),
        "dec": pa.array(dec, type=pa.float64()),
        hp_col: pa.array([npix] * len(source_ids), type=pa.int64()),
        "_cutout_index": pa.array([-1] * len(source_ids), type=pa.int64()),
        "_spectrum_index": pa.array([-1] * len(source_ids), type=pa.int64()),
        extra_col: pa.array(extra_values),
    })
    pq.write_table(table, tile_dir / f"Npix={npix}.parquet")
    (lake / "catalogs" / survey / "catalog_info.json").write_text(json.dumps({
        "hats_order": norder,
        "ra_column": "ra",
        "dec_column": "dec",
        "link_id_mode": "sequential",
        "link_id_column": "_source_id",
        "total_rows": len(source_ids),
        "total_columns": len(table.column_names),
    }))


class TestColumnCrossmatchNaming:
    def test_name_format(self) -> None:
        name = column_crossmatch_name("EUCLID", "DESI", "TARGETID", "TARGETID")
        assert name == "EUCLID_x_DESI__col_TARGETID__TARGETID"

    def test_name_different_cols(self) -> None:
        name = column_crossmatch_name("A", "B", "col1", "col2")
        assert name == "A_x_B__col_col1__col2"

    def test_parse_round_trip(self) -> None:
        name = column_crossmatch_name("A", "B", "col1", "col2")
        parsed = parse_column_crossmatch_dirname(name)
        assert parsed is not None
        a, b, col_a, col_b = parsed
        assert a == "A"
        assert b == "B"
        assert col_a == "col1"
        assert col_b == "col2"

    def test_parse_sky_name_returns_none(self) -> None:
        assert parse_column_crossmatch_dirname("A_x_B__r1.0") is None

    def test_parse_invalid_returns_none(self) -> None:
        assert parse_column_crossmatch_dirname("notaname") is None

    def test_validate_match_column_ok(self) -> None:
        from data_lake.io.crossmatch import validate_match_column

        validate_match_column("TARGETID", "--match-col-a")
        validate_match_column("my.col.v1", "--match-col-a")
        validate_match_column("abc123", "--match-col-a")

    def test_validate_match_column_bad(self) -> None:
        from data_lake.io.crossmatch import validate_match_column

        with pytest.raises(ValueError):
            validate_match_column("", "--match-col-a")
        with pytest.raises(ValueError):
            validate_match_column("bad__col", "--match-col-a")
        with pytest.raises(ValueError):
            validate_match_column("bad_x_col", "--match-col-a")
        with pytest.raises(ValueError):
            validate_match_column("bad col", "--match-col-a")


class TestColumnCrossmatchEngine:
    def test_basic_equality_join(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])

        _write_catalog_tile_with_col(
            lake, "A", norder=norder, npix=npix,
            source_ids=[1, 2], ra=[ra, ra + 0.01], dec=[dec, dec + 0.01],
            extra_col="TARGETID", extra_values=[100, 200],
        )
        _write_catalog_tile_with_col(
            lake, "B", norder=norder, npix=npix,
            source_ids=[10, 20], ra=[ra, ra + 0.02], dec=[dec, dec + 0.02],
            extra_col="TARGETID", extra_values=[100, 999],
        )

        result = build_column_crossmatch(lake, "A", "B", "TARGETID", "TARGETID")
        assert result.n_match_rows == 1
        assert result.match_mode == "column"
        assert result.match_col_a == "TARGETID"
        assert result.match_col_b == "TARGETID"
        assert result.radius_arcsec == 0.0

        tbl = load_crossmatch_table(result.output_root)
        assert tbl.num_rows == 1
        assert tbl["source_id_a"][0].as_py() == 1
        assert tbl["source_id_b"][0].as_py() == 10
        assert tbl["sep_arcsec"][0].as_py() == pytest.approx(0.0)

    def test_many_to_many(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])

        # A source 1 matches two B sources
        _write_catalog_tile_with_col(
            lake, "A", norder=norder, npix=npix,
            source_ids=[1], ra=[ra], dec=[dec],
            extra_col="KEY", extra_values=[42],
        )
        _write_catalog_tile_with_col(
            lake, "B", norder=norder, npix=npix,
            source_ids=[10, 20], ra=[ra, ra + 0.01], dec=[dec, dec + 0.01],
            extra_col="KEY", extra_values=[42, 42],
        )

        result = build_column_crossmatch(lake, "A", "B", "KEY", "KEY")
        assert result.n_match_rows == 2

        tbl = load_crossmatch_table(result.output_root)
        assert tbl.num_rows == 2
        source_ids_b = sorted(tbl["source_id_b"].to_pylist())
        assert source_ids_b == [10, 20]

    def test_different_column_names(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])

        _write_catalog_tile_with_col(
            lake, "A", norder=norder, npix=npix,
            source_ids=[1], ra=[ra], dec=[dec],
            extra_col="ID_A", extra_values=[777],
        )
        _write_catalog_tile_with_col(
            lake, "B", norder=norder, npix=npix,
            source_ids=[10], ra=[ra], dec=[dec],
            extra_col="ID_B", extra_values=[777],
        )

        result = build_column_crossmatch(lake, "A", "B", "ID_A", "ID_B")
        assert result.n_match_rows == 1

    def test_null_keys_excluded(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])

        _write_catalog_tile_with_col(
            lake, "A", norder=norder, npix=npix,
            source_ids=[1, 2], ra=[ra, ra + 0.01], dec=[dec, dec + 0.01],
            extra_col="TARGETID", extra_values=[100, None],
        )
        _write_catalog_tile_with_col(
            lake, "B", norder=norder, npix=npix,
            source_ids=[10], ra=[ra], dec=[dec],
            extra_col="TARGETID", extra_values=[100],
        )

        result = build_column_crossmatch(lake, "A", "B", "TARGETID", "TARGETID")
        assert result.n_match_rows == 1

    def test_tree_name_contains_column_names(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile_with_col(
            lake, "A", norder=norder, npix=npix,
            source_ids=[1], ra=[ra], dec=[dec],
            extra_col="KEY", extra_values=[1],
        )
        _write_catalog_tile_with_col(
            lake, "B", norder=norder, npix=npix,
            source_ids=[10], ra=[ra], dec=[dec],
            extra_col="KEY", extra_values=[1],
        )

        result = build_column_crossmatch(lake, "A", "B", "KEY", "KEY")
        assert "__col_KEY__KEY" in result.crossmatch_name
        # no hex token suffix
        assert not any(result.crossmatch_name.endswith(c) for c in "0123456789abcdef" if len(c) == 1 and result.crossmatch_name[-8:].isalnum())

    def test_crossmatch_info_written(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile_with_col(
            lake, "A", norder=norder, npix=npix,
            source_ids=[1], ra=[ra], dec=[dec],
            extra_col="KEY", extra_values=[5],
        )
        _write_catalog_tile_with_col(
            lake, "B", norder=norder, npix=npix,
            source_ids=[10], ra=[ra], dec=[dec],
            extra_col="KEY", extra_values=[5],
        )

        result = build_column_crossmatch(lake, "A", "B", "KEY", "KEY")
        info_path = result.output_root / "crossmatch_info.json"
        assert info_path.is_file()
        info = json.loads(info_path.read_text())
        assert info["match_mode"] == "column"
        assert "match_id" not in info
        assert info["match_col_a"] == "KEY"
        assert info["match_col_b"] == "KEY"
        assert info["match_radius_arcsec"] is None
        assert info["total_rows"] == 1
        assert info["n_match_rows"] == 1
        assert info["n_tiles"] == 1

    def test_find_and_resolve_column_root(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile_with_col(
            lake, "A", norder=norder, npix=npix,
            source_ids=[1], ra=[ra], dec=[dec],
            extra_col="KEY", extra_values=[9],
        )
        _write_catalog_tile_with_col(
            lake, "B", norder=norder, npix=npix,
            source_ids=[10], ra=[ra], dec=[dec],
            extra_col="KEY", extra_values=[9],
        )

        result = build_column_crossmatch(lake, "A", "B", "KEY", "KEY")
        found = find_column_crossmatch_roots(lake, "A", "B")
        assert len(found) == 1
        col_a, col_b = found[0][2], found[0][3]
        assert col_a == "KEY" and col_b == "KEY"

        resolved = resolve_column_crossmatch_root(lake, "A", "B", "KEY", "KEY")
        assert resolved == result.output_root

    def test_reuse_guard_mismatch(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile_with_col(
            lake, "A", norder=norder, npix=npix,
            source_ids=[1], ra=[ra], dec=[dec],
            extra_col="KEY", extra_values=[1],
        )
        _write_catalog_tile_with_col(
            lake, "B", norder=norder, npix=npix,
            source_ids=[10], ra=[ra], dec=[dec],
            extra_col="KEY", extra_values=[1],
        )
        build_column_crossmatch(lake, "A", "B", "KEY", "KEY")

        # Tamper with the sidecar to simulate a mismatch
        result = build_column_crossmatch(lake, "A", "B", "KEY", "KEY")
        info_path = result.output_root / "crossmatch_info.json"
        info = json.loads(info_path.read_text())
        info["match_col_a"] = "DIFFERENT"
        info_path.write_text(json.dumps(info))

        with pytest.raises(ValueError, match="different"):
            build_column_crossmatch(lake, "A", "B", "KEY", "KEY")
        # overwrite bypasses
        build_column_crossmatch(lake, "A", "B", "KEY", "KEY", overwrite=True)

    def test_non_overlapping_tile_produces_no_match(self, tmp_path: Path) -> None:
        """Key match only in a non-overlapping B tile → no association row (locality)."""
        import healpy as hp_lib
        lake = tmp_path / "lake"
        norder = 3  # low order so pixels are large and far-apart pixels are unambiguous

        ra_a, dec_a = 0.0, 0.0
        ra_b, dec_b = 180.0, 0.0  # antipodal — guaranteed non-overlapping
        npix_a = int(assign_healpix(np.array([ra_a]), np.array([dec_a]), norder)[0])
        npix_b = int(assign_healpix(np.array([ra_b]), np.array([dec_b]), norder)[0])
        assert npix_a != npix_b

        _write_catalog_tile_with_col(
            lake, "A", norder=norder, npix=npix_a,
            source_ids=[1], ra=[ra_a], dec=[dec_a],
            extra_col="KEY", extra_values=[42],
        )
        # B tile placed at antipodal position — shares the key value but not the sky region
        _write_catalog_tile_with_col(
            lake, "B", norder=norder, npix=npix_b,
            source_ids=[10], ra=[ra_b], dec=[dec_b],
            extra_col="KEY", extra_values=[42],
        )

        result = build_column_crossmatch(lake, "A", "B", "locality", "KEY", "KEY")
        assert result.n_match_rows == 0, (
            "Matches in non-overlapping tiles must not be associated under the locality assumption"
        )

    def test_geometric_neighbour_match_found(self, tmp_path: Path) -> None:
        """Key match in geometrically overlapping tiles → association row produced."""
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])

        _write_catalog_tile_with_col(
            lake, "A", norder=norder, npix=npix,
            source_ids=[1], ra=[ra], dec=[dec],
            extra_col="KEY", extra_values=[99],
        )
        _write_catalog_tile_with_col(
            lake, "B", norder=norder, npix=npix,
            source_ids=[20], ra=[ra + 0.001], dec=[dec + 0.001],
            extra_col="KEY", extra_values=[99],
        )

        result = build_column_crossmatch(lake, "A", "B", "KEY", "KEY")
        assert result.n_match_rows == 1
        tbl = load_crossmatch_table(result.output_root)
        assert tbl["healpix_npix_b"][0].as_py() == npix

    def test_half_pixel_pad_includes_neighbour_b_tile(self, tmp_path: Path) -> None:
        """B source in a neighbour A-pixel is still joined (half-pixel pad)."""
        import healpy as hp_lib

        lake = tmp_path / "lake"
        norder = 5
        nside = hp_lib.order2nside(norder)
        ra_a, dec_a = 120.0, 45.0
        npix_a = int(assign_healpix(np.array([ra_a]), np.array([dec_a]), norder)[0])

        neighbours = [
            int(n) for n in hp_lib.get_all_neighbours(nside, npix_a, nest=True)
            if n >= 0
        ]
        assert neighbours, "expected at least one neighbour pixel"
        npix_b = neighbours[0]
        theta, phi = hp_lib.pix2ang(nside, npix_b, nest=True)
        ra_b = float(np.degrees(phi))
        dec_b = float(90.0 - np.degrees(theta))

        _write_catalog_tile_with_col(
            lake, "A", norder=norder, npix=npix_a,
            source_ids=[1], ra=[ra_a], dec=[dec_a],
            extra_col="KEY", extra_values=[77],
        )
        _write_catalog_tile_with_col(
            lake, "B", norder=norder, npix=npix_b,
            source_ids=[20], ra=[ra_b], dec=[dec_b],
            extra_col="KEY", extra_values=[77],
        )

        result = build_column_crossmatch(lake, "A", "B", "KEY", "KEY")
        assert result.n_match_rows == 1
        tbl = load_crossmatch_table(result.output_root)
        assert tbl["healpix_npix_b"][0].as_py() == npix_b

    def test_pad_rad_uses_coarser_survey(self) -> None:
        from data_lake.io.crossmatch import _column_crossmatch_pad_rad
        import healpy as hp_lib

        nside_fine = hp_lib.order2nside(8)
        nside_coarse = hp_lib.order2nside(3)
        pad = _column_crossmatch_pad_rad(nside_fine, nside_coarse)
        assert pad == pytest.approx(0.5 * float(hp_lib.max_pixrad(nside_coarse)))
        assert pad > 0.5 * float(hp_lib.max_pixrad(nside_fine))

    def test_n_workers_sequential_equivalence(self, tmp_path: Path) -> None:
        """n_workers=1 (default) and explicit n_workers=1 both give the same result."""
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile_with_col(
            lake, "A", norder=norder, npix=npix,
            source_ids=[1, 2], ra=[ra, ra + 0.01], dec=[dec, dec + 0.01],
            extra_col="K", extra_values=[1, 2],
        )
        _write_catalog_tile_with_col(
            lake, "B", norder=norder, npix=npix,
            source_ids=[10, 20], ra=[ra, ra + 0.01], dec=[dec, dec + 0.01],
            extra_col="K", extra_values=[1, 2],
        )
        result = build_column_crossmatch(lake, "A", "B", "K", "K", n_workers=1)
        assert result.n_match_rows == 2
        assert result.n_workers == 1


class TestColumnCrossmatchCLI:
    def _setup_lake(self, tmp_path: Path):
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile_with_col(
            lake, "SRC", norder=norder, npix=npix,
            source_ids=[1], ra=[ra], dec=[dec],
            extra_col="OBJ_ID", extra_values=[42],
        )
        _write_catalog_tile_with_col(
            lake, "PARTNER", norder=norder, npix=npix,
            source_ids=[10], ra=[ra], dec=[dec],
            extra_col="OBJ_ID", extra_values=[42],
        )
        return lake, npix

    def test_cli_column_mode(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from data_lake.io.crossmatch import cli

        lake, _ = self._setup_lake(tmp_path)
        result = CliRunner().invoke(cli, [
            "SRC", "PARTNER", str(lake),
            "--match-mode", "column",
            "--match-col-a", "OBJ_ID",
            "--match-col-b", "OBJ_ID",
        ])
        assert result.exit_code == 0, result.output
        assert "1 association row" in result.output
        assert "__col_OBJ_ID__OBJ_ID" in result.output

    def test_cli_column_mode_rejects_from_area(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from data_lake.io.crossmatch import cli

        lake, _ = self._setup_lake(tmp_path)
        result = CliRunner().invoke(cli, [
            "SRC", "PARTNER", str(lake),
            "--match-mode", "column",
            "--match-col-a", "OBJ_ID",
            "--match-col-b", "OBJ_ID",
            "--from-area", "Field1",
        ])
        assert result.exit_code != 0
        assert "column" in result.output.lower()

    def test_cli_column_mode_missing_col(self, tmp_path: Path) -> None:
        """--match-col-a and --match-col-b are both required."""
        from click.testing import CliRunner
        from data_lake.io.crossmatch import cli

        lake, _ = self._setup_lake(tmp_path)
        result = CliRunner().invoke(cli, [
            "SRC", "PARTNER", str(lake),
            "--match-mode", "column",
            "--match-col-a", "OBJ_ID",
        ])
        assert result.exit_code != 0
        assert "--match-col-b" in result.output

    def test_cli_column_mode_with_progress(self, tmp_path: Path) -> None:
        """Default progress (and explicit --no-progress) exit 0 in column mode."""
        from click.testing import CliRunner
        from data_lake.io.crossmatch import cli

        lake, _ = self._setup_lake(tmp_path)
        result = CliRunner().invoke(cli, [
            "SRC", "PARTNER", str(lake),
            "--match-mode", "column",
            "--match-col-a", "OBJ_ID",
            "--match-col-b", "OBJ_ID",
        ])
        assert result.exit_code == 0, result.output

        result_off = CliRunner().invoke(cli, [
            "SRC", "PARTNER", str(lake),
            "--match-mode", "column",
            "--match-col-a", "OBJ_ID",
            "--match-col-b", "OBJ_ID",
            "--no-progress",
            "--overwrite",
        ])
        assert result_off.exit_code == 0, result_off.output

    def test_cli_sky_mode_unchanged(self, tmp_path: Path) -> None:
        """Sky mode still works after adding column-mode flags."""
        from click.testing import CliRunner
        from data_lake.io.crossmatch import cli

        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(lake, "A", norder=norder, npix=npix,
                            source_ids=[1], ra=[ra], dec=[dec])
        _write_catalog_tile(lake, "B", norder=norder, npix=npix,
                            source_ids=[2], ra=[ra + 0.0001], dec=[dec + 0.0001])

        result = CliRunner().invoke(cli, [
            "A", "B", str(lake), "--radius-arcsec", "1.0",
        ])
        assert result.exit_code == 0, result.output
        assert "__r1.0" in result.output
