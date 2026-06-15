"""Tests for dl-gather: derived product catalogs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_lake.discovery.gather import PartnerSpec, gather_product
from data_lake.discovery.region import Region
from data_lake.discovery.selection import selection_from_region
from data_lake.io.catalog import CatalogAccessor
from data_lake.io.crossmatch import build_crossmatch
from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, assign_healpix, healpix_dir


def _write_catalog_tile(lake, survey, *, norder, npix, source_ids, ra, dec, extra=None):
    tile_dir = lake / "catalogs" / survey / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    hp_col = f"_healpix_norder{norder}"
    cols = {
        LAKE_JOIN_ID_COLUMN: pa.array(source_ids, type=pa.int64()),
        "ra": pa.array(ra, type=pa.float64()),
        "dec": pa.array(dec, type=pa.float64()),
        hp_col: pa.array([npix] * len(source_ids), type=pa.int64()),
    }
    if extra:
        for k, v in extra.items():
            cols[k] = pa.array(v)
    pq.write_table(pa.table(cols), tile_dir / f"Npix={npix}.parquet")
    (lake / "catalogs" / survey / "catalog_info.json").write_text(
        json.dumps({
            "hats_order": norder,
            "ra_column": "ra",
            "dec_column": "dec",
            "link_id_mode": "sequential",
            "link_id_column": LAKE_JOIN_ID_COLUMN,
            "total_rows": len(source_ids),
        })
    )


@pytest.fixture
def joined_lake(tmp_path: Path) -> Path:
    lake = tmp_path / "lake"
    norder = 5
    ra, dec = 120.0, 45.0
    npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
    _write_catalog_tile(lake, "EUCLID", norder=norder, npix=npix,
                        source_ids=[1, 2], ra=[ra, ra + 0.0005], dec=[dec, dec + 0.0005])
    _write_catalog_tile(lake, "DESI_DR1", norder=norder, npix=npix,
                        source_ids=[101, 102], ra=[ra + 0.00001, ra + 0.00051],
                        dec=[dec + 0.00001, dec + 0.00051],
                        extra={"z": [0.5, 1.2]})
    build_crossmatch(lake, "EUCLID", "DESI_DR1", radius_arcsec=2.0)
    return lake


class TestGatherProduct:
    def test_nearest_join(self, joined_lake: Path) -> None:
        region = Region.cone(120.0, 45.0, 120.0)
        sel = selection_from_region(joined_lake, "EUCLID", region)
        result = gather_product(
            joined_lake, "EUCLID",
            [PartnerSpec("DESI_DR1", 2.0, ["z"])],
            sel,
            base_columns=["ra", "dec"],
            materialize_as="EUCLID_desi",
        )
        assert result.n_rows == 2
        assert (joined_lake / "catalogs" / "EUCLID_desi" / "catalog_info.json").is_file()

        info = json.loads(
            (joined_lake / "catalogs" / "EUCLID_desi" / "catalog_info.json").read_text()
        )
        assert info["kind"] == "product"
        assert info["provenance"]["base_catalog"] == "EUCLID"

        with CatalogAccessor(joined_lake, "EUCLID_desi") as acc:
            df = acc.query("SELECT * FROM catalog ORDER BY _source_id", fmt="polars")
        assert "DESI_DR1_z" in df.columns
        assert "DESI_DR1_sep_arcsec" in df.columns
        # source 1 -> desi 101 (z=0.5), source 2 -> desi 102 (z=1.2)
        zmap = dict(zip(df["_source_id"].to_list(), df["DESI_DR1_z"].to_list()))
        assert zmap[1] == pytest.approx(0.5)
        assert zmap[2] == pytest.approx(1.2)

    def test_where_joined_filters_partner_column(self, joined_lake: Path) -> None:
        region = Region.cone(120.0, 45.0, 120.0)
        sel = selection_from_region(joined_lake, "EUCLID", region)
        result = gather_product(
            joined_lake, "EUCLID",
            [PartnerSpec("DESI_DR1", 2.0, ["z"])],
            sel,
            base_columns=["ra", "dec"],
            materialize_as="EUCLID_desi_hiz",
            where_joined="DESI_DR1_z > 1.0",
        )
        assert result.n_rows == 1
        with CatalogAccessor(joined_lake, "EUCLID_desi_hiz") as acc:
            df = acc.query("SELECT * FROM catalog", fmt="polars")
        assert df["_source_id"].to_list() == [2]

    def test_missing_crossmatch_tree_errors(self, joined_lake: Path) -> None:
        region = Region.cone(120.0, 45.0, 120.0)
        sel = selection_from_region(joined_lake, "EUCLID", region)
        with pytest.raises(FileNotFoundError):
            gather_product(
                joined_lake, "EUCLID",
                [PartnerSpec("DESI_DR1", 9.9, ["z"])],  # no tree at r9.9
                sel,
                base_columns=["ra"],
                materialize_as="bad_product",
            )

    def test_overwrite_guard(self, joined_lake: Path) -> None:
        region = Region.cone(120.0, 45.0, 120.0)
        sel = selection_from_region(joined_lake, "EUCLID", region)
        kw = dict(base_columns=["ra"], materialize_as="EUCLID_desi_ow")
        gather_product(joined_lake, "EUCLID", [PartnerSpec("DESI_DR1", 2.0, ["z"])], sel, **kw)
        with pytest.raises(FileExistsError):
            gather_product(joined_lake, "EUCLID", [PartnerSpec("DESI_DR1", 2.0, ["z"])], sel, **kw)
        gather_product(joined_lake, "EUCLID", [PartnerSpec("DESI_DR1", 2.0, ["z"])], sel,
                       overwrite=True, **kw)

    def test_tile_bounded_partner_lookup(self, tmp_path: Path) -> None:
        """Partner fetch reads only the requested HEALPix tiles."""
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix_a = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        npix_b = npix_a + 1 if npix_a < 100 else npix_a - 1

        _write_catalog_tile(
            lake, "DESI_DR1", norder=norder, npix=npix_a,
            source_ids=[101], ra=[ra], dec=[dec], extra={"z": [0.42]},
        )
        _write_catalog_tile(
            lake, "DESI_DR1", norder=norder, npix=npix_b,
            source_ids=[999], ra=[ra + 1.0], dec=[dec + 1.0], extra={"z": [9.99]},
        )

        with CatalogAccessor(lake, "DESI_DR1") as acc:
            df = acc.get_sources_by_id_in_healpix_pixels(
                [101, 999], [npix_a], columns=["z"], fmt="polars",
            )
        assert df["z"].to_list() == [pytest.approx(0.42)]
        assert len(df) == 1

    def test_partner_at_finer_healpix_order(self, tmp_path: Path) -> None:
        """Gather tile-bounds partner reads when partner hats_order > base."""
        lake = tmp_path / "lake"
        base_order, partner_order = 5, 6
        ra, dec = 120.0, 45.0
        base_npix = int(assign_healpix(np.array([ra]), np.array([dec]), base_order)[0])
        partner_npix = int(assign_healpix(np.array([ra]), np.array([dec]), partner_order)[0])

        _write_catalog_tile(
            lake, "EUCLID", norder=base_order, npix=base_npix,
            source_ids=[1], ra=[ra], dec=[dec],
        )
        _write_catalog_tile(
            lake, "DESI_DR1", norder=partner_order, npix=partner_npix,
            source_ids=[101], ra=[ra + 0.00001], dec=[dec + 0.00001],
            extra={"z": [1.7]},
        )
        build_crossmatch(lake, "EUCLID", "DESI_DR1", radius_arcsec=2.0)

        sel = selection_from_region(lake, "EUCLID", Region.cone(ra, dec, 120.0))
        result = gather_product(
            lake, "EUCLID",
            [PartnerSpec("DESI_DR1", 2.0, ["z"])],
            sel,
            base_columns=["ra", "dec"],
            materialize_as="EUCLID_desi_fine",
        )
        assert result.n_rows == 1
        with CatalogAccessor(lake, "EUCLID_desi_fine") as acc:
            df = acc.query("SELECT * FROM catalog", fmt="polars")
        assert df["DESI_DR1_z"].to_list() == [pytest.approx(1.7)]

    def test_matches_only_drops_unmatched_base_rows(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        norder = 5
        ra, dec = 120.0, 45.0
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(
            lake, "EUCLID", norder=norder, npix=npix,
            source_ids=[1, 2], ra=[ra, ra + 0.5], dec=[dec, dec + 0.5],
        )
        _write_catalog_tile(
            lake, "DESI_DR1", norder=norder, npix=npix,
            source_ids=[101], ra=[ra + 0.00001], dec=[dec + 0.00001],
            extra={"z": [0.5]},
        )
        build_crossmatch(lake, "EUCLID", "DESI_DR1", radius_arcsec=2.0)
        sel = selection_from_region(lake, "EUCLID", Region.cone(ra, dec, 120.0))

        all_result = gather_product(
            lake, "EUCLID", [PartnerSpec("DESI_DR1", 2.0, ["z"])], sel,
            base_columns=["ra"], materialize_as="all_rows", keep_all=True,
        )
        assert all_result.n_rows == 2

        matched_result = gather_product(
            lake, "EUCLID", [PartnerSpec("DESI_DR1", 2.0, ["z"])], sel,
            base_columns=["ra"], materialize_as="matched_only", keep_all=False,
        )
        assert matched_result.n_rows == 1
        with CatalogAccessor(lake, "matched_only") as acc:
            df = acc.query("SELECT * FROM catalog", fmt="polars")
    def test_partner_fetch_batches_npix_per_tile(self, joined_lake: Path, monkeypatch) -> None:
        """One batched DuckDB read per partner per base tile (not per partner Npix)."""
        calls: list[tuple] = []
        orig = CatalogAccessor.get_sources_by_id_in_healpix_pixels

        def counting(self, source_ids, npixels, columns=None, fmt="polars", **kwargs):
            calls.append((list(npixels), list(source_ids)))
            return orig(self, source_ids, npixels, columns=columns, fmt=fmt, **kwargs)

        monkeypatch.setattr(
            CatalogAccessor, "get_sources_by_id_in_healpix_pixels", counting,
        )
        region = Region.cone(120.0, 45.0, 120.0)
        sel = selection_from_region(joined_lake, "EUCLID", region)
        gather_product(
            joined_lake, "EUCLID",
            [PartnerSpec("DESI_DR1", 2.0, ["z"])],
            sel,
            base_columns=["ra"],
            materialize_as="EUCLID_desi_batch",
        )
        assert len(calls) == 1

    def test_spectra_extraction_for_product(self, tmp_path: Path) -> None:
        from test_extract_subset import SOURCES, SURVEY, _ingest_synthetic_lake
        from data_lake.discovery.gather import extract_modalities_for_product

        lake = tmp_path / "lake"
        _ingest_synthetic_lake(lake)

        # Hand-build a product catalog whose _source_id references SURVEY sources.
        norder = 5
        ids = [s[0] for s in SOURCES[:2]]
        ras = [s[1] for s in SOURCES[:2]]
        decs = [s[2] for s in SOURCES[:2]]
        npix = int(assign_healpix(np.array([ras[0]]), np.array([decs[0]]), norder)[0])
        tile_dir = lake / "catalogs" / "PROD" / healpix_dir(norder, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        hp_col = f"_healpix_norder{norder}"
        pq.write_table(
            pa.table({
                LAKE_JOIN_ID_COLUMN: pa.array(ids, type=pa.int64()),
                hp_col: pa.array([npix] * len(ids), type=pa.int64()),
            }),
            tile_dir / f"Npix={npix}.parquet",
        )
        (lake / "catalogs" / "PROD" / "catalog_info.json").write_text(
            json.dumps({
                "hats_order": norder,
                "kind": "product",
                "link_id_mode": "column:" + LAKE_JOIN_ID_COLUMN,
                "link_id_column": LAKE_JOIN_ID_COLUMN,
                "provenance": {"base_catalog": SURVEY},
            })
        )

        out = tmp_path / "bundle"
        res = extract_modalities_for_product(lake, "PROD", ["spectra"], out, survey=SURVEY)
        assert res["n_sources"] == 2
        assert (out / f"spectra_{SURVEY}.zarr").exists()

    def test_unknown_modality_raises(self, tmp_path: Path) -> None:
        from data_lake.discovery.gather import extract_modalities_for_product

        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "PROD" / healpix_dir(5, 1)
        tile_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({
                LAKE_JOIN_ID_COLUMN: pa.array([1], type=pa.int64()),
                "_healpix_norder5": pa.array([1], type=pa.int64()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        (lake / "catalogs" / "PROD" / "catalog_info.json").write_text(
            json.dumps({"hats_order": 5, "kind": "product",
                        "link_id_column": LAKE_JOIN_ID_COLUMN,
                        "provenance": {"base_catalog": "X"}})
        )
        with pytest.raises(ValueError, match="cannot extract modality"):
            extract_modalities_for_product(lake, "PROD", ["bogus"], tmp_path / "b", survey="X")


class TestGatherCli:
    def test_cli_from_area(self, joined_lake: Path) -> None:
        from click.testing import CliRunner

        from data_lake.discovery.areas import make_area, save_area
        from data_lake.discovery.gather_cli import cli

        area = make_area(
            "WideField",
            Region.cone(120.0, 45.0, 120.0),
            crossmatch_plan={
                "base_catalog": "EUCLID",
                "partners": [{"survey": "DESI_DR1", "radius_arcsec": 2.0}],
            },
            gather={
                "base": "EUCLID",
                "columns": {"EUCLID": ["ra", "dec"], "DESI_DR1": ["z"]},
                "multiplicity": "nearest",
                "materialize_as": "EUCLID_desi_area",
            },
        )
        save_area(joined_lake, area)

        result = CliRunner().invoke(cli, [str(joined_lake), "--from-area", "WideField"])
        assert result.exit_code == 0, result.output
        assert "EUCLID_desi_area" in result.output
        assert (joined_lake / "catalogs" / "EUCLID_desi_area" / "catalog_info.json").is_file()
