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
