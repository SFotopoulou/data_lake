"""Tests for dl-extract-catalog / catalog_extract."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.table import Table

from data_lake.export.catalog_extract import (
    ExtractResult,
    _promote_arrow_type,
    extract_catalog,
    extract_from_catalog_path,
    filter_valid_sky_rows,
    parse_column_spec,
    resolve_column_specs,
    select_catalog_columns,
    specs_from_schema,
    stream_extract_from_lake_catalog,
)


def _write_fits_catalog(path: Path, n: int = 3) -> None:
    tbl = Table({
        "TARGETID": [100, 200, 300],
        "RA": [120.0, 120.1, -9999.0],
        "DEC": [45.0, 45.1, -9999.0],
        "Z": [0.1, 0.2, 0.3],
    })
    tbl.write(path, format="fits", overwrite=True)


class TestColumnSpecs:
    def test_parse_alias(self) -> None:
        assert parse_column_spec("ra:RA") == ("ra", "RA")
        assert parse_column_spec("TARGETID") == ("TARGETID", None)

    def test_resolve_case_insensitive(self) -> None:
        mapping = resolve_column_specs(["RA", "Dec", "id"], ["ra", "dec:DEC", "ID"])
        assert mapping == [("RA", "RA"), ("Dec", "DEC"), ("id", "id")]

    def test_specs_from_schema_all_columns(self) -> None:
        assert specs_from_schema(
            ["a", "b", "c"], all_columns=True, specs=[],
        ) == ["a", "b", "c"]

    def test_specs_from_schema_requires_specs(self) -> None:
        with pytest.raises(ValueError, match="column specs"):
            specs_from_schema(["a"], all_columns=False, specs=[])

    def test_promote_null_with_float(self) -> None:
        assert _promote_arrow_type(pa.null(), pa.float64()) == pa.float64()
        assert _promote_arrow_type(pa.float64(), pa.null()) == pa.float64()

    def test_promote_float_vs_string_prefers_string(self) -> None:
        assert _promote_arrow_type(pa.float64(), pa.large_string()) == pa.large_string()
        assert _promote_arrow_type(pa.string(), pa.float64()) == pa.string()


class TestExtractFromFile:
    def test_select_columns(self, tmp_path: Path) -> None:
        cat = tmp_path / "cat.fits"
        _write_fits_catalog(cat)
        out = select_catalog_columns(
            extract_from_catalog_path(cat, ["TARGETID", "RA", "DEC"]),
            ["TARGETID", "RA", "DEC"],
        )
        assert out.num_rows == 3
        assert out.column_names == ["TARGETID", "RA", "DEC"]

    def test_column_rename(self, tmp_path: Path) -> None:
        cat = tmp_path / "cat.fits"
        _write_fits_catalog(cat)
        tbl = extract_from_catalog_path(cat, ["RA:ra", "DEC:dec", "TARGETID:id"])
        assert tbl.column_names == ["ra", "dec", "id"]

    def test_valid_sky_filter(self, tmp_path: Path) -> None:
        cat = tmp_path / "cat.fits"
        _write_fits_catalog(cat)
        tbl = extract_from_catalog_path(
            cat,
            ["TARGETID", "RA", "DEC"],
            valid_sky_only=True,
        )
        assert tbl.num_rows == 2

    def test_write_parquet(self, tmp_path: Path) -> None:
        cat = tmp_path / "cat.fits"
        out = tmp_path / "sky.parquet"
        _write_fits_catalog(cat)
        table = extract_catalog(
            output=out,
            specs=["TARGETID", "RA", "DEC"],
            paths=[cat],
            valid_sky_only=True,
        )
        assert table.num_rows == 2
        assert pq.read_table(out).num_rows == 2


class TestExtractFromLake:
    def test_lake_catalog(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "TEST_SURVEY" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "source_id": pa.array([1, 2], type=pa.int64()),
                "ra": pa.array([10.0, 11.0], type=pa.float64()),
                "dec": pa.array([0.5, 0.6], type=pa.float64()),
                "_healpix_norder5": pa.array([1, 1], type=pa.int64()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        (lake / "catalogs" / "TEST_SURVEY" / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 2}',
        )
        out = tmp_path / "export.parquet"
        result = extract_catalog(
            output=out,
            specs=["source_id", "ra", "dec"],
            lake_root=lake,
            survey="TEST_SURVEY",
            engine="tiles",
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 2
        assert set(pq.read_table(out).column_names) == {"source_id", "ra", "dec"}

    def test_homogenized_product_export(self, tmp_path: Path) -> None:
        import json

        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "PROD_AB" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "_source_id": pa.array([1], type=pa.int64()),
                "ra": pa.array([10.0], type=pa.float64()),
                "dec": pa.array([0.5], type=pa.float64()),
                "phot_ab_w1": pa.array([12.699], type=pa.float32()),
                "_healpix_norder5": pa.array([1], type=pa.int64()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        (lake / "catalogs" / "PROD_AB" / "catalog_info.json").write_text(
            json.dumps({
                "hats_order": 5,
                "ra_column": "ra",
                "dec_column": "dec",
                "kind": "product",
                "product_subtype": "homogenized",
                "provenance": {
                    "transform_id": "phot_ab_v1",
                    "transform_version": 1,
                    "source_survey": "ALLWISE",
                },
            })
        )
        out = tmp_path / "ml.parquet"
        result = extract_catalog(
            output=out,
            specs=["_source_id", "phot_ab_w1"],
            lake_root=lake,
            survey="PROD_AB",
            require_homogenized=True,
            engine="tiles",
        )
        assert result.n_rows == 1
        sidecar = out.with_name(out.name + ".homogenize_provenance.json")
        assert sidecar.is_file()
        prov = json.loads(sidecar.read_text())
        assert prov["transform_id"] == "phot_ab_v1"

    def test_reject_non_homogenized_when_required(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        native_root = lake / "catalogs" / "NATIVE"
        native_root.mkdir(parents=True)
        (native_root / "catalog_info.json").write_text(
            '{"hats_order": 5, "kind": "ingested"}'
        )
        (native_root / "Norder=5" / "Dir=0").mkdir(parents=True)
        pq.write_table(
            pa.table({"ra": [1.0], "dec": [1.0]}),
            lake / "catalogs" / "NATIVE" / "Norder=5" / "Dir=0" / "Npix=1.parquet",
        )
        with pytest.raises(ValueError, match="homogenized"):
            extract_catalog(
                output=tmp_path / "x.parquet",
                specs=["ra", "dec"],
                lake_root=lake,
                survey="NATIVE",
                require_homogenized=True,
                engine="tiles",
            )

    def test_lake_catalog_tiled_output(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        for npix in (1, 2):
            tile_dir = lake / "catalogs" / "BIG" / "Norder=5" / "Dir=0"
            tile_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.table({
                    "source_id": pa.array([npix], type=pa.int64()),
                    "ra": pa.array([float(npix)], type=pa.float64()),
                    "dec": pa.array([0.0], type=pa.float64()),
                }),
                tile_dir / f"Npix={npix}.parquet",
            )
        (lake / "catalogs" / "BIG" / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 2}',
        )
        out_dir = tmp_path / "tiles"
        result = stream_extract_from_lake_catalog(
            lake,
            "BIG",
            ["source_id", "ra", "dec"],
            output_dir=out_dir,
            engine="tiles",
        )
        assert result.n_rows == 2
        assert result.n_tiles == 2
        assert (out_dir / "Norder=5" / "Dir=0" / "Npix=1.parquet").is_file()
        assert (out_dir / "Norder=5" / "Dir=0" / "Npix=2.parquet").is_file()

    def test_lake_tile_read_avoids_hive_partition_merge(self, tmp_path: Path) -> None:
        """Reading under Norder=/Dir= must not merge partition columns into the export."""
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "PART" / "Norder=1" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "_spectrum_index": pa.array([0, 1], type=pa.int64()),
                "Norder": pa.DictionaryArray.from_arrays(
                    pa.array([0, 0], type=pa.int32()),
                    pa.array([1], type=pa.int32()),
                ),
            }),
            tile_dir / "Npix=42.parquet",
        )
        (lake / "catalogs" / "PART" / "catalog_info.json").write_text(
            '{"hats_order": 1, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 2}',
        )
        out = tmp_path / "idx.parquet"
        result = extract_catalog(
            output=out,
            specs=["_spectrum_index"],
            lake_root=lake,
            survey="PART",
            engine="tiles",
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 2
        tbl = pq.read_table(out)
        assert tbl.column_names == ["_spectrum_index"]

    def test_lake_csv_export(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "S" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "ra": pa.array([10.0], type=pa.float64()),
                "dec": pa.array([0.5], type=pa.float64()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        (lake / "catalogs" / "S" / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 1}',
        )
        out = tmp_path / "out.csv"
        result = extract_catalog(
            output=out,
            specs=["ra", "dec"],
            lake_root=lake,
            survey="S",
            engine="tiles",
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 1
        text = out.read_text()
        assert "ra,dec" in text.splitlines()[0]
        assert "10.0,0.5" in text.splitlines()[1]

    def test_lake_fits_export(self, tmp_path: Path) -> None:
        from astropy.io import fits
        from astropy.table import Table

        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "S" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({"TARGETID": pa.array([42], type=pa.int64())}),
            tile_dir / "Npix=1.parquet",
        )
        (lake / "catalogs" / "S" / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 1}',
        )
        out = tmp_path / "out.fits"
        result = extract_catalog(
            output=out,
            specs=["TARGETID"],
            lake_root=lake,
            survey="S",
            engine="tiles",
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 1
        with fits.open(out) as hdul:
            tbl = Table(hdul[1].data)
        assert int(tbl["TARGETID"][0]) == 42

    def test_lake_fits_nullable_string_column(self, tmp_path: Path) -> None:
        """Nullable Arrow strings must export to FITS without mixed object dtypes."""
        from astropy.io import fits
        from astropy.table import Table

        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "S" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "id": pa.array([1, 2], type=pa.int64()),
                "CLASS": pa.array(["STAR  ", None], type=pa.large_string()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        (lake / "catalogs" / "S" / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 2}',
        )
        out = tmp_path / "class.fits"
        result = extract_catalog(
            output=out,
            specs=["id", "CLASS"],
            lake_root=lake,
            survey="S",
            engine="tiles",
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 2
        with fits.open(out) as hdul:
            tbl = Table(hdul[1].data)
        assert str(tbl["CLASS"][0]).strip() == "STAR"
        assert str(tbl["CLASS"][1]).strip() == ""

    def test_lake_fits_float_pad_and_string_tiles(self, tmp_path: Path) -> None:
        """Legacy float64 null pads + string tiles unify to string for FITS."""
        from astropy.io import fits
        from astropy.table import Table
        from data_lake.ingest.fits_to_parquet import healpix_dir

        lake = tmp_path / "lake"
        root = lake / "catalogs" / "S"
        d0 = root / healpix_dir(5, 0)
        d0.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "id": pa.array([1], type=pa.int64()),
                "CLASS": pa.array([None], type=pa.float64()),
            }),
            d0 / "Npix=0.parquet",
        )
        d1 = root / healpix_dir(5, 10_500)
        d1.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "id": pa.array([2], type=pa.int64()),
                "CLASS": pa.array(["STAR  "], type=pa.large_string()),
            }),
            d1 / "Npix=10500.parquet",
        )
        (root / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 2}',
        )
        out = tmp_path / "mixed.fits"
        result = extract_catalog(
            output=out,
            specs=["id", "CLASS"],
            lake_root=lake,
            survey="S",
            engine="tiles",
        )
        assert result.n_rows == 2
        with fits.open(out) as hdul:
            tbl = Table(hdul[1].data)
        classes = {int(r["id"]): str(r["CLASS"]).strip() for r in tbl}
        assert classes[1] == ""
        assert classes[2] == "STAR"

    def test_lake_all_columns_parquet(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "PROD" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "_source_id": pa.array([1], type=pa.int64()),
                "ra": pa.array([10.0], type=pa.float64()),
                "dec": pa.array([0.5], type=pa.float64()),
                "PARTNER_z": pa.array([0.42], type=pa.float64()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        (lake / "catalogs" / "PROD" / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 1}',
        )
        out = tmp_path / "all.parquet"
        result = extract_catalog(
            output=out,
            all_columns=True,
            lake_root=lake,
            survey="PROD",
            engine="tiles",
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 1
        assert set(pq.read_table(out).column_names) == {
            "_source_id", "ra", "dec", "PARTNER_z",
        }

    def test_lake_all_columns_fits(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "PROD" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "TARGETID": pa.array([42], type=pa.int64()),
                "Z": pa.array([0.5], type=pa.float64()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        (lake / "catalogs" / "PROD" / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 1}',
        )
        out = tmp_path / "all.fits"
        result = extract_catalog(
            output=out,
            all_columns=True,
            lake_root=lake,
            survey="PROD",
            engine="tiles",
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 1
        assert set(result.column_names) == {"TARGETID", "Z"}

    def test_lake_export_unifies_null_and_float_columns(self, tmp_path: Path) -> None:
        """Gather-style tiles: null-type partner cols in one tile, float64 in another."""
        lake = tmp_path / "lake"
        for npix, z_vals in ((1, None), (2, [0.5])):
            tile_dir = lake / "catalogs" / "PROD" / "Norder=5" / "Dir=0"
            tile_dir.mkdir(parents=True, exist_ok=True)
            cols: dict = {
                "_source_id": pa.array([npix], type=pa.int64()),
                "ra": pa.array([10.0], type=pa.float64()),
            }
            if z_vals is None:
                cols["PARTNER_z"] = pa.array([None], type=pa.null())
            else:
                cols["PARTNER_z"] = pa.array(z_vals, type=pa.float64())
            pq.write_table(pa.table(cols), tile_dir / f"Npix={npix}.parquet")
        (lake / "catalogs" / "PROD" / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 2}',
        )
        out = tmp_path / "merged.fits"
        result = extract_catalog(
            output=out,
            all_columns=True,
            lake_root=lake,
            survey="PROD",
            engine="tiles",
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 2
        assert out.is_file()


class TestExtractCatalogCli:
    def test_cli_all_columns(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.export.catalog_extract import cli

        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "S" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({"ra": pa.array([1.0], type=pa.float64())}),
            tile_dir / "Npix=1.parquet",
        )
        (lake / "catalogs" / "S" / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 1}',
        )
        out = tmp_path / "out.parquet"
        result = CliRunner().invoke(
            cli,
            [
                "--lake-root", str(lake), "--survey", "S",
                "--all-columns", "-o", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert pq.read_table(out).column_names == ["ra"]

    def test_cli_rejects_all_columns_and_column(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.export.catalog_extract import cli

        result = CliRunner().invoke(
            cli,
            ["--all-columns", "-c", "ra", "-o", "out.parquet"],
        )
        assert result.exit_code != 0
        assert "not both" in result.output

    def test_cli_requires_column_or_all_columns(self) -> None:
        from click.testing import CliRunner

        from data_lake.export.catalog_extract import cli

        result = CliRunner().invoke(cli, ["-o", "out.parquet"])
        assert result.exit_code != 0
        assert "--all-columns" in result.output

    def test_cli_lake_root_from_config(self, tmp_path: Path, monkeypatch) -> None:
        import textwrap

        from click.testing import CliRunner

        from data_lake.config import CONFIG_FILENAME, SCHEMA_VERSION
        from data_lake.export.catalog_extract import cli

        lake_data = tmp_path / "lake_data"
        tile_dir = lake_data / "catalogs" / "S" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({"ra": pa.array([1.0], type=pa.float64())}),
            tile_dir / "Npix=1.parquet",
        )
        (lake_data / "catalogs" / "S" / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 1}',
        )
        cfg_path = tmp_path / CONFIG_FILENAME
        cfg_path.write_text(textwrap.dedent(f"""
            schema_version = "{SCHEMA_VERSION}"

            [lake]
            name = "test"
            root = "{lake_data}"
            description = "test"
        """).strip() + "\n")
        monkeypatch.setenv("DATA_LAKE_CONFIG", str(cfg_path))
        out = tmp_path / "out.parquet"
        result = CliRunner().invoke(
            cli,
            ["--survey", "S", "--all-columns", "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert pq.read_table(out).column_names == ["ra"]


class TestFilterValidSky:
    def test_requires_sky_columns(self) -> None:
        tbl = pa.table({"x": [1]})
        with pytest.raises(ValueError, match="RA/Dec"):
            filter_valid_sky_rows(tbl)
