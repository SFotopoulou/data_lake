"""Tests for dl-extract-catalog / catalog_extract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.table import Table

from data_lake.export.catalog_extract import (
    ExtractResult,
    _extract_staging_dir,
    _promote_arrow_type,
    expand_column_args,
    extract_catalog,
    extract_from_catalog_path,
    filter_valid_sky_rows,
    map_ids_via_crossmatch,
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
    def test_staging_dir_is_under_cwd(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with _extract_staging_dir() as staging:
            assert staging.is_dir()
            assert staging.parent.resolve() == tmp_path.resolve()
            assert staging.name.startswith("dl_extract_")

    def test_staging_dir_ignores_parent_stays_cwd(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        other = tmp_path / "nested" / "out"
        with _extract_staging_dir(other) as staging:
            assert staging.parent.resolve() == tmp_path.resolve()
            assert not other.exists()

    def test_parse_alias(self) -> None:
        assert parse_column_spec("ra:RA") == ("ra", "RA")
        assert parse_column_spec("TARGETID") == ("TARGETID", None)

    def test_expand_column_args_comma_and_repeat(self) -> None:
        assert expand_column_args(["TARGETID,RA", "DEC"]) == [
            "TARGETID", "RA", "DEC",
        ]
        assert expand_column_args(["ra:RA,dec:DEC"]) == ["ra:RA", "dec:DEC"]
        assert expand_column_args(["  a , b  ", "c"]) == ["a", "b", "c"]

    def test_expand_column_args_rejects_empty_segment(self) -> None:
        with pytest.raises(ValueError, match="empty column spec"):
            expand_column_args(["RA,,DEC"])

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

    def test_where_and_limit_on_raw_file(self, tmp_path: Path) -> None:
        cat = tmp_path / "cat.fits"
        tbl = Table({
            "TARGETID": [100, 200, 300],
            "RA": [120.0, 120.1, 120.2],
            "DEC": [45.0, 45.1, 45.2],
            "Z": [0.1, 0.2, 0.3],
            "SPECTYPE": ["STAR", "GALAXY", "STAR"],
        })
        tbl.write(cat, format="fits", overwrite=True)
        out = tmp_path / "stars.parquet"
        result = extract_catalog(
            output=out,
            specs=["TARGETID", "Z"],
            paths=[cat],
            where_sql="upper(trim(SPECTYPE)) = 'STAR'",
            row_limit=1,
        )
        assert result.num_rows == 1
        assert pq.read_table(out).column("TARGETID").to_pylist() == [100]


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

    def _star_lake(self, tmp_path: Path) -> Path:
        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "STARS" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "TARGETID": pa.array([11, 12, 13, 14], type=pa.int64()),
                "SPECTYPE": pa.array(["STAR", "GALAXY", "STAR", "QSO"], type=pa.string()),
                "Z": pa.array([0.0, 0.5, 0.0, 1.2], type=pa.float64()),
                "ra": pa.array([10.0, 11.0, -9999.0, 13.0], type=pa.float64()),
                "dec": pa.array([0.5, 0.6, -9999.0, 0.8], type=pa.float64()),
                "_healpix_norder5": pa.array([1, 1, 1, 1], type=pa.int64()),
            }),
            tile_dir / "Npix=1.parquet",
        )
        (lake / "catalogs" / "STARS" / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 4}',
        )
        return lake

    def test_where_on_column_not_exported(self, tmp_path: Path) -> None:
        lake = self._star_lake(tmp_path)
        out = tmp_path / "stars.parquet"
        result = extract_catalog(
            output=out,
            specs=["TARGETID", "Z"],
            lake_root=lake,
            survey="STARS",
            engine="tiles",
            where_sql="SPECTYPE = 'STAR'",
        )
        assert isinstance(result, ExtractResult)
        tbl = pq.read_table(out)
        assert tbl.column_names == ["TARGETID", "Z"]
        assert tbl.num_rows == 2
        assert tbl.column("TARGETID").to_pylist() == [11, 13]

    def test_limit(self, tmp_path: Path) -> None:
        lake = self._star_lake(tmp_path)
        out = tmp_path / "lim.parquet"
        result = extract_catalog(
            output=out,
            specs=["TARGETID"],
            lake_root=lake,
            survey="STARS",
            engine="tiles",
            row_limit=2,
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 2
        assert pq.read_table(out).column("TARGETID").to_pylist() == [11, 12]

    def test_limit_with_valid_sky_only(self, tmp_path: Path) -> None:
        lake = self._star_lake(tmp_path)
        out = tmp_path / "sky.parquet"
        result = extract_catalog(
            output=out,
            specs=["TARGETID", "ra", "dec"],
            lake_root=lake,
            survey="STARS",
            engine="tiles",
            valid_sky_only=True,
            row_limit=2,
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 2
        assert pq.read_table(out).column("TARGETID").to_pylist() == [11, 12]

    def test_where_duckdb_engine(self, tmp_path: Path) -> None:
        lake = self._star_lake(tmp_path)
        out = tmp_path / "duck.parquet"
        result = extract_catalog(
            output=out,
            specs=["TARGETID", "Z"],
            lake_root=lake,
            survey="STARS",
            engine="duckdb",
            where_sql="SPECTYPE = 'STAR'",
            row_limit=1,
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 1
        assert pq.read_table(out).column("TARGETID").to_pylist() == [11]

    def _id_lake(self, tmp_path: Path) -> Path:
        lake = tmp_path / "lake"
        survey = lake / "catalogs" / "IDS"
        for npix, ids in ((1, [11, 12]), (2, [13, 14])):
            tile_dir = survey / "Norder=5" / "Dir=0"
            tile_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.table({
                    "_source_id": pa.array(ids, type=pa.int64()),
                    "TARGETID": pa.array(ids, type=pa.int64()),
                    "Z": pa.array([float(i) / 10 for i in ids], type=pa.float64()),
                    "_healpix_norder5": pa.array([npix, npix], type=pa.int64()),
                }),
                tile_dir / f"Npix={npix}.parquet",
            )
        (survey / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "column:TARGETID", "total_rows": 4}',
        )
        return lake

    def test_target_list_filters_source_id(self, tmp_path: Path) -> None:
        lake = self._id_lake(tmp_path)
        out = tmp_path / "subset.parquet"
        result = extract_catalog(
            output=out,
            specs=["TARGETID", "Z"],
            lake_root=lake,
            survey="IDS",
            engine="tiles",
            source_ids=np.array([12, 14, 99], dtype=np.int64),
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 2
        tbl = pq.read_table(out)
        assert tbl.column("TARGETID").to_pylist() == [12, 14]
        assert "_source_id" not in tbl.column_names

    def test_target_list_prunes_to_hit_tiles(self, tmp_path: Path, monkeypatch) -> None:
        """Sparse --target-list must not open every survey tile."""
        lake = self._id_lake(tmp_path)
        from data_lake.export import catalog_extract as ce

        opened: list[Path] = []
        real_read = ce._read_catalog_tile

        def _tracking_read(tile_path: Path, columns):
            opened.append(Path(tile_path))
            return real_read(tile_path, columns)

        monkeypatch.setattr(ce, "_read_catalog_tile", _tracking_read)
        # Force Arrow tile path (no DuckDB filter) so _read_catalog_tile is used.
        # With prune, only the tile holding id 14 (Npix=2) should be opened.
        out_dir = tmp_path / "pruned"
        result = extract_catalog(
            output_dir=out_dir,
            specs=["TARGETID"],
            lake_root=lake,
            survey="IDS",
            engine="tiles",
            source_ids=np.array([14], dtype=np.int64),
        )
        assert result.n_rows == 1
        assert result.n_tiles == 1
        # Parallel output-dir path uses DuckDB per tile; prune still limits work.
        written = list(out_dir.rglob("Npix=*.parquet"))
        assert len(written) == 1
        assert written[0].name == "Npix=2.parquet"

    def test_auto_limit_uses_tile_engine(self, tmp_path: Path, monkeypatch) -> None:
        """engine=auto + --limit must not take the DuckDB ORDER BY path."""
        lake = self._id_lake(tmp_path)
        from data_lake.export import catalog_extract as ce

        called = {"duckdb": False}

        def _boom(*_a, **_k):
            called["duckdb"] = True
            raise AssertionError("DuckDB path should be skipped for auto+limit")

        monkeypatch.setattr(ce, "_stream_lake_via_duckdb", _boom)
        out = tmp_path / "lim.parquet"
        result = extract_catalog(
            output=out,
            specs=["TARGETID"],
            lake_root=lake,
            survey="IDS",
            engine="auto",
            row_limit=1,
        )
        assert not called["duckdb"]
        assert result.n_rows == 1

    def _partner_lake(self, tmp_path: Path) -> Path:
        """DESI-like IDS + LEG partner + DESI_x_LEG crossmatch tree."""
        lake = self._id_lake(tmp_path)
        leg = lake / "catalogs" / "LEG"
        for npix, ids, flux in (
            (1, [101, 102], [1.5, 2.5]),
            (2, [103], [3.5]),
        ):
            tile_dir = leg / "Norder=5" / "Dir=0"
            tile_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.table({
                    "_source_id": pa.array(ids, type=pa.int64()),
                    "ra": pa.array([float(i) for i in ids], type=pa.float64()),
                    "flux_g": pa.array(flux, type=pa.float64()),
                    "_healpix_norder5": pa.array([npix] * len(ids), type=pa.int64()),
                }),
                tile_dir / f"Npix={npix}.parquet",
            )
        (leg / "catalog_info.json").write_text(
            '{"hats_order": 5, "ra_column": "ra", "dec_column": "dec", '
            '"link_id_mode": "sequential", "total_rows": 3}',
        )
        xm = lake / "crossmatch" / "IDS_x_LEG__r1.0"
        xm_tile = xm / "Norder=5" / "Dir=0"
        xm_tile.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "source_id_a": pa.array([11, 12, 13], type=pa.int64()),
                "source_id_b": pa.array([101, 102, 103], type=pa.int64()),
                "sep_arcsec": pa.array([0.1, 0.2, 0.3], type=pa.float64()),
            }),
            xm_tile / "Npix=1.parquet",
        )
        (xm / "crossmatch_info.json").write_text(
            '{"survey_a": "IDS", "survey_b": "LEG", "match_radius_arcsec": 1.0, '
            '"hats_order": 5}',
        )
        return lake

    def test_map_ids_via_crossmatch(self, tmp_path: Path) -> None:
        lake = self._partner_lake(tmp_path)
        mapped = map_ids_via_crossmatch(
            lake,
            "IDS_x_LEG__r1.0",
            extract_survey="LEG",
            source_ids=np.array([11, 13, 99], dtype=np.int64),
        )
        assert sorted(mapped.tolist()) == [101, 103]

    def test_via_crossmatch_extract(self, tmp_path: Path) -> None:
        lake = self._partner_lake(tmp_path)
        ids = tmp_path / "desi_ids.parquet"
        pq.write_table(
            pa.table({"TARGETID": pa.array([12, 11], type=pa.int64())}),
            ids,
        )
        out = tmp_path / "leg.parquet"
        from click.testing import CliRunner

        from data_lake.export.catalog_extract import cli

        result = CliRunner().invoke(
            cli,
            [
                "--lake-root", str(lake),
                "--survey", "LEG",
                "--via-crossmatch", "IDS_x_LEG__r1.0",
                "--target-list", str(ids),
                "--target-id-col", "TARGETID",
                "-c", "_source_id,flux_g",
                "-o", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        tbl = pq.read_table(out)
        assert sorted(tbl.column("_source_id").to_pylist()) == [101, 102]
        assert sorted(tbl.column("flux_g").to_pylist()) == [1.5, 2.5]

    def test_via_crossmatch_requires_target_list(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.export.catalog_extract import cli

        result = CliRunner().invoke(
            cli,
            [
                "--lake-root", str(tmp_path),
                "--survey", "LEG",
                "--via-crossmatch", "IDS_x_LEG__r1.0",
                "--all-columns",
                "-o", str(tmp_path / "out.parquet"),
            ],
        )
        assert result.exit_code != 0
        assert "--via-crossmatch requires --target-list" in result.output

    def test_target_list_duckdb_engine(self, tmp_path: Path, caplog) -> None:
        lake = self._id_lake(tmp_path)
        out = tmp_path / "subset.parquet"
        with caplog.at_level("WARNING"):
            result = extract_catalog(
                output=out,
                specs=["Z"],
                lake_root=lake,
                survey="IDS",
                engine="duckdb",
                source_ids=np.array([11, 11, 13], dtype=np.int64),
                row_limit=1,
            )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 1
        assert pq.read_table(out).column("Z").to_pylist() == [1.1]
        assert "falling back" not in caplog.text

    def test_target_list_output_dir(self, tmp_path: Path) -> None:
        lake = self._id_lake(tmp_path)
        out_dir = tmp_path / "tiles"
        result = extract_catalog(
            output_dir=out_dir,
            specs=["TARGETID"],
            lake_root=lake,
            survey="IDS",
            engine="tiles",
            source_ids=np.array([14], dtype=np.int64),
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 1
        assert result.n_tiles == 1
        written = list(out_dir.rglob("Npix=*.parquet"))
        assert len(written) == 1
        assert pq.read_table(written[0]).column("TARGETID").to_pylist() == [14]

    def test_target_list_empty_writes_no_rows(self, tmp_path: Path) -> None:
        lake = self._id_lake(tmp_path)
        out = tmp_path / "empty.parquet"
        result = extract_catalog(
            output=out,
            specs=["TARGETID"],
            lake_root=lake,
            survey="IDS",
            engine="duckdb",
            source_ids=np.array([], dtype=np.int64),
        )
        assert isinstance(result, ExtractResult)
        assert result.n_rows == 0

    def test_target_list_rejected_with_where(self, tmp_path: Path) -> None:
        lake = self._id_lake(tmp_path)
        with pytest.raises(ValueError, match="not both"):
            extract_catalog(
                output=tmp_path / "bad.parquet",
                specs=["TARGETID"],
                lake_root=lake,
                survey="IDS",
                where_sql="Z > 0",
                source_ids=np.array([11], dtype=np.int64),
            )

    def test_target_list_requires_lake(self, tmp_path: Path) -> None:
        cat = tmp_path / "cat.fits"
        _write_fits_catalog(cat)
        with pytest.raises(ValueError, match="ingested catalog"):
            extract_catalog(
                output=tmp_path / "bad.parquet",
                specs=["TARGETID"],
                paths=[cat],
                source_ids=np.array([100], dtype=np.int64),
            )

    def test_where_rejects_semicolon(self, tmp_path: Path) -> None:
        lake = self._star_lake(tmp_path)
        with pytest.raises(ValueError, match="single boolean"):
            extract_catalog(
                output=tmp_path / "bad.parquet",
                specs=["TARGETID"],
                lake_root=lake,
                survey="STARS",
                engine="tiles",
                where_sql="SPECTYPE = 'STAR'; SELECT 1",
            )

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

    def test_cli_comma_separated_columns(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.export.catalog_extract import cli

        lake = tmp_path / "lake"
        tile_dir = lake / "catalogs" / "S" / "Norder=5" / "Dir=0"
        tile_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "ra": pa.array([1.0], type=pa.float64()),
                "dec": pa.array([2.0], type=pa.float64()),
                "id": pa.array([9], type=pa.int64()),
            }),
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
                "-c", "id,ra,dec", "-o", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert pq.read_table(out).column_names == ["id", "ra", "dec"]

    def test_cli_requires_column_or_all_columns(self) -> None:
        from click.testing import CliRunner

        from data_lake.export.catalog_extract import cli

        result = CliRunner().invoke(cli, ["-o", "out.parquet"])
        assert result.exit_code != 0
        assert "--all-columns" in result.output

    def test_cli_target_list(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.export.catalog_extract import cli

        lake = TestExtractFromLake()._id_lake(tmp_path)
        ids = tmp_path / "ids.parquet"
        pq.write_table(
            pa.table({"TARGETID": pa.array([12, 14, 12], type=pa.int64())}),
            ids,
        )
        out = tmp_path / "out.parquet"
        result = CliRunner().invoke(
            cli,
            [
                "--lake-root", str(lake),
                "--survey", "IDS",
                "-c", "TARGETID",
                "--target-list", str(ids),
                "--target-id-col", "TARGETID",
                "-o", str(out),
                "--engine", "tiles",
            ],
        )
        assert result.exit_code == 0, result.output
        assert pq.read_table(out).column("TARGETID").to_pylist() == [12, 14]

    def test_cli_target_list_rejects_where(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.export.catalog_extract import cli

        ids = tmp_path / "ids.parquet"
        ids.write_bytes(b"")
        result = CliRunner().invoke(
            cli,
            [
                "--survey", "IDS",
                "-c", "TARGETID",
                "--target-list", str(ids),
                "--where", "Z > 0",
                "-o", str(tmp_path / "out.parquet"),
            ],
        )
        assert result.exit_code != 0
        assert "not both" in result.output

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

    def test_cli_where_on_raw_file(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.export.catalog_extract import cli

        cat = tmp_path / "cat.fits"
        _write_fits_catalog(cat)
        out = tmp_path / "out.parquet"
        result = CliRunner().invoke(
            cli,
            [
                str(cat), "-o", str(out),
                "-c", "TARGETID",
                "--where", "Z > 0.15",
                "--limit", "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert pq.read_table(out).column("TARGETID").to_pylist() == [200]


class TestFilterValidSky:
    def test_requires_sky_columns(self) -> None:
        tbl = pa.table({"x": [1]})
        with pytest.raises(ValueError, match="RA/Dec"):
            filter_valid_sky_rows(tbl)
