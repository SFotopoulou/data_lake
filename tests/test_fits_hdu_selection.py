"""``--hdu`` table-extension selection across catalog ingest paths (MEF FITS)."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.table import Table
from click.testing import CliRunner

from data_lake.cli_utils import hdu_option
from data_lake.ingest.catalog_parallel_ingest import ingest_catalogs_parallel
from data_lake.ingest.catalog_shard_ingest import run_map_phase, run_reduce_phase
from data_lake.ingest.check_fits_table_format import cli as check_cli
from data_lake.ingest.check_fits_table_format import inspect_fits_table_format
from data_lake.ingest.fits_to_parquet import (
    _read_fits_catalog_arrow,
    catalog_source_row_count,
    ingest_catalog,
)
from data_lake.ingest.recommend_norder import recommend_catalog_norder
from data_lake.io.fits_read import (
    FitsReadPolicy,
    default_fits_read_policy,
    open_fits,
    parse_hdu_spec,
    resolve_table_hdus,
)

SURVEY = "MEF_TEST"


def _objects(ids: range, *, ra0: float) -> Table:
    n = len(ids)
    return Table({
        "OBJID": np.array(list(ids), dtype=np.int64),
        "RA": np.linspace(ra0, ra0 + 20.0, n),
        "DEC": np.linspace(-10.0, 10.0, n),
        "MAG": np.arange(n, dtype=np.float32),
    })


def _write_mef(path: Path, *, mixed: bool = False) -> Path:
    """PRIMARY, IMAGE 'STAMP', BINTABLE OBJECTS v1 (ids 1–10), OBJECTS v2 (11–20).

    With *mixed*, append a BINTABLE 'SUMMARY' whose columns differ.
    """
    hdus = [
        fits.PrimaryHDU(),
        fits.ImageHDU(np.zeros((4, 4), dtype=np.float32), name="STAMP"),
        fits.BinTableHDU(_objects(range(1, 11), ra0=10.0), name="OBJECTS", ver=1),
        fits.BinTableHDU(_objects(range(11, 21), ra0=200.0), name="OBJECTS", ver=2),
    ]
    if mixed:
        hdus.append(fits.BinTableHDU(Table({"NOBJ": [20]}), name="SUMMARY"))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


@pytest.fixture()
def mef(tmp_path: Path) -> Path:
    return _write_mef(tmp_path / "mef.fits")


@pytest.fixture()
def mef_mixed(tmp_path: Path) -> Path:
    return _write_mef(tmp_path / "mef_mixed.fits", mixed=True)


def _lake_ids(lake: Path, survey: str = SURVEY) -> list[int]:
    tiles = sorted((lake / "catalogs" / survey).rglob("Npix=*.parquet"))
    table = pa.concat_tables([pq.ParquetFile(str(p)).read() for p in tiles])
    return sorted(table.column("OBJID").to_pylist())


def _catalog_info(lake: Path, survey: str = SURVEY) -> dict:
    return json.loads((lake / "catalogs" / survey / "catalog_info.json").read_text())


def _ingest(src: Path, lake: Path, **kw) -> None:
    ingest_catalog(
        src, lake, SURVEY, ra_col="RA", dec_col="DEC", norder=3, link_id_col="OBJID", **kw,
    )


class TestParseAndResolve:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None), ("", None), ("  ", None), ("ALL", "all"), ("2", "2"),
            ("002", "2"), (3, "3"), ("objects", "objects"), ("SCI, 2", "SCI,2"),
        ],
    )
    def test_parse(self, raw, expected) -> None:
        assert parse_hdu_spec(raw) == expected

    @pytest.mark.parametrize("raw", ["-1", "SCI,x", ",2"])
    def test_parse_rejects(self, raw) -> None:
        with pytest.raises(ValueError):
            parse_hdu_spec(raw)

    def test_resolve(self, mef: Path) -> None:
        with open_fits(mef) as hdul:
            assert resolve_table_hdus(hdul, None) == [2]
            assert resolve_table_hdus(hdul, "all") == [2, 3]
            assert resolve_table_hdus(hdul, "3") == [3]
            assert resolve_table_hdus(hdul, "objects,2") == [3]

    @pytest.mark.parametrize(
        ("spec", "match"),
        [
            ("OBJECTS", "ambiguous"),
            ("1", "not a table"),
            ("STAMP", "not a table"),
            ("9", "out of range"),
            ("MISSING", "not found"),
        ],
    )
    def test_resolve_errors_list_hdus(self, mef: Path, spec: str, match: str) -> None:
        with open_fits(mef) as hdul, pytest.raises(ValueError, match=match) as exc:
            resolve_table_hdus(hdul, spec, source=mef.name)
        assert "2:OBJECTS(BinTableHDU)" in str(exc.value)

    def test_policy_env_and_override(self, monkeypatch) -> None:
        monkeypatch.setenv("DATA_LAKE_FITS_HDU", "all")
        assert FitsReadPolicy.from_env().table_hdu == "all"
        assert default_fits_read_policy(table_hdu="OBJECTS").table_hdu == "OBJECTS"


class TestReaders:
    def test_read_arrow_selection(self, mef: Path) -> None:
        def ids(spec):
            policy = FitsReadPolicy(table_hdu=spec)
            return _read_fits_catalog_arrow(mef, fits_read_policy=policy).column("OBJID").to_pylist()

        assert ids(None) == list(range(1, 11))
        assert ids("OBJECTS,2") == list(range(11, 21))
        assert ids("all") == list(range(1, 21))

    def test_all_requires_matching_columns(self, mef_mixed: Path) -> None:
        with pytest.raises(ValueError, match="columns differ"):
            _read_fits_catalog_arrow(mef_mixed, fits_read_policy=FitsReadPolicy(table_hdu="all"))

    def test_row_count(self, mef: Path) -> None:
        assert catalog_source_row_count(mef) == 10
        assert catalog_source_row_count(mef, fits_read_policy=FitsReadPolicy(table_hdu="all")) == 20


class TestIngestCatalog:
    @pytest.mark.parametrize("streaming", [False, True])
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [(None, range(1, 11)), ("OBJECTS,2", range(11, 21)), ("all", range(1, 21))],
    )
    def test_selection(self, mef: Path, tmp_path: Path, streaming, spec, expected) -> None:
        lake = tmp_path / "lake"
        _ingest(mef, lake, streaming=streaming, fits_hdu=spec)
        assert _lake_ids(lake) == list(expected)

    def test_streaming_parallel_all(self, mef: Path, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _ingest(mef, lake, streaming_parallel=3, tile_mode="append", fits_hdu="all")
        assert _lake_ids(lake) == list(range(1, 21))
        assert _catalog_info(lake)["fits_hdu"] == "all"

    def test_streaming_all_mismatch(self, mef_mixed: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="columns differ"):
            _ingest(mef_mixed, tmp_path / "lake", streaming=True, fits_hdu="all")

    def test_provenance_recorded_and_warns(self, mef: Path, tmp_path: Path, caplog) -> None:
        lake = tmp_path / "lake"
        _ingest(mef, lake, fits_hdu="OBJECTS,2")
        assert _catalog_info(lake)["fits_hdu"] == "OBJECTS,2"

        with caplog.at_level(logging.WARNING):
            _ingest(mef, lake, tile_mode="append")
        assert any("--hdu" in r.getMessage() for r in caplog.records)
        # First ingest's selection stays recorded; the run still succeeds.
        assert _catalog_info(lake)["fits_hdu"] == "OBJECTS,2"
        assert _lake_ids(lake) == list(range(1, 21))

    def test_default_not_recorded(self, mef: Path, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _ingest(mef, lake)
        assert "fits_hdu" not in _catalog_info(lake)


class TestParallelAndShard:
    def test_parallel_batch_all(self, tmp_path: Path) -> None:
        files = [_write_mef(tmp_path / f"f{i}.fits") for i in range(2)]
        lake = tmp_path / "lake"
        result = ingest_catalogs_parallel(
            files, output_root=lake, survey_name=SURVEY, n_workers=2,
            ra_col="RA", dec_col="DEC", norder=3, link_id_col="OBJID",
            executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
            show_progress=False, fits_hdu="all",
        )
        assert result["n_files_failed"] == 0
        # Same IDs in both files: dedup keeps 20 unique rows.
        assert _lake_ids(lake) == list(range(1, 21))
        assert _catalog_info(lake)["fits_hdu"] == "all"

    def test_shard_map_reduce(self, mef: Path, tmp_path: Path, caplog) -> None:
        shards, lake = tmp_path / "shards", tmp_path / "lake"
        run_map_phase(
            [mef], shards, SURVEY, "RA", "DEC", 3, link_id_col="OBJID", fits_hdu="all",
        )
        info = json.loads((shards / SURVEY / "shard_info.json").read_text())
        assert info["fits_hdu"] == "all"

        # A later map task with another selection warns but is not rejected.
        other = _write_mef(tmp_path / "other.fits")
        with caplog.at_level(logging.WARNING):
            result = run_map_phase(
                [other], shards, SURVEY, "RA", "DEC", 3,
                link_id_col="OBJID", fits_hdu="OBJECTS,2",
            )
        assert result["n_files_failed"] == 0
        assert any("--hdu" in r.getMessage() for r in caplog.records)

        run_reduce_phase(shards, SURVEY, lake / "catalogs" / SURVEY, 3)
        assert _lake_ids(lake) == list(range(1, 21))
        assert _catalog_info(lake)["fits_hdu"] == "all"


class TestPreflightTools:
    def test_check_format_reports_mef(self, mef: Path) -> None:
        rep = inspect_fits_table_format(mef)
        assert rep.ok and rep.hdu_indices == (2,) and rep.n_table_hdus == 2
        assert "--hdu" in (rep.hint or "")
        rep_all = inspect_fits_table_format(mef, fits_hdu="all")
        assert rep_all.hdu_indices == (2, 3) and rep_all.est_source_count == 20

    def test_check_format_cli(self, mef: Path) -> None:
        result = CliRunner().invoke(check_cli, [str(mef), "--hdu", "all", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["naxis2"] == 20
        bad = CliRunner().invoke(check_cli, [str(mef), "--hdu", "-3"])
        assert bad.exit_code == 2 and "--hdu" in bad.output

    def test_recommend_norder(self, mef: Path) -> None:
        rec = recommend_catalog_norder([mef], ra_col="RA", dec_col="DEC", fits_hdu="all")
        assert rec.total_rows == 20


def test_hdu_option_reads_env(monkeypatch) -> None:
    @click.command()
    @hdu_option
    def cmd(fits_hdu):
        click.echo(repr(fits_hdu))

    runner = CliRunner()
    assert runner.invoke(cmd, []).output.strip() == "None"
    assert runner.invoke(cmd, ["--hdu", "Objects , 2"]).output.strip() == "'Objects,2'"
    monkeypatch.setenv("DATA_LAKE_FITS_HDU", "all")
    assert runner.invoke(cmd, []).output.strip() == "'all'"
