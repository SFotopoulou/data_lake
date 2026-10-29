"""Joined-extension catalog ingest (dl-ingest-catalog-mef).

Synthetic MEF only: three table HDUs, shared keys, RA/DEC on the first,
a length-5 vector column on the second.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.table import Table
from click.testing import CliRunner

from data_lake.ingest.check_fits_table_format import inspect_fits_table_format
from data_lake.ingest.fits_to_parquet import (
    LAKE_JOIN_ID_COLUMN,
    ingest_catalog_mef,
    inspect_joinable_table_hdus,
    read_joined_catalog_hdus,
)
from data_lake.io.fits_read import open_fits

SURVEY = "MEF_JOIN"
N = 12
LINK_ID = "TARGETID,SURVEY,PROGRAM"


def _write_joined_mef(
    path: Path,
    *,
    n: int = N,
    id0: int = 1,
    mismatch_row: int | None = None,
    extra_collision: bool = False,
    unequal: bool = False,
    no_shared: bool = False,
) -> Path:
    """PRIMARY + METADATA + SPECPHOT + FASTSPEC, FastSpec-shaped layout."""
    ids = np.arange(id0, id0 + n, dtype=np.int64)
    survey = np.array(["cmx"] * n)
    program = np.array(["other"] * n)
    ra = np.linspace(10.0, 20.0, n)
    dec = np.linspace(-5.0, 5.0, n)
    meta = Table({
        "TARGETID": ids,
        "SURVEY": survey,
        "PROGRAM": program,
        "RA": ra,
        "DEC": dec,
        "MAG": np.arange(n, dtype=np.float32),
    })
    coeff = np.arange(n * 5, dtype=np.float32).reshape(n, 5)
    specphot = Table({
        "TARGETID": ids.copy(),
        "SURVEY": survey.copy(),
        "PROGRAM": program.copy(),
        "LOGMSTAR": np.linspace(8.0, 11.0, n),
        "COEFF": coeff,
    })
    fast = Table({
        "TARGETID": ids.copy(),
        "SURVEY": survey.copy(),
        "PROGRAM": program.copy(),
        "SNR_B": np.linspace(1.0, 2.0, n),
    })
    if mismatch_row is not None:
        specphot["TARGETID"][mismatch_row] = 999_999
    if extra_collision:
        specphot["MAG"] = meta["MAG"].copy()
        fast["MAG"] = meta["MAG"].copy()
    if unequal:
        fast.remove_row(n - 1)
    if no_shared:
        specphot.remove_columns(["TARGETID", "SURVEY", "PROGRAM"])
        fast.remove_columns(["TARGETID", "SURVEY", "PROGRAM"])
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU(meta, name="METADATA"),
        fits.BinTableHDU(specphot, name="SPECPHOT"),
        fits.BinTableHDU(fast, name="FASTSPEC"),
    ]).writeto(path, overwrite=True)
    return path


def _read_catalog(lake: Path, survey: str = SURVEY) -> pa.Table:
    tiles = sorted((lake / "catalogs" / survey).rglob("Npix=*.parquet"))
    assert tiles, f"no tiles under {lake / 'catalogs' / survey}"
    return pa.concat_tables([pq.ParquetFile(str(p)).read() for p in tiles])


def _ingest(src: Path, lake: Path, **kw) -> None:
    ingest_catalog_mef(
        src, lake, SURVEY,
        ra_col="RA", dec_col="DEC", norder=3,
        link_id_col=LINK_ID, **kw,
    )


@pytest.fixture()
def mef(tmp_path: Path) -> Path:
    return _write_joined_mef(tmp_path / "fastspec.fits")


class TestReadJoinedCatalogHdus:
    def test_join_keeps_keys_once_and_vector(self, mef: Path) -> None:
        table, labels = read_joined_catalog_hdus(mef)
        assert labels == ("METADATA", "SPECPHOT", "FASTSPEC")
        assert table.num_rows == N
        names = table.schema.names
        assert names.count("TARGETID") == 1
        assert names.count("SURVEY") == 1
        assert names.count("PROGRAM") == 1
        for col in ("RA", "DEC", "MAG", "LOGMSTAR", "COEFF", "SNR_B"):
            assert col in names
        coeff = table.schema.field("COEFF").type
        assert pa.types.is_fixed_size_list(coeff)
        assert coeff.list_size == 5
        assert table.column("TARGETID").to_pylist() == list(range(1, N + 1))

    def test_unequal_length(self, tmp_path: Path) -> None:
        path = _write_joined_mef(tmp_path / "unequal.fits", unequal=True)
        with pytest.raises(ValueError, match=r"same row count.*FASTSPEC="):
            read_joined_catalog_hdus(path)

    def test_mismatched_key_row(self, tmp_path: Path) -> None:
        path = _write_joined_mef(tmp_path / "mismatch.fits", mismatch_row=3)
        with pytest.raises(ValueError, match=r"TARGETID.*row 3.*METADATA.*SPECPHOT"):
            read_joined_catalog_hdus(path)

    def test_non_key_name_collision(self, tmp_path: Path) -> None:
        path = _write_joined_mef(tmp_path / "collide.fits", extra_collision=True)
        with pytest.raises(ValueError, match=r"Non-key column.*MAG"):
            read_joined_catalog_hdus(path, key_cols=["TARGETID", "SURVEY", "PROGRAM"])

    def test_empty_intersection(self, tmp_path: Path) -> None:
        path = _write_joined_mef(tmp_path / "noshare.fits", no_shared=True)
        with pytest.raises(ValueError, match="share no column names"):
            read_joined_catalog_hdus(path)

    def test_restrict_extensions(self, mef: Path) -> None:
        table, labels = read_joined_catalog_hdus(
            mef, extensions=["METADATA", "FASTSPEC"],
        )
        assert labels == ("METADATA", "FASTSPEC")
        assert "SNR_B" in table.schema.names
        assert "COEFF" not in table.schema.names


class TestIngestCatalogMef:
    def test_ingest_row_count_source_id_keys_once(self, mef: Path, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _ingest(mef, lake)
        merged = _read_catalog(lake)
        assert merged.num_rows == N
        assert LAKE_JOIN_ID_COLUMN in merged.schema.names
        assert len(set(merged.column(LAKE_JOIN_ID_COLUMN).to_pylist())) == N
        names = list(merged.schema.names)
        assert names.count("TARGETID") == 1
        assert names.count("SURVEY") == 1
        assert names.count("PROGRAM") == 1
        coeff = merged.schema.field("COEFF").type
        assert pa.types.is_fixed_size_list(coeff) and coeff.list_size == 5
        info = json.loads(
            (lake / "catalogs" / SURVEY / "catalog_info.json").read_text()
        )
        assert info["fits_hdu"] == "join:METADATA+SPECPHOT+FASTSPEC"

    def test_file_list_two_files(self, tmp_path: Path) -> None:
        a = _write_joined_mef(tmp_path / "a.fits", n=N, id0=1)
        b = _write_joined_mef(tmp_path / "b.fits", n=N, id0=1 + N)
        lst = tmp_path / "files.txt"
        lst.write_text(f"{a}\n{b}\n")
        lake = tmp_path / "lake"
        from data_lake.ingest.file_list_ingest import _run_file_list

        def one(p: Path) -> None:
            _ingest(p, lake, tile_mode="append")

        code = _run_file_list(
            paths_file=lst,
            survey_name=SURVEY,
            lake_root=lake,
            checkpoint=None,
            failures_log=None,
            show_progress=False,
            skip_completed=True,
            ingest_one=one,
            default_checkpoint=lake / "catalogs" / SURVEY / ".ingest_checkpoint.json",
        )
        assert code == 0
        merged = _read_catalog(lake)
        assert merged.num_rows == 2 * N
        ids = merged.column("TARGETID").to_pylist()
        assert sorted(ids) == list(range(1, 2 * N + 1))
        info = json.loads(
            (lake / "catalogs" / SURVEY / "catalog_info.json").read_text()
        )
        assert info["fits_hdu"] == "join:METADATA+SPECPHOT+FASTSPEC"


class TestPreflightJoinHint:
    def test_inspect_joinable_hint(self, mef: Path) -> None:
        with open_fits(mef) as hdul:
            hint = inspect_joinable_table_hdus(hdul)
        assert hint is not None
        assert "dl-ingest-catalog-mef" in hint
        assert "--hdu all" in hint

    def test_check_format_default_hint(self, mef: Path) -> None:
        rep = inspect_fits_table_format(mef)
        assert rep.ok
        assert "dl-ingest-catalog-mef" in (rep.hint or "")

    def test_check_format_hdu_all_names_mef(self, mef: Path) -> None:
        rep = inspect_fits_table_format(mef, fits_hdu="all")
        assert not rep.ok
        assert "dl-ingest-catalog-mef" in (rep.error or "")

    def test_identical_schema_mef_is_not_a_join(self, tmp_path: Path) -> None:
        t1 = Table({
            "OBJID": np.arange(5, dtype=np.int64),
            "RA": np.linspace(0.0, 1.0, 5),
            "DEC": np.linspace(0.0, 1.0, 5),
        })
        t2 = Table({
            "OBJID": np.arange(5, 10, dtype=np.int64),
            "RA": np.linspace(2.0, 3.0, 5),
            "DEC": np.linspace(2.0, 3.0, 5),
        })
        path = tmp_path / "stack.fits"
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.BinTableHDU(t1, name="OBJECTS", ver=1),
            fits.BinTableHDU(t2, name="OBJECTS", ver=2),
        ]).writeto(path)
        with open_fits(path) as hdul:
            assert inspect_joinable_table_hdus(hdul) is None
        rep = inspect_fits_table_format(path)
        assert "dl-ingest-catalog-mef" not in (rep.hint or "")
        assert "--hdu" in (rep.hint or "")


class TestCli:
    def test_help(self) -> None:
        from data_lake.ingest.catalog_mef_ingest import cli

        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Join row-aligned" in result.output

    def test_rejects_source_and_file_list(self, mef: Path, tmp_path: Path) -> None:
        from data_lake.ingest.catalog_mef_ingest import cli

        lst = tmp_path / "files.txt"
        lst.write_text(f"{mef}\n")
        result = CliRunner().invoke(
            cli,
            [
                str(mef), "--file-list", str(lst),
                "--survey", SURVEY, "--link-id-col", LINK_ID,
            ],
        )
        assert result.exit_code != 0
        assert "SOURCE_PATH or --file-list" in result.output

    def test_requires_two_hdu_flags(self, mef: Path) -> None:
        from data_lake.ingest.catalog_mef_ingest import cli

        result = CliRunner().invoke(
            cli,
            [
                str(mef), "--survey", SURVEY, "--link-id-col", LINK_ID,
                "--hdu", "METADATA",
            ],
        )
        assert result.exit_code != 0
        assert "--hdu at least twice" in result.output
