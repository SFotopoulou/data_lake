"""Tests for dl-check-fits-table-format / inspect_fits_table_format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from data_lake.ingest.check_fits_table_format import (
    inspect_fits_table_format,
    inspect_many,
    resolve_inspect_paths,
)
from data_lake.ingest.fits_to_parquet import estimate_bintable_source_count


def _write_standard_fits(tbl: Table, path: Path) -> None:
    tbl.write(str(path), format="fits", overwrite=True)


class TestInspectFitsTableFormat:
    def test_standard_bintable(self, tmp_path: Path) -> None:
        path = tmp_path / "std.fits"
        _write_standard_fits(
            Table({"id": [1, 2, 3], "ra": [10.0, 11.0, 12.0], "dec": [0.1, 0.2, 0.3]}),
            path,
        )
        rep = inspect_fits_table_format(path)
        assert rep.ok
        assert rep.format == "standard-bintable"
        assert rep.naxis2 == 3
        assert rep.est_source_count == 3
        assert rep.n_columns == 3
        assert "memmap" in (rep.ingest_path or "").lower()

    def test_packed_vector_fits(self, tmp_path: Path) -> None:
        fitsio = pytest.importorskip("fitsio")
        n = 50
        path = tmp_path / "packed.fits"
        row = np.zeros(
            1,
            dtype=[
                ("objid", "i8", (n,)),
                ("ra", "f8", (n,)),
                ("dec", "f8", (n,)),
            ],
        )
        row["objid"][0] = np.arange(n, dtype=np.int64)
        fitsio.write(str(path), row, extname="photoobjall_test")

        rep = inspect_fits_table_format(path)
        assert rep.ok
        assert rep.format == "packed-vector"
        assert rep.naxis2 == 1
        assert rep.est_source_count == n
        assert rep.n_columns == 3
        assert "fitsio" in (rep.ingest_path or "").lower()

    def test_estimate_bintable_source_count_matches_read(self, tmp_path: Path) -> None:
        fitsio = pytest.importorskip("fitsio")
        from data_lake.io.fits_read import open_fits
        from data_lake.ingest.fits_to_parquet import _bintable_hdu_index, _is_packed_vector_bintable

        n = 32
        path = tmp_path / "packed2.fits"
        row = np.zeros(1, dtype=[("x", "f8", (n,))])
        fitsio.write(str(path), row)
        with open_fits(path) as hdul:
            hdu = hdul[_bintable_hdu_index(hdul)]
            assert _is_packed_vector_bintable(hdu)
            assert estimate_bintable_source_count(hdu) == n

    def test_file_list_resolution(self, tmp_path: Path) -> None:
        a = tmp_path / "a.fits"
        b = tmp_path / "b.fits"
        a.write_text("x")
        b.write_text("y")
        lst = tmp_path / "files.txt"
        lst.write_text(f"{a}\n{b}\n")
        got = resolve_inspect_paths([], file_list=lst)
        assert [p.name for p in got] == ["a.fits", "b.fits"]

    def test_cli_json(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.ingest.check_fits_table_format import cli

        if cli is None:
            pytest.skip("click not installed")
        path = tmp_path / "std.fits"
        _write_standard_fits(Table({"id": [1], "ra": [1.0], "dec": [2.0]}), path)
        result = CliRunner().invoke(cli, [str(path), "--json", "--no-summary"])
        assert result.exit_code == 0
        assert '"format": "standard-bintable"' in result.output
