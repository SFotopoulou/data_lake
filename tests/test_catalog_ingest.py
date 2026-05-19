"""
Tests for data_lake.ingest.fits_to_parquet.

Focuses on the FITS → PyArrow conversion path, especially:

* Multidim BINTABLE columns (e.g. DESI's ``COEFF`` shape (N, 10)) survive
  the conversion as Arrow ``FixedSizeList`` arrays — a naive table→rectangular
  dataframe conversion would fail on multidimensional cells.
* Byte-string columns (FITS fixed-width ASCII) decode to UTF-8.
* Format detection works for FITS / Parquet / unknown suffix.
* End-to-end ``ingest_catalog`` writes HEALPix-partitioned Parquet tiles
  and preserves the multidim column on disk.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.table import MaskedColumn, Table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jname_id_table(n_rows: int = 12) -> Table:
    """Alphanumeric object names (not parseable as integers)."""
    rng = np.random.default_rng(11)
    names = np.array(
        [f"J{rng.uniform(0, 360):09.2f}{rng.choice(['+', '-'])}{abs(rng.uniform(0, 90)):06.1f}"
         for _ in range(n_rows)],
        dtype="U24",
    )
    ra = rng.uniform(0.0, 360.0, n_rows).astype(np.float64)
    dec = rng.uniform(-30.0, +30.0, n_rows).astype(np.float64)
    return Table({"NAME": names, "RA": ra, "DEC": dec})


def _make_string_targetid_table(n_rows: int = 20) -> Table:
    """TARGETID as fixed-width ASCII (typical FITS BINTABLE string column)."""
    rng = np.random.default_rng(7)
    targetid = np.array(
        [f"{39627658462934656 + i:>20}" for i in range(n_rows)],
        dtype="U20",
    )
    ra = rng.uniform(0.0, 360.0, n_rows).astype(np.float64)
    dec = rng.uniform(-30.0, +30.0, n_rows).astype(np.float64)
    return Table({
        "TARGETID": targetid,
        "TARGET_RA": ra,
        "TARGET_DEC": dec,
        "Z": rng.uniform(0.0, 3.0, n_rows).astype(np.float64),
    })


def _make_desi_like_table(n_rows: int = 20) -> Table:
    """A synthetic table mimicking the offending DESI zall-pix columns."""
    rng = np.random.default_rng(42)
    targetid = np.arange(1_000_000, 1_000_000 + n_rows, dtype=np.int64)
    ra = rng.uniform(0.0, 360.0, n_rows).astype(np.float64)
    dec = rng.uniform(-30.0, +30.0, n_rows).astype(np.float64)
    z = rng.uniform(0.0, 3.0, n_rows).astype(np.float64)
    spectype = np.array(["QSO    ", "GALAXY ", "STAR   "] * (n_rows // 3 + 1), dtype="|S8")[:n_rows]
    coeff = rng.standard_normal((n_rows, 10)).astype(np.float64)   # 2-D vector column
    fiberflux = rng.standard_normal((n_rows, 4)).astype(np.float32)  # 2-D, different inner size
    return Table({
        "TARGETID": targetid,
        "TARGET_RA": ra,
        "TARGET_DEC": dec,
        "Z": z,
        "SPECTYPE": spectype,
        "COEFF": coeff,
        "FIBERFLUX": fiberflux,
    })


def _write_table_as_fits(tbl: Table, path: Path) -> None:
    """Round-trip an astropy Table through FITS BinTable HDU."""
    tbl.write(str(path), format="fits", overwrite=True)


def _make_same_pixel_table(targetids: list[int], *, ra: float = 120.0, dec: float = 45.0) -> Table:
    """Rows that share one HEALPix pixel at Norder=5 (for tile-mode tests)."""
    n = len(targetids)
    return Table({
        "TARGETID": np.array(targetids, dtype=np.int64),
        "TARGET_RA": np.full(n, ra, dtype=np.float64),
        "TARGET_DEC": np.full(n, dec, dtype=np.float64),
        "Z": np.arange(n, dtype=np.float64),
    })


def _read_merged_catalog(lake_root: Path, survey: str) -> tuple[list[Path], pa.Table]:
    root = lake_root / "catalogs" / survey
    tiles = sorted(root.rglob("Npix=*.parquet"))
    merged = pa.concat_tables([pq.ParquetFile(str(p)).read() for p in tiles])
    return tiles, merged


# ---------------------------------------------------------------------------
# _astropy_table_to_arrow — unit tests on the converter itself
# ---------------------------------------------------------------------------


class TestAstropyToArrow:
    def test_preserves_2d_columns_as_fixed_size_list(self):
        from data_lake.ingest.fits_to_parquet import _astropy_table_to_arrow

        tbl = _make_desi_like_table(n_rows=8)
        arrow_tbl = _astropy_table_to_arrow(tbl)

        assert isinstance(arrow_tbl.schema.field("COEFF").type, pa.FixedSizeListType)
        assert arrow_tbl.schema.field("COEFF").type.list_size == 10
        assert isinstance(arrow_tbl.schema.field("FIBERFLUX").type, pa.FixedSizeListType)
        assert arrow_tbl.schema.field("FIBERFLUX").type.list_size == 4

        # Values round-trip
        coeff_out = np.asarray(arrow_tbl.column("COEFF").to_pylist())
        np.testing.assert_allclose(coeff_out, np.asarray(tbl["COEFF"]))

    def test_decodes_byte_strings(self):
        from data_lake.ingest.fits_to_parquet import _astropy_table_to_arrow

        tbl = _make_desi_like_table(n_rows=6)
        arrow_tbl = _astropy_table_to_arrow(tbl)

        spectype = arrow_tbl.column("SPECTYPE").to_pylist()
        assert all(isinstance(s, str) for s in spectype)
        assert spectype[0].startswith("QSO")

    def test_string_columns_are_large_string(self):
        """All string columns must use large_string (int64 offsets).

        The default `string` type caps total UTF-8 bytes per chunk at 2 GB
        (int32 offsets); table.take()/filter() on a 28M-row DESI catalog
        overflows that limit with `ArrowInvalid: offset overflow ...`.
        """
        from data_lake.ingest.fits_to_parquet import _astropy_table_to_arrow

        tbl = _make_desi_like_table(n_rows=6)
        arrow_tbl = _astropy_table_to_arrow(tbl)

        # 1-D string column
        assert arrow_tbl.schema.field("SPECTYPE").type == pa.large_string()

    def test_take_on_large_string_does_not_overflow_pattern(self):
        """Smoke-test: table.take() must work on our large_string columns.

        This is the exact call site that crashed at 28M-row scale; the
        regression check ensures the conversion path still produces a take-
        compatible Table.
        """
        from data_lake.ingest.fits_to_parquet import _astropy_table_to_arrow

        tbl = _make_desi_like_table(n_rows=50)
        arrow_tbl = _astropy_table_to_arrow(tbl)
        # Pass shuffled indices through take() — same pattern as ingest_catalog
        idx = np.arange(50)[::-1]
        reordered = arrow_tbl.take(pa.array(idx, type=pa.int64()))
        assert reordered.num_rows == 50

    def test_3d_column_flattens_and_records_inner_shape(self):
        """>2-D columns flatten to FixedSizeList(inner_prod) with metadata."""
        from data_lake.ingest.fits_to_parquet import _astropy_table_to_arrow

        n = 4
        cube = np.arange(n * 2 * 3, dtype=np.float32).reshape(n, 2, 3)
        tbl = Table({"id": np.arange(n, dtype=np.int64), "CUBE": cube})

        arrow_tbl = _astropy_table_to_arrow(tbl)
        assert arrow_tbl.schema.field("CUBE").type.list_size == 6

        meta = arrow_tbl.schema.metadata or {}
        inner_shapes = json.loads(meta[b"data_lake.inner_shapes"].decode())
        assert inner_shapes == {"CUBE": [2, 3]}

    def test_handles_big_endian_dtype(self):
        """FITS columns are big-endian; converter must cast to native order."""
        from data_lake.ingest.fits_to_parquet import _astropy_table_to_arrow

        be = np.arange(5, dtype=">f4")    # big-endian float32
        tbl = Table({"x": be})
        arrow_tbl = _astropy_table_to_arrow(tbl)
        assert arrow_tbl.column("x").to_pylist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_masked_1d_column_propagates_nulls(self):
        from data_lake.ingest.fits_to_parquet import _astropy_table_to_arrow

        data = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([False, True, False, True])
        col = MaskedColumn(data=data, mask=mask, name="m")
        tbl = Table([col])

        arrow_tbl = _astropy_table_to_arrow(tbl)
        out = arrow_tbl.column("m").to_pylist()
        assert out[0] == 1.0 and out[2] == 3.0
        assert out[1] is None and out[3] is None


# ---------------------------------------------------------------------------
# _read_source_table — format detection
# ---------------------------------------------------------------------------


class TestShrinkTileStrings:
    def test_shrink_small_tile(self):
        from data_lake.ingest.fits_to_parquet import _shrink_tile_table_for_disk

        tbl = pa.table({"s": pa.array(["QSO", "STAR"], type=pa.large_string())})
        out = _shrink_tile_table_for_disk(tbl)
        assert out.schema.field("s").type == pa.string()

    def test_shrink_skips_huge_row_count(self, monkeypatch):
        from data_lake.ingest.fits_to_parquet import (
            _MAX_ROWS_STRING_SHRINK,
            _shrink_tile_table_for_disk,
        )

        tbl = pa.table({"s": pa.array(["x"], type=pa.large_string())})
        monkeypatch.setattr(
            "data_lake.ingest.fits_to_parquet._MAX_ROWS_STRING_SHRINK", 0,
        )
        out = _shrink_tile_table_for_disk(tbl)
        assert out.schema.field("s").type == pa.large_string()


class TestFormatDetection:
    def test_reads_fits_with_multidim_column(self, tmp_path: Path):
        from data_lake.ingest.fits_to_parquet import _read_source_table

        tbl = _make_desi_like_table(n_rows=10)
        fits_path = tmp_path / "cat.fits"
        _write_table_as_fits(tbl, fits_path)

        arrow_tbl = _read_source_table(fits_path)
        assert arrow_tbl.num_rows == 10
        assert isinstance(arrow_tbl.schema.field("COEFF").type, pa.FixedSizeListType)
        assert arrow_tbl.schema.field("COEFF").type.list_size == 10

    def test_reads_parquet_directly(self, tmp_path: Path):
        from data_lake.ingest.fits_to_parquet import _read_source_table

        pq_path = tmp_path / "cat.parquet"
        pq.write_table(
            pa.table({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]}),
            str(pq_path),
        )
        arrow_tbl = _read_source_table(pq_path)
        assert arrow_tbl.column_names == ["a", "b"]
        assert arrow_tbl.num_rows == 3

    def test_unknown_suffix_uses_autodetect_not_fits(self, tmp_path: Path) -> None:
        """A non-FITS file with an unknown suffix should not be force-read as FITS."""
        from data_lake.ingest.fits_to_parquet import _read_source_table

        ecsv = tmp_path / "cat.ecsv"
        Table({"a": [1, 2, 3]}).write(str(ecsv), format="ascii.ecsv", overwrite=True)
        arrow_tbl = _read_source_table(ecsv)
        assert arrow_tbl.num_rows == 3

    def test_read_csv_gz(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import _read_source_table

        csv_gz = tmp_path / "cat.csv.gz"
        with gzip.open(csv_gz, "wt", encoding="utf-8") as fh:
            fh.write("RAdeg,DEdeg,WISEA\n10.5,-20.3,J000000.00-314627.5\n")
        arrow_tbl = _read_source_table(csv_gz)
        assert arrow_tbl.num_rows == 1
        assert "RAdeg" in arrow_tbl.column_names

    def test_ingest_csv_gz(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        csv_gz = tmp_path / "cat.csv.gz"
        with gzip.open(csv_gz, "wt", encoding="utf-8") as fh:
            fh.write(
                "RAdeg,DEdeg,WISEA\n"
                "10.0,20.0,J000000.00+085706.6\n"
                "10.1,20.1,J000001.00+085707.6\n"
            )
        lake = tmp_path / "lake"
        ingest_catalog(
            source_path=csv_gz,
            output_root=lake,
            survey_name="csv_gz_survey",
            ra_col="RAdeg",
            dec_col="DEdeg",
            source_id_col="WISEA",
            norder=5,
            overwrite=True,
        )
        tiles = list((lake / "catalogs" / "csv_gz_survey").rglob("Npix=*.parquet"))
        assert tiles
        _, merged = _read_merged_catalog(lake, "csv_gz_survey")
        assert merged.num_rows == 2


# ---------------------------------------------------------------------------
# End-to-end ingest_catalog with the offending column
# ---------------------------------------------------------------------------


class TestCastObjectIdColumn:
    def test_string_column_to_int64(self) -> None:
        from data_lake.ingest.fits_to_parquet import cast_object_id_column_to_int64

        col = pa.array(["39627658462934656", "39627658462934657"], type=pa.large_string())
        out = cast_object_id_column_to_int64(col)
        assert out.type == pa.int64()
        assert out.to_pylist() == [39627658462934656, 39627658462934657]


class TestIngestCatalogEndToEnd:
    def test_ingest_jname_label_column(self, tmp_path: Path) -> None:
        """Non-integer NAME labels: column kept, source_id = stable hash."""
        import json

        from data_lake.ingest.fits_to_parquet import (
            ingest_catalog,
            stable_object_id_from_string,
        )

        tbl = _make_jname_id_table(n_rows=8)
        fits_path = tmp_path / "jnames.fits"
        _write_table_as_fits(tbl, fits_path)

        lake_root = tmp_path / "lake"
        ingest_catalog(
            source_path=fits_path,
            output_root=lake_root,
            survey_name="syn_j",
            ra_col="RA",
            dec_col="DEC",
            source_id_col="NAME",
            overwrite=True,
            streaming=False,
        )

        info = json.loads(
            (lake_root / "catalogs" / "syn_j" / "catalog_info.json").read_text()
        )
        assert info["source_id_mode"] == "label:NAME"

        _, merged = _read_merged_catalog(lake_root, "syn_j")
        assert pa.types.is_string(merged.schema.field("NAME").type) or pa.types.is_large_string(
            merged.schema.field("NAME").type
        )
        assert merged.schema.field("source_id").type == pa.int64()
        name0 = str(merged.column("NAME")[0].as_py()).strip()
        sid0 = int(merged.column("source_id")[0].as_py())
        assert sid0 == stable_object_id_from_string(name0)

    def test_ingest_string_targetid_column(self, tmp_path: Path) -> None:
        """String/object TARGETID in FITS is parsed to int64 in Parquet."""
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        tbl = _make_string_targetid_table(n_rows=12)
        fits_path = tmp_path / "string_ids.fits"
        _write_table_as_fits(tbl, fits_path)

        lake_root = tmp_path / "lake"
        ingest_catalog(
            source_path=fits_path,
            output_root=lake_root,
            survey_name="syn_str",
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            source_id_col="TARGETID",
            overwrite=True,
            streaming=False,
        )

        _, merged = _read_merged_catalog(lake_root, "syn_str")
        assert merged.schema.field("TARGETID").type == pa.int64()
        tids = np.asarray(merged.column("TARGETID"))
        expected = np.array([int(s) for s in tbl["TARGETID"]], dtype=np.int64)
        np.testing.assert_array_equal(np.sort(tids), np.sort(expected))

    def test_ingest_string_targetid_streaming(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        tbl = _make_string_targetid_table(n_rows=16)
        fits_path = tmp_path / "string_ids.fits"
        _write_table_as_fits(tbl, fits_path)

        lake_root = tmp_path / "lake"
        ingest_catalog(
            source_path=fits_path,
            output_root=lake_root,
            survey_name="syn_str",
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            source_id_col="TARGETID",
            overwrite=True,
            streaming=True,
        )

        _, merged = _read_merged_catalog(lake_root, "syn_str")
        assert merged.schema.field("TARGETID").type == pa.int64()

    def test_ingest_preserves_multidim_column_on_disk(self, tmp_path: Path):
        """Ingest a DESI-like FITS catalog and verify COEFF survives on disk."""
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        tbl = _make_desi_like_table(n_rows=24)
        fits_path = tmp_path / "zall_like.fits"
        _write_table_as_fits(tbl, fits_path)

        lake_root = tmp_path / "lake"
        ingest_catalog(
            source_path=fits_path,
            output_root=lake_root,
            survey_name="syn_desi",
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            source_id_col="TARGETID",
            overwrite=True,
        )

        # Find all written parquet tiles, read them back
        tile_files = sorted((lake_root / "catalogs" / "syn_desi").rglob("Npix=*.parquet"))
        assert tile_files, "No parquet tiles were written"

        merged = pa.concat_tables([pq.read_table(str(p)) for p in tile_files])
        assert merged.num_rows == 24
        # Schema preserved
        coeff_type = merged.schema.field("COEFF").type
        assert isinstance(coeff_type, pa.FixedSizeListType)
        assert coeff_type.list_size == 10
        # Values preserved (set-wise; row order may differ due to HEALPix sort)
        original_coeff = np.asarray(tbl["COEFF"])
        readback_coeff = np.asarray(merged.column("COEFF").to_pylist())
        # Sort both by TARGETID for comparison
        tids = np.asarray(merged.column("TARGETID"))
        order = np.argsort(tids)
        np.testing.assert_allclose(readback_coeff[order], original_coeff)

        # Required _spectrum_index / _cutout_index columns present
        names = set(merged.schema.names)
        assert "_spectrum_index" in names
        assert "_cutout_index" in names
        assert "_healpix_norder5" in names


class TestStreamingIngest:
    """Streaming and in-memory paths must produce equivalent per-tile output.

    This is the key correctness gate for the --streaming flag: the schema
    set, tile partitioning, multidim columns, and per-row values must all
    round-trip identically.  Row *order* may differ within a tile because
    streaming uses argsort+fancy-index while the in-memory path uses
    table.sort+slice; we compare set-wise by TARGETID.
    """

    def _read_merged(self, lake_root: Path, survey: str) -> pa.Table:
        files = sorted((lake_root / "catalogs" / survey).rglob("Npix=*.parquet"))
        return pa.concat_tables([pq.read_table(str(p)) for p in files])

    def test_streaming_equals_in_memory(self, tmp_path: Path):
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        tbl = _make_desi_like_table(n_rows=40)
        fits_path = tmp_path / "zall_like.fits"
        _write_table_as_fits(tbl, fits_path)

        mem_root = tmp_path / "mem"
        ingest_catalog(
            source_path=fits_path, output_root=mem_root, survey_name="syn",
            ra_col="TARGET_RA", dec_col="TARGET_DEC", norder=5,
            source_id_col="TARGETID", overwrite=True, streaming=False,
        )

        stream_root = tmp_path / "stream"
        ingest_catalog(
            source_path=fits_path, output_root=stream_root, survey_name="syn",
            ra_col="TARGET_RA", dec_col="TARGET_DEC", norder=5,
            source_id_col="TARGETID", overwrite=True, streaming=True,
        )

        # Same tile set
        mem_tiles = {p.relative_to(mem_root) for p in
                     (mem_root / "catalogs" / "syn").rglob("Npix=*.parquet")}
        stream_tiles = {p.relative_to(stream_root) for p in
                        (stream_root / "catalogs" / "syn").rglob("Npix=*.parquet")}
        assert mem_tiles == stream_tiles, "tile partitioning differs between paths"

        mem_tbl = self._read_merged(mem_root, "syn")
        stream_tbl = self._read_merged(stream_root, "syn")

        # Same row counts
        assert mem_tbl.num_rows == stream_tbl.num_rows == 40

        # Same set of columns (schemas may differ only in nullability/order)
        assert set(mem_tbl.schema.names) == set(stream_tbl.schema.names)

        # Required bookkeeping columns
        for col in ("_healpix_norder5", "_cutout_index", "_spectrum_index"):
            assert col in mem_tbl.schema.names
            assert col in stream_tbl.schema.names

        # COEFF preserved as FixedSizeList(10) in both paths
        for t in (mem_tbl, stream_tbl):
            f = t.schema.field("COEFF").type
            assert isinstance(f, pa.FixedSizeListType) and f.list_size == 10

        # SPECTYPE preserved as a string column (large_string in RAM, string per tile on disk)
        for t in (mem_tbl, stream_tbl):
            stype = t.schema.field("SPECTYPE").type
            assert pa.types.is_string(stype) or pa.types.is_large_string(stype)

        # Set-wise value equivalence (order may differ within a tile)
        def by_tid(t):
            tids = np.asarray(t.column("TARGETID"))
            order = np.argsort(tids)
            return order, tids[order]

        mo, mt = by_tid(mem_tbl)
        so, st = by_tid(stream_tbl)
        np.testing.assert_array_equal(mt, st)
        np.testing.assert_allclose(
            np.asarray(mem_tbl.column("COEFF").to_pylist())[mo],
            np.asarray(stream_tbl.column("COEFF").to_pylist())[so],
        )
        np.testing.assert_allclose(
            np.asarray(mem_tbl.column("Z"))[mo],
            np.asarray(stream_tbl.column("Z"))[so],
        )

    def test_streaming_rejects_non_fits(self, tmp_path: Path):
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        pq_path = tmp_path / "cat.parquet"
        pq.write_table(pa.table({"ra": [1.0], "dec": [2.0]}), str(pq_path))

        with pytest.raises(ValueError, match="streaming=True is supported only for FITS"):
            ingest_catalog(
                source_path=pq_path, output_root=tmp_path / "lake",
                survey_name="x", ra_col="ra", dec_col="dec",
                streaming=True,
            )

    def test_streaming_with_auto_generated_source_id(self, tmp_path: Path):
        """When source_id_col is None, streaming should auto-generate sequential IDs."""
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        # Build a minimal table without an explicit ID column
        tbl = Table({
            "RA":  np.array([10.0, 20.0, 30.0, 40.0]),
            "DEC": np.array([1.0, 2.0, 3.0, 4.0]),
            "MAG": np.array([18.0, 19.0, 20.0, 21.0]),
        })
        fits_path = tmp_path / "noid.fits"
        _write_table_as_fits(tbl, fits_path)

        lake_root = tmp_path / "lake"
        ingest_catalog(
            source_path=fits_path, output_root=lake_root, survey_name="noid",
            ra_col="RA", dec_col="DEC", norder=5,
            source_id_col=None, overwrite=True, streaming=True,
        )

        merged = self._read_merged(lake_root, "noid")
        assert merged.num_rows == 4
        assert "source_id" in merged.schema.names
        assert set(np.asarray(merged.column("source_id")).tolist()) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# tile_mode: skip | overwrite | append
# ---------------------------------------------------------------------------


class TestTileMode:
    def _ingest_two(
        self,
        tmp_path: Path,
        ids_a: list[int],
        ids_b: list[int],
        *,
        second_tile_mode: str,
        on_duplicate_id: str = "skip",
    ) -> pa.Table:
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        lake = tmp_path / "lake"
        fits_a = tmp_path / "a.fits"
        fits_b = tmp_path / "b.fits"
        _write_table_as_fits(_make_same_pixel_table(ids_a), fits_a)
        _write_table_as_fits(_make_same_pixel_table(ids_b), fits_b)

        common = dict(
            output_root=lake,
            survey_name="tile_test",
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            source_id_col="TARGETID",
        )
        ingest_catalog(source_path=fits_a, tile_mode="overwrite", **common)
        ingest_catalog(
            source_path=fits_b,
            tile_mode=second_tile_mode,
            on_duplicate_id=on_duplicate_id,
            **common,
        )
        _, merged = _read_merged_catalog(lake, "tile_test")
        return merged

    def test_append_concatenates_rows(self, tmp_path: Path):
        merged = self._ingest_two(tmp_path, [1, 2, 3], [4, 5], second_tile_mode="append")
        assert merged.num_rows == 5
        assert set(np.asarray(merged.column("TARGETID")).tolist()) == {1, 2, 3, 4, 5}

    def test_skip_leaves_existing_tile(self, tmp_path: Path):
        merged = self._ingest_two(tmp_path, [1, 2, 3], [4, 5], second_tile_mode="skip")
        assert merged.num_rows == 3
        assert set(np.asarray(merged.column("TARGETID")).tolist()) == {1, 2, 3}

    def test_overwrite_replaces_tile(self, tmp_path: Path):
        merged = self._ingest_two(tmp_path, [1, 2, 3], [4, 5], second_tile_mode="overwrite")
        assert merged.num_rows == 2
        assert set(np.asarray(merged.column("TARGETID")).tolist()) == {4, 5}

    def test_append_duplicate_id_error(self, tmp_path: Path):
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        lake = tmp_path / "lake"
        fits_a = tmp_path / "a.fits"
        fits_b = tmp_path / "b.fits"
        _write_table_as_fits(_make_same_pixel_table([1, 2]), fits_a)
        _write_table_as_fits(_make_same_pixel_table([2, 3]), fits_b)
        common = dict(
            output_root=lake,
            survey_name="dup",
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            source_id_col="TARGETID",
        )
        ingest_catalog(source_path=fits_a, tile_mode="overwrite", **common)
        with pytest.raises(ValueError, match="Duplicate object ID"):
            ingest_catalog(
                source_path=fits_b,
                tile_mode="append",
                on_duplicate_id="error",
                **common,
            )

    def test_reingest_same_file_append_skip_is_idempotent(self, tmp_path: Path) -> None:
        """Second ingest of the same catalog: append + skip duplicates → no change."""
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        tbl = _make_jname_id_table(n_rows=8)
        fits_path = tmp_path / "same.fits"
        _write_table_as_fits(tbl, fits_path)
        lake = tmp_path / "lake"
        common = dict(
            output_root=lake,
            survey_name="reingest",
            ra_col="RA",
            dec_col="DEC",
            norder=5,
            source_id_col="NAME",
            tile_mode="append",
            on_duplicate_id="skip",
        )
        ingest_catalog(source_path=fits_path, overwrite=True, **common)
        _, first = _read_merged_catalog(lake, "reingest")
        ingest_catalog(source_path=fits_path, **common)
        _, second = _read_merged_catalog(lake, "reingest")
        assert second.num_rows == first.num_rows
        np.testing.assert_array_equal(
            np.sort(np.asarray(second.column("source_id"))),
            np.sort(np.asarray(first.column("source_id"))),
        )

    def test_append_duplicate_id_last(self, tmp_path: Path):
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        lake = tmp_path / "lake"
        fits_a = tmp_path / "a.fits"
        fits_b = tmp_path / "b.fits"
        _write_table_as_fits(_make_same_pixel_table([1, 2]), fits_a)
        _write_table_as_fits(_make_same_pixel_table([2, 3]), fits_b)
        common = dict(
            output_root=lake,
            survey_name="last",
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            source_id_col="TARGETID",
        )
        ingest_catalog(source_path=fits_a, tile_mode="overwrite", **common)
        ingest_catalog(
            source_path=fits_b,
            tile_mode="append",
            on_duplicate_id="last",
            **common,
        )
        _, merged = _read_merged_catalog(lake, "last")
        assert merged.num_rows == 3
        tids = np.asarray(merged.column("TARGETID"))
        z = np.asarray(merged.column("Z"))
        by_id = {int(t): float(zv) for t, zv in zip(tids, z)}
        assert by_id[1] == 0.0
        assert by_id[2] == 0.0  # second file's row for ID 2 (Z index 0 in 2-row table)
        assert by_id[3] == 1.0

    def test_metadata_lists_all_on_disk_tiles(self, tmp_path: Path):
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        lake = tmp_path / "lake"
        # File A: one shared pixel; file B: different sky → second tile
        _write_table_as_fits(_make_same_pixel_table([1, 2]), tmp_path / "a.fits")
        tbl_b = Table({
            "TARGETID": np.array([10, 11], dtype=np.int64),
            "TARGET_RA": np.array([200.0, 201.0]),
            "TARGET_DEC": np.array([10.0, 11.0]),
            "Z": np.array([0.0, 1.0]),
        })
        _write_table_as_fits(tbl_b, tmp_path / "b.fits")

        common = dict(
            output_root=lake,
            survey_name="meta",
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            source_id_col="TARGETID",
        )
        ingest_catalog(source_path=tmp_path / "a.fits", tile_mode="overwrite", **common)
        ingest_catalog(source_path=tmp_path / "b.fits", tile_mode="append", **common)

        catalog_root = lake / "catalogs" / "meta"
        n_tiles = len(list(catalog_root.rglob("Npix=*.parquet")))
        assert n_tiles >= 2
        combined = pq.read_metadata(str(catalog_root / "_metadata"))
        assert combined.num_row_groups == n_tiles

        with open(catalog_root / "catalog_info.json") as fh:
            info = json.load(fh)
        assert info["total_rows"] == 4
