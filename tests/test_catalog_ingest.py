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


def _make_filename_label_table(n_rows: int = 24) -> Table:
    """String filename labels (non-integer link IDs)."""
    rng = np.random.default_rng(3)
    filenames = np.array(
        [f"spec_{i:04d}.fits" for i in range(n_rows)],
        dtype="U32",
    )
    ra = rng.uniform(0.0, 360.0, n_rows).astype(np.float64)
    dec = rng.uniform(-30.0, +30.0, n_rows).astype(np.float64)
    return Table({"filename": filenames, "alpha": ra, "delta": dec})


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

    def test_read_tab_separated_csv_gz(self, tmp_path: Path) -> None:
        """Tab-separated tables named .csv.gz (common for GAIA exports)."""
        from data_lake.ingest.fits_to_parquet import _read_source_table

        path = tmp_path / "gaia_like.csv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write("source_id\tra\tdec\n123\t10.5\t-20.3\n")
        tbl = _read_source_table(path)
        assert tbl.num_rows == 1
        assert set(tbl.column_names) >= {"source_id", "ra", "dec"}

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
            link_id_col="WISEA",
            norder=5,
            tile_mode="overwrite",
        )
        tiles = list((lake / "catalogs" / "csv_gz_survey").rglob("Npix=*.parquet"))
        assert tiles
        _, merged = _read_merged_catalog(lake, "csv_gz_survey")
        assert merged.num_rows == 2


# ---------------------------------------------------------------------------
# End-to-end ingest_catalog with the offending column
# ---------------------------------------------------------------------------


class TestCastObjectIdColumn:
    def test_vector_link_id_col_rejected(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import ensure_catalog_source_ids

        tbl = pa.table({
            "OBJID": pa.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], type=pa.list_(pa.int64(), 5)),
            "objid": pa.array([100, 200], type=pa.int64()),
            "RA": pa.array([0.0, 1.0], type=pa.float64()),
        })
        with pytest.raises(ValueError, match="vector column"):
            ensure_catalog_source_ids(tbl, "OBJID")

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
            link_id_col="NAME",
            tile_mode="overwrite",
            streaming=False,
        )

        info = json.loads(
            (lake_root / "catalogs" / "syn_j" / "catalog_info.json").read_text()
        )
        assert info["link_id_mode"] == "label:NAME"

        _, merged = _read_merged_catalog(lake_root, "syn_j")
        assert pa.types.is_string(merged.schema.field("NAME").type) or pa.types.is_large_string(
            merged.schema.field("NAME").type
        )
        from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN

        assert merged.schema.field(LAKE_JOIN_ID_COLUMN).type == pa.int64()
        name0 = str(merged.column("NAME")[0].as_py()).strip()
        sid0 = int(merged.column(LAKE_JOIN_ID_COLUMN)[0].as_py())
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
            link_id_col="TARGETID",
            tile_mode="overwrite",
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
            link_id_col="TARGETID",
            tile_mode="overwrite",
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
            link_id_col="TARGETID",
            tile_mode="overwrite",
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


class TestColumnNameResolution:
    """FITS TTYPE names are often padded with leading/trailing spaces."""

    def test_match_schema_column_strip_and_case(self) -> None:
        from data_lake.ingest.fits_to_parquet import match_schema_column

        # match_schema_column still resolves padded names (used for query-time compat)
        names = ["designation", " ra", " dec", " source_id"]
        assert match_schema_column("ra", names) == " ra"
        assert match_schema_column("RA", names) == " ra"
        assert match_schema_column(" dec", names) == " dec"
        assert match_schema_column("source_id", names) == " source_id"
        assert match_schema_column("missing", names) is None

    def test_resolve_catalog_column_name_raises(self) -> None:
        from data_lake.ingest.fits_to_parquet import resolve_catalog_column_name

        with pytest.raises(KeyError, match="not in catalog"):
            resolve_catalog_column_name([" ra", " dec"], "glon")

    def test_strip_catalog_column_names(self) -> None:
        import pyarrow as pa

        from data_lake.ingest.fits_to_parquet import strip_catalog_column_names

        tbl = pa.table({" ra": [1.0, 2.0], " dec": [-1.0, -2.0], "  z  ": [0.1, 0.2]})
        out = strip_catalog_column_names(tbl)
        assert out.schema.names == ["ra", "dec", "z"]
        # clean names are a no-op
        clean = pa.table({"ra": [1.0], "dec": [0.0]})
        assert strip_catalog_column_names(clean) is clean

    def test_strip_catalog_column_names_duplicate_error(self) -> None:
        import pyarrow as pa

        from data_lake.ingest.fits_to_parquet import strip_catalog_column_names

        tbl = pa.table({" ra": [1.0], "ra": [2.0]})
        with pytest.raises(ValueError, match="duplicates"):
            strip_catalog_column_names(tbl)

    def test_ingest_padded_fits_column_names(self, tmp_path: Path) -> None:
        """Ingest strips FITS TTYPE padding so Parquet has clean column names."""
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        n = 12
        rng = np.random.default_rng(99)
        tbl = Table({
            " ra": rng.uniform(0.0, 360.0, n).astype(np.float64),
            " dec": rng.uniform(-30.0, 30.0, n).astype(np.float64),
            " source_id": np.arange(1, n + 1, dtype=np.int64),
        })
        fits_path = tmp_path / "padded_cols.fits"
        _write_table_as_fits(tbl, fits_path)

        lake = tmp_path / "lake"
        for streaming in (False, True):
            ingest_catalog(
                source_path=fits_path,
                output_root=lake / ("stream" if streaming else "mem"),
                survey_name="padded",
                ra_col="ra",
                dec_col="dec",
                link_id_col="source_id",
                norder=5,
                tile_mode="overwrite",
                streaming=streaming,
            )

        _, mem = _read_merged_catalog(lake / "mem", "padded")
        _, stream = _read_merged_catalog(lake / "stream", "padded")
        assert mem.num_rows == stream.num_rows == n
        # Padding must have been stripped — clean names in Parquet
        assert "ra" in mem.schema.names
        assert " ra" not in mem.schema.names
        assert "dec" in mem.schema.names
        assert " dec" not in mem.schema.names
        np.testing.assert_array_equal(
            np.sort(np.asarray(mem.column("source_id"))),
            np.arange(1, n + 1, dtype=np.int64),
        )


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
            link_id_col="TARGETID", tile_mode="overwrite", streaming=False,
        )

        stream_root = tmp_path / "stream"
        ingest_catalog(
            source_path=fits_path, output_root=stream_root, survey_name="syn",
            ra_col="TARGET_RA", dec_col="TARGET_DEC", norder=5,
            link_id_col="TARGETID", tile_mode="overwrite", streaming=True,
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
        for col in (
            "_healpix_norder5",
            "_cutout_index", "_cutout_npix",
            "_spectrum_index", "_spectrum_npix",
        ):
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

    def test_streaming_parallel_finalize_label_link_id(self, tmp_path: Path) -> None:
        """Parallel streaming shards must propagate link_id_mode through finalize."""
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        tbl = _make_filename_label_table(n_rows=24)
        fits_path = tmp_path / "label_ids.fits"
        _write_table_as_fits(tbl, fits_path)

        lake_root = tmp_path / "lake"
        ingest_catalog(
            source_path=fits_path,
            output_root=lake_root,
            survey_name="syn_parallel",
            ra_col="alpha",
            dec_col="delta",
            norder=1,
            link_id_col="filename",
            tile_mode="append",
            on_duplicate_id="skip",
            allow_incomplete_link_id=True,
            streaming_parallel=2,
        )

        info_path = lake_root / "catalogs" / "syn_parallel" / "catalog_info.json"
        assert info_path.is_file()
        with open(info_path) as fh:
            info = json.load(fh)
        assert info["link_id_mode"] == "label:filename"
        assert info.get("native_id_column") == "filename"

        tiles, merged = _read_merged_catalog(lake_root, "syn_parallel")
        assert tiles
        assert merged.num_rows == 24
        assert "Norder" not in merged.schema.names
        assert "Dir" not in merged.schema.names

    def test_streaming_parallel_schema_matches_in_memory_append(self, tmp_path: Path) -> None:
        """Parallel streaming append must not inject hive partition columns."""
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        tbl = _make_desi_like_table(n_rows=24)
        fits_path = tmp_path / "desi_like.fits"
        _write_table_as_fits(tbl, fits_path)

        lake_root = tmp_path / "lake"
        ingest_catalog(
            source_path=fits_path,
            output_root=lake_root,
            survey_name="schema_cmp",
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            link_id_col="TARGETID",
            tile_mode="overwrite",
            streaming=False,
        )
        _, mem_tbl = _read_merged_catalog(lake_root, "schema_cmp")
        mem_cols = set(mem_tbl.schema.names)

        ingest_catalog(
            source_path=fits_path,
            output_root=lake_root,
            survey_name="schema_cmp",
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            link_id_col="TARGETID",
            tile_mode="append",
            on_duplicate_id="skip",
            streaming_parallel=2,
        )
        _, par_tbl = _read_merged_catalog(lake_root, "schema_cmp")
        assert set(par_tbl.schema.names) == mem_cols
        assert "Norder" not in par_tbl.schema.names
        assert "Dir" not in par_tbl.schema.names

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

    def test_streaming_requires_link_id_col(self, tmp_path: Path):
        """Catalog ingest must have --link-id-col; no sequential auto-IDs."""
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        tbl = Table({
            "RA":  np.array([10.0, 20.0, 30.0, 40.0]),
            "DEC": np.array([1.0, 2.0, 3.0, 4.0]),
            "MAG": np.array([18.0, 19.0, 20.0, 21.0]),
        })
        fits_path = tmp_path / "noid.fits"
        _write_table_as_fits(tbl, fits_path)

        with pytest.raises(ValueError, match="--link-id-col"):
            ingest_catalog(
                source_path=fits_path,
                output_root=tmp_path / "lake",
                survey_name="noid",
                ra_col="RA",
                dec_col="DEC",
                norder=5,
                link_id_col=None,
                tile_mode="overwrite",
                streaming=True,
            )

    def test_invalid_sky_coordinates_rejected(self, tmp_path: Path):
        """Rows with NaN or pipeline sentinels must fail before HEALPix assignment."""
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        tbl = Table({
            "TARGETID": np.array([1, 2], dtype=np.int64),
            "TARGET_RA": np.array([10.0, np.nan]),
            "TARGET_DEC": np.array([1.0, 2.0]),
        })
        fits_path = tmp_path / "bad_sky.fits"
        _write_table_as_fits(tbl, fits_path)

        with pytest.raises(ValueError, match="invalid sky"):
            ingest_catalog(
                source_path=fits_path,
                output_root=tmp_path / "lake",
                survey_name="bad",
                ra_col="TARGET_RA",
                dec_col="TARGET_DEC",
                link_id_col="TARGETID",
                tile_mode="overwrite",
            )


# ---------------------------------------------------------------------------
# tile_mode: skip | overwrite | append
# ---------------------------------------------------------------------------


class TestParquetTileIntegrity:
    def test_corrupt_tile_recovered_on_append(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import (
            _write_catalog_parquet_tile,
            _write_tile_for_mode,
            CatalogParquetOptions,
        )

        out = tmp_path / "tiles" / "Npix=42.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"not-parquet")
        incoming = pa.table({"TARGETID": [1], "TARGET_RA": [10.0], "TARGET_DEC": [20.0]})
        meta = _write_tile_for_mode(
            out,
            incoming,
            tile_mode="append",
            on_duplicate_id="skip",
            link_id_col="TARGETID",
            parquet_options=CatalogParquetOptions(),
        )
        assert meta is not None
        assert pq.read_metadata(str(out)).num_rows == 1

    def test_atomic_write_leaves_no_tmp(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import (
            _parquet_tile_tmp_path,
            _write_catalog_parquet_tile,
            CatalogParquetOptions,
        )

        out = tmp_path / "tile.parquet"
        tbl = pa.table({"a": [1, 2, 3]})
        _write_catalog_parquet_tile(tbl, out, CatalogParquetOptions())
        tmp = _parquet_tile_tmp_path(out)
        assert not tmp.exists()
        pq.read_metadata(str(out))


class TestNumericTypeNormalization:
    """AllWISE/GALEX: same column as float32 in one file and float64 in another."""

    def test_canonical_merge_types_null_promotes_to_float64(self) -> None:
        from data_lake.ingest.fits_to_parquet import _canonical_merge_types

        assert _canonical_merge_types(pa.null(), pa.float64()) == pa.float64()
        assert _canonical_merge_types(pa.float64(), pa.null()) == pa.float64()
        assert _canonical_merge_types(pa.null(), pa.float32()) == pa.float64()
        assert _canonical_merge_types(pa.null(), pa.null()) == pa.float64()

    def test_append_float32_then_float64_same_column(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import ingest_catalog

        ra, dec = 120.0, 45.0
        tbl_a = Table({
            "TARGETID": np.array([1, 2], dtype=np.int64),
            "TARGET_RA": np.full(2, ra, dtype=np.float64),
            "TARGET_DEC": np.full(2, dec, dtype=np.float64),
            "MAG": np.array([18.0, 19.0], dtype=np.float32),
        })
        tbl_b = Table({
            "TARGETID": np.array([3, 4], dtype=np.int64),
            "TARGET_RA": np.full(2, ra, dtype=np.float64),
            "TARGET_DEC": np.full(2, dec, dtype=np.float64),
            "MAG": np.array([20.0, 21.0], dtype=np.float64),
        })
        fits_a, fits_b = tmp_path / "a.fits", tmp_path / "b.fits"
        _write_table_as_fits(tbl_a, fits_a)
        _write_table_as_fits(tbl_b, fits_b)
        lake = tmp_path / "lake"
        common = dict(
            output_root=lake,
            survey_name="dtype_mix",
            ra_col="TARGET_RA",
            dec_col="TARGET_DEC",
            norder=5,
            link_id_col="TARGETID",
        )
        ingest_catalog(source_path=fits_a, tile_mode="overwrite", **common)
        ingest_catalog(source_path=fits_b, tile_mode="append", **common)

        _, merged = _read_merged_catalog(lake, "dtype_mix")
        assert merged.num_rows == 4
        assert merged.schema.field("MAG").type == pa.float64()
        mags = np.asarray(merged.column("MAG"))
        assert np.allclose(mags, [18.0, 19.0, 20.0, 21.0])

    def test_metadata_regen_after_mixed_float_tiles(self, tmp_path: Path) -> None:
        """Legacy tiles: float32 vs float64 across different Npix files break _metadata."""
        from data_lake.ingest.fits_to_parquet import (
            _regenerate_metadata_from_all_tiles,
            healpix_dir,
        )

        catalog_root = tmp_path / "catalogs" / "legacy_mix"
        norder = 5
        for npix, flux_type in ((100, pa.float32()), (10_500, pa.float64())):
            tile_dir = catalog_root / healpix_dir(norder, npix)
            tile_dir.mkdir(parents=True, exist_ok=True)
            tbl = pa.table({
                "_source_id": pa.array([npix], type=pa.int64()),
                f"_healpix_norder{norder}": pa.array([npix], type=pa.int64()),
                "flux": pa.array([1.0], type=flux_type),
            })
            pq.write_table(tbl, tile_dir / f"Npix={npix}.parquet")

        _regenerate_metadata_from_all_tiles(catalog_root)
        assert (catalog_root / "_metadata").is_file()
        for path in catalog_root.rglob("Npix=*.parquet"):
            assert pq.read_schema(str(path)).field("flux").type == pa.float64()


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
            link_id_col="TARGETID",
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
            link_id_col="TARGETID",
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
        from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, ingest_catalog

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
            link_id_col="NAME",
            on_duplicate_id="skip",
        )
        ingest_catalog(source_path=fits_path, tile_mode="overwrite", **common)
        _, first = _read_merged_catalog(lake, "reingest")
        ingest_catalog(source_path=fits_path, tile_mode="append", **common)
        _, second = _read_merged_catalog(lake, "reingest")
        assert second.num_rows == first.num_rows
        np.testing.assert_array_equal(
            np.sort(np.asarray(second.column(LAKE_JOIN_ID_COLUMN))),
            np.sort(np.asarray(first.column(LAKE_JOIN_ID_COLUMN))),
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
            link_id_col="TARGETID",
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
            link_id_col="TARGETID",
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


class TestAllowIncompleteLinkId:
    def test_composite_null_when_part_missing(self) -> None:
        from data_lake.ingest.fits_to_parquet import (
            LAKE_JOIN_ID_COLUMN,
            composite_link_label,
            ensure_catalog_source_ids,
            stable_object_id_from_string,
        )

        table = pa.table({
            "SPFILE": pa.array(["a.fits", "", "c.fits"]),
            "FIBRE": pa.array([1, 2, 3]),
        })
        out, mode = ensure_catalog_source_ids(
            table, "SPFILE,FIBRE", allow_incomplete_link_id=True,
        )
        assert mode == "composite:SPFILE,FIBRE"
        sids = out[LAKE_JOIN_ID_COLUMN].to_pylist()
        assert sids[0] == stable_object_id_from_string(
            composite_link_label("a.fits", 1),
        )
        assert sids[1] is None
        assert sids[2] == stable_object_id_from_string(
            composite_link_label("c.fits", 3),
        )

    def test_composite_partial_join_without_flag(self) -> None:
        from data_lake.ingest.fits_to_parquet import (
            LAKE_JOIN_ID_COLUMN,
            composite_link_label,
            ensure_catalog_source_ids,
            stable_object_id_from_string,
        )

        table = pa.table({
            "SPFILE": pa.array(["a.fits", "", "c.fits"]),
            "FIBRE": pa.array([1, 2, 3]),
        })
        out, _ = ensure_catalog_source_ids(table, "SPFILE,FIBRE")
        sids = out[LAKE_JOIN_ID_COLUMN].to_pylist()
        assert sids[1] == stable_object_id_from_string(composite_link_label(2))


class TestPackedVectorFits:
    """GALEX photoobjall: one FITS row, each column a vector of sources."""

    def test_read_packed_vector_fits(self, tmp_path: Path) -> None:
        fitsio = pytest.importorskip("fitsio")
        from data_lake.ingest.fits_to_parquet import _read_fits_catalog_table

        n = 50
        objid = np.arange(1_000, 1_000 + n, dtype=np.int64)
        ra = np.linspace(10.0, 20.0, n)
        dec = np.linspace(-5.0, 5.0, n)
        path = tmp_path / "packed.fits"
        # One FITS row; each field is a length-n vector (GALEX-style repeat TFORM).
        row = np.zeros(
            1,
            dtype=[
                ("objid", "i8", (n,)),
                ("ra", "f8", (n,)),
                ("dec", "f8", (n,)),
            ],
        )
        row["objid"][0] = objid
        row["ra"][0] = ra
        row["dec"][0] = dec
        fitsio.write(str(path), row, extname="photoobjall_test")

        tbl = _read_fits_catalog_table(path)
        assert len(tbl) == n
        assert int(tbl["objid"][0]) == 1000


# ---------------------------------------------------------------------------
# --set-column: parse_set_column_spec
# ---------------------------------------------------------------------------


class TestParseSetColumnSpec:
    from data_lake.ingest.fits_to_parquet import parse_set_column_spec

    def test_auto_int(self):
        from data_lake.ingest.fits_to_parquet import parse_set_column_spec
        name, scalar = parse_set_column_spec("VISIT=7")
        assert name == "VISIT"
        assert scalar.as_py() == 7
        import pyarrow as pa
        assert scalar.type == pa.int64()

    def test_auto_float(self):
        from data_lake.ingest.fits_to_parquet import parse_set_column_spec
        import pyarrow as pa
        name, scalar = parse_set_column_spec("WEIGHT=1.5")
        assert name == "WEIGHT"
        assert scalar.as_py() == 1.5
        assert scalar.type == pa.float64()

    def test_auto_string(self):
        from data_lake.ingest.fits_to_parquet import parse_set_column_spec
        import pyarrow as pa
        name, scalar = parse_set_column_spec("EPOCH=J2000")
        assert name == "EPOCH"
        assert scalar.as_py() == "J2000"
        assert scalar.type == pa.string()

    def test_explicit_str_type_preserves_zero_padding(self):
        from data_lake.ingest.fits_to_parquet import parse_set_column_spec
        import pyarrow as pa
        name, scalar = parse_set_column_spec("VISIT:str=007")
        assert name == "VISIT"
        assert scalar.as_py() == "007"
        assert scalar.type == pa.string()

    def test_explicit_int_type(self):
        from data_lake.ingest.fits_to_parquet import parse_set_column_spec
        import pyarrow as pa
        name, scalar = parse_set_column_spec("N:int=42")
        assert scalar.as_py() == 42
        assert scalar.type == pa.int64()

    def test_explicit_bool_type(self):
        from data_lake.ingest.fits_to_parquet import parse_set_column_spec
        name, scalar = parse_set_column_spec("FLAG:bool=true")
        assert scalar.as_py() is True

    def test_value_with_colon(self):
        from data_lake.ingest.fits_to_parquet import parse_set_column_spec
        name, scalar = parse_set_column_spec("URL=http://example.com:8080/path")
        assert name == "URL"
        assert scalar.as_py() == "http://example.com:8080/path"

    def test_value_with_equals(self):
        from data_lake.ingest.fits_to_parquet import parse_set_column_spec
        name, scalar = parse_set_column_spec("KV=a=b=c")
        assert name == "KV"
        assert scalar.as_py() == "a=b=c"

    def test_bad_spec_no_equals(self):
        from data_lake.ingest.fits_to_parquet import parse_set_column_spec
        with pytest.raises(ValueError, match="NAME=VALUE"):
            parse_set_column_spec("VISIT7")

    def test_bad_type_tag(self):
        from data_lake.ingest.fits_to_parquet import parse_set_column_spec
        with pytest.raises(ValueError, match="unknown type tag"):
            parse_set_column_spec("X:complex=1+2j")

    def test_reserved_source_id(self):
        from data_lake.ingest.fits_to_parquet import parse_set_column_spec
        with pytest.raises(ValueError, match="lake-internal"):
            parse_set_column_spec("_source_id=123")

    def test_reserved_healpix_prefix(self):
        from data_lake.ingest.fits_to_parquet import parse_set_column_spec
        with pytest.raises(ValueError, match="reserved lake-internal prefix"):
            parse_set_column_spec("_healpix_norder5=1")

    def test_empty_name(self):
        from data_lake.ingest.fits_to_parquet import parse_set_column_spec
        with pytest.raises(ValueError, match="empty"):
            parse_set_column_spec("=VALUE")


# ---------------------------------------------------------------------------
# --set-column: end-to-end ingest
# ---------------------------------------------------------------------------


class TestSetColumnEndToEnd:
    """Integration tests: inject VISIT, use in composite link-id."""

    def _write_fits(self, path: Path, visit: int, base_id: int = 1000) -> None:
        """Write a small catalog FITS file with object IDs and sky coords."""
        n = 8
        rng = np.random.default_rng(base_id)
        tbl = Table({
            "OBJECT_ID": np.arange(base_id, base_id + n, dtype=np.int64),
            "TILE_ID": np.full(n, 42, dtype=np.int64),
            "RA": rng.uniform(120.0, 121.0, n),
            "DEC": rng.uniform(44.0, 46.0, n),
        })
        tbl.write(str(path), format="fits", overwrite=True)

    def test_visit_column_present_and_correct_value(self, tmp_path):
        from data_lake.ingest.fits_to_parquet import ingest_catalog, parse_set_column_spec
        fits_path = tmp_path / "visit7.fits"
        self._write_fits(fits_path, visit=7)
        _, scalar = parse_set_column_spec("VISIT=7")
        ingest_catalog(
            fits_path,
            tmp_path / "lake",
            survey_name="MYSURVEY",
            ra_col="RA",
            dec_col="DEC",
            link_id_col="OBJECT_ID,TILE_ID,VISIT",
            set_columns={"VISIT": scalar},
        )
        _, table = _read_merged_catalog(tmp_path / "lake", "MYSURVEY")
        assert "VISIT" in table.schema.names
        assert set(table.column("VISIT").to_pylist()) == {7}

    def test_composite_link_id_mode_recorded(self, tmp_path):
        from data_lake.ingest.fits_to_parquet import ingest_catalog, parse_set_column_spec
        import pyarrow as pa
        fits_path = tmp_path / "v1.fits"
        self._write_fits(fits_path, visit=1)
        _, scalar = parse_set_column_spec("VISIT=1")
        ingest_catalog(
            fits_path,
            tmp_path / "lake",
            survey_name="MYSURVEY",
            ra_col="RA",
            dec_col="DEC",
            link_id_col="OBJECT_ID,TILE_ID,VISIT",
            set_columns={"VISIT": scalar},
        )
        info_path = tmp_path / "lake" / "catalogs" / "MYSURVEY" / "catalog_info.json"
        import json
        info = json.loads(info_path.read_text())
        assert "composite" in info["link_id_mode"]
        assert info["set_columns"] == {"VISIT": 1}

    def test_distinct_source_ids_across_visits(self, tmp_path):
        """Same OBJECT_ID+TILE_ID in two different visits → different _source_id."""
        from data_lake.ingest.fits_to_parquet import ingest_catalog, parse_set_column_spec
        lake = tmp_path / "lake"
        for visit in (1, 2):
            fits_path = tmp_path / f"v{visit}.fits"
            self._write_fits(fits_path, visit=visit, base_id=1000)
            _, scalar = parse_set_column_spec(f"VISIT={visit}")
            ingest_catalog(
                fits_path,
                lake,
                survey_name="MYSURVEY",
                ra_col="RA",
                dec_col="DEC",
                link_id_col="OBJECT_ID,TILE_ID,VISIT",
                tile_mode="append",
                on_duplicate_id="error",
                set_columns={"VISIT": scalar},
            )
        _, table = _read_merged_catalog(lake, "MYSURVEY")
        src_ids = table.column("_source_id").to_pylist()
        assert len(src_ids) == len(set(src_ids)), "Duplicate _source_id across visits"

    def test_collision_raises(self, tmp_path):
        """Injecting a column name that already exists in the source table raises."""
        from data_lake.ingest.fits_to_parquet import ingest_catalog, parse_set_column_spec
        fits_path = tmp_path / "coll.fits"
        # TILE_ID already exists in the FITS file
        self._write_fits(fits_path, visit=1)
        _, scalar = parse_set_column_spec("TILE_ID=99")
        with pytest.raises(ValueError, match="already exists"):
            ingest_catalog(
                fits_path,
                tmp_path / "lake",
                survey_name="MYSURVEY",
                ra_col="RA",
                dec_col="DEC",
                link_id_col="OBJECT_ID",
                set_columns={"TILE_ID": scalar},
            )

    def test_streaming_path_same_source_ids(self, tmp_path):
        """Streaming and non-streaming paths produce identical _source_id values."""
        from data_lake.ingest.fits_to_parquet import ingest_catalog, parse_set_column_spec
        fits_path = tmp_path / "src.fits"
        self._write_fits(fits_path, visit=3)
        _, scalar = parse_set_column_spec("VISIT=3")

        lake_std = tmp_path / "lake_std"
        lake_stream = tmp_path / "lake_stream"
        kwargs = dict(
            ra_col="RA",
            dec_col="DEC",
            link_id_col="OBJECT_ID,TILE_ID,VISIT",
            set_columns={"VISIT": scalar},
            survey_name="MYSURVEY",
        )
        ingest_catalog(fits_path, lake_std, **kwargs)
        ingest_catalog(fits_path, lake_stream, streaming=True, **kwargs)

        _, t_std = _read_merged_catalog(lake_std, "MYSURVEY")
        _, t_stream = _read_merged_catalog(lake_stream, "MYSURVEY")
        ids_std = sorted(t_std.column("_source_id").to_pylist())
        ids_stream = sorted(t_stream.column("_source_id").to_pylist())
        assert ids_std == ids_stream

    def test_append_missing_injected_column_raises(self, tmp_path):
        """Appending without --set-column into tiles that have an injected column raises."""
        from data_lake.ingest.fits_to_parquet import ingest_catalog, parse_set_column_spec
        fits_path = tmp_path / "src.fits"
        self._write_fits(fits_path, visit=1)
        lake = tmp_path / "lake"
        _, scalar = parse_set_column_spec("VISIT=1")
        # First ingest: with VISIT column
        ingest_catalog(
            fits_path,
            lake,
            survey_name="MYSURVEY",
            ra_col="RA",
            dec_col="DEC",
            link_id_col="OBJECT_ID",
            tile_mode="append",
            set_columns={"VISIT": scalar},
        )
        # Second ingest: without VISIT → should raise on schema mismatch
        with pytest.raises(ValueError, match="VISIT"):
            ingest_catalog(
                fits_path,
                lake,
                survey_name="MYSURVEY",
                ra_col="RA",
                dec_col="DEC",
                link_id_col="OBJECT_ID",
                tile_mode="append",
            )

    def test_set_columns_in_catalog_info_preserved_after_finalize(self, tmp_path):
        """dl-finalize-catalog (finalize_catalog_survey) preserves set_columns."""
        from data_lake.ingest.fits_to_parquet import (
            finalize_catalog_survey, ingest_catalog, parse_set_column_spec,
        )
        fits_path = tmp_path / "src.fits"
        self._write_fits(fits_path, visit=5)
        lake = tmp_path / "lake"
        _, scalar = parse_set_column_spec("VISIT=5")
        ingest_catalog(
            fits_path,
            lake,
            survey_name="MYSURVEY",
            ra_col="RA",
            dec_col="DEC",
            link_id_col="OBJECT_ID,TILE_ID,VISIT",
            set_columns={"VISIT": scalar},
        )
        catalog_root = lake / "catalogs" / "MYSURVEY"
        finalize_catalog_survey(catalog_root, "MYSURVEY", norder=5)
        import json
        info = json.loads((catalog_root / "catalog_info.json").read_text())
        assert info.get("set_columns") == {"VISIT": 5}


# ---------------------------------------------------------------------------
# Tests for the shared resolve_link_object_id / resolve_link_object_ids helpers
# ---------------------------------------------------------------------------

class TestResolveLinkObjectId:
    """Unit tests for the canonical scalar + vector resolvers in fits_to_parquet."""

    def test_single_part_scalar(self):
        from data_lake.ingest.fits_to_parquet import (
            normalize_object_id,
            resolve_link_object_id,
        )
        result = resolve_link_object_id("TARGETID", lambda name: 12345)
        assert result == normalize_object_id(12345)

    def test_composite_scalar_matches_composite_link_label(self):
        from data_lake.ingest.fits_to_parquet import (
            composite_link_label,
            normalize_object_id,
            resolve_link_object_id,
        )
        mapping = {"TARGETID": 9999, "SURVEY": "main", "PROGRAM": "dark"}
        result = resolve_link_object_id(
            "TARGETID,SURVEY,PROGRAM", lambda name: mapping[name]
        )
        expected = normalize_object_id(
            composite_link_label(9999, "main", "dark")
        )
        assert result == expected

    def test_missing_key_raises_keyerror_with_context(self):
        from data_lake.ingest.fits_to_parquet import resolve_link_object_id

        def _raise(name: str):
            raise KeyError(name)

        import pytest
        with pytest.raises(KeyError, match="TARGETID"):
            resolve_link_object_id("TARGETID", _raise, context="test.fits")

    def test_empty_spec_raises_valueerror(self):
        import pytest
        from data_lake.ingest.fits_to_parquet import resolve_link_object_id

        with pytest.raises(ValueError, match="Empty"):
            resolve_link_object_id("  ,  ", lambda n: n)

    def test_vector_single_part(self):
        import numpy as np

        from data_lake.ingest.fits_to_parquet import (
            normalize_object_id,
            resolve_link_object_ids,
        )
        cols = {"TARGETID": [101, 202, 303]}
        ids = resolve_link_object_ids("TARGETID", cols)
        expected = np.asarray(
            [normalize_object_id(v) for v in cols["TARGETID"]], dtype=np.int64
        )
        np.testing.assert_array_equal(ids, expected)

    def test_vector_composite_matches_scalar(self):
        import numpy as np

        from data_lake.ingest.fits_to_parquet import (
            composite_link_label,
            normalize_object_id,
            resolve_link_object_id,
            resolve_link_object_ids,
        )
        cols = {
            "TARGETID": [1, 2],
            "SURVEY": ["main", "sv3"],
            "PROGRAM": ["dark", "bright"],
        }
        ids = resolve_link_object_ids("TARGETID,SURVEY,PROGRAM", cols)
        for i in range(2):
            scalar = resolve_link_object_id(
                "TARGETID,SURVEY,PROGRAM",
                lambda name, _i=i: cols[name][_i],
            )
            assert int(ids[i]) == scalar

    def test_vector_missing_column_raises_keyerror(self):
        import pytest
        from data_lake.ingest.fits_to_parquet import resolve_link_object_ids

        with pytest.raises(KeyError, match="PROGRAM"):
            resolve_link_object_ids(
                "TARGETID,SURVEY,PROGRAM",
                {"TARGETID": [1], "SURVEY": ["main"]},
            )


class TestObjectIdFromFitsHeaderComposite:
    """Composite keyword spec in object_id_from_fits_header."""

    def test_composite_header_matches_resolver(self):
        from astropy.io import fits

        from data_lake.ingest.fits_to_parquet import (
            composite_link_label,
            normalize_object_id,
            object_id_from_fits_header,
        )
        hdr = fits.Header()
        hdr["TARGETID"] = 9876543210123456
        hdr["SURVEY"] = "main"
        hdr["PROGRAM"] = "dark"

        result = object_id_from_fits_header(hdr, "TARGETID,SURVEY,PROGRAM")
        expected = normalize_object_id(
            composite_link_label(9876543210123456, "main", "dark")
        )
        assert result == expected

    def test_single_keyword_unchanged(self):
        from astropy.io import fits

        from data_lake.ingest.fits_to_parquet import (
            normalize_object_id,
            object_id_from_fits_header,
        )
        hdr = fits.Header()
        hdr["TARGETID"] = 12345

        result = object_id_from_fits_header(hdr, "TARGETID")
        assert result == normalize_object_id(12345)

    def test_missing_composite_part_raises_keyerror(self):
        import pytest
        from astropy.io import fits

        from data_lake.ingest.fits_to_parquet import object_id_from_fits_header

        hdr = fits.Header()
        hdr["TARGETID"] = 1
        hdr["SURVEY"] = "main"
        # PROGRAM is absent

        with pytest.raises(KeyError, match="PROGRAM"):
            object_id_from_fits_header(hdr, "TARGETID,SURVEY,PROGRAM")
