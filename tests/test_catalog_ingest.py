"""
Tests for data_lake.ingest.fits_to_parquet.

Focuses on the FITS → PyArrow conversion path, especially:

* Multidim BINTABLE columns (e.g. DESI's ``COEFF`` shape (N, 10)) survive
  the conversion as Arrow ``FixedSizeList`` arrays — the original
  ``to_pandas`` path crashed with ``ValueError: Cannot convert a table with
  multidimensional columns``.
* Byte-string columns (FITS fixed-width ASCII) decode to UTF-8.
* Format detection works for FITS / Parquet / unknown suffix.
* End-to-end ``ingest_catalog`` writes HEALPix-partitioned Parquet tiles
  and preserves the multidim column on disk.
"""

from __future__ import annotations

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

    def test_unknown_suffix_uses_autodetect_not_fits(self, tmp_path: Path):
        """A non-FITS file with an unknown suffix should not be force-read as FITS."""
        from data_lake.ingest.fits_to_parquet import _read_source_table

        ecsv = tmp_path / "cat.ecsv"
        Table({"a": [1, 2, 3]}).write(str(ecsv), format="ascii.ecsv", overwrite=True)
        arrow_tbl = _read_source_table(ecsv)
        assert arrow_tbl.num_rows == 3


# ---------------------------------------------------------------------------
# End-to-end ingest_catalog with the offending column
# ---------------------------------------------------------------------------


class TestIngestCatalogEndToEnd:
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
