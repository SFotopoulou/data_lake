"""Tests for bulk source_id lookup (spectra and cutouts)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from data_lake.io.id_lookup import bulk_tile_index_from_catalog, bulk_tile_index_with_scan
from data_lake.io.spectra import SpectrumAccessor

# Re-use synthetic lake helpers from extract_subset tests
from synthetic_lake_helpers import (
    NORDER,
    SOURCES,
    SURVEY,
    ingest_synthetic_spectrum_lake,
)


@pytest.fixture
def synthetic_lake(tmp_path: Path) -> Path:
    ingest_synthetic_spectrum_lake(tmp_path)
    return tmp_path


class TestBulkTileIndexFromCatalog:
    def test_resolves_via_sql(self, synthetic_lake: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from data_lake.ingest.fits_to_parquet import healpix_dir
        from data_lake.io.catalog import CatalogAccessor

        cat_root = synthetic_lake / "catalogs" / SURVEY
        cat_root.mkdir(parents=True)
        # Minimal catalog tile with spectrum indices
        acc_spec = SpectrumAccessor(synthetic_lake, SURVEY)
        lookup = acc_spec._build_source_id_lookup(
            np.array([s[0] for s in SOURCES], dtype=np.int64), show_progress=False,
        )
        rows = []
        for sid in (101, 102, 201, 202):
            npix, lidx = lookup[sid]
            rows.append({
                "_source_id": sid,
                "_healpix_norder5": npix,
                "_spectrum_index": lidx,
                "_spectrum_npix": npix,
            })
        table = pa.Table.from_pylist(rows)
        npix = int(rows[0]["_healpix_norder5"])
        out_dir = cat_root / healpix_dir(NORDER, npix)
        out_dir.mkdir(parents=True)
        pq.write_table(table, out_dir / f"Npix={npix}.parquet")
        (cat_root / "catalog_info.json").write_text(
            '{"hats_order": 5, "link_id_column": "_source_id"}'
        )

        cat = CatalogAccessor(synthetic_lake, SURVEY, norder=NORDER)
        result = bulk_tile_index_from_catalog(
            cat, [101, 201], "spectrum", show_progress=False,
        )
        assert result[101] == lookup[101]
        assert result[201] == lookup[201]

    def test_empty_when_index_column_missing(self, synthetic_lake: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from data_lake.ingest.fits_to_parquet import healpix_dir
        from data_lake.io.catalog import CatalogAccessor

        cat_root = synthetic_lake / "catalogs" / "empty_idx"
        cat_root.mkdir(parents=True)
        table = pa.table({"_source_id": pa.array([1], type=pa.int64())})
        out_dir = cat_root / healpix_dir(NORDER, 1)
        out_dir.mkdir(parents=True)
        pq.write_table(table, out_dir / "Npix=1.parquet")
        (cat_root / "catalog_info.json").write_text(
            '{"hats_order": 5, "link_id_column": "_source_id"}'
        )
        cat = CatalogAccessor(synthetic_lake, "empty_idx", norder=NORDER)
        assert bulk_tile_index_from_catalog(cat, [1], "spectrum") == {}


class TestSpectrumGetBatchBulkLookup:
    def test_get_batch_matches_individual(self, synthetic_lake: Path) -> None:
        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        ids = [101, 103, 201]
        batch = acc.get_batch(ids)
        flux_b, ivar_b, mask_b, wave_b = batch
        for i, sid in enumerate(ids):
            spec = acc.get_spectrum(sid)
            np.testing.assert_array_equal(flux_b[i], spec.flux)
            np.testing.assert_array_equal(ivar_b[i], spec.ivar)
            np.testing.assert_array_equal(mask_b[i], spec.mask)
        assert wave_b.ndim == 1

    def test_bulk_lookup_uses_single_catalog_query_batch(self, synthetic_lake: Path) -> None:
        """Bulk path should not call get_tile_index per ID."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        from data_lake.ingest.fits_to_parquet import healpix_dir
        from data_lake.io.catalog import CatalogAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        base_lookup = acc._build_source_id_lookup(
            np.array([s[0] for s in SOURCES], dtype=np.int64), show_progress=False,
        )
        cat_root = synthetic_lake / "catalogs" / SURVEY
        cat_root.mkdir(parents=True)
        rows = []
        for sid, _, _, _ in SOURCES:
            npix, lidx = base_lookup[sid]
            rows.append({
                "_source_id": sid,
                "_healpix_norder5": npix,
                "_spectrum_index": lidx,
                "_spectrum_npix": npix,
            })
        table = pa.Table.from_pylist(rows)
        for npix in sorted({r["_healpix_norder5"] for r in rows}):
            sub = table.filter(
                pa.compute.equal(table["_healpix_norder5"], npix)
            )
            out_dir = cat_root / healpix_dir(NORDER, npix)
            out_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(sub, out_dir / f"Npix={npix}.parquet")
        (cat_root / "catalog_info.json").write_text(
            '{"hats_order": 5, "link_id_column": "_source_id"}'
        )

        cat = CatalogAccessor(synthetic_lake, SURVEY, norder=NORDER)
        acc_cat = SpectrumAccessor(synthetic_lake, SURVEY, catalog_accessor=cat)

        with patch.object(cat, "get_tile_index", wraps=cat.get_tile_index) as spy:
            ids = [101, 102, 103, 201, 202]
            acc_cat.get_batch(ids)
            assert spy.call_count == 0

    def test_tile_scan_fallback_without_catalog(self, synthetic_lake: Path) -> None:
        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        result = acc._build_source_id_lookup(
            np.array([101, 999], dtype=np.int64), show_progress=False,
        )
        assert 101 in result
        assert 999 not in result


class TestBulkTileIndexWithScan:
    def test_scan_callback_used_for_misses(self) -> None:
        calls: list[int] = []

        def scan(npix: int, remaining: np.ndarray) -> dict[int, tuple[int, int]]:
            calls.append(npix)
            if npix == 5 and 42 in remaining:
                return {42: (5, 0)}
            return {}

        result = bulk_tile_index_with_scan(
            [42],
            catalog=None,
            kind="spectrum",
            scan_tile=scan,
            available_tiles=lambda: [3, 5, 7],
            show_progress=False,
        )
        assert result == {42: (5, 0)}
        assert calls == [3, 5]
