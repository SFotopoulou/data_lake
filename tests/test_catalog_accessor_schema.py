"""CatalogAccessor schema reads: small schema file, read once, never the aggregate _metadata."""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import finalize_catalog_survey
from data_lake.io.catalog import CatalogAccessor
from data_lake.schema_registry import catalog_schema_source, read_catalog_arrow_schema
from tests.test_crossmatch import _write_catalog_tile


def _counting_read_schema(monkeypatch) -> list[str]:
    calls: list[str] = []
    real = pq.read_schema

    def counting(path, *args, **kwargs):
        calls.append(Path(str(path)).name)
        return real(path, *args, **kwargs)

    monkeypatch.setattr("data_lake.io.catalog.pq.read_schema", counting)
    return calls


def _finalized_lake(tmp_path: Path) -> tuple[Path, Path]:
    lake = tmp_path / "lake"
    for npix in (100, 101):
        _write_catalog_tile(
            lake, "B", norder=5, npix=npix, source_ids=[npix], ra=[30.0], dec=[5.0],
        )
    root = lake / "catalogs" / "B"
    assert finalize_catalog_survey(root, "B", 5)
    return lake, root


def test_finalize_writes_common_metadata(tmp_path: Path) -> None:
    _, root = _finalized_lake(tmp_path)
    common = root / "_common_metadata"
    assert common.is_file()
    assert (root / "_metadata").is_file()
    assert catalog_schema_source(root) == common
    assert pq.read_metadata(str(common)).num_row_groups == 0
    assert "ra" in read_catalog_arrow_schema(root).names


def test_schema_read_once_across_column_lookups(tmp_path: Path, monkeypatch) -> None:
    lake, root = _finalized_lake(tmp_path)
    calls = _counting_read_schema(monkeypatch)

    with CatalogAccessor(lake, "B", norder=5) as acc:
        n_init = len(calls)
        for _ in range(5):
            acc.sources_in_tile(100, columns=["ra", "dec"])
            acc.resolve_column("ra")
        assert len(calls) == n_init
        assert "_metadata" not in calls
        assert calls.count("_common_metadata") == 1

        common = root / "_common_metadata"
        st = common.stat()
        os.utime(common, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        acc.resolve_column("dec")
        assert calls.count("_common_metadata") == 2


def test_schema_falls_back_to_one_tile_without_common_metadata(
    tmp_path: Path, monkeypatch,
) -> None:
    lake, root = _finalized_lake(tmp_path)
    (root / "_common_metadata").unlink()
    calls = _counting_read_schema(monkeypatch)
    with CatalogAccessor(lake, "B", norder=5) as acc:
        assert acc.columns
    assert "_metadata" not in calls
    assert all(name.startswith("Npix=") for name in calls)


def test_common_metadata_unifies_null_columns(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    _write_catalog_tile(lake, "B", norder=5, npix=100, source_ids=[1], ra=[30.0], dec=[5.0])
    root = lake / "catalogs" / "B"
    tile = next(root.rglob("Npix=100.parquet"))
    t = pq.read_table(tile)
    pq.write_table(t.append_column("extra", pa.nulls(t.num_rows)), tile)
    _write_catalog_tile(lake, "B", norder=5, npix=101, source_ids=[2], ra=[31.0], dec=[5.0])
    tile2 = next(root.rglob("Npix=101.parquet"))
    t2 = pq.read_table(tile2)
    pq.write_table(t2.append_column("extra", pa.array([1.5])), tile2)

    assert finalize_catalog_survey(root, "B", 5)
    assert pq.read_schema(str(root / "_common_metadata")).field("extra").type == pa.float64()
