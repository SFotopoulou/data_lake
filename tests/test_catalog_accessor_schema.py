"""CatalogAccessor schema caching (aggregate _metadata is expensive to parse)."""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow.parquet as pq

from data_lake.io.catalog import CatalogAccessor
from tests.test_crossmatch import _write_catalog_tile


def _write_aggregate_metadata(lake: Path, survey: str) -> Path:
    root = lake / "catalogs" / survey
    tiles = sorted(root.rglob("Npix=*.parquet"))
    agg = pq.read_metadata(str(tiles[0]))
    for tile in tiles[1:]:
        agg.append_row_groups(pq.read_metadata(str(tile)))
    path = root / "_metadata"
    agg.write_metadata_file(str(path))
    return path


def test_metadata_read_once_across_column_lookups(tmp_path: Path, monkeypatch) -> None:
    lake = tmp_path / "lake"
    _write_catalog_tile(lake, "B", norder=5, npix=100, source_ids=[1], ra=[30.0], dec=[5.0])
    meta = _write_aggregate_metadata(lake, "B")

    calls: list[str] = []
    real = pq.read_metadata

    def counting(path, *args, **kwargs):
        if str(path).endswith("_metadata"):
            calls.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr("data_lake.io.catalog.pq.read_metadata", counting)

    with CatalogAccessor(lake, "B", norder=5) as acc:
        n_init = len(calls)
        for _ in range(5):
            acc.sources_in_tile(100, columns=["ra", "dec"])
            acc.resolve_column("ra")
        assert len(calls) == n_init <= 1

        st = meta.stat()
        os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        acc.resolve_column("dec")
        assert len(calls) == n_init + 1
