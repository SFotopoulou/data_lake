"""CatalogAccessor ID lookups: chunked semi-join, duplicates and missing IDs."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_lake.io.catalog import CatalogAccessor
from tests.test_crossmatch import _write_catalog_tile


@pytest.fixture
def lake(tmp_path: Path) -> Path:
    root = tmp_path / "lake"
    for npix, ids in ((100, [1, 2, 3]), (101, [4, 5])):
        _write_catalog_tile(
            root, "S", norder=5, npix=npix, source_ids=ids,
            ra=[30.0] * len(ids), dec=[5.0] * len(ids),
        )
    return root


@pytest.mark.parametrize("batch_size", [1, 2, 1_000_000])
def test_get_sources_by_id_chunks(lake: Path, batch_size: int) -> None:
    with CatalogAccessor(lake, "S", norder=5) as cat:
        df = cat.get_sources_by_id([5, 1, 1, 3, 999], columns=["_source_id"], batch_size=batch_size)
    assert sorted(df["_source_id"].to_list()) == [1, 3, 5]


def test_get_sources_by_id_empty(lake: Path) -> None:
    with CatalogAccessor(lake, "S", norder=5) as cat:
        assert cat.get_sources_by_id([]).height == 0
        assert cat.get_sources_by_id([2], fmt="arrow").num_rows == 1
