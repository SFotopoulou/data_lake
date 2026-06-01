"""Unit tests for catalog_cli_options.resolve_catalog_tile_mode."""

import pytest

from data_lake.ingest.catalog_cli_options import resolve_catalog_tile_mode


class TestResolveCatalogTileMode:
    def test_none_sequential_returns_none(self):
        assert resolve_catalog_tile_mode(None, parallel=False) is None

    def test_none_parallel_returns_append(self):
        assert resolve_catalog_tile_mode(None, parallel=True) == "append"

    def test_explicit_append_sequential(self):
        assert resolve_catalog_tile_mode("append", parallel=False) == "append"

    def test_explicit_append_parallel(self):
        assert resolve_catalog_tile_mode("append", parallel=True) == "append"

    def test_explicit_skip(self):
        assert resolve_catalog_tile_mode("skip", parallel=False) == "skip"
        assert resolve_catalog_tile_mode("skip", parallel=True) == "skip"

    def test_explicit_overwrite(self):
        assert resolve_catalog_tile_mode("overwrite", parallel=False) == "overwrite"

    def test_uppercase_normalized(self):
        assert resolve_catalog_tile_mode("Append", parallel=False) == "append"
        assert resolve_catalog_tile_mode("SKIP", parallel=True) == "skip"
        assert resolve_catalog_tile_mode("Overwrite", parallel=False) == "overwrite"
