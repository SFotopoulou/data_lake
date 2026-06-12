"""Tests for deferred finalize (lifecycle=live) catalog ingest ergonomics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import (
    LAKE_JOIN_ID_COLUMN,
    _finalize_catalog_writes,
    assign_healpix,
    finalize_catalog_survey,
    healpix_dir,
)


def _write_tile(catalog_root: Path, norder: int, npix: int, n: int = 3) -> None:
    tile_dir = catalog_root / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    hp_col = f"_healpix_norder{norder}"
    pq.write_table(
        pa.table({
            LAKE_JOIN_ID_COLUMN: pa.array(list(range(n)), type=pa.int64()),
            "ra": pa.array([120.0] * n, type=pa.float64()),
            "dec": pa.array([45.0] * n, type=pa.float64()),
            hp_col: pa.array([npix] * n, type=pa.int64()),
        }),
        tile_dir / f"Npix={npix}.parquet",
    )


class TestDeferredFinalize:
    def test_defer_skips_metadata_and_marks_live(self, tmp_path: Path) -> None:
        norder = 5
        npix = int(assign_healpix(np.array([120.0]), np.array([45.0]), norder)[0])
        catalog_root = tmp_path / "lake" / "catalogs" / "LIVE"
        _write_tile(catalog_root, norder, npix)

        _finalize_catalog_writes(
            catalog_root, "LIVE", norder,
            ra_col="ra", dec_col="dec", link_id_mode="sequential",
            streaming=False, fallback_n_cols=4,
            regenerate_metadata=False, lifecycle="live", finalized=False,
        )

        info = json.loads((catalog_root / "catalog_info.json").read_text())
        assert info["lifecycle"] == "live"
        assert info["finalized"] is False
        # The expensive aggregate _metadata was NOT rebuilt.
        assert not (catalog_root / "_metadata").exists()

    def test_finalize_flips_finalized_true(self, tmp_path: Path) -> None:
        norder = 5
        npix = int(assign_healpix(np.array([120.0]), np.array([45.0]), norder)[0])
        catalog_root = tmp_path / "lake" / "catalogs" / "LIVE"
        _write_tile(catalog_root, norder, npix)
        _finalize_catalog_writes(
            catalog_root, "LIVE", norder,
            ra_col="ra", dec_col="dec", link_id_mode="sequential",
            streaming=False, fallback_n_cols=4,
            regenerate_metadata=False, lifecycle="live", finalized=False,
        )

        assert finalize_catalog_survey(catalog_root, "LIVE", norder, ra_col="ra", dec_col="dec")
        info = json.loads((catalog_root / "catalog_info.json").read_text())
        assert info["finalized"] is True
        # lifecycle is preserved across finalize.
        assert info["lifecycle"] == "live"
        assert (catalog_root / "_metadata").exists()
