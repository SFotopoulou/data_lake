"""Tests for catalog ↔ spectrum Zarr link validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, healpix_dir
from data_lake.ingest.validate_catalog_spectra_link import run_validation


NORDER = 5
N_PIX = 8
SURVEY = "LINK_TEST"


def _write_spectrum_info(survey_root: Path) -> None:
    survey_root.mkdir(parents=True, exist_ok=True)
    (survey_root / "spectrum_info.json").write_text(
        '{"n_pix": 8, "hats_order": 5, "wavelength_mode": "shared", "wcs": {}}'
    )


def _write_min_zarr(
    zarr_path: Path,
    *,
    source_ids: list[int],
) -> None:
    from data_lake.ingest.fits_to_spectra_zarr import (
        _META_DTYPE,
        _open_or_create_spectrum_tile,
    )
    from data_lake.ingest.zarr_ids import zarr_join_array

    n = len(source_ids)
    wcs_attrs = {
        "ctype": "WAVE",
        "crval": 3600.0,
        "cdelt": 1.0,
        "crpix": 1.0,
        "unit": "Angstrom",
        "air_or_vacuum": "vacuum",
        "n_pix": N_PIX,
    }
    root = _open_or_create_spectrum_tile(
        zarr_path, N_PIX, "shared", np.dtype(np.uint8), wcs_attrs,
    )
    root["flux"].append(np.zeros((n, N_PIX), dtype=np.float32))
    root["ivar"].append(np.zeros((n, N_PIX), dtype=np.float32))
    root["mask"].append(np.zeros((n, N_PIX), dtype=np.uint8))
    zarr_join_array(root).append(np.array(source_ids, dtype=np.int64))
    meta = np.zeros(n, dtype="|V" + str(_META_DTYPE.itemsize))
    root["meta"].append(meta)
    root["wavelength"][:] = np.linspace(3600.0, 3600.0 + N_PIX - 1, N_PIX)


def _write_catalog_tile(
    cat_path: Path,
    *,
    source_ids: list[int],
    spectrum_indices: list[int],
    npix: int,
) -> None:
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            LAKE_JOIN_ID_COLUMN: pa.array(source_ids, type=pa.int64()),
            f"_healpix_norder{NORDER}": pa.array([npix] * len(source_ids), type=pa.int64()),
            "_spectrum_index": pa.array(spectrum_indices, type=pa.int64()),
        }),
        cat_path,
    )


def _make_lake(tmp_path: Path, npix: int = 42) -> Path:
    lake = tmp_path / "lake"
    spec_root = lake / "spectra" / SURVEY
    _write_spectrum_info(spec_root)
    zarr_path = spec_root / healpix_dir(NORDER, npix) / f"Npix={npix}.zarr"
    _write_min_zarr(zarr_path, source_ids=[101, 102])

    cat_root = lake / "catalogs" / SURVEY
    cat_path = cat_root / healpix_dir(NORDER, npix) / f"Npix={npix}.parquet"
    _write_catalog_tile(
        cat_path,
        source_ids=[101, 102],
        spectrum_indices=[0, 1],
        npix=npix,
    )
    (cat_root / "catalog_info.json").write_text(
        '{"hats_order": 5, "link_id_mode": "sequential", '
        '"link_id_column": "_source_id", "ra_column": "ra", "dec_column": "dec"}'
    )
    return lake


class TestValidateCatalogSpectraLink:
    def test_happy_path(self, tmp_path: Path) -> None:
        lake = _make_lake(tmp_path)
        rep = run_validation(lake, SURVEY)
        assert rep.ok(strict=True)
        assert rep.stats.n_linked == 2
        assert rep.stats.n_orphan_zarr == 0

    def test_wrong_index_id_mismatch(self, tmp_path: Path) -> None:
        lake = _make_lake(tmp_path)
        npix = 42
        cat_path = (
            lake / "catalogs" / SURVEY / healpix_dir(NORDER, npix) / f"Npix={npix}.parquet"
        )
        _write_catalog_tile(
            cat_path,
            source_ids=[101, 999],
            spectrum_indices=[0, 1],
            npix=npix,
        )
        rep = run_validation(lake, SURVEY)
        assert not rep.ok(strict=False)
        assert rep.stats.n_wrong_id >= 1

    def test_orphan_zarr_row(self, tmp_path: Path) -> None:
        lake = _make_lake(tmp_path)
        npix = 42
        zarr_path = (
            lake / "spectra" / SURVEY / healpix_dir(NORDER, npix) / f"Npix={npix}.zarr"
        )
        _write_min_zarr(zarr_path, source_ids=[101, 102, 777])
        rep = run_validation(lake, SURVEY)
        assert not rep.ok(strict=True)
        assert rep.stats.n_orphan_zarr >= 1

    def test_unpatched_catalog_index(self, tmp_path: Path) -> None:
        lake = _make_lake(tmp_path)
        npix = 42
        cat_path = (
            lake / "catalogs" / SURVEY / healpix_dir(NORDER, npix) / f"Npix={npix}.parquet"
        )
        _write_catalog_tile(
            cat_path,
            source_ids=[101, 102],
            spectrum_indices=[0, -1],
            npix=npix,
        )
        rep = run_validation(lake, SURVEY)
        assert rep.ok(strict=False)
        assert not rep.ok(strict=True)
        assert rep.stats.n_unpatched_catalog >= 1

    def test_sample_mode_limits_linked_checks(self, tmp_path: Path) -> None:
        lake = _make_lake(tmp_path)
        rep = run_validation(lake, SURVEY, sample=1)
        assert rep.ok(strict=True)
        assert rep.stats.n_linked == 1

    def test_cli_unpatched_emits_rebuild_hint(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.ingest.validate_catalog_spectra_link import cli

        if cli is None:
            pytest.skip("click not available")

        lake = _make_lake(tmp_path)
        npix = 42
        cat_path = (
            lake / "catalogs" / SURVEY / healpix_dir(NORDER, npix) / f"Npix={npix}.parquet"
        )
        _write_catalog_tile(
            cat_path,
            source_ids=[101, 102],
            spectrum_indices=[-1, -1],
            npix=npix,
        )

        result = CliRunner().invoke(
            cli,
            ["--survey", SURVEY, str(lake)],
        )
        assert result.exit_code == 0
        assert "unpatched catalog:" in result.output
        assert "dl-rebuild-catalog-indices" in result.output
        assert "OK (with" in result.output
