"""Tests for dl-widen-spectrum-tiles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from data_lake.ingest.fits_to_parquet import healpix_dir
from data_lake.ingest.widen_spectrum_tiles import widen_survey_tiles


NORDER = 5
N_PIX_NARROW = 8
N_PIX_SURVEY = 12
SURVEY = "WIDEN_TEST"


def _write_survey_with_tile(
    lake: Path,
    *,
    tile_n_pix: int = N_PIX_NARROW,
    survey_n_pix: int = N_PIX_SURVEY,
) -> Path:
    from data_lake.ingest.fits_to_spectra_zarr import (
        _META_DTYPE,
        _open_or_create_spectrum_tile,
    )
    from data_lake.ingest.zarr_ids import zarr_join_array

    survey_root = lake / "spectra" / SURVEY
    survey_root.mkdir(parents=True)
    (survey_root / "spectrum_info.json").write_text(
        json.dumps({
            "n_pix": survey_n_pix,
            "hats_order": NORDER,
            "wavelength_mode": "shared",
            "mask_dtype": "uint8",
            "wcs": {
                "ctype": "WAVE",
                "crval": 3600.0,
                "cdelt": 1.0,
                "crpix": 1.0,
                "unit": "Angstrom",
                "air_or_vacuum": "vacuum",
                "n_pix": survey_n_pix,
            },
        })
    )
    npix = 34
    zpath = survey_root / healpix_dir(NORDER, npix) / f"Npix={npix}.zarr"
    zpath.parent.mkdir(parents=True)
    wcs_attrs = {
        "ctype": "WAVE",
        "crval": 3600.0,
        "cdelt": 1.0,
        "crpix": 1.0,
        "unit": "Angstrom",
        "air_or_vacuum": "vacuum",
        "n_pix": tile_n_pix,
    }
    root = _open_or_create_spectrum_tile(
        zpath, tile_n_pix, "shared", np.dtype(np.uint8), wcs_attrs,
    )
    n_row = 3
    root["flux"].append(np.zeros((n_row, tile_n_pix), dtype=np.float32))
    root["ivar"].append(np.zeros((n_row, tile_n_pix), dtype=np.float32))
    root["mask"].append(np.zeros((n_row, tile_n_pix), dtype=np.uint8))
    zarr_join_array(root).append(np.array([101, 102, 103], dtype=np.int64))
    root["meta"].append(np.zeros(n_row, dtype="|V" + str(_META_DTYPE.itemsize)))
    root["wavelength"][:] = np.linspace(3600.0, 3600.0 + tile_n_pix - 1, tile_n_pix)
    return zpath


class TestWidenSurveyTiles:
    def test_widens_narrow_tile(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        zpath = _write_survey_with_tile(lake)

        result = widen_survey_tiles(lake, SURVEY)
        assert result.ok
        assert result.tiles_widened == 1
        assert result.tiles_already_ok == 0

        root = zarr.open_group(
            store=zarr.storage.LocalStore(str(zpath)),
            mode="r",
            zarr_format=3,
        )
        assert root["flux"].shape == (3, N_PIX_SURVEY)
        assert root["wavelength"].shape == (N_PIX_SURVEY,)

    def test_dry_run_leaves_tile_unchanged(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        zpath = _write_survey_with_tile(lake)

        result = widen_survey_tiles(lake, SURVEY, dry_run=True)
        assert result.ok
        assert result.tiles_widened == 1

        root = zarr.open_group(
            store=zarr.storage.LocalStore(str(zpath)),
            mode="r",
            zarr_format=3,
        )
        assert root["flux"].shape == (3, N_PIX_NARROW)

    def test_already_wide_skipped(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_survey_with_tile(lake, tile_n_pix=N_PIX_SURVEY, survey_n_pix=N_PIX_SURVEY)

        result = widen_survey_tiles(lake, SURVEY)
        assert result.ok
        assert result.tiles_widened == 0
        assert result.tiles_already_ok == 1


class TestWidenCli:
    def test_cli_all(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.ingest.widen_spectrum_tiles import cli

        if cli is None:
            pytest.skip("click not available")

        lake = tmp_path / "lake"
        _write_survey_with_tile(lake)
        result = CliRunner().invoke(cli, ["--all", str(lake)])
        assert result.exit_code == 0, result.output
        assert "Widened 1" in result.output or "widened 1" in result.output
        assert SURVEY in result.output

    def test_cli_dry_run(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.ingest.widen_spectrum_tiles import cli

        if cli is None:
            pytest.skip("click not available")

        lake = tmp_path / "lake"
        _write_survey_with_tile(lake)
        result = CliRunner().invoke(
            cli, ["--survey", SURVEY, "--dry-run", str(lake)],
        )
        assert result.exit_code == 0, result.output
        assert "Would widen" in result.output or "would widen" in result.output
