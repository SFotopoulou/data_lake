"""Tests for shared validation CLI helpers and --all survey discovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data_lake.ingest.fits_to_parquet import healpix_dir
from data_lake.ingest.validate_cli import (
    discover_catalog_ingest_surveys,
    discover_catalog_spectra_link_surveys,
    discover_cutout_ingest_surveys,
    discover_spectra_ingest_surveys,
    resolve_validation_survey_names,
)

NORDER = 5
N_PIX = 8
SURVEY_A = "SURVEY_A"
SURVEY_B = "SURVEY_B"


def _write_catalog_survey(lake: Path, name: str) -> None:
    root = lake / "catalogs" / name
    root.mkdir(parents=True)
    (root / "catalog_info.json").write_text(
        json.dumps({
            "hats_order": NORDER,
            "total_rows": 1,
            "total_columns": 3,
            "ra_column": "ra",
            "dec_column": "dec",
        })
    )


def _write_spectra_survey(lake: Path, name: str) -> None:
    from data_lake.ingest.fits_to_spectra_zarr import (
        _META_DTYPE,
        _open_or_create_spectrum_tile,
    )
    from data_lake.ingest.zarr_ids import zarr_join_array

    root = lake / "spectra" / name
    root.mkdir(parents=True)
    (root / "spectrum_info.json").write_text(
        json.dumps({
            "n_pix": N_PIX,
            "hats_order": NORDER,
            "wavelength_mode": "shared",
            "wcs": {},
        })
    )
    npix = 17
    zpath = root / healpix_dir(NORDER, npix) / f"Npix={npix}.zarr"
    zpath.parent.mkdir(parents=True)
    wcs_attrs = {
        "ctype": "WAVE",
        "crval": 3600.0,
        "cdelt": 1.0,
        "crpix": 1.0,
        "unit": "Angstrom",
        "air_or_vacuum": "vacuum",
        "n_pix": N_PIX,
    }
    zroot = _open_or_create_spectrum_tile(
        zpath, N_PIX, "shared", np.dtype(np.uint8), wcs_attrs,
    )
    zroot["flux"].append(np.zeros((1, N_PIX), dtype=np.float32))
    zroot["ivar"].append(np.zeros((1, N_PIX), dtype=np.float32))
    zroot["mask"].append(np.zeros((1, N_PIX), dtype=np.uint8))
    zarr_join_array(zroot).append(np.array([101], dtype=np.int64))
    zroot["meta"].append(np.zeros(1, dtype="|V" + str(_META_DTYPE.itemsize)))
    zroot["wavelength"][:] = np.linspace(3600.0, 3600.0 + N_PIX - 1, N_PIX)


def _write_cutout_survey(lake: Path, name: str) -> None:
    root = lake / "cutouts" / name
    root.mkdir(parents=True)
    (root / "cutout_info.json").write_text(
        json.dumps({
            "hats_order": NORDER,
            "n_bands": 1,
            "height": 4,
            "width": 4,
        })
    )


def _make_multi_survey_lake(tmp_path: Path) -> Path:
    lake = tmp_path / "lake"
    _write_catalog_survey(lake, SURVEY_A)
    _write_catalog_survey(lake, SURVEY_B)
    _write_spectra_survey(lake, SURVEY_A)
    _write_spectra_survey(lake, SURVEY_B)
    return lake


class TestDiscoverSurveys:
    def test_catalog_discovery(self, tmp_path: Path) -> None:
        lake = _make_multi_survey_lake(tmp_path)
        assert discover_catalog_ingest_surveys(lake) == [SURVEY_A, SURVEY_B]

    def test_spectra_discovery(self, tmp_path: Path) -> None:
        lake = _make_multi_survey_lake(tmp_path)
        assert discover_spectra_ingest_surveys(lake) == [SURVEY_A, SURVEY_B]

    def test_link_discovery_requires_both(self, tmp_path: Path) -> None:
        lake = _make_multi_survey_lake(tmp_path)
        assert discover_catalog_spectra_link_surveys(lake) == [SURVEY_A, SURVEY_B]
        (lake / "catalogs" / "CAT_ONLY").mkdir()
        (lake / "catalogs" / "CAT_ONLY" / "catalog_info.json").write_text(
            '{"hats_order": 5, "total_rows": 0, "total_columns": 1, '
            '"ra_column": "ra", "dec_column": "dec"}'
        )
        assert "CAT_ONLY" not in discover_catalog_spectra_link_surveys(lake)

    def test_cutout_discovery(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_cutout_survey(lake, "CUT1")
        assert discover_cutout_ingest_surveys(lake) == ["CUT1"]


class TestResolveValidationSurveys:
    def test_rejects_survey_and_all(self) -> None:
        from click.exceptions import ClickException

        with pytest.raises(ClickException, match="not both"):
            resolve_validation_survey_names(
                surveys=("a",),
                validate_all=True,
                discovered=["a"],
                empty_message="empty",
            )

    def test_requires_survey_or_all(self) -> None:
        from click.exceptions import ClickException

        with pytest.raises(ClickException, match="Provide --survey"):
            resolve_validation_survey_names(
                surveys=(),
                validate_all=False,
                discovered=["a"],
                empty_message="empty",
            )


class TestValidateIngestCliAll:
    def test_catalog_ingest_cli_all(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.ingest.validate_catalog_ingest import cli

        if cli is None:
            pytest.skip("click not available")

        lake = _make_multi_survey_lake(tmp_path)
        result = CliRunner().invoke(cli, ["--all", str(lake)])
        assert result.exit_code == 0, result.output
        assert SURVEY_A in result.output
        assert SURVEY_B in result.output

    def test_spectra_ingest_cli_all(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.ingest.validate_spectra_ingest import cli

        if cli is None:
            pytest.skip("click not available")

        lake = _make_multi_survey_lake(tmp_path)
        result = CliRunner().invoke(cli, ["--all", str(lake)])
        assert result.exit_code == 0, result.output
        assert SURVEY_A in result.output
        assert SURVEY_B in result.output
