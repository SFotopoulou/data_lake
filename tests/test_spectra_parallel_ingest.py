"""Tests for parallel spectrum file-list ingest."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import zarr

from data_lake.ingest.checkpoint_sidecars import paths_from_file_list_file
from data_lake.ingest.fits_to_spectra_zarr import (
    SpectrumDecodeConfig,
    decode_spectrum_file_to_worker_result,
    ingest_spectra_from_fits,
)
from data_lake.ingest.spectra_parallel_ingest import ingest_spectra_files_parallel


def _repo_data(name: str) -> Path:
    p = Path(__file__).resolve().parents[1] / "data" / name
    if not p.exists():
        pytest.skip(f"missing fixture {p}")
    return p


def _thread_executor(n_workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=n_workers)


class TestDecodeSpectrumFile:
    def test_6df_multi_vr_same_tile_decodes_with_pad(self) -> None:
        """6dF files with two VR HDUs of different n_pix on the same sky must stack."""
        fits_path = _repo_data("g1512153-393712.fits")
        cfg = SpectrumDecodeConfig(
            survey_name="SIXDF_DR3",
            output_root="/tmp",
            norder=1,
            fmt="6df",
            ra_col="ra",
            dec_col="dec",
            link_id_col=None,
            wavelength_mode="per_source",
            on_length_mismatch="pad",
            mask_dtype="uint8",
        )
        res = decode_spectrum_file_to_worker_result(str(fits_path), cfg)
        assert res.ok, res.error
        assert res.n_spectra == 2
        assert len(res.batches) == 1
        batch = res.batches[0]
        assert batch.flux.shape == (2, 2899)
        assert batch.wavelength_rows is not None
        assert batch.wavelength_rows.shape == (2, 2899)

    def test_2df_fixture_yields_tile_batches(self) -> None:
        fits_path = _repo_data("389442.fits")
        cfg = SpectrumDecodeConfig(
            survey_name="TEST_2DF",
            output_root="/tmp",
            norder=5,
            fmt="2df",
            ra_col="RA",
            dec_col="DEC",
            link_id_col=None,
            wavelength_mode="shared",
            on_length_mismatch="pad",
            mask_dtype="uint8",
        )
        res = decode_spectrum_file_to_worker_result(str(fits_path), cfg)
        assert res.ok
        assert res.n_spectra == 2
        assert len(res.batches) >= 1
        total_rows = sum(int(b.source_ids.size) for b in res.batches)
        assert total_rows == 2


class TestParallelSpectrumIngest:
    def test_single_2df_file_parallel_matches_sequential_count(self, tmp_path: Path) -> None:
        fits_path = _repo_data("389442.fits")
        list_file = tmp_path / "files.txt"
        list_file.write_text(str(fits_path.resolve()) + "\n")

        lake_par = tmp_path / "lake_par"
        cfg = SpectrumDecodeConfig(
            survey_name="TEST_2DF",
            output_root=str(lake_par),
            norder=5,
            fmt="2df",
            ra_col="RA",
            dec_col="DEC",
            link_id_col=None,
            wavelength_mode="shared",
            on_length_mismatch="pad",
            mask_dtype="uint8",
        )
        result = ingest_spectra_files_parallel(
            paths_from_file_list_file(list_file),
            output_root=lake_par,
            survey_name="TEST_2DF",
            n_workers=2,
            decode_config=cfg,
            checkpoint_path=tmp_path / "ckpt.json",
            show_progress=False,
            executor_factory=_thread_executor,
        )
        assert result["n_files_failed"] == 0
        assert result["n_spectra"] == 2

        lake_seq = tmp_path / "lake_seq"
        seq_map = ingest_spectra_from_fits(
            fits_path,
            lake_seq,
            "TEST_2DF",
            fmt="2df",
            on_length_mismatch="pad",
        )
        assert len(seq_map) == 2

        # Parallel writer should have written flux for both source_ids
        survey_root = lake_par / "spectra" / "TEST_2DF"
        zarr_tiles = list(survey_root.rglob("Npix=*.zarr"))
        assert zarr_tiles
        total_zarr_rows = 0
        for zpath in zarr_tiles:
            root = zarr.open_group(zarr.storage.LocalStore(str(zpath)), mode="r")
            total_zarr_rows += int(root["flux"].shape[0])
        assert total_zarr_rows == 2

    def test_6df_multi_vr_same_tile_parallel_ingest(self, tmp_path: Path) -> None:
        fits_path = _repo_data("g1512153-393712.fits")
        list_file = tmp_path / "files.txt"
        list_file.write_text(str(fits_path.resolve()) + "\n")

        lake = tmp_path / "lake"
        cfg = SpectrumDecodeConfig(
            survey_name="SIXDF_DR3",
            output_root=str(lake),
            norder=1,
            fmt="6df",
            ra_col="ra",
            dec_col="dec",
            link_id_col=None,
            wavelength_mode="per_source",
            on_length_mismatch="pad",
            mask_dtype="uint8",
        )
        result = ingest_spectra_files_parallel(
            paths_from_file_list_file(list_file),
            output_root=lake,
            survey_name="SIXDF_DR3",
            n_workers=2,
            decode_config=cfg,
            checkpoint_path=tmp_path / "ckpt.json",
            show_progress=False,
            executor_factory=_thread_executor,
        )
        assert result["n_files_failed"] == 0
        assert result["n_spectra"] == 2

    def test_checkpoint_resume_skips_completed(self, tmp_path: Path) -> None:
        fits_path = _repo_data("389442.fits")
        list_file = tmp_path / "files.txt"
        list_file.write_text(str(fits_path.resolve()) + "\n")
        ckpt = tmp_path / "ckpt.json"
        ckpt.write_text(
            json.dumps({"completed": [str(fits_path.resolve())]}),
            encoding="utf-8",
        )

        lake = tmp_path / "lake"
        cfg = SpectrumDecodeConfig(
            survey_name="TEST_2DF",
            output_root=str(lake),
            norder=5,
            fmt="2df",
            ra_col="RA",
            dec_col="DEC",
            link_id_col=None,
            wavelength_mode="shared",
            on_length_mismatch="pad",
            mask_dtype="uint8",
        )
        result = ingest_spectra_files_parallel(
            paths_from_file_list_file(list_file),
            output_root=lake,
            survey_name="TEST_2DF",
            n_workers=2,
            decode_config=cfg,
            checkpoint_path=ckpt,
            show_progress=False,
            executor_factory=_thread_executor,
        )
        assert result["n_files_processed"] == 0
        assert result["n_files_skipped"] == 1
