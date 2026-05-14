"""Tests for ``data_lake.ingest.validate_spectra_ingest``."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from data_lake.ingest.desi_parallel_ingest import (
    TileBatch,
    WorkerResult,
    _atomic_write_json,
    ingest_spectra_parallel,
)
from data_lake.ingest.fits_to_spectra_zarr import _meta_to_bytes
from data_lake.ingest.validate_spectra_ingest import run_validation

N_PIX = 8


def _mini_decoder(path_str: str, norder: int) -> WorkerResult:
    label = Path(path_str).stem
    npix = 100 if label == "fileA" else 200
    sid = 101 if label == "fileA" else 202
    flux = np.full((1, N_PIX), float(sid), dtype=np.float32)
    ivar = np.ones_like(flux)
    mask = np.zeros((1, N_PIX), dtype=np.uint8)
    sids = np.array([sid], dtype=np.int64)
    mb = _meta_to_bytes({
        "z": 0.1, "z_err": 0.01, "snr": 5.0,
        "exptime": 100.0, "R": 3000.0, "instr": "TEST",
    })
    wave = np.linspace(3600.0, 3700.0, N_PIX, dtype=np.float64)
    wcs = {
        "ctype": "WAVE", "crval": float(wave[0]),
        "cdelt": float(wave[1] - wave[0]), "crpix": 1.0,
        "unit": "Angstrom", "air_or_vacuum": "vacuum", "n_pix": N_PIX,
    }
    return WorkerResult(
        path=path_str,
        ok=True,
        batches=[
            TileBatch(
                npix=npix,
                flux=flux,
                ivar=ivar,
                mask=mask,
                source_ids=sids,
                meta_bytes=mb,
            ),
        ],
        wavelength=wave,
        wcs_attrs=wcs,
        n_pix=N_PIX,
        n_spectra=1,
        elapsed_s=0.0,
    )


def _thread_executor(n_workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=n_workers)


@pytest.fixture
def two_fake_coadds(tmp_path: Path) -> list[Path]:
    paths = []
    for label in ("fileA", "fileB"):
        p = tmp_path / f"{label}.fits"
        p.touch()
        paths.append(p)
    return paths


def test_run_validation_ok_after_minimal_ingest(tmp_path: Path, two_fake_coadds) -> None:
    lake = tmp_path / "lake"
    ingest_spectra_parallel(
        file_paths=two_fake_coadds,
        output_root=lake,
        survey_name="syn",
        n_workers=1,
        norder=5,
        checkpoint_path=tmp_path / "ckpt.json",
        failures_log=tmp_path / "fail.jsonl",
        show_progress=False,
        decoder=_mini_decoder,
        executor_factory=_thread_executor,
    )
    rep = run_validation(lake, "syn", max_tiles=10)
    assert rep.ok(strict=True), (rep.errors, rep.warnings)


def test_strict_fails_on_stale_inflight(tmp_path: Path, two_fake_coadds) -> None:
    lake = tmp_path / "lake"
    survey_root = lake / "spectra" / "syn"
    ingest_spectra_parallel(
        file_paths=two_fake_coadds,
        output_root=lake,
        survey_name="syn",
        n_workers=1,
        norder=5,
        checkpoint_path=tmp_path / "ckpt.json",
        failures_log=tmp_path / "fail.jsonl",
        show_progress=False,
        decoder=_mini_decoder,
        executor_factory=_thread_executor,
    )
    inflight = survey_root / ".ingest_inflight.json"
    _atomic_write_json(
        inflight,
        {"commit": {"path": "/nope.fits", "tiles": {"100": 0}, "norder": 5}},
    )
    rep = run_validation(lake, "syn")
    assert not rep.ok(strict=True)
    assert any("inflight" in w.lower() for w in rep.warnings)
