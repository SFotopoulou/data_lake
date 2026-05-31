"""Tests for lake registry and master association metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.table import Table

from concurrent.futures import ThreadPoolExecutor

from data_lake.ingest.desi_parallel_ingest import TileBatch, WorkerResult, ingest_spectra_parallel
from data_lake.ingest.fits_to_parquet import ingest_catalog
from data_lake.ingest.fits_to_spectra_zarr import _meta_to_bytes
from data_lake.lake_registry import (
    REGISTRY_FILENAME,
    build_lake_registry_table,
    guess_master_meta,
    load_lake_registry,
    load_master_meta,
    master_meta_path,
    refresh_lake_registry,
    registry_path,
    write_master_meta,
)


def _ingest_mini_catalog(
    tmp_path: Path,
    survey: str,
    *,
    ra: float = 10.0,
    dec: float = 20.0,
    id_offset: int = 0,
) -> None:
    n = 6
    tbl = Table({
        "TARGETID": np.arange(id_offset, id_offset + n, dtype=np.int64),
        "TARGET_RA": np.full(n, ra),
        "TARGET_DEC": np.full(n, dec),
        "Z": np.linspace(0.1, 0.3, n),
        "MAG_G": np.linspace(18.0, 19.0, n),
    })
    fits = tmp_path / f"{survey}.fits"
    tbl.write(fits, overwrite=True)
    lake = tmp_path / "lake"
    ingest_catalog(
        source_path=fits,
        output_root=lake,
        survey_name=survey,
        ra_col="TARGET_RA",
        dec_col="TARGET_DEC",
        link_id_col="TARGETID",
        norder=5,
        tile_mode="overwrite",
    )


def test_refresh_lake_registry(tmp_path: Path) -> None:
    _ingest_mini_catalog(tmp_path, "SURV_A", id_offset=0)
    _ingest_mini_catalog(tmp_path, "SURV_B", id_offset=100, ra=50.0, dec=60.0)
    lake = tmp_path / "lake"

    out = refresh_lake_registry(lake)
    assert out == registry_path(lake)
    table = load_lake_registry(lake)
    names = set(table.column("survey").to_pylist())
    assert "SURV_A" in names
    assert "SURV_B" in names


def test_master_meta_guess_and_write(tmp_path: Path) -> None:
    _ingest_mini_catalog(tmp_path, "DESI_DR1", id_offset=0)
    _ingest_mini_catalog(tmp_path, "EUCLID_DR1", id_offset=1000, ra=50.0)
    lake = tmp_path / "lake"

    master = pa.table({
        "desi_targetid": pa.array([1, 2], type=pa.int64()),
        "euclid_source_id": pa.array([1000, 1001], type=pa.int64()),
        "sep_arcsec": pa.array([0.1, 0.2], type=pa.float32()),
    })
    master_path = lake / "associations" / "master.parquet"
    master_path.parent.mkdir(parents=True)
    pq.write_table(master, str(master_path))

    meta = guess_master_meta(master_path, lake)
    assert len(meta["partners"]) >= 1
    surveys = {p["survey"] for p in meta["partners"]}
    assert "DESI_DR1" in surveys or "EUCLID_DR1" in surveys

    write_master_meta(master_path, meta)
    assert master_meta_path(master_path).is_file()
    loaded = load_master_meta(master_path, lake, allow_guess=False)
    assert loaded["partners"][0]["master_column"]


def test_build_registry_table_empty_lake(tmp_path: Path) -> None:
    lake = tmp_path / "empty_lake"
    (lake / "catalogs").mkdir(parents=True)
    table = build_lake_registry_table(lake)
    assert table.num_rows == 0


def _mini_spectrum_decoder(path_str: str, norder: int) -> WorkerResult:
    label = Path(path_str).stem
    npix = 100 if label == "fileA" else 200
    sid = 101 if label == "fileA" else 202
    n_pix = 8
    flux = np.full((1, n_pix), float(sid), dtype=np.float32)
    ivar = np.ones_like(flux)
    mask = np.zeros((1, n_pix), dtype=np.uint8)
    sids = np.array([sid], dtype=np.int64)
    mb = _meta_to_bytes({
        "z": 0.1, "z_err": 0.01, "snr": 5.0,
        "exptime": 100.0, "R": 3000.0, "instr": "TEST",
    })
    wave = np.linspace(3600.0, 3700.0, n_pix, dtype=np.float64)
    wcs = {
        "ctype": "WAVE", "crval": float(wave[0]),
        "cdelt": float(wave[1] - wave[0]), "crpix": 1.0,
        "unit": "Angstrom", "air_or_vacuum": "vacuum", "n_pix": n_pix,
    }
    return WorkerResult(
        path=path_str,
        ok=True,
        batches=[TileBatch(npix=npix, flux=flux, ivar=ivar, mask=mask, source_ids=sids, meta_bytes=mb)],
        wavelength=wave,
        wcs_attrs=wcs,
        n_pix=n_pix,
        n_spectra=1,
        elapsed_s=0.0,
    )


def test_registry_spectra_total_rows(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    paths = []
    for label in ("fileA", "fileB"):
        p = tmp_path / f"{label}.fits"
        p.touch()
        paths.append(p)

    ingest_spectra_parallel(
        file_paths=paths,
        output_root=lake,
        survey_name="SPEC_SURV",
        n_workers=1,
        norder=5,
        checkpoint_path=tmp_path / "ckpt.json",
        failures_log=tmp_path / "fail.jsonl",
        show_progress=False,
        decoder=_mini_spectrum_decoder,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )

    refresh_lake_registry(lake)
    table = load_lake_registry(lake)
    spec_rows = [
        r
        for r in table.to_pylist()
        if r["survey"] == "SPEC_SURV" and r["modality"] == "spectra"
    ]
    assert len(spec_rows) == 1
    assert spec_rows[0]["total_rows"] == 2
