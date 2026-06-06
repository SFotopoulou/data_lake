"""Shared synthetic lake fixtures for IO tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_lake.ingest.fits_to_spectra_zarr import (
    _META_DTYPE,
    SpectrumRecord,
    _meta_to_bytes,
    _open_or_create_spectrum_tile,
    _write_spectrum_info,
)
from data_lake.ingest.fits_to_parquet import assign_healpix, healpix_dir

N_PIX = 32
NORDER = 5
SURVEY = "synthetic"

SOURCES = [
    (101, 30.0, +5.0, 0.10),
    (102, 30.1, +5.1, 0.20),
    (103, 30.2, +5.2, 0.30),
    (201, 210.0, -10.0, 1.10),
    (202, 210.1, -10.1, 1.20),
]


def _make_record(sid: int, ra: float, dec: float, z: float) -> SpectrumRecord:
    flux = np.full(N_PIX, sid, dtype=np.float32)
    ivar = np.full(N_PIX, 1.0 / (sid + 1), dtype=np.float32)
    mask = np.full(N_PIX, sid % 7, dtype=np.uint8)
    return SpectrumRecord(
        source_id=sid,
        ra=ra,
        dec=dec,
        flux=flux,
        ivar=ivar,
        mask=mask,
        wavelength=np.linspace(3600.0, 9800.0, N_PIX),
        meta={
            "z": z, "z_err": 0.001, "snr": 10.0,
            "exptime": 1000.0, "R": 3000.0, "instr": "TEST",
        },
    )


def ingest_synthetic_spectrum_lake(lake_root: Path) -> dict[int, int]:
    records = [_make_record(*row) for row in SOURCES]
    wave = records[0].wavelength
    wcs_attrs = {
        "ctype": "WAVE",
        "crval": float(wave[0]),
        "cdelt": float(wave[1] - wave[0]),
        "crpix": 1.0,
        "unit": "Angstrom",
        "air_or_vacuum": "vacuum",
        "n_pix": N_PIX,
    }
    tile_groups: dict[int, list[SpectrumRecord]] = {}
    for rec in records:
        pix = int(assign_healpix(np.array([rec.ra]), np.array([rec.dec]), NORDER)[0])
        tile_groups.setdefault(pix, []).append(rec)

    survey_root = lake_root / "spectra" / SURVEY
    index_map: dict[int, int] = {}
    for npix, recs in tile_groups.items():
        tile_dir = survey_root / healpix_dir(NORDER, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path = tile_dir / f"Npix={npix}.zarr"
        root = _open_or_create_spectrum_tile(
            tile_path, N_PIX, "shared", np.dtype(np.uint8), wcs_attrs,
        )
        start_idx = root["flux"].shape[0]
        root["flux"].append(np.stack([r.flux for r in recs]))
        root["ivar"].append(np.stack([r.ivar for r in recs]))
        root["mask"].append(np.stack([r.mask for r in recs]))
        from data_lake.ingest.zarr_ids import zarr_join_array

        zarr_join_array(root).append(np.array([r.source_id for r in recs], dtype=np.int64))
        meta_buf = np.frombuffer(
            b"".join(_meta_to_bytes(r.meta) for r in recs),
            dtype="|V" + str(_META_DTYPE.itemsize),
        )
        root["meta"].append(meta_buf)
        if start_idx == 0:
            root["wavelength"][:] = wave.astype(np.float64)
        for i, r in enumerate(recs):
            index_map[r.source_id] = start_idx + i

    _write_spectrum_info(
        survey_root, SURVEY, NORDER, N_PIX, "shared", "uint8", wcs_attrs,
    )
    assert len(tile_groups) >= 2
    return index_map
