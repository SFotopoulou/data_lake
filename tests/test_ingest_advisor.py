"""Tests for ingest command / Slurm advisor."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_lake.ingest_advisor import recommend_ingest


def test_catalog_batch_many_files() -> None:
    rec = recommend_ingest("catalog", "GAIA_DR3", n_files=5000, file_list="gaia.txt")
    assert "dl-ingest-catalog-batch" in rec.command
    assert rec.slurm_script is not None
    assert "SBATCH" in rec.slurm_script
    assert any("parallel" in r.lower() for r in rec.rationale)


def test_catalog_live_defer_finalize() -> None:
    rec = recommend_ingest("catalog", "EUCLID", n_files=1, lifecycle="live")
    assert "--defer-finalize" in rec.command
    assert "--lifecycle live" in rec.command


def test_spectra_2df_matches_existing_slurm() -> None:
    rec = recommend_ingest(
        "spectra", "2DFGRS_DR3", fmt="2df", n_files=100, file_list="2df.txt",
    )
    assert "dl-ingest-spectra-from-list" in rec.command
    assert "--fmt 2df" in rec.command
    assert rec.slurm_script_path == "scripts/slurm_ingest_2df_spectra.sh"
    assert rec.slurm_script is not None
    assert "dl-2df-spectra" in rec.slurm_script


def test_spectra_6df_matches_existing_slurm() -> None:
    rec = recommend_ingest("spectra", "SIXDF_DR3", fmt="6df", n_files=50)
    assert rec.slurm_script_path == "scripts/slurm_ingest_6df_spectra.sh"


def test_spectra_desi_batch() -> None:
    rec = recommend_ingest("spectra", "DESI_DR1", fmt="desi_coadd", n_files=1000)
    assert "dl-ingest-spectra-batch-desi-coadds" in rec.command


def test_cutout_from_list() -> None:
    rec = recommend_ingest("cutout", "DESI_DR1", n_files=20, file_list="cuts.txt")
    assert "dl-ingest-cutouts-from-list" in rec.command
    assert "--on-duplicate skip" in rec.command


def test_catalog_streaming_large_file() -> None:
    rec = recommend_ingest(
        "catalog", "DESI", n_files=1, total_size_gb=12.0,
    )
    assert "--streaming" in rec.command


def test_packed_vector_warning(tmp_path: Path) -> None:
    pytest.importorskip("astropy")
    from astropy.io import fits
    import numpy as np

    # Minimal packed-vector style: NAXIS2=1 with TDIM on a column
    col = fits.Column(name="FLUX", format="100E", array=np.zeros((1, 100), dtype=np.float32))
    hdu = fits.BinTableHDU.from_columns([col])
    hdu.header["NAXIS2"] = 1
    path = tmp_path / "packed.fits"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)

    rec = recommend_ingest("catalog", "GALEX", sample_file=path, n_files=10)
    assert any("packed-vector" in w for w in rec.warnings)
