"""Tests for ingest command / Slurm advisor."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_lake.ingest_advisor import (
    estimate_files_per_tile,
    recommend_ingest,
    tile_area_deg2,
)


def test_catalog_batch_many_files() -> None:
    rec = recommend_ingest("catalog", "GAIA_DR3", n_files=5000, file_list="gaia.txt")
    assert "dl-ingest-catalog-batch" in rec.command
    assert rec.slurm_script is not None
    assert "SBATCH" in rec.slurm_script
    assert any("parallel" in r.lower() for r in rec.rationale)
    assert any("layout unknown" in w.lower() for w in rec.warnings)
    assert rec.extra_slurm_scripts == {}


def test_tile_area_matches_docs_table() -> None:
    assert tile_area_deg2(5) == pytest.approx(3.36, abs=0.01)
    assert tile_area_deg2(7) == pytest.approx(0.21, abs=0.001)


def test_files_per_tile_legacy_bricks() -> None:
    brick = 0.25 * 0.25
    assert estimate_files_per_tile(4, brick) == pytest.approx(215, rel=0.01)
    assert estimate_files_per_tile(7, brick) == pytest.approx(3.36, rel=0.01)
    assert estimate_files_per_tile(4, 100.0) == 1.0
    with pytest.raises(ValueError):
        estimate_files_per_tile(5, 0.0)


@pytest.mark.parametrize(
    ("kwargs", "shard"),
    [
        ({"layout": "spatial"}, True),
        ({"layout": "unsorted"}, True),
        ({"layout": "partitioned"}, False),
        ({"file_area_deg2": 0.0625, "norder": 4}, True),
        ({"file_area_deg2": 0.0625, "norder": 8}, False),
        # Area estimate wins over a contradicting layout hint.
        ({"layout": "partitioned", "file_area_deg2": 0.0625, "norder": 4}, True),
    ],
)
def test_catalog_layout_selects_helper(kwargs, shard) -> None:
    rec = recommend_ingest(
        "catalog", "LEGACY_DR11", n_files=93_000, file_list="bricks.txt", **kwargs,
    )
    assert ("dl-ingest-from-shard" in rec.command) is shard
    assert ("dl-ingest-catalog-batch" in rec.command) is not shard
    assert not any("layout unknown" in w.lower() for w in rec.warnings)


def test_shard_recommendation_commands_and_scripts() -> None:
    rec = recommend_ingest(
        "catalog", "LEGACY_DR11", n_files=93_000, file_list="bricks.txt",
        layout="spatial", norder=7,
    )
    assert "--phase map" in rec.command and "--phase reduce" in rec.command
    assert "--norder 7" in rec.command
    assert "--file-list bricks.txt" in rec.command
    map_script = rec.slurm_script
    reduce_script = rec.extra_slurm_scripts["reduce"]
    assert "--array=0-$((N-1))%64" in map_script
    assert '--file-list "${CHUNK}"' in map_script
    assert "LINK_ID_COL:?" in map_script
    assert "<" not in map_script.split("set -euo pipefail")[1].replace("<<", "")
    assert "--dependency=afterok:${MAP_JID}" in reduce_script
    assert "--phase reduce" in reduce_script and "--failures-log" in reduce_script
    assert "dl-finalize-catalog" not in reduce_script
    assert any("~47 chunk" in r for r in rec.rationale)


def test_shard_not_used_for_live_or_single_or_packed(tmp_path: Path) -> None:
    live = recommend_ingest(
        "catalog", "EUCLID", n_files=500, layout="spatial", lifecycle="live",
    )
    assert "dl-ingest-catalog-batch" in live.command
    assert "--defer-finalize" in live.command

    single = recommend_ingest("catalog", "ONE", n_files=1, layout="spatial")
    assert "dl-ingest-catalog " in single.command


def test_invalid_layout_rejected() -> None:
    with pytest.raises(ValueError, match="layout"):
        recommend_ingest("catalog", "X", n_files=10, layout="bricks")  # type: ignore[arg-type]


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
