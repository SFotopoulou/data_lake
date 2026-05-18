"""Phase 0 tests: DESI coadd exposure-multiplicity characterization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from data_lake.ingest.desi_coadd_characterize import (
    CoaddExposureBatchSummary,
    characterize_coadd_fits_quick,
    characterize_targetid_multiplicity,
)


def test_single_row_per_target_pipeline_like() -> None:
    """Pipeline coadd: one fibermap row per TARGETID."""
    tids = np.array([100, 200, 300], dtype=np.int64)
    rep = characterize_targetid_multiplicity(tids)
    assert rep.n_fibermap_rows == 3
    assert rep.n_unique_targetid == 3
    assert rep.max_rows_per_targetid == 1
    assert rep.n_targets_multi_row == 0
    assert not rep.needs_exposure_coadd
    assert rep.is_pipeline_single_row
    assert rep.rows_per_targetid_histogram == {1: 3}


def test_multi_row_per_target_needs_exposure_coadd() -> None:
    """Per-exposure rows: two spectra for target 100, one for 200."""
    tids = np.array([100, 100, 200], dtype=np.int64)
    rep = characterize_targetid_multiplicity(tids, n_exp_fibermap_rows=6, n_fibermap_rows=3)
    assert rep.max_rows_per_targetid == 2
    assert rep.n_targets_multi_row == 1
    assert rep.needs_exposure_coadd
    assert not rep.is_pipeline_single_row
    assert rep.rows_per_targetid_histogram == {1: 1, 2: 1}
    assert rep.exp_fibermap_to_fibermap_ratio == 2.0


def test_empty_fibermap() -> None:
    rep = characterize_targetid_multiplicity(np.array([], dtype=np.int64))
    assert rep.n_fibermap_rows == 0
    assert rep.max_rows_per_targetid == 0
    assert not rep.needs_exposure_coadd


def test_batch_summary_counts() -> None:
    summary = CoaddExposureBatchSummary()
    summary.absorb(
        characterize_targetid_multiplicity(np.array([1, 2], dtype=np.int64), path="/a.fits")
    )
    summary.absorb(
        characterize_targetid_multiplicity(np.array([1, 1], dtype=np.int64), path="/b.fits")
    )
    assert summary.n_files == 2
    assert summary.n_files_single_row_per_target == 1
    assert summary.n_files_needing_exposure_coadd == 1
    assert summary.max_rows_per_targetid_overall == 2
    assert "/b.fits" in summary.files_needing_coadd


def _write_minimal_coadd_fits(
    path: Path,
    targetids: list[int],
    *,
    with_exp_fibermap: bool = False,
) -> None:
    """Minimal DESI-like coadd FITS with FIBERMAP (+ optional EXP_FIBERMAP)."""
    n = len(targetids)
    fmap = Table()
    fmap["TARGETID"] = np.array(targetids, dtype=np.int64)
    fmap["FIBER"] = np.arange(n, dtype=np.int32)
    hdus = [fits.PrimaryHDU(), fits.BinTableHDU(fmap, name="FIBERMAP")]
    if with_exp_fibermap:
        exp = Table()
        exp["TARGETID"] = np.repeat(np.array(targetids, dtype=np.int64), 2)
        exp["EXPID"] = np.arange(len(exp["TARGETID"]), dtype=np.int32)
        hdus.append(fits.BinTableHDU(exp, name="EXP_FIBERMAP"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def test_characterize_coadd_fits_quick_single_row(tmp_path: Path) -> None:
    path = tmp_path / "coadd.fits"
    _write_minimal_coadd_fits(path, [111, 222, 333], with_exp_fibermap=True)
    rep = characterize_coadd_fits_quick(path)
    assert rep.mode == "quick"
    assert not rep.needs_exposure_coadd
    assert rep.n_exp_fibermap_rows == 6  # 3 targets * 2 exp rows each


def test_characterize_coadd_fits_quick_multi_row(tmp_path: Path) -> None:
    path = tmp_path / "multi.fits"
    _write_minimal_coadd_fits(path, [10, 10, 20], with_exp_fibermap=False)
    rep = characterize_coadd_fits_quick(path)
    assert rep.needs_exposure_coadd
    assert rep.max_rows_per_targetid == 2
    assert rep.n_targets_multi_row == 1


def test_characterize_coadd_fits_quick_missing_fibermap(tmp_path: Path) -> None:
    path = tmp_path / "bad.fits"
    fits.PrimaryHDU().writeto(path, overwrite=True)
    with pytest.raises(KeyError, match="FIBERMAP"):
        characterize_coadd_fits_quick(path)


@pytest.mark.skipif(
    not Path("/data/DESI/fits").is_dir(),
    reason="DESI coadd tree not available on this host",
)
def test_characterize_real_coadd_sample() -> None:
    """Optional integration: one real coadd from /data/DESI/fits if mounted."""
    root = Path("/data/DESI/fits")
    files = sorted(root.glob("coadd-*.fits"))
    if not files:
        pytest.skip("no coadd-*.fits under /data/DESI/fits")
    rep = characterize_coadd_fits_quick(files[0])
    assert rep.n_fibermap_rows > 0
    assert rep.n_unique_targetid > 0
    # Report only — do not assert needs_exposure_coadd; that's Phase 0 discovery
