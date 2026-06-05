"""
Phase 0: characterize DESI coadd FITS for per-target exposure multiplicity.

Answers whether ingest needs ``desispec.coaddition.coadd()`` before
``coadd_cameras()`` — i.e. whether ``read_spectra`` yields multiple fibermap
rows per ``TARGETID`` (per-exposure spectra still present in the file).

Quick mode reads only ``FIBERMAP`` / ``EXP_FIBERMAP`` HDUs (fast for 10k+ files).
Full mode calls ``desispec.io.read_spectra`` with the same ``skip_hdus`` as ingest.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

Mode = Literal["quick", "desispec"]


@dataclass(frozen=True, slots=True)
class CoaddExposureReport:
    """Exposure-multiplicity summary for one coadd FITS file."""

    path: str
    mode: Mode
    n_fibermap_rows: int
    n_unique_targetid: int
    max_rows_per_targetid: int
    n_targets_multi_row: int
    rows_per_targetid_histogram: dict[int, int]
    n_exp_fibermap_rows: int | None
    exp_fibermap_to_fibermap_ratio: float | None
    needs_exposure_coadd: bool
    targetid_col: str = "TARGETID"

    @property
    def is_pipeline_single_row(self) -> bool:
        """True when every TARGETID appears exactly once in FIBERMAP."""
        return self.max_rows_per_targetid <= 1


def _targetid_counts(targetids: np.ndarray) -> Counter[int]:
    """Return occurrence counts per TARGETID (Python ints for JSON/reporting)."""
    flat = np.asarray(targetids).ravel()
    if flat.size == 0:
        return Counter()
    return Counter(int(t) for t in flat)


def characterize_targetid_multiplicity(
    targetids: np.ndarray,
    *,
    path: str = "",
    mode: Mode = "quick",
    n_exp_fibermap_rows: int | None = None,
    n_fibermap_rows: int | None = None,
    targetid_col: str = "TARGETID",
) -> CoaddExposureReport:
    """Build a report from a TARGETID vector (one value per fibermap row)."""
    counts = _targetid_counts(targetids)
    n_fibermap = int(n_fibermap_rows if n_fibermap_rows is not None else len(targetids))
    n_unique = len(counts)
    max_per = max(counts.values()) if counts else 0
    hist = dict(sorted(Counter(counts.values()).items()))
    n_multi = sum(1 for c in counts.values() if c > 1)
    ratio = None
    if n_exp_fibermap_rows is not None and n_fibermap > 0:
        ratio = n_exp_fibermap_rows / n_fibermap

    return CoaddExposureReport(
        path=path,
        mode=mode,
        n_fibermap_rows=n_fibermap,
        n_unique_targetid=n_unique,
        max_rows_per_targetid=max_per,
        n_targets_multi_row=n_multi,
        rows_per_targetid_histogram=hist,
        n_exp_fibermap_rows=n_exp_fibermap_rows,
        exp_fibermap_to_fibermap_ratio=ratio,
        needs_exposure_coadd=max_per > 1,
        targetid_col=targetid_col,
    )


def _read_fibermap_targetids(path: Path) -> tuple[np.ndarray, int | None, str]:
    """Read TARGETID from FIBERMAP and row count from EXP_FIBERMAP if present."""
    from data_lake.io.fits_read import open_fits

    with open_fits(path) as hdul:
        names = {hdu.name for hdu in hdul}
        if "FIBERMAP" not in names:
            raise KeyError(f"{path}: no FIBERMAP extension (found {sorted(names)[:12]}...)")

        fmap = hdul["FIBERMAP"].data
        n_fibermap = len(fmap)
        for col in ("TARGETID", "TARGET_ID"):
            if col in fmap.dtype.names:
                return np.asarray(fmap[col]), _exp_fibermap_nrow(hdul), col
        raise KeyError(
            f"{path}: FIBERMAP has no TARGETID column; "
            f"columns={list(fmap.dtype.names)[:20]}"
        )


def _exp_fibermap_nrow(hdul) -> int | None:
    if "EXP_FIBERMAP" not in {h.name for h in hdul}:
        return None
    return len(hdul["EXP_FIBERMAP"].data)


def characterize_coadd_fits_quick(path: str | Path) -> CoaddExposureReport:
    """Fast characterization: FIBERMAP + EXP_FIBERMAP HDUs only."""
    path = Path(path).expanduser().resolve()
    targetids, n_exp, col = _read_fibermap_targetids(path)
    return characterize_targetid_multiplicity(
        targetids,
        path=str(path),
        mode="quick",
        n_exp_fibermap_rows=n_exp,
        n_fibermap_rows=len(targetids),
        targetid_col=col,
    )


def characterize_coadd_fits_desispec(path: str | Path) -> CoaddExposureReport:
    """Full characterization via ``read_spectra`` (same row set as ingest decode)."""
    from data_lake.ingest.fits_to_spectra_zarr import (
        _desi_read_spectra_skip_hdus,
        _import_desispec,
    )

    path = Path(path).expanduser().resolve()
    desispec = _import_desispec()
    spectra = desispec.io.read_spectra(
        str(path),
        single=True,
        skip_hdus=_desi_read_spectra_skip_hdus(with_resolution=False),
    )
    fmap = spectra.fibermap
    if "TARGETID" not in fmap.colnames:
        raise KeyError(f"{path}: fibermap missing TARGETID; cols={fmap.colnames[:20]}")
    n_exp = len(spectra.exp_fibermap) if getattr(spectra, "exp_fibermap", None) is not None else None
    return characterize_targetid_multiplicity(
        np.asarray(fmap["TARGETID"]),
        path=str(path),
        mode="desispec",
        n_exp_fibermap_rows=n_exp,
        n_fibermap_rows=len(fmap),
    )


def characterize_coadd_fits(
    path: str | Path,
    *,
    mode: Mode = "quick",
) -> CoaddExposureReport:
    """Characterize one coadd FITS file."""
    if mode == "quick":
        return characterize_coadd_fits_quick(path)
    if mode == "desispec":
        return characterize_coadd_fits_desispec(path)
    raise ValueError(f"unknown mode {mode!r}")


@dataclass
class CoaddExposureBatchSummary:
    """Aggregate over many coadd files."""

    n_files: int = 0
    n_files_needing_exposure_coadd: int = 0
    n_files_single_row_per_target: int = 0
    max_rows_per_targetid_overall: int = 0
    pooled_histogram: dict[int, int] = field(default_factory=dict)
    files_needing_coadd: list[str] = field(default_factory=list)

    def absorb(self, report: CoaddExposureReport) -> None:
        self.n_files += 1
        if report.needs_exposure_coadd:
            self.n_files_needing_exposure_coadd += 1
            if len(self.files_needing_coadd) < 50:
                self.files_needing_coadd.append(report.path)
        else:
            self.n_files_single_row_per_target += 1
        self.max_rows_per_targetid_overall = max(
            self.max_rows_per_targetid_overall,
            report.max_rows_per_targetid,
        )
        for k, v in report.rows_per_targetid_histogram.items():
            self.pooled_histogram[k] = self.pooled_histogram.get(k, 0) + v


def characterize_coadd_paths(
    paths: Iterable[str | Path],
    *,
    mode: Mode = "quick",
) -> tuple[list[CoaddExposureReport], CoaddExposureBatchSummary]:
    """Characterize many files; skip missing paths with a warning in reports."""
    reports: list[CoaddExposureReport] = []
    summary = CoaddExposureBatchSummary()
    for p in paths:
        path = Path(p).expanduser()
        if not path.is_file():
            continue
        rep = characterize_coadd_fits(path, mode=mode)
        reports.append(rep)
        summary.absorb(rep)
    return reports, summary
