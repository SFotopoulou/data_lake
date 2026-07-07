"""
BandpassRegistry — per-band effective wavelength, FWHM, and optional
transmission-curve loading (ECSV files in data_lake/homogenize/bandpasses/).

Resolution order (same pattern as survey_registry):
  1. <lake_root>/shared/registry/bandpasses/<curve_file>   (lake override)
  2. data_lake/homogenize/bandpasses/<curve_file>           (bundled default)

Usage
-----
>>> from data_lake.homogenize.bandpass import BandpassRegistry
>>> bp = BandpassRegistry(lake_root="/shared/como")
>>> bp.lambda_eff_um("phot_ab_w1")
3.368
>>> wave, throughput = bp.load_curve("phot_ab_w1")  # or None if unavailable
>>> bp.available_bands()
['phot_ab_bp', 'phot_ab_g', ...]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_PKG_BANDPASS_JSON = Path(__file__).resolve().parent / "bandpass.json"
_PKG_BANDPASSES_DIR = Path(__file__).resolve().parent / "bandpasses"

PHOT_AB_PREFIX = "phot_ab_"


class BandpassRegistry:
    """
    Resolves per-band effective wavelength metadata and optional
    transmission-curve arrays from ECSV files.

    Parameters
    ----------
    lake_root:
        Data lake root. When provided, checks
        ``<lake_root>/shared/registry/bandpasses/`` for lake-local overrides
        before falling back to the bundled ``data_lake/homogenize/bandpasses/``
        directory. Pass ``None`` to use only bundled data.
    """

    def __init__(self, lake_root: Path | str | None = None) -> None:
        self._lake_root = Path(lake_root) if lake_root is not None else None
        self._meta: dict[str, dict[str, Any]] = self._load_metadata()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        """Load bandpass.json (bundled only; lake overrides are per-curve, not per-json)."""
        if not _PKG_BANDPASS_JSON.is_file():
            log.warning("bandpass.json not found at %s", _PKG_BANDPASS_JSON)
            return {}
        with open(_PKG_BANDPASS_JSON, encoding="utf-8") as fh:
            data = json.load(fh)
        return dict(data.get("bands", {}))

    def _lake_bandpasses_dir(self) -> Path | None:
        if self._lake_root is None:
            return None
        d = self._lake_root / "shared" / "registry" / "bandpasses"
        return d if d.is_dir() else None

    def _resolve_curve_path(self, filename: str) -> Path | None:
        """Lake override → bundled default; None when neither exists."""
        lake_dir = self._lake_bandpasses_dir()
        if lake_dir is not None:
            candidate = lake_dir / filename
            if candidate.is_file():
                return candidate
        bundled = _PKG_BANDPASSES_DIR / filename
        return bundled if bundled.is_file() else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available_bands(self) -> list[str]:
        """Return sorted list of all band names with metadata."""
        return sorted(self._meta.keys())

    def band_meta(self, band: str) -> dict[str, Any] | None:
        """Return the metadata dict for *band*, or ``None`` if unknown."""
        return self._meta.get(band)

    def lambda_eff_um(self, band: str) -> float | None:
        """Effective wavelength in micron for *band*, or ``None`` if unknown."""
        meta = self._meta.get(band)
        if meta is None:
            return None
        return float(meta["lambda_eff_um"])

    def fwhm_um(self, band: str) -> float | None:
        """FWHM in micron for *band*, or ``None`` if unknown."""
        meta = self._meta.get(band)
        if meta is None:
            return None
        val = meta.get("fwhm_um")
        return float(val) if val is not None else None

    def curve_filename(self, band: str) -> str | None:
        """Registered ECSV curve filename for *band*, or ``None``."""
        meta = self._meta.get(band)
        if meta is None:
            return None
        return meta.get("curve")

    def load_curve(
        self, band: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Load the transmission curve for *band*.

        Returns
        -------
        (wavelength_angstrom, throughput) as float64 numpy arrays, or
        ``None`` when no curve file is registered or the file is missing.
        """
        fname = self.curve_filename(band)
        if fname is None:
            return None
        path = self._resolve_curve_path(fname)
        if path is None:
            log.debug("Bandpass curve file %r not found for band %r", fname, band)
            return None
        try:
            from astropy.io import ascii as astropy_ascii

            tbl = astropy_ascii.read(str(path), format="ecsv")
            wave = np.asarray(tbl["wavelength"], dtype=np.float64)
            throughput = np.asarray(tbl["throughput"], dtype=np.float64)
            return wave, throughput
        except Exception as exc:
            log.warning("Failed to load curve %s: %s", path, exc)
            return None

    def all_metadata(self) -> dict[str, dict[str, Any]]:
        """Return a copy of the full metadata dict keyed by band name."""
        return dict(self._meta)

    def __repr__(self) -> str:
        lake = str(self._lake_root) if self._lake_root else "None"
        return f"BandpassRegistry(lake_root={lake!r}, n_bands={len(self._meta)})"
