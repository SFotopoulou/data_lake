"""Per-survey flux calibration for spectrum subset exports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FluxCalibration:
    """Constant multiplicative flux calibration applied at extract time."""

    flux_scale: float
    native_flux_unit: str | None = None
    output_flux_unit: str | None = None
    reference: str | None = None
    survey: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if v is not None}
        return out


def _calibration_from_spectra_block(spectra: dict[str, Any], survey: str) -> FluxCalibration | None:
    cal = spectra.get("flux_calibration")
    if isinstance(cal, dict) and cal.get("flux_scale") is not None:
        return FluxCalibration(
            survey=survey,
            flux_scale=float(cal["flux_scale"]),
            native_flux_unit=cal.get("native_flux_unit"),
            output_flux_unit=cal.get("output_flux_unit"),
            reference=cal.get("reference"),
        )
    spec = spectra.get("spec_observed_v1")
    if isinstance(spec, dict):
        transform = spec.get("transform") or {}
        if transform.get("type") == "flux_scale" and transform.get("factor") is not None:
            return FluxCalibration(
                survey=survey,
                flux_scale=float(transform["factor"]),
                native_flux_unit=spec.get("native_flux_unit"),
                output_flux_unit=spec.get("output_flux_unit"),
                reference=transform.get("reference"),
            )
    return None


def load_spectrum_flux_calibration(
    lake_root: Path | str | None,
    survey: str,
) -> FluxCalibration | None:
    """Load bundled or lake-local calibration from homogenize survey JSON."""
    from data_lake.homogenize.survey_registry import load_survey_homogenize

    doc = load_survey_homogenize(lake_root, survey)
    if doc is None:
        return None
    spectra = doc.get("spectra")
    if not isinstance(spectra, dict):
        return None
    return _calibration_from_spectra_block(spectra, survey)


def resolve_flux_calibration(
    lake_root: Path | str | None,
    survey: str,
    *,
    flux_scale: float | None = None,
    apply_survey_calibration: bool = False,
) -> FluxCalibration | None:
    """Resolve extract-time flux calibration (explicit scale wins over registry)."""
    if flux_scale is not None:
        if flux_scale <= 0:
            raise ValueError(f"flux_scale must be positive, got {flux_scale}")
        return FluxCalibration(survey=survey, flux_scale=float(flux_scale))
    if not apply_survey_calibration:
        return None
    cal = load_spectrum_flux_calibration(lake_root, survey)
    if cal is None:
        raise LookupError(
            f"No spectra.flux_calibration for survey {survey!r}. "
            f"Add homogenize/surveys/{survey}.json or pass --flux-scale."
        )
    return cal


def apply_flux_scale(
    flux: np.ndarray,
    ivar: np.ndarray,
    factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale flux and ivar arrays (ivar /= factor²)."""
    if factor == 1.0:
        return flux, ivar
    flux_out = np.asarray(flux, dtype=np.float32) * factor
    ivar_out = np.asarray(ivar, dtype=np.float32)
    safe = ivar_out > 0
    scaled_ivar = np.zeros_like(ivar_out)
    scaled_ivar[safe] = ivar_out[safe] / (factor * factor)
    return flux_out, scaled_ivar


def calibration_sidecar_path(output: Path | str) -> Path:
    """Sidecar path beside a subset export (``*.calibration.json``)."""
    out = Path(output)
    if out.suffix.lower() == ".zarr":
        return out.parent / f"{out.stem}.calibration.json"
    if out.suffix == "" or out.is_dir():
        return out / "calibration.json"
    return out.with_suffix(".calibration.json")


def write_calibration_sidecar(output: Path | str, calibration: FluxCalibration) -> Path:
    """Write extract provenance for flux calibration."""
    path = calibration_sidecar_path(output)
    path.write_text(json.dumps(calibration.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
