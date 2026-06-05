"""
crossmatch_matchers – sky nearest-neighbour matching backends for dl-crossmatch.

Backends
--------
astropy (default)
    ``SkyCoord.match_to_catalog_sky`` — portable, no GPU required.
rapids
    cuML ``NearestNeighbors`` on unit-sphere XYZ — optional ``[rapids]`` extra.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

MatchBackend = Literal["astropy", "rapids"]


def rapids_available() -> bool:
    """Return True when cuML and CuPy are importable."""
    try:
        import cuml  # noqa: F401
        import cupy  # noqa: F401
        return True
    except ImportError:
        return False


def validate_match_backend(backend: MatchBackend) -> None:
    if backend not in ("astropy", "rapids"):
        raise ValueError(f"match_backend must be 'astropy' or 'rapids', got {backend!r}")
    if backend == "rapids" and not rapids_available():
        raise ImportError(
            "match_backend='rapids' requires the optional [rapids] extra "
            "(cuML + CuPy). Install with: uv sync --extra rapids"
        )


def warn_rapids_worker_config(*, n_workers: int, gpu_id: int) -> None:
    """Log when multi-process + single-GPU RAPIDS is likely to contend."""
    if n_workers <= 1:
        return
    log.warning(
        "match_backend=rapids with --n-workers=%d on gpu_id=%d: multiple "
        "processes may contend for one GPU. Prefer --n-workers 1 unless each "
        "worker has a dedicated GPU.",
        n_workers,
        gpu_id,
    )


def _radec_to_unit_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg.astype(np.float64, copy=False))
    dec = np.deg2rad(dec_deg.astype(np.float64, copy=False))
    cos_dec = np.cos(dec)
    return np.column_stack(
        (cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec))
    )


def _angular_sep_deg_from_unit_xyz(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    idx_b: np.ndarray,
) -> np.ndarray:
    dots = np.sum(xyz_a * xyz_b[idx_b], axis=1)
    sep_rad = np.arccos(np.clip(dots, -1.0, 1.0))
    return np.degrees(sep_rad)


def _match_sky_astropy(
    ra_a: np.ndarray,
    dec_a: np.ndarray,
    ids_a: np.ndarray,
    ra_b: np.ndarray,
    dec_b: np.ndarray,
    ids_b: np.ndarray,
    radius_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coords_a = SkyCoord(ra=ra_a * u.deg, dec=dec_a * u.deg)
    coords_b = SkyCoord(ra=ra_b * u.deg, dec=dec_b * u.deg)

    idx_b, sep2d, _ = coords_a.match_to_catalog_sky(coords_b)
    sep_deg = sep2d.deg

    mask = sep_deg <= radius_deg
    return ids_a[mask], ids_b[idx_b[mask]], sep_deg[mask]


def _match_sky_rapids(
    ra_a: np.ndarray,
    dec_a: np.ndarray,
    ids_a: np.ndarray,
    ra_b: np.ndarray,
    dec_b: np.ndarray,
    ids_b: np.ndarray,
    radius_deg: float,
    *,
    gpu_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cupy as cp
    from cuml.neighbors import NearestNeighbors

    xyz_a = _radec_to_unit_xyz(ra_a, dec_a)
    xyz_b = _radec_to_unit_xyz(ra_b, dec_b)

    with cp.cuda.Device(gpu_id):
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(cp.asarray(xyz_b, dtype=cp.float32))
        _, idx = nn.kneighbors(cp.asarray(xyz_a, dtype=cp.float32))

    idx_b = cp.asnumpy(idx).ravel().astype(np.int64, copy=False)
    sep_deg = _angular_sep_deg_from_unit_xyz(xyz_a, xyz_b, idx_b)

    mask = sep_deg <= radius_deg
    return ids_a[mask], ids_b[idx_b[mask]], sep_deg[mask]


def match_sky_nn_within_radius(
    ra_a: np.ndarray,
    dec_a: np.ndarray,
    ids_a: np.ndarray,
    ra_b: np.ndarray,
    dec_b: np.ndarray,
    ids_b: np.ndarray,
    radius_deg: float,
    *,
    backend: MatchBackend = "astropy",
    gpu_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Nearest-neighbour sky match: each A source → closest B, keep if within radius.

    Returns ``(ids_a_matched, ids_b_matched, separations_deg)``.
    """
    validate_match_backend(backend)
    if ra_a.size == 0 or ra_b.size == 0:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        return empty_i, empty_i, empty_f

    if backend == "astropy":
        return _match_sky_astropy(
            ra_a, dec_a, ids_a, ra_b, dec_b, ids_b, radius_deg,
        )
    return _match_sky_rapids(
        ra_a, dec_a, ids_a, ra_b, dec_b, ids_b, radius_deg, gpu_id=gpu_id,
    )
