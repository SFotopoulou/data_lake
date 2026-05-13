"""
spectrum_dataset – PyTorch Dataset / IterableDataset for 1-D spectra.

Two modes
---------
SpectrumDataset (map-style)
    Loads a fixed list of source_ids.  Supports random access, shuffling,
    and DataLoader with ``num_workers > 0``.

TileSpectrumDataset (iterable-style)
    Streams complete HEALPix tiles sequentially.  Optimal for full-epoch
    training with minimal random I/O.

Built-in transforms (composable)
---------------------------------
    mask_to_nan          – zero-out masked pixels (set flux NaN, ivar 0)
    median_normalise     – divide flux by its good-pixel median
    log_flux             – log1p stretch on flux
    rest_frame_resample  – resample onto a common rest-frame grid using z

Usage
-----
>>> from data_lake.ml.spectrum_dataset import SpectrumDataset, mask_to_nan
>>> ds = SpectrumDataset(
...     lake_root="/data/lake",
...     survey="sdss_dr17",
...     source_ids=my_ids,
...     transform=mask_to_nan,
... )
>>> loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset, get_worker_info
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

    class Dataset:  # type: ignore[no-redef]
        pass

    class IterableDataset:  # type: ignore[no-redef]
        pass

    def get_worker_info():  # type: ignore[misc]
        return None

from data_lake.io.spectra import Spectrum, SpectrumAccessor

log = logging.getLogger(__name__)

Transform = Callable[[Spectrum], Any]


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------


def mask_to_nan(sp: Spectrum) -> Spectrum:
    """Set flux to NaN and ivar to 0 for masked pixels (mask != 0)."""
    bad = sp.mask != 0
    flux = sp.flux.copy()
    ivar = sp.ivar.copy()
    flux[bad] = np.nan
    ivar[bad] = 0.0
    return Spectrum(
        source_id=sp.source_id,
        flux=flux, ivar=ivar, mask=sp.mask,
        wavelength=sp.wavelength, meta=sp.meta, wcs_attrs=sp.wcs_attrs,
    )


def median_normalise(sp: Spectrum) -> Spectrum:
    """Divide flux by its median over good pixels; ivar scaled accordingly."""
    good = sp.good
    if not good.any():
        return sp
    med = float(np.median(sp.flux[good]))
    if med == 0:
        return sp
    flux = sp.flux / med
    ivar = sp.ivar * (med ** 2)
    return Spectrum(
        source_id=sp.source_id,
        flux=flux.astype(np.float32), ivar=ivar.astype(np.float32),
        mask=sp.mask, wavelength=sp.wavelength, meta=sp.meta, wcs_attrs=sp.wcs_attrs,
    )


def log_flux(sp: Spectrum) -> Spectrum:
    """Apply log1p stretch to flux (clipped to ≥0 first)."""
    flux = np.log1p(np.clip(sp.flux, 0.0, None)).astype(np.float32)
    return Spectrum(
        source_id=sp.source_id,
        flux=flux, ivar=sp.ivar, mask=sp.mask,
        wavelength=sp.wavelength, meta=sp.meta, wcs_attrs=sp.wcs_attrs,
    )


def rest_frame_resample(
    target_wave: np.ndarray,
) -> Callable[[Spectrum], Spectrum]:
    """
    Return a transform that resamples a spectrum onto ``target_wave`` (Å)
    in the rest frame defined by the stored redshift ``z``.

    Uses linear interpolation; pixels outside the coverage are set to NaN/0.

    Parameters
    ----------
    target_wave:
        1-D array of target wavelengths in Angstrom (rest frame).
    """

    def _transform(sp: Spectrum) -> Spectrum:
        z = float(sp.meta.get("z", 0.0))
        obs_wave_rest = sp.wavelength / (1.0 + z)

        # Interpolate flux and ivar onto target grid
        flux_new = np.interp(target_wave, obs_wave_rest, sp.flux,
                             left=np.nan, right=np.nan).astype(np.float32)
        ivar_new = np.interp(target_wave, obs_wave_rest, sp.ivar,
                             left=0.0, right=0.0).astype(np.float32)
        mask_new = (np.isnan(flux_new) | (ivar_new <= 0)).astype(np.uint8)
        flux_new = np.nan_to_num(flux_new, nan=0.0)

        return Spectrum(
            source_id=sp.source_id,
            flux=flux_new, ivar=ivar_new, mask=mask_new,
            wavelength=target_wave.astype(np.float64),
            meta=sp.meta, wcs_attrs=sp.wcs_attrs,
        )

    return _transform


def compose(*transforms: Transform) -> Transform:
    """Chain multiple transforms into one."""
    def _composed(sp: Spectrum) -> Any:
        result = sp
        for t in transforms:
            result = t(result)
        return result
    return _composed


# ---------------------------------------------------------------------------
# Map-style dataset
# ---------------------------------------------------------------------------


class SpectrumDataset(Dataset):
    """
    Map-style PyTorch Dataset over a fixed list of source_ids.

    Parameters
    ----------
    lake_root:
        Data lake root.
    survey:
        Survey identifier.
    source_ids:
        Ordered list of source IDs to include.
    norder:
        HEALPix partitioning order.
    transform:
        Optional callable ``(Spectrum) -> any``.  Applied after loading.
        If None and PyTorch is available, the flux array is returned as a
        ``torch.Tensor`` of shape ``(N_pix,)``.
    return_full:
        If True, return the full ``Spectrum`` dataclass (not just flux).
    catalog_accessor:
        Optional ``CatalogAccessor`` for O(1) tile lookup.
    """

    def __init__(
        self,
        lake_root: Path | str,
        survey: str,
        source_ids: Sequence[int],
        norder: int = 5,
        transform: Transform | None = None,
        return_full: bool = False,
        catalog_accessor=None,
    ) -> None:
        self.lake_root = Path(lake_root)
        self.survey = survey
        self.norder = norder
        self.source_ids: list[int] = list(source_ids)
        self.transform = transform
        self.return_full = return_full
        self._catalog = catalog_accessor
        self._accessor: SpectrumAccessor | None = None
        self._worker_id: int | None = None

    def _get_accessor(self) -> SpectrumAccessor:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        if self._accessor is None or worker_id != self._worker_id:
            self._accessor = SpectrumAccessor(
                self.lake_root, self.survey,
                norder=self.norder,
                catalog_accessor=self._catalog,
            )
            self._worker_id = worker_id
        return self._accessor

    def __len__(self) -> int:
        return len(self.source_ids)

    def __getitem__(self, idx: int):
        source_id = self.source_ids[idx]
        sp = self._get_accessor().get_spectrum(source_id)

        if self.transform is not None:
            return self.transform(sp)

        if self.return_full:
            return sp

        # Default: return flux tensor
        if _TORCH_AVAILABLE:
            return torch.from_numpy(np.ascontiguousarray(sp.flux))
        return sp.flux

    def __repr__(self) -> str:
        return (
            f"SpectrumDataset(survey={self.survey!r}, n={len(self)}, "
            f"norder={self.norder})"
        )


# ---------------------------------------------------------------------------
# Iterable dataset
# ---------------------------------------------------------------------------


class TileSpectrumDataset(IterableDataset):
    """
    Iterable dataset that streams complete HEALPix tiles in order.

    Each tile is read sequentially (maximally efficient for sharded Zarr).
    Tiles are distributed across DataLoader workers so each tile belongs to
    exactly one worker.

    Parameters
    ----------
    lake_root, survey, norder:
        Standard data lake parameters.
    tile_pixels:
        Optional explicit list of HEALPix pixel indices to include.
        Defaults to all tiles on disk.
    transform:
        Transform applied to each ``Spectrum``.
    shuffle_tiles:
        Shuffle the tile order at the start of each epoch (default False).
    """

    def __init__(
        self,
        lake_root: Path | str,
        survey: str,
        norder: int = 5,
        tile_pixels: list[int] | None = None,
        transform: Transform | None = None,
        shuffle_tiles: bool = False,
    ) -> None:
        self.lake_root = Path(lake_root)
        self.survey = survey
        self.norder = norder
        self.transform = transform
        self.shuffle_tiles = shuffle_tiles

        acc = SpectrumAccessor(lake_root, survey, norder=norder)
        self._all_tiles: list[int] = (
            tile_pixels if tile_pixels is not None else acc.available_tiles()
        )

    def __iter__(self):
        worker = get_worker_info()
        tiles = list(self._all_tiles)

        if self.shuffle_tiles:
            np.random.shuffle(tiles)

        if worker is not None:
            tiles = tiles[worker.id :: worker.num_workers]

        acc = SpectrumAccessor(self.lake_root, self.survey, norder=self.norder)

        for npix in tiles:
            try:
                for sid, flux, ivar, mask, wave, meta in acc.iter_tile(npix):
                    sp = Spectrum(
                        source_id=sid,
                        flux=flux, ivar=ivar, mask=mask,
                        wavelength=wave, meta=meta,
                        wcs_attrs=acc._get_tile_store(npix).wcs_attrs,
                    )
                    if self.transform is not None:
                        result = self.transform(sp)
                    elif _TORCH_AVAILABLE:
                        result = torch.from_numpy(np.ascontiguousarray(sp.flux))
                    else:
                        result = sp.flux
                    yield result
            except FileNotFoundError:
                log.warning("Spectrum tile %d not found, skipping.", npix)
