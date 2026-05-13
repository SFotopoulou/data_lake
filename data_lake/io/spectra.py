"""
spectra – Zarr v3 accessor for 1-D spectrum stacks.

Core API
--------
>>> from data_lake.io.spectra import SpectrumAccessor
>>> acc = SpectrumAccessor("/data/lake", "sdss_dr17")
>>> sp = acc.get_spectrum(source_id=1237654321098)
>>> sp.flux          # (N_pix,) float32
>>> sp.wavelength    # (N_pix,) float64 in Angstrom
>>> sp.wcs_attrs     # dict with crval, cdelt, ctype …
>>> batch = acc.get_batch([id1, id2, id3])   # (N, N_pix) float32
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from data_lake.ingest.fits_to_parquet import healpix_dir
from data_lake.ingest.fits_to_spectra_zarr import _META_DTYPE

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spectrum container
# ---------------------------------------------------------------------------


@dataclass
class Spectrum:
    """
    Container for one 1-D spectrum returned by SpectrumAccessor.

    Attributes
    ----------
    source_id:
        Stable integer source identifier.
    flux:
        Flux array, shape ``(N_pix,)``, dtype float32.
    ivar:
        Inverse-variance array, shape ``(N_pix,)``, dtype float32.
    mask:
        Bitmask array, shape ``(N_pix,)``, dtype uint8 / uint16.
    wavelength:
        Wavelength array in Angstrom, shape ``(N_pix,)``, dtype float64.
    meta:
        Dict of per-source scalars (z, z_err, snr, exptime, R, instr).
    wcs_attrs:
        Dict with spectral WCS parameters (ctype, crval, cdelt, crpix, unit).
    resolution:
        Banded resolution matrix in diagonal storage, shape ``(n_diag, N_pix)``,
        dtype float32.  ``None`` when the tile was ingested without
        ``--with-resolution``.
    resolution_offsets:
        Integer diagonal offsets, shape ``(n_diag,)``.  ``None`` when
        ``resolution`` is ``None``.
    """
    source_id: int
    flux: np.ndarray
    ivar: np.ndarray
    mask: np.ndarray
    wavelength: np.ndarray
    meta: dict[str, Any]
    wcs_attrs: dict[str, Any]
    resolution: np.ndarray | None = None          # (n_diag, N_pix) float32
    resolution_offsets: np.ndarray | None = None  # (n_diag,) int

    @property
    def err(self) -> np.ndarray:
        """1-sigma flux uncertainty (ivar → sigma, avoiding divide-by-zero)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.ivar > 0, 1.0 / np.sqrt(self.ivar), np.inf).astype(np.float32)

    @property
    def good(self) -> np.ndarray:
        """Boolean mask: True where both ivar > 0 and mask == 0."""
        return (self.ivar > 0) & (self.mask == 0)

    def rest_frame_wavelength(self) -> np.ndarray:
        """Return wavelength shifted to rest frame using stored redshift z."""
        z = float(self.meta.get("z", 0.0))
        return self.wavelength / (1.0 + z)

    def resolution_operator(self):
        """
        Rebuild the sparse banded LSF operator for forward-modelling.

        Returns a ``scipy.sparse.dia_matrix`` of shape ``(N_pix, N_pix)`` such
        that ``R @ model_grid`` gives the predicted observed spectrum.

        Raises
        ------
        ValueError
            If this spectrum was ingested without ``--with-resolution``.
        ImportError
            If ``scipy`` is not installed.

        Example
        -------
        >>> R = spec.resolution_operator()
        >>> model_obs = R @ template_resampled   # forward through the LSF
        >>> chi2 = np.sum((spec.flux - model_obs) ** 2 * spec.ivar)
        """
        if self.resolution is None or self.resolution_offsets is None:
            raise ValueError(
                "Resolution matrix was not stored for this spectrum. "
                "Re-ingest the tile with --with-resolution."
            )
        try:
            from scipy.sparse import dia_matrix
        except ImportError as exc:
            raise ImportError(
                "scipy is required for resolution_operator(). "
                "Install it with: pip install scipy"
            ) from exc
        n_pix = self.flux.shape[0]
        return dia_matrix(
            (self.resolution, self.resolution_offsets), shape=(n_pix, n_pix)
        )


def _decode_meta(raw: bytes | np.ndarray) -> dict[str, Any]:
    """Decode a single structured meta byte entry into a plain dict."""
    if isinstance(raw, (bytes, bytearray, np.bytes_)):
        arr = np.frombuffer(raw, dtype=_META_DTYPE)[0]
    else:
        arr = np.frombuffer(bytes(raw), dtype=_META_DTYPE)[0]
    result: dict[str, Any] = {}
    for name in _META_DTYPE.names:
        val = arr[name].item()
        if isinstance(val, bytes):
            val = val.rstrip(b"\x00").decode("ascii", errors="replace")
        result[name] = val
    return result


# ---------------------------------------------------------------------------
# Per-tile lazy store
# ---------------------------------------------------------------------------


class SpectrumTileStore:
    """Lazily opened Zarr v3 group for a single HEALPix tile."""

    def __init__(self, tile_path: Path) -> None:
        self._path = tile_path
        self._root: zarr.Group | None = None

    def _open(self) -> zarr.Group:
        if self._root is None:
            store = zarr.storage.LocalStore(str(self._path))
            self._root = zarr.open_group(store=store, mode="r", zarr_format=3)
        return self._root

    @property
    def wcs_attrs(self) -> dict[str, Any]:
        root = self._open()
        return dict(root.attrs)

    @property
    def wavelength_mode(self) -> str:
        return str(self.wcs_attrs.get("wavelength_mode", "shared"))

    def _get_shared_wavelength(self) -> np.ndarray:
        root = self._open()
        return np.array(root["wavelength"])

    def _get_per_source_wavelength(self, idx: int) -> np.ndarray:
        root = self._open()
        return np.array(root["wavelength"][idx], dtype=np.float64)

    @property
    def has_resolution(self) -> bool:
        """True if this tile stores a banded resolution array."""
        return "resolution_n_diag" in self.wcs_attrs

    @property
    def resolution_offsets(self) -> np.ndarray | None:
        """Return the stored resolution diagonal offsets, or None."""
        offsets = self.wcs_attrs.get("resolution_offsets")
        return np.asarray(offsets, dtype=np.int32) if offsets is not None else None

    def get_spectrum(
        self, idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray | None]:
        """Return (flux, ivar, mask, wavelength, meta_dict, resolution|None) for idx."""
        root = self._open()
        flux  = np.array(root["flux"][idx])
        ivar  = np.array(root["ivar"][idx])
        mask  = np.array(root["mask"][idx])
        meta  = _decode_meta(root["meta"][idx])

        if self.wavelength_mode == "shared":
            wave = self._get_shared_wavelength()
        else:
            wave = self._get_per_source_wavelength(idx)

        res = np.array(root["resolution"][idx], dtype=np.float32) if self.has_resolution else None

        return flux, ivar, mask, wave, meta, res

    def get_spectra(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray | None]:
        """Return batch arrays for sorted indices."""
        root = self._open()
        order = np.argsort(indices)
        sorted_idx = indices[order]

        flux_list  = [np.array(root["flux"][int(i)])  for i in sorted_idx]
        ivar_list  = [np.array(root["ivar"][int(i)])  for i in sorted_idx]
        mask_list  = [np.array(root["mask"][int(i)])  for i in sorted_idx]
        meta_list  = [_decode_meta(root["meta"][int(i)]) for i in sorted_idx]

        flux_arr = np.stack(flux_list)
        ivar_arr = np.stack(ivar_list)
        mask_arr = np.stack(mask_list)

        if self.wavelength_mode == "shared":
            wave = self._get_shared_wavelength()
        else:
            wave_list = [np.array(root["wavelength"][int(i)], dtype=np.float64) for i in sorted_idx]
            wave = np.stack(wave_list)

        res_arr: np.ndarray | None = None
        if self.has_resolution:
            res_list = [np.array(root["resolution"][int(i)], dtype=np.float32) for i in sorted_idx]
            res_arr = np.stack(res_list)

        # Unshuffle to original order
        unshuffle = np.empty_like(order)
        unshuffle[order] = np.arange(len(order))
        return (
            flux_arr[unshuffle],
            ivar_arr[unshuffle],
            mask_arr[unshuffle],
            wave if wave.ndim == 1 else wave[unshuffle],
            [meta_list[i] for i in unshuffle],
            res_arr[unshuffle] if res_arr is not None else None,
        )

    def get_source_ids(self) -> np.ndarray:
        root = self._open()
        return np.array(root["source_id"])

    def build_index(self) -> dict[int, int]:
        """Return {source_id: local_index} mapping."""
        ids = self.get_source_ids()
        return {int(sid): i for i, sid in enumerate(ids)}

    @property
    def n_sources(self) -> int:
        root = self._open()
        return root["flux"].shape[0]


# ---------------------------------------------------------------------------
# Survey-level accessor
# ---------------------------------------------------------------------------


class SpectrumAccessor:
    """
    Random-access interface to the Zarr v3 spectrum stacks for one survey.

    Parameters
    ----------
    lake_root:
        Data lake root.
    survey_name:
        Survey identifier (must match the directory under ``spectra/``).
    norder:
        HEALPix partitioning order (read from ``spectrum_info.json`` if omitted).
    catalog_accessor:
        Optional ``CatalogAccessor`` for fast O(1) tile lookup via the
        ``_spectrum_index`` catalog column.
    """

    def __init__(
        self,
        lake_root: Path | str,
        survey_name: str,
        norder: int | None = None,
        catalog_accessor=None,
    ) -> None:
        self.lake_root = Path(lake_root)
        self.survey_name = survey_name
        self._spectra_root = self.lake_root / "spectra" / survey_name
        if not self._spectra_root.exists():
            raise FileNotFoundError(f"Spectrum store not found: {self._spectra_root}")

        info_path = self._spectra_root / "spectrum_info.json"
        self._info: dict = {}
        if info_path.exists():
            with open(info_path) as fh:
                self._info = json.load(fh)

        self.norder: int = norder if norder is not None else int(self._info.get("hats_order", 5))
        self._catalog = catalog_accessor

        self._tile_stores: dict[int, SpectrumTileStore] = {}
        self._tile_indices: dict[int, dict[int, int]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tile_path(self, npix: int) -> Path:
        return self._spectra_root / healpix_dir(self.norder, npix) / f"Npix={npix}.zarr"

    def _get_tile_store(self, npix: int) -> SpectrumTileStore:
        if npix not in self._tile_stores:
            path = self._tile_path(npix)
            if not path.exists():
                raise FileNotFoundError(f"Spectrum tile not found: {path}")
            self._tile_stores[npix] = SpectrumTileStore(path)
        return self._tile_stores[npix]

    def _get_tile_index(self, npix: int) -> dict[int, int]:
        if npix not in self._tile_indices:
            self._tile_indices[npix] = self._get_tile_store(npix).build_index()
        return self._tile_indices[npix]

    def _source_id_to_tile_and_local(self, source_id: int) -> tuple[int, int]:
        """Return (npix, local_index) for a given source_id."""
        if self._catalog is not None:
            try:
                npix, local_idx = self._catalog.get_tile_index(source_id, kind="spectrum")
                if local_idx >= 0:
                    return npix, local_idx
            except (KeyError, Exception):
                pass

        # Search open tiles
        for npix, idx_map in self._tile_indices.items():
            if source_id in idx_map:
                return npix, idx_map[source_id]

        # Scan unvisited tiles
        for tile_path in self._spectra_root.rglob("*.zarr"):
            npix_str = tile_path.stem.split("=")[-1]
            try:
                npix = int(npix_str)
            except ValueError:
                continue
            if npix in self._tile_indices:
                continue
            idx_map = self._get_tile_index(npix)
            if source_id in idx_map:
                return npix, idx_map[source_id]

        raise KeyError(f"source_id={source_id} not found in spectrum store.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_spectrum(self, source_id: int) -> Spectrum:
        """
        Return a ``Spectrum`` for a single source.

        The wavelength array is reconstructed from WCS attrs when
        ``wavelength_mode == "shared"`` so no extra disk read is needed.
        If the tile was ingested with ``--with-resolution``, the returned
        ``Spectrum`` will have ``.resolution`` and ``.resolution_offsets``
        populated; call ``.resolution_operator()`` to get a scipy sparse matrix.
        """
        npix, local_idx = self._source_id_to_tile_and_local(source_id)
        store = self._get_tile_store(npix)
        flux, ivar, mask, wave, meta, res = store.get_spectrum(local_idx)
        return Spectrum(
            source_id=source_id,
            flux=flux, ivar=ivar, mask=mask,
            wavelength=wave, meta=meta,
            wcs_attrs=store.wcs_attrs,
            resolution=res,
            resolution_offsets=store.resolution_offsets if res is not None else None,
        )

    def get_batch(
        self,
        source_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | list[np.ndarray]]:
        """
        Return ``(flux, ivar, mask, wavelength)`` for a batch of source_ids.

        Reads are grouped per tile to minimise seeks.

        Returns
        -------
        flux:
            ``(N, N_pix)`` float32.
        ivar:
            ``(N, N_pix)`` float32.
        mask:
            ``(N, N_pix)`` uint8.
        wavelength:
            ``(N_pix,)`` float64 in shared mode, or list of per-source arrays.
        """
        from collections import defaultdict
        tile_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for orig_pos, sid in enumerate(source_ids):
            npix, local_idx = self._source_id_to_tile_and_local(sid)
            tile_groups[npix].append((orig_pos, local_idx))

        # Infer output shape from first tile
        sample_npix = next(iter(tile_groups))
        sample_store = self._get_tile_store(sample_npix)
        sample_idx = tile_groups[sample_npix][0][1]
        sample_flux, _, _, sample_wave, _, _ = sample_store.get_spectrum(sample_idx)

        n = len(source_ids)
        n_pix = len(sample_flux)
        flux_out = np.empty((n, n_pix), dtype=np.float32)
        ivar_out = np.empty((n, n_pix), dtype=np.float32)
        mask_out = np.empty((n, n_pix), dtype=np.uint8)
        shared_wavelength: np.ndarray | None = sample_wave if sample_wave.ndim == 1 else None
        per_source_wavelength: list[np.ndarray | None] = [None] * n

        for npix, positions in tile_groups.items():
            store = self._get_tile_store(npix)
            orig_pos_arr = np.array([p[0] for p in positions])
            local_idx_arr = np.array([p[1] for p in positions])
            f, iv, mk, wave, _, _res = store.get_spectra(local_idx_arr)
            flux_out[orig_pos_arr] = f
            ivar_out[orig_pos_arr] = iv
            mask_out[orig_pos_arr] = mk
            if wave.ndim == 2:
                shared_wavelength = None
                for i, orig in enumerate(orig_pos_arr):
                    per_source_wavelength[orig] = wave[i]

        if shared_wavelength is not None:
            return flux_out, ivar_out, mask_out, shared_wavelength
        return flux_out, ivar_out, mask_out, per_source_wavelength

    def iter_tile(self, npix: int):
        """
        Iterate over all (source_id, flux, ivar, mask, wavelength, meta) tuples in a tile.

        Sequential read order; ideal for full-tile ML or export tasks.
        """
        store = self._get_tile_store(npix)
        root = store._open()
        ids = store.get_source_ids()
        shared_wave = store._get_shared_wavelength() if store.wavelength_mode == "shared" else None

        res_offsets = store.resolution_offsets

        for i, sid in enumerate(ids):
            flux  = np.array(root["flux"][i])
            ivar  = np.array(root["ivar"][i])
            mask  = np.array(root["mask"][i])
            meta  = _decode_meta(root["meta"][i])
            wave  = shared_wave if shared_wave is not None else np.array(root["wavelength"][i], dtype=np.float64)
            res   = np.array(root["resolution"][i], dtype=np.float32) if store.has_resolution else None
            yield int(sid), flux, ivar, mask, wave, meta, res, res_offsets

    def available_tiles(self) -> list[int]:
        """Return sorted list of HEALPix pixel indices that have spectrum data."""
        tiles = []
        for p in self._spectra_root.rglob("*.zarr"):
            try:
                tiles.append(int(p.stem.split("=")[-1]))
            except ValueError:
                pass
        return sorted(tiles)

    @property
    def info(self) -> dict:
        return self._info

    def __repr__(self) -> str:
        return (
            f"SpectrumAccessor(survey={self.survey_name!r}, "
            f"norder={self.norder}, root={self.lake_root})"
        )
