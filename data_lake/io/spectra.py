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

Bulk subset extraction
----------------------
>>> result = acc.extract_subset_to_zarr(
...     source_ids=my_129k_targetids,
...     output_zarr="/scratch/desi_subset.zarr",
... )
>>> result["n_written"], len(result["missing_ids"])
"""

from __future__ import annotations

import json
import logging
import time
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

SubsetFormat = Literal["zarr", "parquet", "fits", "hdf5"]
FitsLayout = Literal["per-file", "catalog"]

import numpy as np
import zarr
import zarr.codecs

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


@dataclass
class _SubsetExtractionPlan:
    """Resolved tile locations and shapes for a subset extract."""

    by_tile: dict[int, list[tuple[int, int]]]
    missing_ids: list[int]
    n_requested: int
    n_written: int
    n_pix: int
    wavelength_mode: str  # "shared" | "per_source"
    wavelength: np.ndarray | None  # 1-D shared grid, or None for per_source
    flux_dtype: Any
    ivar_dtype: Any
    mask_dtype: Any
    src_wcs: dict[str, Any]


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
        from data_lake.io.zarr_batch import read_zarr_rows

        root = self._open()
        flux_arr = read_zarr_rows(root["flux"], indices)
        ivar_arr = read_zarr_rows(root["ivar"], indices)
        mask_arr = read_zarr_rows(root["mask"], indices)
        order = np.argsort(indices, kind="stable")
        sorted_idx = indices[order]
        meta_sorted = [_decode_meta(root["meta"][int(i)]) for i in sorted_idx]
        unshuffle = np.empty_like(order)
        unshuffle[order] = np.arange(len(order))
        meta_list = [meta_sorted[i] for i in unshuffle]

        if self.wavelength_mode == "shared":
            wave = self._get_shared_wavelength()
        else:
            wave = np.asarray(
                read_zarr_rows(root["wavelength"], indices), dtype=np.float64,
            )

        res_arr: np.ndarray | None = None
        if self.has_resolution:
            res_arr = read_zarr_rows(root["resolution"], indices).astype(np.float32)

        return flux_arr, ivar_arr, mask_arr, wave, meta_list, res_arr

    def get_source_ids(self) -> np.ndarray:
        root = self._open()
        from data_lake.ingest.zarr_ids import zarr_join_array

        return np.array(zarr_join_array(root))

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

        lookup = self._build_source_id_lookup(source_ids, show_progress=False)
        tile_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for orig_pos, sid in enumerate(source_ids):
            loc = lookup.get(int(sid))
            if loc is None:
                raise KeyError(f"source_id={sid} not found in spectrum store.")
            npix, local_idx = loc
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

    # ------------------------------------------------------------------
    # Bulk subset extraction
    # ------------------------------------------------------------------

    def _scan_tile_for_ids(
        self, npix: int, remaining: np.ndarray,
    ) -> dict[int, tuple[int, int]]:
        store = self._get_tile_store(npix)
        sids_in_tile = store.get_source_ids()
        hit_mask = np.isin(sids_in_tile, remaining, assume_unique=False)
        if not hit_mask.any():
            return {}
        local_idxs = np.nonzero(hit_mask)[0]
        hit_sids = sids_in_tile[local_idxs]
        return {
            int(sid): (npix, int(lidx))
            for sid, lidx in zip(hit_sids.tolist(), local_idxs.tolist())
        }

    def _build_source_id_lookup(
        self,
        requested: np.ndarray,
        show_progress: bool = True,
    ) -> dict[int, tuple[int, int]]:
        """Resolve ``source_id -> (npix, local_idx)`` for a batch of IDs."""
        from data_lake.io.id_lookup import bulk_tile_index_with_scan

        requested_arr = np.asarray(requested, dtype=np.int64).ravel()
        return bulk_tile_index_with_scan(
            requested_arr,
            catalog=self._catalog,
            kind="spectrum",
            scan_tile=self._scan_tile_for_ids,
            available_tiles=self.available_tiles,
            show_progress=show_progress,
        )

    def _plan_subset_extraction(
        self,
        source_ids: Sequence[int] | np.ndarray,
        *,
        missing: str = "skip",
        show_progress: bool = True,
    ) -> _SubsetExtractionPlan:
        """Resolve source IDs to tile locations and capture wavelength layout."""
        if missing not in ("skip", "error"):
            raise ValueError(f"missing must be 'skip' or 'error', got {missing!r}")

        wave_mode = str(self._info.get("wavelength_mode", "shared"))

        requested = np.asarray(source_ids, dtype=np.int64).ravel()
        if requested.size == 0:
            raise ValueError("source_ids is empty.")
        n_requested = int(np.unique(requested).size)

        log.info(
            "Resolving %d unique source_ids in %s …",
            n_requested, self.survey_name,
        )
        sid_to_loc = self._build_source_id_lookup(requested, show_progress=show_progress)

        by_tile: dict[int, list[tuple[int, int]]] = {}
        missing_ids: list[int] = []
        seen: set[int] = set()
        for sid_raw in requested.tolist():
            sid = int(sid_raw)
            if sid in seen:
                continue
            seen.add(sid)
            loc = sid_to_loc.get(sid)
            if loc is None:
                missing_ids.append(sid)
                continue
            npix, lidx = loc
            by_tile.setdefault(npix, []).append((lidx, sid))

        if missing == "error" and missing_ids:
            raise KeyError(
                f"{len(missing_ids)} source_id(s) not found in lake "
                f"(first few: {missing_ids[:10]})."
            )

        n_written = sum(len(v) for v in by_tile.values())
        if n_written == 0:
            raise ValueError("None of the requested source_ids were found in the lake.")

        # Scan all tiles to find the maximum n_pix (per_source tiles can differ after widen).
        first_npix = next(iter(by_tile))
        first_store = self._get_tile_store(first_npix)
        first_root = first_store._open()
        flux_dtype = first_root["flux"].dtype
        ivar_dtype = first_root["ivar"].dtype
        mask_dtype = first_root["mask"].dtype
        src_wcs = dict(first_store.wcs_attrs)

        n_pix = int(first_root["flux"].shape[1])
        for npix in by_tile:
            if npix == first_npix:
                continue
            t_root = self._get_tile_store(npix)._open()
            tile_n_pix = int(t_root["flux"].shape[1])
            if wave_mode == "shared" and tile_n_pix != n_pix:
                raise ValueError(
                    f"Tile {npix} has N_pix={tile_n_pix} but tile {first_npix} has "
                    f"N_pix={n_pix}; non-uniform wavelength grid is not supported for "
                    f"wavelength_mode='shared'."
                )
            n_pix = max(n_pix, tile_n_pix)

        if wave_mode == "shared":
            wavelength: np.ndarray | None = np.asarray(
                first_root["wavelength"][:], dtype=np.float64
            )
        else:
            wavelength = None

        return _SubsetExtractionPlan(
            by_tile=by_tile,
            missing_ids=missing_ids,
            n_requested=n_requested,
            n_written=n_written,
            n_pix=n_pix,
            wavelength_mode=wave_mode,
            wavelength=wavelength,
            flux_dtype=flux_dtype,
            ivar_dtype=ivar_dtype,
            mask_dtype=mask_dtype,
            src_wcs=src_wcs,
        )

    @staticmethod
    def _source_ids_in_plan(plan: _SubsetExtractionPlan) -> np.ndarray:
        """All source IDs that will be written for this extraction plan."""
        sids: list[int] = []
        for items in plan.by_tile.values():
            sids.extend(int(sid) for _, sid in items)
        return np.asarray(sids, dtype=np.int64)

    def _build_catalog_redshift_map(
        self,
        plan: _SubsetExtractionPlan,
    ) -> dict[int, float] | None:
        """Load redshifts from the survey catalog when available."""
        if self._catalog is None:
            return None
        if self._catalog.redshift_column is None:
            log.warning(
                "Catalog %r has no redshift column (%s); using spectrum-tile meta.z.",
                self.survey_name,
                ", ".join(("Z", "ZCOSMO", "REDSHIFT", "...")),
            )
            return None
        sids = self._source_ids_in_plan(plan)
        z_map = self._catalog.redshifts_for_source_ids(sids)
        log.info(
            "Redshift for %d spectra from catalog column %r.",
            len(z_map),
            self._catalog.redshift_column,
        )
        return z_map

    @staticmethod
    def _redshift_batch_for_sources(
        sorted_sids: np.ndarray,
        z_map: dict[int, float] | None,
        meta_seq,
    ) -> np.ndarray:
        """Per-row redshift: catalog map when provided, else spectrum tile meta.

        ``meta_seq`` may be a list of already-decoded dicts (from ``get_spectra``) or
        a raw zarr array of structured byte buffers decoded via ``_decode_meta``.
        """
        if z_map is not None:
            return np.array(
                [z_map.get(int(s), np.nan) for s in sorted_sids.tolist()],
                dtype=np.float32,
            )
        z_batch = np.empty(len(sorted_sids), dtype=np.float32)
        for i, item in enumerate(meta_seq):
            if isinstance(item, dict):
                z_batch[i] = float(item.get("z", np.nan))
            else:
                z_batch[i] = _decode_meta(item).get("z", np.nan)
        return z_batch

    @staticmethod
    def _pad_to_n_pix(
        arr: np.ndarray,
        n_pix: int,
        pad_value: float,
    ) -> np.ndarray:
        """Right-pad a 2-D ``(N, tile_n_pix)`` array to ``(N, n_pix)``."""
        tile_n_pix = arr.shape[1]
        if tile_n_pix == n_pix:
            return arr
        pad_width = n_pix - tile_n_pix
        return np.pad(arr, ((0, 0), (0, pad_width)), constant_values=pad_value)

    def _iter_subset_tile_batches(
        self,
        plan: _SubsetExtractionPlan,
        *,
        show_progress: bool,
        z_map: dict[int, float] | None = None,
        flux_scale: float | None = None,
    ):
        """Yield per-tile (sorted_sids, flux, ivar, mask, wave_batch|None, redshift) batches.

        ``wave_batch`` is ``(batch, plan.n_pix) float32`` for per_source surveys and
        ``None`` for shared surveys (callers use ``plan.wavelength`` for the grid).
        """
        try:
            from tqdm.auto import tqdm
        except ImportError:
            def tqdm(x, **_kw):
                return x

        per_source = plan.wavelength_mode == "per_source"

        for npix, items in tqdm(
            plan.by_tile.items(),
            desc="extract",
            disable=not show_progress,
            unit="tile",
            total=len(plan.by_tile),
        ):
            local_idxs = np.fromiter((t[0] for t in items), dtype=np.int64, count=len(items))
            sids = np.fromiter((t[1] for t in items), dtype=np.int64, count=len(items))
            order = np.argsort(local_idxs)
            sorted_local = local_idxs[order]
            sorted_sids = sids[order]

            t_store = self._get_tile_store(npix)
            # get_spectra handles shared/per_source wavelength internally.
            flux_batch, ivar_batch, mask_batch, wave_or_1d, meta_list, _res = (
                t_store.get_spectra(sorted_local)
            )

            # For shared surveys verify grid width consistency (already enforced by
            # _plan_subset_extraction, but guard against concurrent tile widen).
            if not per_source and flux_batch.shape[1] != plan.n_pix:
                raise ValueError(
                    f"Tile {npix} has N_pix={flux_batch.shape[1]} but expected "
                    f"{plan.n_pix}; non-uniform wavelength grid is not supported "
                    f"for wavelength_mode='shared'."
                )

            # Pad to plan.n_pix when this tile is narrower (per_source cross-tile).
            if per_source:
                flux_batch = self._pad_to_n_pix(
                    np.asarray(flux_batch, dtype=np.float32), plan.n_pix, np.nan
                )
                ivar_batch = self._pad_to_n_pix(
                    np.asarray(ivar_batch, dtype=np.float32), plan.n_pix, 0.0
                )
                mask_batch = self._pad_to_n_pix(
                    np.asarray(mask_batch), plan.n_pix, 0
                )
                wave_batch: np.ndarray | None = self._pad_to_n_pix(
                    np.asarray(wave_or_1d, dtype=np.float32), plan.n_pix, 0.0
                )
            else:
                flux_batch = np.asarray(flux_batch, dtype=np.float32)
                ivar_batch = np.asarray(ivar_batch, dtype=np.float32)
                mask_batch = np.asarray(mask_batch)
                wave_batch = None

            z_batch = self._redshift_batch_for_sources(
                sorted_sids, z_map, meta_list,
            )

            if flux_scale is not None and flux_scale != 1.0:
                from data_lake.export.spectra_calibration import apply_flux_scale

                flux_batch, ivar_batch = apply_flux_scale(flux_batch, ivar_batch, flux_scale)

            yield sorted_sids, flux_batch, ivar_batch, mask_batch, wave_batch, z_batch

    def _collect_subset_stacks(
        self,
        plan: _SubsetExtractionPlan,
        *,
        show_progress: bool,
        z_map: dict[int, float] | None = None,
        flux_scale: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
        """Read all subset rows into contiguous numpy arrays (catalog FITS / Parquet)."""
        sid_chunks: list[np.ndarray] = []
        flux_chunks: list[np.ndarray] = []
        ivar_chunks: list[np.ndarray] = []
        mask_chunks: list[np.ndarray] = []
        z_chunks: list[np.ndarray] = []
        id_to_row: dict[int, int] = {}
        row_offset = 0

        for sorted_sids, flux_batch, ivar_batch, mask_batch, _wave_batch, z_batch in self._iter_subset_tile_batches(
            plan, show_progress=show_progress, z_map=z_map, flux_scale=flux_scale,
        ):
            sid_chunks.append(sorted_sids)
            flux_chunks.append(flux_batch)
            ivar_chunks.append(ivar_batch)
            mask_chunks.append(mask_batch)
            z_chunks.append(z_batch)
            for j, sid in enumerate(sorted_sids.tolist()):
                id_to_row[int(sid)] = row_offset + j
            row_offset += len(sorted_sids)

        source_id = np.concatenate(sid_chunks)
        flux = np.concatenate(flux_chunks, axis=0)
        ivar = np.concatenate(ivar_chunks, axis=0)
        mask = np.concatenate(mask_chunks, axis=0)
        redshift = np.concatenate(z_chunks, axis=0)
        return source_id, flux, ivar, mask, redshift, id_to_row

    def extract_subset(
        self,
        source_ids: Sequence[int] | np.ndarray,
        output: Path | str,
        *,
        fmt: SubsetFormat = "zarr",
        missing: str = "skip",
        chunks_per_shard: int = 512,
        show_progress: bool = True,
        overwrite: bool = False,
        fits_filename_template: str = "spec_{source_id}.fits",
        fits_layout: FitsLayout = "per-file",
        flux_scale: float | None = None,
    ) -> dict[str, Any]:
        """Extract a subset to Zarr, Parquet, or FITS (per-file or catalog)."""
        if fmt == "zarr":
            return self.extract_subset_to_zarr(
                source_ids,
                output,
                missing=missing,
                chunks_per_shard=chunks_per_shard,
                show_progress=show_progress,
                overwrite=overwrite,
                flux_scale=flux_scale,
            )
        if fmt == "parquet":
            return self.extract_subset_to_parquet(
                source_ids,
                output,
                missing=missing,
                show_progress=show_progress,
                overwrite=overwrite,
                flux_scale=flux_scale,
            )
        if fmt == "hdf5":
            return self.extract_subset_to_hdf5(
                source_ids,
                output,
                missing=missing,
                show_progress=show_progress,
                overwrite=overwrite,
                flux_scale=flux_scale,
            )
        if fmt == "fits":
            return self.extract_subset_to_fits(
                source_ids,
                output,
                missing=missing,
                show_progress=show_progress,
                overwrite=overwrite,
                filename_template=fits_filename_template,
                layout=fits_layout,
                flux_scale=flux_scale,
            )
        raise ValueError(f"unknown format {fmt!r}; use zarr, parquet, hdf5, or fits")

    def extract_subset_to_zarr(
        self,
        source_ids: Sequence[int] | np.ndarray,
        output_zarr: Path | str,
        *,
        missing: str = "skip",
        chunks_per_shard: int = 512,
        show_progress: bool = True,
        overwrite: bool = False,
        flux_scale: float | None = None,
    ) -> dict[str, Any]:
        """Extract a curated subset of spectra into a single flat Zarr v3 group.

        Writes one self-contained Zarr group containing only the requested
        spectra, with rows in **tile-traversal order** (contiguous writes →
        each output shard is written exactly once, no read-modify-write).
        The returned ``id_to_row`` mapping lets callers reorder if needed.

        Output layout
        -------------
        ::

            output_zarr/
              flux/        (N_written, N_pix)  float32  sharded
              ivar/        (N_written, N_pix)  float32  sharded
              mask/        (N_written, N_pix)  uint8/16 sharded
              wavelength/  (N_pix,)            float64  shared grid
              source_id/   (N_written,)        int64
              redshift/    (N_written,)        float32  (from catalog Z when available)

        Parameters
        ----------
        source_ids:
            Requested source identifiers.  Duplicates are deduplicated.
        output_zarr:
            Destination ``.zarr`` directory (must not exist unless
            ``overwrite=True``).
        missing:
            ``"skip"`` (default) silently omits IDs not found in the lake and
            reports them in the result; ``"error"`` raises ``KeyError``.
        chunks_per_shard:
            Output shard rows (same convention as ingest; default 512).
        show_progress:
            Show tqdm progress bars for the lookup and extract phases.
        overwrite:
            Remove an existing ``output_zarr`` before writing.

        Returns
        -------
        dict with keys:
            ``n_requested``  – unique IDs requested
            ``n_written``    – rows actually written
            ``missing_ids``  – list[int] of IDs not found in the lake
            ``id_to_row``    – dict[int, int] source_id → row in output
            ``output_zarr``  – str path to the destination

        Notes
        -----
        * Requires the source survey to be stored with
          ``wavelength_mode='shared'`` (single common wavelength grid across
          tiles).  Per-source wavelength support is a deliberate follow-up.
        * Reads each tile's flux / ivar / mask with a single
          ``get_orthogonal_selection`` call so each shard is decompressed at
          most once per tile.  This is typically 10–50× faster than looping
          ``root["flux"][i]`` per source.
        """
        output_zarr = Path(output_zarr)
        if output_zarr.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{output_zarr} already exists. Pass overwrite=True to replace it."
                )
            shutil.rmtree(output_zarr)

        plan = self._plan_subset_extraction(
            source_ids, missing=missing, show_progress=show_progress,
        )
        z_map = self._build_catalog_redshift_map(plan)
        n_written = plan.n_written
        n_pix = plan.n_pix
        n_requested = plan.n_requested
        missing_ids = plan.missing_ids
        src_wcs = plan.src_wcs
        per_source = plan.wavelength_mode == "per_source"

        # --- Create destination Zarr ---
        store = zarr.storage.LocalStore(str(output_zarr))
        out_root = zarr.open_group(store=store, mode="w", zarr_format=3)

        compressors = zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=3,
            shuffle=zarr.codecs.BloscShuffle.bitshuffle,
        )
        shard_rows = max(1, min(chunks_per_shard, n_written))

        def _arr2d(name: str, dtype, fill):
            out_root.create_array(
                name,
                shape=(n_written, n_pix),
                chunks=(1, n_pix),
                shards=(shard_rows, n_pix),
                dtype=dtype,
                compressors=compressors,
                fill_value=fill,
            )

        _arr2d("flux", plan.flux_dtype, np.nan)
        _arr2d("ivar", plan.ivar_dtype, 0.0)
        _arr2d("mask", plan.mask_dtype, 0)

        if per_source:
            # Per-source: wavelength stored as (N_written, N_pix) float32.
            _arr2d("wavelength", np.float32, 0.0)
        else:
            out_root.create_array(
                "wavelength", shape=(n_pix,), chunks=(n_pix,), dtype=np.float64, fill_value=0.0,
            )
            out_root["wavelength"][:] = plan.wavelength

        from data_lake.ingest.zarr_ids import create_zarr_join_array

        create_zarr_join_array(
            out_root,
            shape=(n_written,),
            chunks=(min(4096, n_written),),
            dtype=np.int64,
            fill_value=-1,
        )
        out_root.create_array(
            "redshift",
            shape=(n_written,),
            chunks=(min(4096, n_written),),
            dtype=np.float32,
            fill_value=np.nan,
        )

        attrs: dict[str, Any] = dict(src_wcs)
        attrs.update({
            "source_survey": self.survey_name,
            "source_lake_root": str(self.lake_root),
            "wavelength_mode": plan.wavelength_mode,
            "n_sources": n_written,
            "n_pix": n_pix,
            "n_requested": n_requested,
            "n_missing": len(missing_ids),
            "extract_created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema_version": "1",
        })
        if flux_scale is not None and flux_scale != 1.0:
            attrs["flux_scale"] = flux_scale
            attrs["flux_calibrated"] = True
        out_root.attrs.update(attrs)

        id_to_row: dict[int, int] = {}
        write_offset = 0
        for sorted_sids, flux_batch, ivar_batch, mask_batch, wave_batch, z_batch in self._iter_subset_tile_batches(
            plan, show_progress=show_progress, z_map=z_map, flux_scale=flux_scale,
        ):
            k = len(sorted_sids)
            slc = slice(write_offset, write_offset + k)
            out_root["flux"][slc, :] = flux_batch
            out_root["ivar"][slc, :] = ivar_batch
            out_root["mask"][slc, :] = mask_batch
            if per_source and wave_batch is not None:
                out_root["wavelength"][slc, :] = wave_batch
            from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN

            out_root[LAKE_JOIN_ID_COLUMN][slc] = sorted_sids
            out_root["redshift"][slc] = z_batch

            for j, sid in enumerate(sorted_sids.tolist()):
                id_to_row[int(sid)] = write_offset + j
            write_offset += k

        log.info(
            "Extracted %d/%d spectra → %s (skipped %d missing across %d tiles)",
            n_written, n_requested, output_zarr, len(missing_ids), len(plan.by_tile),
        )

        return {
            "n_requested": n_requested,
            "n_written": n_written,
            "missing_ids": missing_ids,
            "id_to_row": id_to_row,
            "output": str(output_zarr),
            "output_zarr": str(output_zarr),
            "format": "zarr",
            "flux_scale": flux_scale,
        }

    def extract_subset_to_parquet(
        self,
        source_ids: Sequence[int] | np.ndarray,
        output_parquet: Path | str,
        *,
        missing: str = "skip",
        show_progress: bool = True,
        overwrite: bool = False,
        flux_scale: float | None = None,
    ) -> dict[str, Any]:
        """Extract a subset into one Parquet file (one row per spectrum)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        output_parquet = Path(output_parquet)
        if output_parquet.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{output_parquet} already exists. Pass overwrite=True to replace it."
                )
            output_parquet.unlink()

        plan = self._plan_subset_extraction(
            source_ids, missing=missing, show_progress=show_progress,
        )
        z_map = self._build_catalog_redshift_map(plan)
        n_pix = plan.n_pix
        per_source = plan.wavelength_mode == "per_source"
        mask_pa = pa.uint16() if plan.mask_dtype == np.dtype(np.uint16) else pa.uint8()
        flt_vec = pa.list_(pa.float32())
        mask_vec = pa.list_(mask_pa)
        meta: dict[bytes, bytes] = {
            b"source_survey": self.survey_name.encode(),
            b"wavelength_mode": plan.wavelength_mode.encode(),
            b"wcs_attrs": json.dumps(plan.src_wcs).encode(),
            b"n_pix": str(n_pix).encode(),
        }
        if not per_source:
            meta[b"wavelength"] = json.dumps(plan.wavelength.tolist()).encode()  # type: ignore[union-attr]
        if flux_scale is not None and flux_scale != 1.0:
            meta[b"flux_scale"] = str(flux_scale).encode()
            meta[b"flux_calibrated"] = b"true"

        fields = [
            ("source_id", pa.int64()),
            ("redshift", pa.float32()),
            ("flux", flt_vec),
            ("ivar", flt_vec),
            ("mask", mask_vec),
        ]
        if per_source:
            fields.append(("wavelength", flt_vec))
        schema = pa.schema(fields).with_metadata(meta)

        id_to_row: dict[int, int] = {}
        row_offset = 0
        writer = pq.ParquetWriter(
            str(output_parquet), schema, compression="zstd", compression_level=3,
        )
        try:
            for sorted_sids, flux_batch, ivar_batch, mask_batch, wave_batch, z_batch in self._iter_subset_tile_batches(
                plan, show_progress=show_progress, z_map=z_map, flux_scale=flux_scale,
            ):
                arrays = [
                    pa.array(sorted_sids, type=pa.int64()),
                    pa.array(z_batch, type=pa.float32()),
                    pa.array(flux_batch.tolist(), type=flt_vec),
                    pa.array(ivar_batch.tolist(), type=flt_vec),
                    pa.array(mask_batch.tolist(), type=mask_vec),
                ]
                if per_source and wave_batch is not None:
                    arrays.append(pa.array(wave_batch.tolist(), type=flt_vec))
                table = pa.Table.from_arrays(arrays, schema=schema)
                writer.write_table(table)
                for j, sid in enumerate(sorted_sids.tolist()):
                    id_to_row[int(sid)] = row_offset + j
                row_offset += len(sorted_sids)
        finally:
            writer.close()

        log.info(
            "Extracted %d/%d spectra → %s (Parquet)",
            plan.n_written, plan.n_requested, output_parquet,
        )
        return {
            "n_requested": plan.n_requested,
            "n_written": plan.n_written,
            "missing_ids": plan.missing_ids,
            "id_to_row": id_to_row,
            "output": str(output_parquet),
            "output_parquet": str(output_parquet),
            "format": "parquet",
            "flux_scale": flux_scale,
        }

    def extract_subset_to_hdf5(
        self,
        source_ids: Sequence[int] | np.ndarray,
        output_hdf5: Path | str,
        *,
        missing: str = "skip",
        show_progress: bool = True,
        overwrite: bool = False,
        compression: str | None = "gzip",
        flux_scale: float | None = None,
    ) -> dict[str, Any]:
        """Extract a subset into one HDF5 file (stacked arrays, shared wavelength).

        Output layout (single file, root-level datasets)::

            flux         (N_written, N_pix)  float32
            ivar         (N_written, N_pix)  float32
            mask         (N_written, N_pix)  uint8/uint16
            wavelength   (N_pix,)            float64
            source_id    (N_written,)        int64
            redshift     (N_written,)        float32

        Root attributes mirror the flat Zarr subset (survey name, WCS JSON, etc.).
        """
        import h5py

        output_hdf5 = Path(output_hdf5)
        if output_hdf5.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{output_hdf5} already exists. Pass overwrite=True to replace it."
                )
            output_hdf5.unlink()

        plan = self._plan_subset_extraction(
            source_ids, missing=missing, show_progress=show_progress,
        )
        z_map = self._build_catalog_redshift_map(plan)
        n_written = plan.n_written
        n_pix = plan.n_pix
        per_source = plan.wavelength_mode == "per_source"
        dset_kw: dict[str, Any] = {}
        if compression:
            dset_kw = {
                "compression": compression,
                "compression_opts": 4,
                "shuffle": True,
            }

        id_to_row: dict[int, int] = {}
        with h5py.File(output_hdf5, "w") as f:
            f.attrs["source_survey"] = self.survey_name
            f.attrs["source_lake_root"] = str(self.lake_root)
            f.attrs["wavelength_mode"] = plan.wavelength_mode
            f.attrs["n_sources"] = n_written
            f.attrs["n_pix"] = n_pix
            f.attrs["n_requested"] = plan.n_requested
            f.attrs["n_missing"] = len(plan.missing_ids)
            f.attrs["extract_created_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
            )
            f.attrs["schema_version"] = "1"
            f.attrs["wcs_attrs"] = json.dumps(plan.src_wcs)
            if flux_scale is not None and flux_scale != 1.0:
                f.attrs["flux_scale"] = flux_scale
                f.attrs["flux_calibrated"] = True

            if per_source:
                wave_ds = f.create_dataset(
                    "wavelength", shape=(n_written, n_pix), dtype=np.float32, **dset_kw,
                )
            else:
                f.create_dataset("wavelength", data=plan.wavelength, dtype=np.float64)
                wave_ds = None

            flux_ds = f.create_dataset(
                "flux", shape=(n_written, n_pix), dtype=plan.flux_dtype, **dset_kw,
            )
            ivar_ds = f.create_dataset(
                "ivar", shape=(n_written, n_pix), dtype=plan.ivar_dtype, **dset_kw,
            )
            mask_ds = f.create_dataset(
                "mask", shape=(n_written, n_pix), dtype=plan.mask_dtype, **dset_kw,
            )
            from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN

            sid_ds = f.create_dataset(
                LAKE_JOIN_ID_COLUMN, shape=(n_written,), dtype=np.int64,
            )
            z_ds = f.create_dataset("redshift", shape=(n_written,), dtype=np.float32)

            write_offset = 0
            for sorted_sids, flux_batch, ivar_batch, mask_batch, wave_batch, z_batch in (
                self._iter_subset_tile_batches(
                    plan, show_progress=show_progress, z_map=z_map, flux_scale=flux_scale,
                )
            ):
                k = len(sorted_sids)
                slc = slice(write_offset, write_offset + k)
                flux_ds[slc, :] = flux_batch
                ivar_ds[slc, :] = ivar_batch
                mask_ds[slc, :] = mask_batch
                if per_source and wave_ds is not None and wave_batch is not None:
                    wave_ds[slc, :] = wave_batch
                sid_ds[slc] = sorted_sids
                z_ds[slc] = z_batch
                for j, sid in enumerate(sorted_sids.tolist()):
                    id_to_row[int(sid)] = write_offset + j
                write_offset += k

        log.info(
            "Extracted %d/%d spectra → %s (HDF5)",
            n_written, plan.n_requested, output_hdf5,
        )
        return {
            "n_requested": plan.n_requested,
            "n_written": n_written,
            "missing_ids": plan.missing_ids,
            "id_to_row": id_to_row,
            "output": str(output_hdf5),
            "output_hdf5": str(output_hdf5),
            "format": "hdf5",
            "flux_scale": flux_scale,
        }

    def extract_subset_to_fits(
        self,
        source_ids: Sequence[int] | np.ndarray,
        output: Path | str,
        *,
        missing: str = "skip",
        show_progress: bool = True,
        overwrite: bool = False,
        filename_template: str = "spec_{source_id}.fits",
        layout: FitsLayout = "per-file",
        flux_scale: float | None = None,
    ) -> dict[str, Any]:
        """Extract a subset to FITS (one file per spectrum or one catalog file)."""
        if layout == "catalog":
            return self.extract_subset_to_fits_catalog(
                source_ids,
                output,
                missing=missing,
                show_progress=show_progress,
                overwrite=overwrite,
                flux_scale=flux_scale,
            )
        return self._extract_subset_to_fits_per_file(
            source_ids,
            output,
            missing=missing,
            show_progress=show_progress,
            overwrite=overwrite,
            filename_template=filename_template,
            flux_scale=flux_scale,
        )

    def extract_subset_to_fits_catalog(
        self,
        source_ids: Sequence[int] | np.ndarray,
        output_fits: Path | str,
        *,
        missing: str = "skip",
        show_progress: bool = True,
        overwrite: bool = False,
        flux_scale: float | None = None,
    ) -> dict[str, Any]:
        """Extract a subset into one multi-row FITS catalog (BINTABLE + WAVELENGTH HDU).

        Requires ``wavelength_mode='shared'``; use ``--fits-layout per-file`` for
        per-source wavelength surveys.
        """
        from data_lake.export.to_spectrum_fits import write_spectra_catalog_fits

        output_fits = Path(output_fits)
        plan = self._plan_subset_extraction(
            source_ids, missing=missing, show_progress=show_progress,
        )
        if plan.wavelength_mode != "shared":
            raise ValueError(
                f"FITS catalog layout requires wavelength_mode='shared' "
                f"(survey {self.survey_name!r} uses {plan.wavelength_mode!r}). "
                "Use --fits-layout per-file instead."
            )
        z_map = self._build_catalog_redshift_map(plan)
        source_id, flux, ivar, mask, redshift, id_to_row = self._collect_subset_stacks(
            plan, show_progress=show_progress, z_map=z_map, flux_scale=flux_scale,
        )
        write_spectra_catalog_fits(
            output_fits,
            source_id=source_id,
            flux=flux,
            ivar=ivar,
            mask=mask,
            wavelength=plan.wavelength,  # type: ignore[arg-type]  # shared mode guarantees non-None
            redshift=redshift,
            wcs_attrs=plan.src_wcs,
            survey=self.survey_name,
            overwrite=overwrite,
        )
        log.info(
            "Extracted %d/%d spectra → %s (FITS catalog)",
            plan.n_written, plan.n_requested, output_fits,
        )
        return {
            "n_requested": plan.n_requested,
            "n_written": plan.n_written,
            "missing_ids": plan.missing_ids,
            "id_to_row": id_to_row,
            "output": str(output_fits),
            "output_fits": str(output_fits),
            "fits_layout": "catalog",
            "format": "fits",
            "flux_scale": flux_scale,
        }

    def _extract_subset_to_fits_per_file(
        self,
        source_ids: Sequence[int] | np.ndarray,
        output_dir: Path | str,
        *,
        missing: str = "skip",
        show_progress: bool = True,
        overwrite: bool = False,
        filename_template: str = "spec_{source_id}.fits",
        flux_scale: float | None = None,
    ) -> dict[str, Any]:
        """Extract a subset into one FITS file per spectrum."""
        from astropy.io import fits

        from data_lake.export.to_spectrum_fits import _build_primary_header, _build_table_hdu

        output_dir = Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"{output_dir} is not empty. Pass overwrite=True to write into it."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        plan = self._plan_subset_extraction(
            source_ids, missing=missing, show_progress=show_progress,
        )
        z_map = self._build_catalog_redshift_map(plan)
        shared_wave = plan.wavelength  # None for per_source surveys
        id_to_row: dict[int, int] = {}
        paths: list[str] = []
        row_offset = 0

        for sorted_sids, flux_batch, ivar_batch, mask_batch, wave_batch, z_batch in self._iter_subset_tile_batches(
            plan, show_progress=show_progress, z_map=z_map, flux_scale=flux_scale,
        ):
            for j, sid in enumerate(sorted_sids.tolist()):
                meta = {"z": float(z_batch[j])}
                row_wave = (
                    np.asarray(wave_batch[j], dtype=np.float64)
                    if wave_batch is not None
                    else shared_wave
                )
                sp = Spectrum(
                    source_id=int(sid),
                    flux=np.asarray(flux_batch[j], dtype=np.float32),
                    ivar=np.asarray(ivar_batch[j], dtype=np.float32),
                    mask=np.asarray(mask_batch[j], dtype=plan.mask_dtype),
                    wavelength=row_wave,
                    meta=meta,
                    wcs_attrs=plan.src_wcs,
                )
                out_path = output_dir / filename_template.format(source_id=sid)
                if out_path.exists() and not overwrite:
                    raise FileExistsError(f"{out_path} already exists.")
                hdul = fits.HDUList([
                    fits.PrimaryHDU(
                        data=sp.flux,
                        header=_build_primary_header(sp, self.survey_name),
                    ),
                    _build_table_hdu(sp),
                ])
                hdul.writeto(str(out_path), overwrite=overwrite)
                paths.append(str(out_path))
                id_to_row[int(sid)] = row_offset + j
            row_offset += len(sorted_sids)

        log.info(
            "Extracted %d/%d spectra → %s (%d FITS files)",
            plan.n_written, plan.n_requested, output_dir, len(paths),
        )
        return {
            "n_requested": plan.n_requested,
            "n_written": plan.n_written,
            "missing_ids": plan.missing_ids,
            "id_to_row": id_to_row,
            "output": str(output_dir),
            "output_dir": str(output_dir),
            "fits_paths": paths,
            "fits_layout": "per-file",
            "format": "fits",
            "flux_scale": flux_scale,
        }

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
