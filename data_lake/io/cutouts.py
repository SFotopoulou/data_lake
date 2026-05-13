"""
cutouts – Zarr v3 accessor for galaxy image cutout stacks.

Core API
--------
>>> from data_lake.io.cutouts import CutoutAccessor
>>> acc = CutoutAccessor("/data/lake", "des_dr2")
>>> image, wcs = acc.get_cutout(source_id=12345678)
>>> batch = acc.get_batch([12345678, 87654321, ...])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from data_lake.ingest.fits_to_parquet import healpix_dir
from data_lake.ingest.fits_to_zarr import _WCS_DTYPE

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WCS recovery
# ---------------------------------------------------------------------------


class CutoutWCS:
    """
    Thin container for the WCS scalars stored per cutout.

    Call ``.to_astropy()`` to get a full ``astropy.wcs.WCS`` object.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self._p = params

    @property
    def crval(self) -> tuple[float, float]:
        return self._p["crval1"], self._p["crval2"]

    @property
    def crpix(self) -> tuple[float, float]:
        return self._p["crpix1"], self._p["crpix2"]

    @property
    def cd_matrix(self) -> np.ndarray:
        return np.array([
            [self._p["cd1_1"], self._p["cd1_2"]],
            [self._p["cd2_1"], self._p["cd2_2"]],
        ])

    @property
    def shape(self) -> tuple[int, int]:
        return int(self._p["naxis2"]), int(self._p["naxis1"])

    def to_astropy(self):
        """Convert to an ``astropy.wcs.WCS`` object."""
        from astropy.wcs import WCS
        wcs = WCS(naxis=2)
        wcs.wcs.crval = list(self.crval)
        wcs.wcs.crpix = list(self.crpix)
        wcs.wcs.cd = self.cd_matrix
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return wcs

    def to_fits_header(self):
        """Return an ``astropy.io.fits.Header`` with WCS keywords."""
        return self.to_astropy().to_header()

    def __repr__(self) -> str:
        return f"CutoutWCS(crval={self.crval}, shape={self.shape})"


def _decode_wcs(raw_bytes: bytes | np.ndarray) -> CutoutWCS:
    """Decode a single raw WCS structured array entry."""
    if isinstance(raw_bytes, (bytes, bytearray, np.bytes_)):
        arr = np.frombuffer(raw_bytes, dtype=_WCS_DTYPE)[0]
    else:
        arr = np.frombuffer(bytes(raw_bytes), dtype=_WCS_DTYPE)[0]
    params = {name: arr[name].item() for name in _WCS_DTYPE.names}
    return CutoutWCS(params)


# ---------------------------------------------------------------------------
# Tile-level store
# ---------------------------------------------------------------------------


class TileStore:
    """Lazily opened Zarr v3 group for a single HEALPix tile."""

    def __init__(self, tile_path: Path) -> None:
        self._path = tile_path
        self._root: zarr.Group | None = None

    def _open(self) -> zarr.Group:
        if self._root is None:
            store = zarr.storage.LocalStore(str(self._path))
            self._root = zarr.open_group(store=store, mode="r", zarr_format=3)
        return self._root

    def get_image(self, idx: int) -> np.ndarray:
        """Return image at position idx; shape = (B, H, W)."""
        root = self._open()
        return np.array(root["images"][idx])

    def get_images(self, indices: np.ndarray) -> np.ndarray:
        """Return images for multiple indices; shape = (N, B, H, W)."""
        root = self._open()
        images = root["images"]
        # Sort indices for more sequential reads, then unshuffle
        order = np.argsort(indices)
        sorted_idx = indices[order]
        out_sorted = np.stack([np.array(images[int(i)]) for i in sorted_idx])
        # Restore original order
        unshuffle = np.empty_like(order)
        unshuffle[order] = np.arange(len(order))
        return out_sorted[unshuffle]

    def get_wcs(self, idx: int) -> CutoutWCS:
        root = self._open()
        raw = root["wcs"][idx]
        return _decode_wcs(raw)

    def get_source_ids(self) -> np.ndarray:
        root = self._open()
        return np.array(root["source_id"])

    def build_index(self) -> dict[int, int]:
        """Build source_id → local index mapping for this tile."""
        ids = self.get_source_ids()
        return {int(sid): i for i, sid in enumerate(ids)}

    @property
    def n_sources(self) -> int:
        root = self._open()
        return root["images"].shape[0]

    @property
    def band_names(self) -> list[str]:
        root = self._open()
        return list(root.attrs.get("band_names", []))


# ---------------------------------------------------------------------------
# Survey-level accessor
# ---------------------------------------------------------------------------


class CutoutAccessor:
    """
    Random-access interface to the Zarr v3 cutout stacks for one survey.

    Lookup path
    -----------
    1. ``source_id`` → HEALPix tile pixel (via catalog column or healpy)
    2. tile pixel → ``TileStore``
    3. O(1) in-memory index ``source_id → local position`` per tile

    The per-tile index is built lazily and cached so the first access to a
    tile pays ~O(N_tile) once, subsequent accesses are O(1).

    Parameters
    ----------
    lake_root:
        Data lake root.
    survey_name:
        Survey identifier.
    norder:
        HEALPix partitioning order (read from ``cutout_info.json`` if omitted).
    catalog_accessor:
        Optional ``CatalogAccessor`` used for source_id → tile lookup.
        If provided it is preferred over healpy-based lookup (more accurate
        when sources span tile borders).
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
        self._cutout_root = self.lake_root / "cutouts" / survey_name
        if not self._cutout_root.exists():
            raise FileNotFoundError(f"Cutout store not found: {self._cutout_root}")

        info_path = self._cutout_root / "cutout_info.json"
        self._info: dict = {}
        if info_path.exists():
            import json
            with open(info_path) as fh:
                self._info = json.load(fh)

        self.norder: int = norder if norder is not None else int(self._info.get("hats_order", 5))
        self._catalog = catalog_accessor

        self._tile_stores: dict[int, TileStore] = {}
        self._tile_indices: dict[int, dict[int, int]] = {}  # npix → {source_id: local_idx}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tile_path(self, npix: int) -> Path:
        return self._cutout_root / healpix_dir(self.norder, npix) / f"Npix={npix}.zarr"

    def _get_tile_store(self, npix: int) -> TileStore:
        if npix not in self._tile_stores:
            path = self._tile_path(npix)
            if not path.exists():
                raise FileNotFoundError(f"Tile not found: {path}")
            self._tile_stores[npix] = TileStore(path)
        return self._tile_stores[npix]

    def _get_tile_index(self, npix: int) -> dict[int, int]:
        if npix not in self._tile_indices:
            store = self._get_tile_store(npix)
            self._tile_indices[npix] = store.build_index()
        return self._tile_indices[npix]

    def _source_id_to_tile_and_local(self, source_id: int) -> tuple[int, int]:
        """Return (npix, local_index) for a given source_id."""
        # Fast path: use catalog accessor which has _cutout_index stored
        if self._catalog is not None:
            npix, local_idx = self._catalog.get_tile_index(source_id)
            return npix, local_idx

        # Slow path: search all open tiles, then all on-disk tiles
        for npix, idx_map in self._tile_indices.items():
            if source_id in idx_map:
                return npix, idx_map[source_id]

        # Scan on-disk tiles we haven't loaded yet
        for tile_path in self._cutout_root.rglob("*.zarr"):
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

        raise KeyError(f"source_id={source_id} not found in any tile.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cutout(self, source_id: int) -> tuple[np.ndarray, CutoutWCS]:
        """
        Return ``(image, wcs)`` for a single source.

        Returns
        -------
        image:
            NumPy array of shape ``(B, H, W)``, dtype float32.
        wcs:
            ``CutoutWCS`` with ``.to_astropy()`` for coordinate transforms.
        """
        npix, local_idx = self._source_id_to_tile_and_local(source_id)
        store = self._get_tile_store(npix)
        image = store.get_image(local_idx)
        wcs = store.get_wcs(local_idx)
        return image, wcs

    def get_image(self, source_id: int) -> np.ndarray:
        """Return only the image array ``(B, H, W)`` for a source."""
        npix, local_idx = self._source_id_to_tile_and_local(source_id)
        return self._get_tile_store(npix).get_image(local_idx)

    def get_batch(
        self,
        source_ids: list[int],
        return_wcs: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[CutoutWCS]]:
        """
        Return a batch of images as a single array of shape ``(N, B, H, W)``.

        Reads are grouped per tile to minimise random seeks.

        Parameters
        ----------
        source_ids:
            Ordered list of source IDs.
        return_wcs:
            If True, also return a list of ``CutoutWCS`` objects.
        """
        # Group by tile
        from collections import defaultdict
        tile_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)  # npix → [(orig_pos, local_idx)]
        for orig_pos, sid in enumerate(source_ids):
            npix, local_idx = self._source_id_to_tile_and_local(sid)
            tile_groups[npix].append((orig_pos, local_idx))

        # Allocate output (shape inferred from first tile)
        sample_npix = next(iter(tile_groups))
        sample_store = self._get_tile_store(sample_npix)
        sample_img = sample_store.get_image(tile_groups[sample_npix][0][1])
        out = np.empty((len(source_ids), *sample_img.shape), dtype=sample_img.dtype)
        wcs_list: list[CutoutWCS | None] = [None] * len(source_ids)

        for npix, positions in tile_groups.items():
            store = self._get_tile_store(npix)
            orig_positions = np.array([p[0] for p in positions])
            local_indices = np.array([p[1] for p in positions])
            images = store.get_images(local_indices)
            out[orig_positions] = images
            if return_wcs:
                for i, (orig_pos, local_idx) in enumerate(positions):
                    wcs_list[orig_pos] = store.get_wcs(local_idx)

        if return_wcs:
            return out, wcs_list  # type: ignore[return-value]
        return out

    def iter_tile(self, npix: int):
        """
        Iterate over all (source_id, image, wcs) tuples in a tile.

        Useful for full tile processing / ML dataset building without
        random-access overhead.
        """
        store = self._get_tile_store(npix)
        ids = store.get_source_ids()
        root = store._open()
        images_arr = root["images"]
        for i, sid in enumerate(ids):
            img = np.array(images_arr[i])
            wcs = store.get_wcs(i)
            yield int(sid), img, wcs

    def available_tiles(self) -> list[int]:
        """Return sorted list of HEALPix pixel indices with cutout data."""
        tiles = []
        for p in self._cutout_root.rglob("*.zarr"):
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
            f"CutoutAccessor(survey={self.survey_name!r}, "
            f"norder={self.norder}, root={self.lake_root})"
        )
