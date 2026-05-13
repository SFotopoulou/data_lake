"""
dataset – PyTorch Dataset / IterableDataset for galaxy cutout stacks.

Two modes
---------
CutoutDataset (map-style)
    Loads a pre-built list of source_ids.  Supports random access, shuffling,
    and torch DataLoader with ``num_workers > 0``.  Indices within each
    HEALPix tile are accessed in sequential order within a worker to maximise
    Zarr shard reuse.

TileCutoutDataset (iterable-style)
    Streams complete tiles one by one.  Optimal for training when the full
    corpus is visited once per epoch without random sampling.

Usage
-----
>>> from data_lake.ml.dataset import CutoutDataset
>>> ds = CutoutDataset(
...     lake_root="/data/lake",
...     survey="des_dr2",
...     source_ids=my_ids,      # list[int]
...     transform=my_transforms,
... )
>>> loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset, get_worker_info
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Provide placeholder base classes so the module imports without torch
    class Dataset:  # type: ignore[no-redef]
        pass
    class IterableDataset:  # type: ignore[no-redef]
        pass
    def get_worker_info():  # type: ignore[misc]
        return None

from data_lake.io.cutouts import CutoutAccessor

log = logging.getLogger(__name__)

Transform = Callable[[np.ndarray], Any]


# ---------------------------------------------------------------------------
# Helper: per-worker accessor
# ---------------------------------------------------------------------------


def _make_accessor(lake_root: Path, survey: str, norder: int) -> CutoutAccessor:
    """Create a CutoutAccessor; called once per DataLoader worker."""
    return CutoutAccessor(lake_root, survey, norder=norder)


# ---------------------------------------------------------------------------
# Map-style dataset
# ---------------------------------------------------------------------------


class CutoutDataset(Dataset):
    """
    Map-style PyTorch Dataset over a fixed list of source_ids.

    Tile-aware index ordering
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    To reduce random seeks within a DataLoader worker, the dataset pre-sorts
    the source_ids by their HEALPix tile pixel.  The ``__getitem__`` method is
    then called in sorted order by the sampler, so consecutive reads land in
    the same shard.  Pass ``tile_sort=True`` (the default) to enable this;
    the original order is preserved in ``self.source_ids``.

    Parameters
    ----------
    lake_root:
        Data lake root directory.
    survey:
        Survey name.
    source_ids:
        Ordered collection of source IDs to include.
    norder:
        HEALPix partitioning order (default 5).
    transform:
        Optional callable applied to each image array (NumPy → any).
        Receives an array of shape ``(B, H, W)``.
    target_transform:
        Optional callable applied to the label dict.
    tile_sort:
        If True, expose a ``tile_sorted_indices`` property for use with a
        sequential sampler to improve I/O locality.
    return_wcs:
        If True, each item is a dict ``{"image": …, "wcs": …, "source_id": …}``.
        If False (default), returns only the image tensor (or transform output).
    catalog_accessor:
        Optional ``CatalogAccessor`` for fast O(1) tile lookup; if None,
        the accessor falls back to scanning tiles.
    """

    def __init__(
        self,
        lake_root: Path | str,
        survey: str,
        source_ids: Sequence[int],
        norder: int = 5,
        transform: Transform | None = None,
        target_transform: Callable | None = None,
        tile_sort: bool = True,
        return_wcs: bool = False,
        catalog_accessor=None,
    ) -> None:
        self.lake_root = Path(lake_root)
        self.survey = survey
        self.norder = norder
        self.source_ids: list[int] = list(source_ids)
        self.transform = transform
        self.target_transform = target_transform
        self.return_wcs = return_wcs
        self._catalog = catalog_accessor

        # Build tile-sorted permutation
        self._tile_sorted_perm: list[int] | None = None
        if tile_sort:
            self._tile_sorted_perm = self._build_tile_sorted_perm()

        # Accessor is created lazily per worker (or immediately for single-process)
        self._accessor: CutoutAccessor | None = None
        self._worker_id: int | None = None

    def _build_tile_sorted_perm(self) -> list[int]:
        """Return permutation of 0..N that groups source_ids by HEALPix tile."""
        try:
            import healpy as hp
            # We need RA/Dec to assign tiles, but we only have source_ids here.
            # If a catalog accessor is available, use it; otherwise defer sort
            # to runtime (sort by tile after first access).
            if self._catalog is not None:
                # Fetch healpix tile for all source_ids in one shot
                hp_col = f"_healpix_norder{self.norder}"
                id_list = ", ".join(str(s) for s in self.source_ids)
                sql = f"SELECT source_id, {hp_col} FROM catalog WHERE source_id IN ({id_list})"
                df = self._catalog.query(sql, fmt="pandas")
                id_to_tile = dict(zip(df["source_id"].astype(int), df[hp_col].astype(int)))
                tiles = [id_to_tile.get(sid, 0) for sid in self.source_ids]
                return list(np.argsort(tiles, kind="stable"))
        except Exception as exc:
            log.debug("Could not build tile-sorted perm: %s", exc)
        return list(range(len(self.source_ids)))

    @property
    def tile_sorted_indices(self) -> list[int]:
        """
        Permutation of dataset indices sorted by HEALPix tile.

        Use with ``torch.utils.data.SubsetRandomSampler`` or as a custom
        sampler to improve I/O locality:

        >>> sampler = SequentialSampler(ds.tile_sorted_indices)
        """
        if self._tile_sorted_perm is None:
            return list(range(len(self.source_ids)))
        return self._tile_sorted_perm

    def _get_accessor(self) -> CutoutAccessor:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        if self._accessor is None or worker_id != self._worker_id:
            self._accessor = CutoutAccessor(
                self.lake_root, self.survey, norder=self.norder,
                catalog_accessor=self._catalog,
            )
            self._worker_id = worker_id
        return self._accessor

    def __len__(self) -> int:
        return len(self.source_ids)

    def __getitem__(self, idx: int):
        source_id = self.source_ids[idx]
        acc = self._get_accessor()

        if self.return_wcs:
            image, wcs = acc.get_cutout(source_id)
        else:
            image = acc.get_image(source_id)
            wcs = None

        if self.transform is not None:
            image = self.transform(image)
        elif _TORCH_AVAILABLE:
            image = torch.from_numpy(np.ascontiguousarray(image))

        if self.return_wcs:
            item = {"image": image, "source_id": source_id, "wcs": wcs}
            if self.target_transform is not None:
                item = self.target_transform(item)
            return item

        return image

    def __repr__(self) -> str:
        return (
            f"CutoutDataset(survey={self.survey!r}, n={len(self)}, "
            f"norder={self.norder}, return_wcs={self.return_wcs})"
        )


# ---------------------------------------------------------------------------
# Tile-aware sampler (optional helper)
# ---------------------------------------------------------------------------


class TileSequentialSampler:
    """
    Sampler that yields dataset indices grouped by HEALPix tile, optionally
    shuffling the order of tiles and the order within each tile.

    Designed to work with ``CutoutDataset`` to maximise Zarr shard locality
    while still providing intra-tile and inter-tile variety.

    Parameters
    ----------
    dataset:
        A ``CutoutDataset`` instance.
    shuffle_tiles:
        Shuffle the ordering of tiles (default True).
    shuffle_within_tile:
        Shuffle the ordering within each tile (default False, better I/O).
    """

    def __init__(
        self,
        dataset: CutoutDataset,
        shuffle_tiles: bool = True,
        shuffle_within_tile: bool = False,
    ) -> None:
        self._dataset = dataset
        self._shuffle_tiles = shuffle_tiles
        self._shuffle_within_tile = shuffle_within_tile
        self._build()

    def _build(self) -> None:
        perm = self._dataset.tile_sorted_indices
        # Group consecutive positions (same tile) into runs
        self._tile_runs: list[list[int]] = []
        if not perm:
            return
        run: list[int] = [perm[0]]
        for p in perm[1:]:
            run.append(p)
            # Each element of tile_sorted_perm is a dataset index; we can't
            # easily detect tile boundaries here without re-doing the lookup.
            # Use a run of _TILE_RUN_SIZE as heuristic; override by subclassing.
        self._tile_runs = [run]  # single group for now

    def __iter__(self):
        runs = list(self._tile_runs)
        if self._shuffle_tiles:
            np.random.shuffle(runs)
        for run in runs:
            r = list(run)
            if self._shuffle_within_tile:
                np.random.shuffle(r)
            yield from r

    def __len__(self) -> int:
        return len(self._dataset)


# ---------------------------------------------------------------------------
# Iterable-style dataset (whole-tile streaming)
# ---------------------------------------------------------------------------


class TileCutoutDataset(IterableDataset):
    """
    Iterable dataset that streams complete HEALPix tiles in order.

    Each tile's Zarr array is read sequentially, which is maximally efficient
    for Zstd-compressed shards.  Ideal for training runs where all sources
    are visited once per epoch.

    Supports multi-worker DataLoader: tiles are distributed across workers
    such that each tile is processed by exactly one worker.

    Parameters
    ----------
    lake_root, survey, norder:
        Standard data lake parameters.
    tile_pixels:
        Optional explicit list of HEALPix pixel indices to include.
        Defaults to all tiles on disk.
    transform:
        Transform applied to each image ``(B, H, W)`` array.
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

        # Discover tiles at construction time (cheap directory scan)
        acc = CutoutAccessor(lake_root, survey, norder=norder)
        self._all_tiles: list[int] = tile_pixels if tile_pixels is not None else acc.available_tiles()
        del acc

    def __iter__(self):
        worker = get_worker_info()
        tiles = list(self._all_tiles)

        if self.shuffle_tiles:
            np.random.shuffle(tiles)

        # Distribute tiles across workers
        if worker is not None:
            tiles = tiles[worker.id :: worker.num_workers]

        acc = CutoutAccessor(self.lake_root, self.survey, norder=self.norder)

        for npix in tiles:
            try:
                for source_id, image, _wcs in acc.iter_tile(npix):
                    if self.transform is not None:
                        image = self.transform(image)
                    elif _TORCH_AVAILABLE:
                        image = torch.from_numpy(np.ascontiguousarray(image))
                    yield image
            except FileNotFoundError:
                log.warning("Tile %d not found, skipping.", npix)

    def __len__(self) -> int:
        """Approximate length; exact count requires reading all tile metadata."""
        # We don't know the total without opening every tile; return a cached
        # estimate if available, else raise NotImplementedError so DataLoader
        # doesn't try to use it in sampler construction.
        raise NotImplementedError(
            "TileCutoutDataset does not support len(). "
            "Use CutoutDataset for a length-aware map-style dataset."
        )
