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
from collections import OrderedDict
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

    def close(self) -> None:
        """Release Zarr file handles (idempotent)."""
        if self._root is not None:
            from data_lake.io.zarr_close import close_zarr_group

            close_zarr_group(self._root)
            self._root = None

    def get_image(self, idx: int) -> np.ndarray:
        """Return image at position idx; shape = (B, H, W)."""
        root = self._open()
        return np.array(root["images"][idx])

    def get_images(self, indices: np.ndarray) -> np.ndarray:
        """Return images for multiple indices; shape = (N, B, H, W)."""
        from data_lake.io.zarr_batch import read_zarr_rows

        root = self._open()
        return read_zarr_rows(root["images"], indices)

    def get_wcs(self, idx: int) -> CutoutWCS:
        root = self._open()
        raw = root["wcs"][idx]
        return _decode_wcs(raw)

    def get_source_ids(self) -> np.ndarray:
        root = self._open()
        from data_lake.ingest.zarr_ids import zarr_join_array

        return np.array(zarr_join_array(root))

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
        max_open_tiles: int = 8,
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
        self._max_open_tiles: int = max_open_tiles

        self._tile_stores: OrderedDict[int, TileStore] = OrderedDict()
        self._tile_indices: dict[int, dict[int, int]] = {}  # npix → {source_id: local_idx}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tile_path(self, npix: int) -> Path:
        return self._cutout_root / healpix_dir(self.norder, npix) / f"Npix={npix}.zarr"

    def _get_tile_store(self, npix: int) -> TileStore:
        if npix in self._tile_stores:
            self._tile_stores.move_to_end(npix)
            return self._tile_stores[npix]
        path = self._tile_path(npix)
        if not path.exists():
            raise FileNotFoundError(f"Tile not found: {path}")
        store = TileStore(path)
        self._tile_stores[npix] = store
        if self._max_open_tiles > 0:
            while len(self._tile_stores) > self._max_open_tiles:
                _, evicted = self._tile_stores.popitem(last=False)
                evicted.close()
        return store

    def close(self) -> None:
        """Close all cached tile stores and clear caches."""
        for store in self._tile_stores.values():
            store.close()
        self._tile_stores.clear()
        self._tile_indices.clear()

    def __enter__(self) -> "CutoutAccessor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get_tile_index(self, npix: int) -> dict[int, int]:
        if npix not in self._tile_indices:
            store = self._get_tile_store(npix)
            self._tile_indices[npix] = store.build_index()
        return self._tile_indices[npix]

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
            kind="cutout",
            scan_tile=self._scan_tile_for_ids,
            available_tiles=self.available_tiles,
            show_progress=show_progress,
        )

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

        lookup = self._build_source_id_lookup(source_ids, show_progress=False)
        tile_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for orig_pos, sid in enumerate(source_ids):
            loc = lookup.get(int(sid))
            if loc is None:
                raise KeyError(f"source_id={sid} not found in any tile.")
            npix, local_idx = loc
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

    # ------------------------------------------------------------------
    # Subset extraction
    # ------------------------------------------------------------------

    def _iter_subset_batches(
        self,
        sid_to_loc: "dict[int, tuple[int, int]]",
        show_progress: bool,
    ):
        """Yield ``(sorted_sids, images, wcs_raw_list)`` per tile.

        Images have shape ``(k, B, H, W)``; *wcs_raw_list* is a list of
        raw structured-array entries (one per source) that can be decoded
        with :func:`data_lake.io.cutouts._decode_wcs`.
        """
        from collections import defaultdict

        tile_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for sid, (npix, local_idx) in sid_to_loc.items():
            tile_groups[npix].append((sid, local_idx))

        items = list(tile_groups.items())
        if show_progress:
            try:
                from tqdm.auto import tqdm
                items = tqdm(items, desc="cutout tiles", unit="tile")
            except ImportError:
                pass

        for npix, pairs in items:
            sids = np.array([p[0] for p in pairs], dtype=np.int64)
            local_idxs = np.array([p[1] for p in pairs], dtype=np.int64)
            store = self._get_tile_store(npix)
            images = store.get_images(local_idxs)
            root = store._open()
            wcs_raw = [root["wcs"][int(i)] for i in local_idxs]
            yield sids, images, wcs_raw

    def extract_subset_to_zarr(
        self,
        source_ids: "list[int]",
        output_zarr: "Path | str",
        *,
        missing: str = "skip",
        show_progress: bool = True,
        overwrite: bool = False,
    ) -> dict:
        """Extract a subset of cutouts into a self-contained flat Zarr group.

        Output layout
        -------------
        ::

            output_zarr/
              images/      (N, B, H, W)  float32    sharded
              wcs/         (N,)           bytes      WCS structured array (preserve)
              _source_id/  (N,)           int64

        Group attributes mirror those of the source tiles plus
        ``source_survey``, ``source_lake_root``, ``n_sources``,
        ``extract_created_utc``.

        Returns
        -------
        dict with ``n_requested``, ``n_written``, ``missing_ids``,
        ``id_to_row``, ``output_zarr``.
        """
        import shutil
        import time

        import zarr
        import zarr.codecs

        from data_lake.ingest.fits_to_zarr import _WCS_DTYPE
        from data_lake.ingest.zarr_ids import create_zarr_join_array

        output_zarr = Path(output_zarr)
        if output_zarr.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{output_zarr} already exists. Pass overwrite=True to replace it."
                )
            shutil.rmtree(output_zarr)

        unique_ids = list(dict.fromkeys(int(s) for s in source_ids))
        n_requested = len(unique_ids)

        sid_to_loc = self._build_source_id_lookup(unique_ids, show_progress=show_progress)
        found_ids = set(sid_to_loc)
        missing_ids = [s for s in unique_ids if s not in found_ids]

        if missing_ids and missing == "error":
            raise KeyError(
                f"{len(missing_ids)} source_id(s) not found in {self.survey_name}: "
                f"{missing_ids[:10]}{'…' if len(missing_ids) > 10 else ''}"
            )

        sid_to_loc = {s: v for s, v in sid_to_loc.items() if s in found_ids}

        # Determine image geometry from the first tile
        sample_npix = next(iter(sid_to_loc.values()))[0]
        sample_store = self._get_tile_store(sample_npix)
        sample_root = sample_store._open()
        _, n_bands, h, w = sample_root["images"].shape
        band_names: list[str] = list(sample_root.attrs.get("band_names", []))

        # Create output Zarr
        store = zarr.storage.LocalStore(str(output_zarr))
        out = zarr.open_group(store=store, mode="w", zarr_format=3)

        blosc = zarr.codecs.BloscCodec(
            cname="zstd", clevel=3,
            shuffle=zarr.codecs.BloscShuffle.bitshuffle,
        )
        n_out = len(sid_to_loc)
        chunk_img = (1, n_bands, h, w)
        shard_img = (min(512, n_out), n_bands, h, w)

        images_arr = out.create_array(
            "images", shape=(n_out, n_bands, h, w),
            chunks=chunk_img, shards=shard_img,
            dtype=np.float32, compressors=blosc, fill_value=np.nan,
        )
        create_zarr_join_array(out, shape=(n_out,), chunks=(4096,), dtype=np.int64, fill_value=-1)
        wcs_arr = out.create_array(
            "wcs", shape=(n_out,), chunks=(512,),
            dtype="|V" + str(_WCS_DTYPE.itemsize),
            fill_value=b"\x00" * _WCS_DTYPE.itemsize,
        )

        id_to_row: dict[int, int] = {}
        row = 0
        for sids_batch, images_batch, wcs_raw_batch in self._iter_subset_batches(
            sid_to_loc, show_progress=show_progress
        ):
            k = len(sids_batch)
            slc = slice(row, row + k)
            images_arr[slc] = images_batch
            out["_source_id"][slc] = sids_batch
            for j, raw in enumerate(wcs_raw_batch):
                wcs_arr[row + j] = raw
            for j, sid in enumerate(sids_batch):
                id_to_row[int(sid)] = row + j
            row += k

        out.attrs.update({
            "source_survey": self.survey_name,
            "source_lake_root": str(self.lake_root),
            "n_sources": n_out,
            "n_requested": n_requested,
            "n_missing": len(missing_ids),
            "n_bands": n_bands,
            "height": h,
            "width": w,
            "band_names": band_names,
            "extract_created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema_version": "1",
        })

        return {
            "n_requested": n_requested,
            "n_written": n_out,
            "missing_ids": missing_ids,
            "id_to_row": id_to_row,
            "output_zarr": str(output_zarr),
        }

    def extract_subset_to_fits(
        self,
        source_ids: "list[int]",
        output_dir: "Path | str",
        *,
        missing: str = "skip",
        show_progress: bool = True,
        overwrite: bool = False,
        filename_template: str = "cutout_{source_id}.fits",
        id_hdu_key: str = "SOURCE_ID",
    ) -> dict:
        """Extract cutouts to per-source FITS files, each with full WCS.

        Each FITS has a PrimaryHDU with shape ``(B, H, W)`` float32 and
        WCS keywords (CTYPE1/2, CRVAL1/2, CRPIX1/2, CD1_1 etc.) populated
        from the stored WCS structured array.

        Returns
        -------
        dict with ``n_written``, ``missing_ids``, ``id_to_path``.
        """
        from astropy.io import fits as apfits

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        unique_ids = list(dict.fromkeys(int(s) for s in source_ids))
        n_requested = len(unique_ids)

        sid_to_loc = self._build_source_id_lookup(unique_ids, show_progress=show_progress)
        found_ids = set(sid_to_loc)
        missing_ids = [s for s in unique_ids if s not in found_ids]

        if missing_ids and missing == "error":
            raise KeyError(
                f"{len(missing_ids)} source_id(s) not found: "
                f"{missing_ids[:10]}{'…' if len(missing_ids) > 10 else ''}"
            )
        sid_to_loc = {s: v for s, v in sid_to_loc.items() if s in found_ids}

        # Band names from the first tile
        sample_store = self._get_tile_store(next(iter(sid_to_loc.values()))[0])
        band_names: list[str] = list(sample_store._open().attrs.get("band_names", []))

        id_to_path: dict[int, str] = {}
        n_written = 0

        for sids_batch, images_batch, wcs_raw_batch in self._iter_subset_batches(
            sid_to_loc, show_progress=show_progress
        ):
            for j, (sid, img, raw) in enumerate(
                zip(sids_batch.tolist(), images_batch, wcs_raw_batch)
            ):
                wcs_obj = _decode_wcs(raw)
                fname = output_dir / filename_template.format(source_id=sid)
                if fname.exists() and not overwrite:
                    log.warning("Skipping existing %s (pass overwrite=True to replace)", fname.name)
                    continue

                hdu = apfits.PrimaryHDU(data=img.astype(np.float32, copy=False))
                hdr = hdu.header
                hdr[id_hdu_key] = int(sid)
                # WCS keywords — preserve full TAN projection
                hdr["CTYPE1"] = "RA---TAN"
                hdr["CTYPE2"] = "DEC--TAN"
                hdr["CRVAL1"] = wcs_obj.crval[0]
                hdr["CRVAL2"] = wcs_obj.crval[1]
                hdr["CRPIX1"] = wcs_obj.crpix[0]
                hdr["CRPIX2"] = wcs_obj.crpix[1]
                cd = wcs_obj.cd_matrix
                hdr["CD1_1"] = cd[0, 0]
                hdr["CD1_2"] = cd[0, 1]
                hdr["CD2_1"] = cd[1, 0]
                hdr["CD2_2"] = cd[1, 1]
                if band_names:
                    hdr["NBANDS"] = len(band_names)
                    hdr["BANDLIST"] = ",".join(band_names)[:68]
                hdu.writeto(str(fname), overwrite=True)
                id_to_path[int(sid)] = str(fname)
                n_written += 1

        return {
            "n_requested": n_requested,
            "n_written": n_written,
            "missing_ids": missing_ids,
            "id_to_path": id_to_path,
        }

    def extract_subset_to_hdf5(
        self,
        source_ids: "list[int]",
        output_hdf5: "Path | str",
        *,
        missing: str = "skip",
        show_progress: bool = True,
        overwrite: bool = False,
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> dict:
        """Extract cutouts to a single HDF5 file.

        Datasets
        --------
        images       (N, B, H, W)  float32
        _source_id   (N,)           int64
        wcs_crval1   (N,)           float64   RA of reference pixel (deg)
        wcs_crval2   (N,)           float64   Dec of reference pixel (deg)
        wcs_crpix1   (N,)           float64
        wcs_crpix2   (N,)           float64
        wcs_cd1_1 …  (N,)           float64   CD matrix elements
        wcs_naxis1   (N,)           int32
        wcs_naxis2   (N,)           int32

        Root attributes: survey, band_names, n_sources, extract_created_utc.
        """
        import time

        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 export: pip install h5py")

        from data_lake.ingest.fits_to_zarr import _WCS_DTYPE

        output_hdf5 = Path(output_hdf5)
        if output_hdf5.exists():
            if not overwrite:
                raise FileExistsError(f"{output_hdf5} already exists.")
            output_hdf5.unlink()

        unique_ids = list(dict.fromkeys(int(s) for s in source_ids))
        n_requested = len(unique_ids)

        sid_to_loc = self._build_source_id_lookup(unique_ids, show_progress=show_progress)
        found_ids = set(sid_to_loc)
        missing_ids = [s for s in unique_ids if s not in found_ids]

        if missing_ids and missing == "error":
            raise KeyError(
                f"{len(missing_ids)} source_id(s) not found: "
                f"{missing_ids[:10]}{'…' if len(missing_ids) > 10 else ''}"
            )
        sid_to_loc = {s: v for s, v in sid_to_loc.items() if s in found_ids}

        sample_store = self._get_tile_store(next(iter(sid_to_loc.values()))[0])
        sample_root = sample_store._open()
        _, n_bands, h, w = sample_root["images"].shape
        band_names: list[str] = list(sample_root.attrs.get("band_names", []))
        n_out = len(sid_to_loc)

        ck = dict(compression=compression, compression_opts=compression_opts)
        wcs_fields = [f for f in _WCS_DTYPE.names]

        id_to_row: dict[int, int] = {}
        row = 0

        output_hdf5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(output_hdf5), "w") as hf:
            ds_img = hf.create_dataset(
                "images", shape=(n_out, n_bands, h, w), dtype=np.float32, **ck
            )
            ds_sid = hf.create_dataset("_source_id", shape=(n_out,), dtype=np.int64, **ck)
            wcs_ds = {
                f: hf.create_dataset(
                    f"wcs_{f}",
                    shape=(n_out,),
                    dtype=_WCS_DTYPE[f],
                    **ck,
                )
                for f in wcs_fields
            }

            for sids_batch, images_batch, wcs_raw_batch in self._iter_subset_batches(
                sid_to_loc, show_progress=show_progress
            ):
                k = len(sids_batch)
                slc = slice(row, row + k)
                ds_img[slc] = images_batch
                ds_sid[slc] = sids_batch
                for j, raw in enumerate(wcs_raw_batch):
                    wcs_obj = _decode_wcs(raw)
                    for field in wcs_fields:
                        wcs_ds[field][row + j] = wcs_obj._p[field]
                for j, sid in enumerate(sids_batch):
                    id_to_row[int(sid)] = row + j
                row += k

            hf.attrs["source_survey"] = self.survey_name
            hf.attrs["source_lake_root"] = str(self.lake_root)
            hf.attrs["n_sources"] = n_out
            hf.attrs["n_bands"] = n_bands
            hf.attrs["band_names"] = band_names
            hf.attrs["extract_created_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )

        return {
            "n_requested": n_requested,
            "n_written": n_out,
            "missing_ids": missing_ids,
            "id_to_row": id_to_row,
            "output_hdf5": str(output_hdf5),
        }
