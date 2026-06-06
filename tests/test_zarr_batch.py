"""Tests for Zarr batch row reads."""

from __future__ import annotations

import numpy as np
import pytest

from data_lake.io.fits_read import contiguous_runs
from data_lake.io.zarr_batch import read_zarr_rows


class TestContiguousRuns:
    def test_single_and_runs(self) -> None:
        assert contiguous_runs(np.array([0, 1, 2, 5, 6])) == [(0, 3), (5, 7)]


class TestReadZarrRows:
    def test_preserves_order(self) -> None:
        data = np.arange(80).reshape(20, 4)
        got = read_zarr_rows(data, [7, 1, 2, 3, 15])
        want = data[[7, 1, 2, 3, 15]]
        np.testing.assert_array_equal(got, want)

    def test_contiguous_slice_equivalent(self) -> None:
        data = np.arange(30).reshape(10, 3)
        indices = [2, 3, 4, 8]
        got = read_zarr_rows(data, indices)
        np.testing.assert_array_equal(got, data[indices])

    def test_empty(self) -> None:
        data = np.zeros((5, 2))
        got = read_zarr_rows(data, [])
        assert got.size == 0

    def test_spectrum_tile_store_integration(self, tmp_path) -> None:
        import zarr

        from data_lake.ingest.fits_to_spectra_zarr import _META_DTYPE, _meta_to_bytes
        from data_lake.ingest.zarr_ids import create_zarr_join_array
        from data_lake.io.spectra import SpectrumTileStore

        path = tmp_path / "tile.zarr"
        root = zarr.open_group(store=zarr.storage.LocalStore(str(path)), mode="w", zarr_format=3)
        n, n_pix = 8, 16
        root.create_array("flux", shape=(n, n_pix), chunks=(1, n_pix), dtype=np.float32)
        root.create_array("ivar", shape=(n, n_pix), chunks=(1, n_pix), dtype=np.float32)
        root.create_array("mask", shape=(n, n_pix), chunks=(1, n_pix), dtype=np.uint8)
        meta_itemsize = _META_DTYPE.itemsize
        root.create_array(
            "meta",
            shape=(n,),
            chunks=(n,),
            dtype=f"|V{meta_itemsize}",
        )
        create_zarr_join_array(root, shape=(n,), chunks=(n,), dtype=np.int64)
        root["flux"][:] = np.arange(n * n_pix, dtype=np.float32).reshape(n, n_pix)
        root["ivar"][:] = 1.0
        root["mask"][:] = 0
        for i in range(n):
            root["meta"][i] = _meta_to_bytes({"z": float(i)})
        root.create_array(
            "wavelength", shape=(n_pix,), chunks=(n_pix,), dtype=np.float64,
        )
        root["wavelength"][:] = np.linspace(3600, 9800, n_pix)
        root.attrs["wavelength_mode"] = "shared"
        root.attrs["wcs"] = {
            "ctype": "WAVE", "crval": 3600.0, "cdelt": 1.0, "crpix": 1.0,
            "unit": "Angstrom", "air_or_vacuum": "vacuum", "n_pix": n_pix,
        }

        store = SpectrumTileStore(path)
        idx = np.array([0, 2, 3, 7, 1])
        flux, ivar, mask, wave, meta, res = store.get_spectra(idx)
        assert flux.shape == (5, n_pix)
        assert wave.ndim == 1
        assert len(meta) == 5
        assert res is None
