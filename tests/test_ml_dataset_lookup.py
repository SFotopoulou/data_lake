"""ML datasets resolve source IDs once at construction, never per sample."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_lake.ingest.fits_to_parquet import healpix_dir
from data_lake.io.catalog import CatalogAccessor
from data_lake.io.cutouts import CutoutAccessor
from data_lake.io.spectra import SpectrumAccessor
from data_lake.ml.dataset import CutoutDataset, TileSequentialSampler
from data_lake.ml.spectrum_dataset import SpectrumDataset

from synthetic_lake_helpers import NORDER, SOURCES, SURVEY, ingest_synthetic_spectrum_lake
from test_cutout_lookup import _make_cutout_tile


def _write_index_catalog(lake: Path, lookup: dict[int, tuple[int, int]], kind: str) -> None:
    rows = [
        {
            "_source_id": sid,
            f"_healpix_norder{NORDER}": npix,
            f"_{kind}_index": lidx,
            f"_{kind}_npix": npix,
        }
        for sid, (npix, lidx) in lookup.items()
    ]
    table = pa.Table.from_pylist(rows)
    root = lake / "catalogs" / SURVEY
    for npix in sorted({r[f"_healpix_norder{NORDER}"] for r in rows}):
        sub = table.filter(pa.compute.equal(table[f"_healpix_norder{NORDER}"], npix))
        out = root / healpix_dir(NORDER, npix)
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(sub, out / f"Npix={npix}.parquet")
    (root / "catalog_info.json").write_text(
        f'{{"hats_order": {NORDER}, "link_id_column": "_source_id"}}'
    )


@pytest.fixture
def spectrum_lake(tmp_path: Path) -> Path:
    ingest_synthetic_spectrum_lake(tmp_path)
    return tmp_path


@pytest.fixture
def cutout_lake(tmp_path: Path) -> Path:
    _make_cutout_tile(tmp_path, 100, [501, 502])
    _make_cutout_tile(tmp_path, 200, [601])
    return tmp_path


class TestLocate:
    def test_spectrum_locate_marks_missing(self, spectrum_lake: Path) -> None:
        with SpectrumAccessor(spectrum_lake, SURVEY) as acc:
            npix, local = acc.locate([101, 999, 201])
            assert npix[1] == -1 and local[1] == -1
            sp = acc.get_spectrum_at(npix[0], local[0])
            assert sp.source_id == 101
            np.testing.assert_array_equal(sp.flux, acc.get_spectrum(101).flux)

    def test_cutout_locate_and_read(self, cutout_lake: Path) -> None:
        with CutoutAccessor(cutout_lake, SURVEY) as acc:
            npix, local = acc.locate([601, 502])
            assert npix.tolist() == [200, 100]
            np.testing.assert_array_equal(acc.get_image_at(npix[1], local[1]), acc.get_image(502))


class TestSpectrumDataset:
    def test_items_match_accessor(self, spectrum_lake: Path) -> None:
        ids = [s[0] for s in SOURCES]
        ds = SpectrumDataset(spectrum_lake, SURVEY, ids, norder=NORDER, return_full=True)
        with SpectrumAccessor(spectrum_lake, SURVEY) as acc:
            for i, sid in enumerate(ids):
                assert ds[i].source_id == sid
                np.testing.assert_array_equal(ds[i].flux, acc.get_spectrum(sid).flux)

    def test_missing_id_raises_on_access(self, spectrum_lake: Path) -> None:
        ds = SpectrumDataset(spectrum_lake, SURVEY, [101, 999], norder=NORDER)
        with pytest.raises(KeyError, match="999"):
            ds[1]

    def test_catalog_used_once_and_not_kept(self, spectrum_lake: Path) -> None:
        ids = [s[0] for s in SOURCES]
        with SpectrumAccessor(spectrum_lake, SURVEY) as acc:
            npix, local = acc.locate(ids)
        _write_index_catalog(
            spectrum_lake, dict(zip(ids, zip(npix.tolist(), local.tolist()))), "spectrum",
        )
        cat = CatalogAccessor(spectrum_lake, SURVEY, norder=NORDER)
        with patch.object(cat, "get_tile_index", wraps=cat.get_tile_index) as spy:
            ds = SpectrumDataset(
                spectrum_lake, SURVEY, ids, norder=NORDER, catalog_accessor=cat,
            )
            for i in range(len(ds)):
                ds[i]
        assert spy.call_count == 0
        cat.close()
        pickle.dumps(ds)


class TestCutoutDataset:
    def test_tile_sort_and_sampler_runs(self, cutout_lake: Path) -> None:
        ds = CutoutDataset(cutout_lake, SURVEY, [601, 501, 502], norder=NORDER)
        assert ds.tile_sorted_indices == [1, 2, 0]
        sampler = TileSequentialSampler(ds, shuffle_tiles=False)
        assert sorted(map(sorted, sampler._tile_runs)) == [[0], [1, 2]]
        assert sorted(sampler) == [0, 1, 2]

    def test_items_match_accessor(self, cutout_lake: Path) -> None:
        ds = CutoutDataset(cutout_lake, SURVEY, [601, 502], norder=NORDER)
        with CutoutAccessor(cutout_lake, SURVEY) as acc:
            np.testing.assert_array_equal(np.asarray(ds[1]), acc.get_image(502))
            np.testing.assert_array_equal(np.asarray(ds[0]), acc.get_image(601))

    def test_catalog_lookup_without_per_sample_queries(self, cutout_lake: Path) -> None:
        ids = [501, 502, 601]
        with CutoutAccessor(cutout_lake, SURVEY) as acc:
            npix, local = acc.locate(ids)
        _write_index_catalog(
            cutout_lake, dict(zip(ids, zip(npix.tolist(), local.tolist()))), "cutout",
        )
        cat = CatalogAccessor(cutout_lake, SURVEY, norder=NORDER)
        with patch.object(cat, "get_tile_index", wraps=cat.get_tile_index) as spy:
            ds = CutoutDataset(cutout_lake, SURVEY, ids, norder=NORDER, catalog_accessor=cat)
            for i in range(len(ds)):
                ds[i]
        assert spy.call_count == 0
        assert ds.tile_npix.tolist() == npix.tolist()
        cat.close()
