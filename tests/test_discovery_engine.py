"""Tests for the discovery engine (resolve_region)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from data_lake.discovery import tile_index as ti
from data_lake.discovery.engine import resolve_region, round_count
from data_lake.discovery.region import Region
from data_lake.ingest.fits_to_parquet import assign_healpix, healpix_dir
from data_lake.ingest.zarr_ids import create_zarr_join_array, zarr_join_array
from data_lake.schema_registry import MODALITY_CROSSMATCH, MODALITY_SPECTRA


def _write_catalog_tile(lake: Path, survey: str, norder: int, npix: int, n_rows: int) -> None:
    tile_dir = lake / "catalogs" / survey / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"x": pa.array(list(range(n_rows)), type=pa.int64())}),
        tile_dir / f"Npix={npix}.parquet",
    )


def _write_info(lake: Path, survey: str, norder: int, total_rows: int) -> None:
    (lake / "catalogs" / survey / "catalog_info.json").write_text(
        json.dumps({"hats_order": norder, "total_rows": total_rows})
    )


def _write_crossmatch_tile(
    lake: Path, tree: str, norder: int, npix: int, n_rows: int
) -> None:
    tile_dir = lake / "crossmatch" / tree / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "source_id_a": pa.array(list(range(n_rows)), type=pa.int64()),
            "source_id_b": pa.array(list(range(n_rows)), type=pa.int64()),
        }),
        tile_dir / f"Npix={npix}.parquet",
    )


def _write_crossmatch_info(lake: Path, tree: str, norder: int, total_rows: int) -> None:
    (lake / "crossmatch" / tree / "crossmatch_info.json").write_text(
        json.dumps({
            "catalog_name": tree,
            "modality": "crossmatch",
            "match_mode": "sky",
            "survey_a": "A",
            "survey_b": "B",
            "match_radius_arcsec": 1.0,
            "hats_order": norder,
            "total_rows": total_rows,
            "n_match_rows": total_rows,
            "schema_version": "1",
        })
    )


def _write_spectra_tile(
    lake: Path, survey: str, norder: int, npix: int, source_ids: list[int]
) -> None:
    """Minimal Zarr tile with ``_source_id`` only (enough for row estimates)."""
    tile_dir = lake / "spectra" / survey / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_path = tile_dir / f"Npix={npix}.zarr"
    root = zarr.open_group(
        store=zarr.storage.LocalStore(str(tile_path)),
        mode="w",
        zarr_format=3,
    )
    create_zarr_join_array(root, shape=(0,), chunks=(4096,), dtype=np.int64, fill_value=-1)
    zarr_join_array(root).append(np.array(source_ids, dtype=np.int64))


def _write_spectrum_info(
    lake: Path,
    survey: str,
    norder: int,
    *,
    total_rows: int | None = None,
    total_spectra: int | None = None,
) -> None:
    info: dict = {"hats_order": norder, "n_pix": 8, "wavelength_mode": "shared"}
    if total_rows is not None:
        info["total_rows"] = total_rows
    if total_spectra is not None:
        info["total_spectra"] = total_spectra
    (lake / "spectra" / survey / "spectrum_info.json").write_text(json.dumps(info))


class TestRoundCount:
    def test_formatting(self) -> None:
        assert round_count(12) == "~12"
        assert round_count(12000) == "~12.0k"
        assert round_count(750_000_000) == "~750M"
        assert round_count(2_000_000_000) == "~2.0G"


class TestResolveRegion:
    def test_cone_overlap_single_tile(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        ra, dec, norder = 120.0, 45.0, 5
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(lake, "ALLWISE", norder, npix, n_rows=100)
        _write_info(lake, "ALLWISE", norder, total_rows=100)
        ti.write_tile_index(lake, "ALLWISE", "catalog")

        region = Region.cone(ra, dec, radius_arcsec=30.0)
        rows = resolve_region(lake, region, surveys="all", modalities=["catalog"])
        assert len(rows) == 1
        row = rows[0]
        assert row.survey == "ALLWISE"
        assert row.n_tiles_overlap == 1
        assert row.est_rows == 100
        assert row.exact_rows is None  # estimate by default

    def test_exact_count_footer_sum(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        ra, dec, norder = 10.0, -10.0, 5
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(lake, "S", norder, npix, n_rows=42)
        _write_info(lake, "S", norder, total_rows=999)  # deliberately wrong total
        ti.write_tile_index(lake, "S", "catalog")

        region = Region.cone(ra, dec, radius_arcsec=30.0)
        rows = resolve_region(lake, region, modalities=["catalog"], count=True)
        assert len(rows) == 1
        assert rows[0].exact_rows == 42  # footer truth, not the estimate

    def test_no_overlap_returns_empty(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        ra, dec, norder = 120.0, 45.0, 5
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_catalog_tile(lake, "S", norder, npix, n_rows=10)
        _write_info(lake, "S", norder, total_rows=10)
        ti.write_tile_index(lake, "S", "catalog")

        # Opposite side of the sky.
        region = Region.cone(300.0, -45.0, radius_arcsec=30.0)
        rows = resolve_region(lake, region, modalities=["catalog"])
        assert rows == []

    def test_explicit_surveys(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        ra, dec, norder = 50.0, 5.0, 5
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        for s in ("A", "B"):
            _write_catalog_tile(lake, s, norder, npix, n_rows=5)
            _write_info(lake, s, norder, total_rows=5)
            ti.write_tile_index(lake, s, "catalog")

        region = Region.cone(ra, dec, radius_arcsec=30.0)
        rows = resolve_region(lake, region, surveys=["A"], modalities=["catalog"])
        assert {r.survey for r in rows} == {"A"}

    def test_crossmatch_count_scopes_to_region(self, tmp_path: Path) -> None:
        """Two XM tiles (in/out of cone); exact count reports only in-area rows."""
        lake = tmp_path / "lake"
        tree = "A_x_B__r1.0"
        norder = 5
        ra_in, dec_in = 120.0, 45.0
        ra_out, dec_out = 300.0, -45.0
        npix_in = int(assign_healpix(np.array([ra_in]), np.array([dec_in]), norder)[0])
        npix_out = int(assign_healpix(np.array([ra_out]), np.array([dec_out]), norder)[0])
        assert npix_in != npix_out

        _write_crossmatch_tile(lake, tree, norder, npix_in, n_rows=7)
        _write_crossmatch_tile(lake, tree, norder, npix_out, n_rows=11)
        _write_crossmatch_info(lake, tree, norder, total_rows=18)
        ti.write_tile_index(lake, tree, MODALITY_CROSSMATCH)

        region = Region.cone(ra_in, dec_in, radius_arcsec=30.0)
        rows = resolve_region(
            lake, region, modalities=[MODALITY_CROSSMATCH], count=True
        )
        assert len(rows) == 1
        row = rows[0]
        assert row.survey == tree
        assert row.modality == MODALITY_CROSSMATCH
        assert row.n_tiles_overlap == 1
        assert row.exact_rows == 7

    def test_spectra_estimate_from_zarr_when_info_lacks_totals(
        self, tmp_path: Path
    ) -> None:
        """spectrum_info without total_rows must not yield est_rows=0 when tiles exist."""
        lake = tmp_path / "lake"
        ra, dec, norder = 120.0, 45.0, 5
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_spectra_tile(lake, "SPEC", norder, npix, source_ids=[1, 2, 3])
        _write_spectrum_info(lake, "SPEC", norder)  # no total_rows / total_spectra
        ti.write_tile_index(lake, "SPEC", MODALITY_SPECTRA)

        region = Region.cone(ra, dec, radius_arcsec=30.0)
        rows = resolve_region(lake, region, modalities=[MODALITY_SPECTRA])
        assert len(rows) == 1
        row = rows[0]
        assert row.n_tiles_overlap == 1
        assert row.est_rows == 3
        assert row.exact_rows is None

    def test_spectra_estimate_uses_total_spectra_key(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        ra, dec, norder = 10.0, -10.0, 5
        npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        _write_spectra_tile(lake, "SPEC", norder, npix, source_ids=[1])
        _write_spectrum_info(lake, "SPEC", norder, total_spectra=50)
        ti.write_tile_index(lake, "SPEC", MODALITY_SPECTRA)

        region = Region.cone(ra, dec, radius_arcsec=30.0)
        rows = resolve_region(lake, region, modalities=[MODALITY_SPECTRA])
        assert len(rows) == 1
        assert rows[0].est_rows == 50

    def test_spectra_exact_count_zarr_overlap(self, tmp_path: Path) -> None:
        """--count sums ``_source_id`` lengths over overlap tiles only."""
        lake = tmp_path / "lake"
        norder = 5
        ra_in, dec_in = 120.0, 45.0
        ra_out, dec_out = 300.0, -45.0
        npix_in = int(assign_healpix(np.array([ra_in]), np.array([dec_in]), norder)[0])
        npix_out = int(assign_healpix(np.array([ra_out]), np.array([dec_out]), norder)[0])
        assert npix_in != npix_out

        _write_spectra_tile(lake, "SPEC", norder, npix_in, source_ids=[1, 2, 3, 4])
        _write_spectra_tile(lake, "SPEC", norder, npix_out, source_ids=[10, 11])
        # Deliberately wrong sidecar total; exact count must use Zarr metadata.
        _write_spectrum_info(lake, "SPEC", norder, total_rows=999)
        ti.write_tile_index(lake, "SPEC", MODALITY_SPECTRA)

        region = Region.cone(ra_in, dec_in, radius_arcsec=30.0)
        rows = resolve_region(
            lake, region, modalities=[MODALITY_SPECTRA], count=True
        )
        assert len(rows) == 1
        assert rows[0].n_tiles_overlap == 1
        assert rows[0].exact_rows == 4
        # Estimate uses sidecar total_rows scaled by overlap/total tiles.
        assert rows[0].est_rows == round(999 / 2)
