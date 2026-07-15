"""Tests for catalog ↔ spectrum Zarr link validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, healpix_dir
from data_lake.ingest.validate_catalog_spectra_link import (
    CatalogLinkIndex,
    build_catalog_link_index,
    discover_surveys_for_spectra_link_validation,
    run_validation,
)


NORDER = 5
N_PIX = 8
SURVEY = "LINK_TEST"


def _write_spectrum_info(survey_root: Path) -> None:
    survey_root.mkdir(parents=True, exist_ok=True)
    (survey_root / "spectrum_info.json").write_text(
        '{"n_pix": 8, "hats_order": 5, "wavelength_mode": "shared", "wcs": {}}'
    )


def _write_min_zarr(
    zarr_path: Path,
    *,
    source_ids: list[int],
) -> None:
    from data_lake.ingest.fits_to_spectra_zarr import (
        _META_DTYPE,
        _open_or_create_spectrum_tile,
    )
    from data_lake.ingest.zarr_ids import zarr_join_array

    n = len(source_ids)
    wcs_attrs = {
        "ctype": "WAVE",
        "crval": 3600.0,
        "cdelt": 1.0,
        "crpix": 1.0,
        "unit": "Angstrom",
        "air_or_vacuum": "vacuum",
        "n_pix": N_PIX,
    }
    root = _open_or_create_spectrum_tile(
        zarr_path, N_PIX, "shared", np.dtype(np.uint8), wcs_attrs,
    )
    root["flux"].append(np.zeros((n, N_PIX), dtype=np.float32))
    root["ivar"].append(np.zeros((n, N_PIX), dtype=np.float32))
    root["mask"].append(np.zeros((n, N_PIX), dtype=np.uint8))
    zarr_join_array(root).append(np.array(source_ids, dtype=np.int64))
    meta = np.zeros(n, dtype="|V" + str(_META_DTYPE.itemsize))
    root["meta"].append(meta)
    root["wavelength"][:] = np.linspace(3600.0, 3600.0 + N_PIX - 1, N_PIX)


def _write_catalog_tile(
    cat_path: Path,
    *,
    source_ids: list[int | None],
    spectrum_indices: list[int],
    npix: int,
) -> None:
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            LAKE_JOIN_ID_COLUMN: pa.array(source_ids, type=pa.int64()),
            f"_healpix_norder{NORDER}": pa.array([npix] * len(source_ids), type=pa.int64()),
            "_spectrum_index": pa.array(spectrum_indices, type=pa.int64()),
        }),
        cat_path,
    )


def _write_catalog_tile_with_npix(
    cat_path: Path,
    *,
    source_ids: list[int],
    spectrum_indices: list[int],
    cat_npix: int,
    spec_npix_values: list[int],
) -> None:
    """Write a catalog tile that includes the ``_spectrum_npix`` column."""
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            LAKE_JOIN_ID_COLUMN: pa.array(source_ids, type=pa.int64()),
            f"_healpix_norder{NORDER}": pa.array([cat_npix] * len(source_ids), type=pa.int64()),
            "_spectrum_index": pa.array(spectrum_indices, type=pa.int64()),
            "_spectrum_npix": pa.array(spec_npix_values, type=pa.int64()),
        }),
        cat_path,
    )


def _make_lake(tmp_path: Path, npix: int = 42) -> Path:
    lake = tmp_path / "lake"
    spec_root = lake / "spectra" / SURVEY
    _write_spectrum_info(spec_root)
    zarr_path = spec_root / healpix_dir(NORDER, npix) / f"Npix={npix}.zarr"
    _write_min_zarr(zarr_path, source_ids=[101, 102])

    cat_root = lake / "catalogs" / SURVEY
    cat_path = cat_root / healpix_dir(NORDER, npix) / f"Npix={npix}.parquet"
    _write_catalog_tile(
        cat_path,
        source_ids=[101, 102],
        spectrum_indices=[0, 1],
        npix=npix,
    )
    (cat_root / "catalog_info.json").write_text(
        '{"hats_order": 5, "link_id_mode": "sequential", '
        '"link_id_column": "_source_id", "ra_column": "ra", "dec_column": "dec"}'
    )
    return lake


class TestValidateCatalogSpectraLink:
    def test_happy_path(self, tmp_path: Path) -> None:
        lake = _make_lake(tmp_path)
        rep = run_validation(lake, SURVEY)
        assert rep.ok(strict=True)
        assert rep.stats.n_linked == 2
        assert rep.stats.n_orphan_zarr == 0

    def test_wrong_index_id_mismatch(self, tmp_path: Path) -> None:
        lake = _make_lake(tmp_path)
        npix = 42
        cat_path = (
            lake / "catalogs" / SURVEY / healpix_dir(NORDER, npix) / f"Npix={npix}.parquet"
        )
        _write_catalog_tile(
            cat_path,
            source_ids=[101, 999],
            spectrum_indices=[0, 1],
            npix=npix,
        )
        rep = run_validation(lake, SURVEY)
        assert not rep.ok(strict=False)
        assert rep.stats.n_wrong_id >= 1

    def test_orphan_zarr_row(self, tmp_path: Path) -> None:
        lake = _make_lake(tmp_path)
        npix = 42
        zarr_path = (
            lake / "spectra" / SURVEY / healpix_dir(NORDER, npix) / f"Npix={npix}.zarr"
        )
        _write_min_zarr(zarr_path, source_ids=[101, 102, 777])
        rep = run_validation(lake, SURVEY)
        assert not rep.ok(strict=True)
        assert rep.stats.n_orphan_zarr >= 1

    def test_quiet_mode_skips_per_row_warnings(self, tmp_path: Path) -> None:
        lake = _make_lake(tmp_path)
        npix = 42
        zarr_path = (
            lake / "spectra" / SURVEY / healpix_dir(NORDER, npix) / f"Npix={npix}.zarr"
        )
        _write_min_zarr(zarr_path, source_ids=[101, 102, 777, 888])
        rep_verbose = run_validation(lake, SURVEY, quiet=False)
        rep_quiet = run_validation(lake, SURVEY, quiet=True)
        assert rep_quiet.stats.n_orphan_zarr == rep_verbose.stats.n_orphan_zarr >= 1
        assert len(rep_quiet.warnings) < len(rep_verbose.warnings)
        assert not rep_quiet.ok(strict=True)

    def test_cli_quiet_suppresses_warning_lines(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.ingest.validate_catalog_spectra_link import cli

        if cli is None:
            pytest.skip("click not available")

        lake = _make_lake(tmp_path)
        npix = 42
        zarr_path = (
            lake / "spectra" / SURVEY / healpix_dir(NORDER, npix) / f"Npix={npix}.zarr"
        )
        _write_min_zarr(zarr_path, source_ids=[101, 102, 777])

        verbose = CliRunner().invoke(cli, ["--survey", SURVEY, str(lake)])
        quiet = CliRunner().invoke(cli, ["-q", "--survey", SURVEY, str(lake)])

        assert verbose.exit_code == 0
        assert quiet.exit_code == 0
        assert "WARNING:" in verbose.output
        assert "WARNING:" not in quiet.output
        assert "orphan zarr:" in quiet.output

    def test_unpatched_catalog_index(self, tmp_path: Path) -> None:
        lake = _make_lake(tmp_path)
        npix = 42
        cat_path = (
            lake / "catalogs" / SURVEY / healpix_dir(NORDER, npix) / f"Npix={npix}.parquet"
        )
        _write_catalog_tile(
            cat_path,
            source_ids=[101, 102],
            spectrum_indices=[0, -1],
            npix=npix,
        )
        rep = run_validation(lake, SURVEY)
        assert rep.ok(strict=False)
        assert not rep.ok(strict=True)
        assert rep.stats.n_unpatched_catalog >= 1

    def test_sample_mode_limits_linked_checks(self, tmp_path: Path) -> None:
        lake = _make_lake(tmp_path)
        rep = run_validation(lake, SURVEY, sample=1)
        assert rep.ok(strict=True)
        assert rep.stats.n_linked == 1

    def test_null_source_id_unlinked_rows_ok(self, tmp_path: Path) -> None:
        """Rows with null _source_id (--allow-incomplete-link-id) must not crash validation."""
        lake = _make_lake(tmp_path)
        npix = 42
        cat_path = (
            lake / "catalogs" / SURVEY / healpix_dir(NORDER, npix) / f"Npix={npix}.parquet"
        )
        _write_catalog_tile(
            cat_path,
            source_ids=[101, None, 102],
            spectrum_indices=[0, -1, 1],
            npix=npix,
        )
        rep = run_validation(lake, SURVEY)
        assert rep.ok(strict=True)
        assert rep.stats.n_linked == 2
        assert rep.stats.n_null_source_id == 1
        assert rep.stats.n_null_source_id_linked == 0

    def test_null_source_id_with_linked_index_is_error(self, tmp_path: Path) -> None:
        lake = _make_lake(tmp_path)
        npix = 42
        cat_path = (
            lake / "catalogs" / SURVEY / healpix_dir(NORDER, npix) / f"Npix={npix}.parquet"
        )
        _write_catalog_tile(
            cat_path,
            source_ids=[101, None],
            spectrum_indices=[0, 0],
            npix=npix,
        )
        rep = run_validation(lake, SURVEY)
        assert not rep.ok(strict=False)
        assert rep.stats.n_null_source_id_linked >= 1

    def test_discover_surveys_requires_both_modalities(self, tmp_path: Path) -> None:
        lake = _make_lake(tmp_path)
        assert discover_surveys_for_spectra_link_validation(lake) == [SURVEY]

        (lake / "catalogs" / "CAT_ONLY").mkdir(parents=True)
        (lake / "catalogs" / "CAT_ONLY" / "catalog_info.json").write_text(
            '{"hats_order": 5}'
        )
        assert "CAT_ONLY" not in discover_surveys_for_spectra_link_validation(lake)

    def test_cli_all_validates_every_paired_survey(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.ingest.validate_catalog_spectra_link import cli

        if cli is None:
            pytest.skip("click not available")

        lake = _make_lake(tmp_path)
        result = CliRunner().invoke(cli, ["--all", str(lake)])
        assert result.exit_code == 0
        assert "OK:" in result.output
        assert SURVEY in result.output

    def test_cli_unpatched_emits_rebuild_hint(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.ingest.validate_catalog_spectra_link import cli

        if cli is None:
            pytest.skip("click not available")

        lake = _make_lake(tmp_path)
        npix = 42
        cat_path = (
            lake / "catalogs" / SURVEY / healpix_dir(NORDER, npix) / f"Npix={npix}.parquet"
        )
        _write_catalog_tile(
            cat_path,
            source_ids=[101, 102],
            spectrum_indices=[-1, -1],
            npix=npix,
        )

        result = CliRunner().invoke(
            cli,
            ["--survey", SURVEY, str(lake)],
        )
        assert result.exit_code == 0
        assert "unpatched catalog:" in result.output
        assert "dl-rebuild-catalog-indices" in result.output
        assert "OK (with" in result.output

    def test_mixed_order_with_spectrum_npix_col(self, tmp_path: Path) -> None:
        """Catalog at norder=5, spectra at norder=1 — different Npix values.

        The catalog tile uses _spectrum_npix (the new format) to record which
        Zarr tile each row belongs to.  Validation must pass even though the
        catalog _healpix_norder5 pixel differs from the Zarr tile Npix.
        """
        import json

        cat_order = 5
        spec_order = 1
        cat_npix = 42
        spec_npix = 0  # different from cat_npix

        lake = tmp_path / "lake"

        # Write spectrum Zarr tile at spec_order=1, Npix=0.
        spec_root = lake / "spectra" / SURVEY
        spec_root.mkdir(parents=True, exist_ok=True)
        (spec_root / "spectrum_info.json").write_text(json.dumps({
            "n_pix": N_PIX, "hats_order": spec_order, "wavelength_mode": "shared", "wcs": {}
        }))
        zarr_path = spec_root / healpix_dir(spec_order, spec_npix) / f"Npix={spec_npix}.zarr"
        _write_min_zarr(zarr_path, source_ids=[201, 202])

        # Write catalog tile at cat_order=5, with _spectrum_npix pointing to the zarr tile.
        cat_root = lake / "catalogs" / SURVEY
        cat_path = cat_root / healpix_dir(cat_order, cat_npix) / f"Npix={cat_npix}.parquet"
        cat_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({
                LAKE_JOIN_ID_COLUMN: pa.array([201, 202], type=pa.int64()),
                f"_healpix_norder{cat_order}": pa.array([cat_npix, cat_npix], type=pa.int64()),
                "_spectrum_index": pa.array([0, 1], type=pa.int64()),
                "_spectrum_npix": pa.array([spec_npix, spec_npix], type=pa.int64()),
            }),
            cat_path,
        )
        (cat_root / "catalog_info.json").write_text(json.dumps({
            "hats_order": cat_order, "link_id_mode": "sequential",
            "link_id_column": "_source_id", "ra_column": "ra", "dec_column": "dec",
        }))

        rep = run_validation(lake, SURVEY)
        # strict=False because the order-mismatch warning is informational.
        assert rep.ok(strict=False), f"errors={rep.errors} warnings={rep.warnings}"
        assert rep.stats.n_linked == 2
        assert rep.stats.n_orphan_zarr == 0
        # Confirm the warning is about order difference, not a linkage error.
        assert any("hats_order differs" in w for w in rep.warnings)

    def test_legacy_mode_same_order(self, tmp_path: Path) -> None:
        """No _spectrum_npix column — legacy path using same-Npix pairing."""
        lake = _make_lake(tmp_path)
        # _make_lake creates a catalog without _spectrum_npix; validation must still pass.
        rep = run_validation(lake, SURVEY)
        assert rep.ok(strict=True), f"errors={rep.errors} warnings={rep.warnings}"
        assert rep.stats.n_linked == 2


def _make_indexed_lake(tmp_path: Path) -> tuple[Path, int]:
    """Three catalog tiles (cat_npix 100/101/102) → two Zarr tiles (spec_npix 10/20).

    Zarr 10: source_ids [1, 2, 3]
    Zarr 20: source_ids [4, 5]

    Catalog layout:
      tile 100: id=1 → spec_npix=10 idx=0,  id=4 → spec_npix=20 idx=0
      tile 101: id=2 → spec_npix=10 idx=1,  id=3 → spec_npix=10 idx=2
      tile 102: id=5 → spec_npix=20 idx=1
    """
    lake = tmp_path / "lake"
    spec_root = lake / "spectra" / SURVEY
    spec_root.mkdir(parents=True, exist_ok=True)
    (spec_root / "spectrum_info.json").write_text(
        json.dumps({"n_pix": N_PIX, "hats_order": NORDER, "wavelength_mode": "shared", "wcs": {}})
    )

    for spec_npix, ids in [(10, [1, 2, 3]), (20, [4, 5])]:
        _write_min_zarr(
            spec_root / healpix_dir(NORDER, spec_npix) / f"Npix={spec_npix}.zarr",
            source_ids=ids,
        )

    cat_root = lake / "catalogs" / SURVEY
    cat_root.mkdir(parents=True, exist_ok=True)
    (cat_root / "catalog_info.json").write_text(json.dumps({
        "hats_order": NORDER,
        "link_id_mode": "sequential",
        "link_id_column": "_source_id",
        "ra_column": "ra",
        "dec_column": "dec",
    }))

    _write_catalog_tile_with_npix(
        cat_root / healpix_dir(NORDER, 100) / "Npix=100.parquet",
        source_ids=[1, 4],
        spectrum_indices=[0, 0],
        cat_npix=100,
        spec_npix_values=[10, 20],
    )
    _write_catalog_tile_with_npix(
        cat_root / healpix_dir(NORDER, 101) / "Npix=101.parquet",
        source_ids=[2, 3],
        spectrum_indices=[1, 2],
        cat_npix=101,
        spec_npix_values=[10, 10],
    )
    _write_catalog_tile_with_npix(
        cat_root / healpix_dir(NORDER, 102) / "Npix=102.parquet",
        source_ids=[5],
        spectrum_indices=[1],
        cat_npix=102,
        spec_npix_values=[20],
    )
    return lake, 5  # total expected linked rows


class TestIndexedPathRegression:
    """Regression tests for the single-pass catalog index path."""

    def test_build_catalog_link_index_unit(self, tmp_path: Path) -> None:
        """build_catalog_link_index groups rows correctly by _spectrum_npix."""
        lake, _ = _make_indexed_lake(tmp_path)
        cat_root = lake / "catalogs" / SURVEY
        all_parquet = sorted(cat_root.rglob("Npix=*.parquet"))

        idx: CatalogLinkIndex = build_catalog_link_index(
            all_parquet, "_source_id", NORDER, has_npix_col=True
        )

        # Three catalog tiles → five rows total, split across two spec_npix keys.
        assert set(idx.by_spec_npix.keys()) == {10, 20}
        total_linked = sum(
            sum(tld.row_indices.size for tld in tiles)
            for tiles in idx.by_spec_npix.values()
        )
        assert total_linked == 5

        # spec_npix=10 must be covered by tiles 100 and 101; spec_npix=20 by 100 and 102.
        npix10_tiles = {tld.cat_tile.name for tld in idx.by_spec_npix[10]}
        assert npix10_tiles == {"Npix=100.parquet", "Npix=101.parquet"}
        npix20_tiles = {tld.cat_tile.name for tld in idx.by_spec_npix[20]}
        assert npix20_tiles == {"Npix=100.parquet", "Npix=102.parquet"}

        # Global reverse sets cover all five source_ids; none are unpatched.
        assert idx.all_catalog_sids == {1, 2, 3, 4, 5}
        assert idx.unpatched_sids == set()
        assert idx.n_null_source_id == 0

    def test_indexed_path_matches_many_tiles(self, tmp_path: Path) -> None:
        """3 catalog tiles + 2 Zarr tiles — only relevant catalog rows per Zarr npix."""
        lake, expected_linked = _make_indexed_lake(tmp_path)
        rep = run_validation(lake, SURVEY)

        assert rep.ok(strict=True), f"errors={rep.errors!r} warnings={rep.warnings!r}"
        assert rep.stats.n_linked == expected_linked
        assert rep.stats.n_orphan_zarr == 0
        assert rep.stats.n_wrong_id == 0
        assert rep.stats.n_stale_index == 0
        assert rep.stats.n_tiles_checked == 2  # two Zarr tiles

    def test_large_fanout_simulated(self, tmp_path: Path) -> None:
        """Many catalog tiles all pointing at one Zarr tile — assert correct counts."""
        n_cat_tiles = 20
        n_rows_each = 5  # rows per catalog tile pointing to the same spec_npix
        spec_npix = 0

        lake = tmp_path / "lake"
        spec_root = lake / "spectra" / SURVEY
        spec_root.mkdir(parents=True, exist_ok=True)
        (spec_root / "spectrum_info.json").write_text(
            json.dumps({"n_pix": N_PIX, "hats_order": NORDER, "wavelength_mode": "shared", "wcs": {}})
        )
        total_spectra = n_cat_tiles * n_rows_each
        _write_min_zarr(
            spec_root / healpix_dir(NORDER, spec_npix) / f"Npix={spec_npix}.zarr",
            source_ids=list(range(total_spectra)),
        )

        cat_root = lake / "catalogs" / SURVEY
        cat_root.mkdir(parents=True, exist_ok=True)
        (cat_root / "catalog_info.json").write_text(json.dumps({
            "hats_order": NORDER,
            "link_id_mode": "sequential",
            "link_id_column": "_source_id",
            "ra_column": "ra",
            "dec_column": "dec",
        }))

        # Each catalog tile covers a disjoint slice of Zarr rows.
        for tile_i in range(n_cat_tiles):
            base = tile_i * n_rows_each
            cat_npix = 200 + tile_i
            _write_catalog_tile_with_npix(
                cat_root / healpix_dir(NORDER, cat_npix) / f"Npix={cat_npix}.parquet",
                source_ids=list(range(base, base + n_rows_each)),
                spectrum_indices=list(range(base, base + n_rows_each)),
                cat_npix=cat_npix,
                spec_npix_values=[spec_npix] * n_rows_each,
            )

        rep = run_validation(lake, SURVEY)
        assert rep.ok(strict=True), f"errors={rep.errors!r} warnings={rep.warnings!r}"
        assert rep.stats.n_linked == total_spectra
        assert rep.stats.n_orphan_zarr == 0
        assert rep.stats.n_tiles_checked == 1

    def test_parallel_matches_serial(self, tmp_path: Path) -> None:
        """n_workers=2 must produce identical stats to n_workers=1."""
        lake, _ = _make_indexed_lake(tmp_path)

        serial = run_validation(lake, SURVEY, n_workers=1)
        parallel = run_validation(lake, SURVEY, n_workers=2)

        assert serial.stats.n_linked == parallel.stats.n_linked
        assert serial.stats.n_orphan_zarr == parallel.stats.n_orphan_zarr
        assert serial.stats.n_wrong_id == parallel.stats.n_wrong_id
        assert serial.stats.n_tiles_checked == parallel.stats.n_tiles_checked
        assert sorted(serial.errors) == sorted(parallel.errors)

    def test_cli_n_workers_and_progress_flags(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from data_lake.ingest.validate_catalog_spectra_link import cli

        if cli is None:
            pytest.skip("click not available")

        lake, _ = _make_indexed_lake(tmp_path)
        result = CliRunner().invoke(
            cli,
            ["--survey", SURVEY, "--n-workers", "2", "--no-progress", str(lake)],
        )
        assert result.exit_code == 0, result.output
        assert "OK:" in result.output

    def test_indexed_wrong_id_across_tiles(self, tmp_path: Path) -> None:
        """A catalog row pointing at the wrong Zarr row must produce a wrong_id error."""
        lake, _ = _make_indexed_lake(tmp_path)

        # Corrupt one catalog tile: row 0 has id=1 but spectrum_index=2
        # (Zarr[2]=3, so 1 != 3 → wrong_id).
        cat_root = lake / "catalogs" / SURVEY
        tile100 = cat_root / healpix_dir(NORDER, 100) / "Npix=100.parquet"
        _write_catalog_tile_with_npix(
            tile100,
            source_ids=[1, 4],
            spectrum_indices=[2, 0],  # id=1 points at zarr[2]=3, mismatch
            cat_npix=100,
            spec_npix_values=[10, 20],
        )

        rep = run_validation(lake, SURVEY)
        assert not rep.ok(strict=False)
        assert rep.stats.n_wrong_id >= 1
