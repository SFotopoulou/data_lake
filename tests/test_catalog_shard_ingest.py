"""
Tests for data_lake.ingest.catalog_shard_ingest (dl-ingest-from-shard).

Scenarios:
- 3-file round-trip: map then reduce produces tiles matching ingest_catalog.
- Map phase is idempotent: second run writes no extra part files.
- Reduce dedup: duplicate part (simulated re-map) drops duplicates with skip policy.
- Reduce with on_duplicate_id='error' raises on true duplicate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.table import Table
from astropy.io import fits

from data_lake.ingest.fits_to_parquet import (
    ingest_catalog,
    LAKE_JOIN_ID_COLUMN,
)
from data_lake.ingest.catalog_shard_ingest import (
    _shard_part_path,
    _load_checkpoint,
    _append_checkpoint,
    _collect_fits_files,
    run_map_phase,
    run_reduce_phase,
    _dedup_table,
    _iter_shard_parts,
)


# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic FITS catalogs
# ---------------------------------------------------------------------------

_RA_BANDS = [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0)]


def _make_catalog_table(ra_min: float, ra_max: float, n: int = 30) -> Table:
    rng = np.random.default_rng(int(ra_min * 100 + 7))
    ra = rng.uniform(ra_min, ra_max, n).astype(np.float64)
    dec = rng.uniform(-20.0, 20.0, n).astype(np.float64)
    ls_id = np.arange(int(ra_min * 1000), int(ra_min * 1000) + n, dtype=np.int64)
    return Table({"ra": ra, "dec": dec, "ls_id": ls_id})


def _write_fits_catalog(path: Path, table: Table) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(str(path), overwrite=True)
    return path


@pytest.fixture()
def catalog_fits_files(tmp_path: Path) -> list[Path]:
    """Three tiny FITS files in a single RA folder."""
    fits_dir = tmp_path / "ra_000"
    files = []
    for i, (ra_min, ra_max) in enumerate(_RA_BANDS):
        t = _make_catalog_table(ra_min, ra_max)
        p = _write_fits_catalog(fits_dir / f"brick_{i:03d}.fits", t)
        files.append(p)
    return files


@pytest.fixture()
def shard_root(tmp_path: Path) -> Path:
    return tmp_path / "shards"


@pytest.fixture()
def lake_root(tmp_path: Path) -> Path:
    return tmp_path / "lake"


_SURVEY = "test_survey"
_NORDER = 3
_RA_COL = "ra"
_DEC_COL = "dec"
_LINK_ID_COL = "ls_id"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_map(catalog_fits_files, shard_root):
    run_map_phase(
        source_paths=catalog_fits_files,
        shard_root=shard_root,
        survey=_SURVEY,
        ra_col=_RA_COL,
        dec_col=_DEC_COL,
        norder=_NORDER,
        link_id_col=_LINK_ID_COL,
        n_workers=1,
    )


def _run_reduce(shard_root, lake_root, on_duplicate_id="skip"):
    catalog_root = lake_root / "catalogs" / _SURVEY
    run_reduce_phase(
        shard_root=shard_root,
        survey=_SURVEY,
        catalog_root=catalog_root,
        norder=_NORDER,
        on_duplicate_id=on_duplicate_id,
        n_workers=1,
        ra_col=_RA_COL,
        dec_col=_DEC_COL,
    )
    return catalog_root


def _read_tile(path: Path) -> pa.Table:
    """Read a single tile file without Hive-partition schema inference."""
    return pq.ParquetFile(str(path)).read()


def _collect_sids(catalog_root: Path) -> set[int]:
    sids: set[int] = set()
    for pq_file in sorted(catalog_root.rglob("Npix=*.parquet")):
        t = _read_tile(pq_file)
        if LAKE_JOIN_ID_COLUMN in t.schema.names:
            sids.update(t.column(LAKE_JOIN_ID_COLUMN).to_pylist())
    return sids


# ---------------------------------------------------------------------------
# Test: checkpoint helpers
# ---------------------------------------------------------------------------


class TestCheckpointHelpers:
    def test_empty_checkpoint(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        assert _load_checkpoint(cp) == set()

    def test_append_and_reload(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        _append_checkpoint(cp, "/data/a.fits")
        _append_checkpoint(cp, "/data/b.fits")
        done = _load_checkpoint(cp)
        assert done == {"/data/a.fits", "/data/b.fits"}


# ---------------------------------------------------------------------------
# Test: collect_fits_files
# ---------------------------------------------------------------------------


class TestCollectFitsFiles:
    def test_from_dir(self, tmp_path):
        d = tmp_path / "dir"
        d.mkdir()
        (d / "a.fits").write_bytes(b"")
        (d / "b.fits").write_bytes(b"")
        (d / "ignore.txt").write_bytes(b"")
        files = _collect_fits_files(d, None)
        assert len(files) == 2
        assert all(f.suffix == ".fits" for f in files)

    def test_from_file_list(self, tmp_path):
        lst = tmp_path / "files.txt"
        lst.write_text("/data/a.fits\n# comment\n/data/b.fits\n")
        files = _collect_fits_files(None, lst)
        assert files == [Path("/data/a.fits"), Path("/data/b.fits")]

    def test_both_raises(self, tmp_path):
        d = tmp_path / "dir"
        d.mkdir()
        with pytest.raises(ValueError, match="not both"):
            _collect_fits_files(d, tmp_path / "list.txt")

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="required"):
            _collect_fits_files(None, None)


# ---------------------------------------------------------------------------
# Test: map phase produces parts
# ---------------------------------------------------------------------------


class TestMapPhase:
    def test_parts_created(self, catalog_fits_files, shard_root):
        _run_map(catalog_fits_files, shard_root)
        parts = list((shard_root / _SURVEY).rglob("part-*.parquet"))
        assert len(parts) >= len(catalog_fits_files)

    def test_checkpoint_records_all_files(self, catalog_fits_files, shard_root):
        _run_map(catalog_fits_files, shard_root)
        from data_lake.ingest.catalog_shard_ingest import _checkpoint_path
        cp = _checkpoint_path(shard_root, _SURVEY)
        done = _load_checkpoint(cp)
        for f in catalog_fits_files:
            assert str(f.resolve()) in done

    def test_idempotent_no_extra_parts(self, catalog_fits_files, shard_root):
        """Second map run must not create duplicate part files."""
        _run_map(catalog_fits_files, shard_root)
        parts_after_first = set(str(p) for p in (shard_root / _SURVEY).rglob("part-*.parquet"))

        _run_map(catalog_fits_files, shard_root)
        parts_after_second = set(str(p) for p in (shard_root / _SURVEY).rglob("part-*.parquet"))

        assert parts_after_first == parts_after_second, (
            "Second map run created extra part files (not idempotent)."
        )

    def test_part_schema_has_source_id(self, catalog_fits_files, shard_root):
        _run_map(catalog_fits_files, shard_root)
        parts = list((shard_root / _SURVEY).rglob("part-*.parquet"))
        for part in parts:
            t = _read_tile(part)
            assert LAKE_JOIN_ID_COLUMN in t.schema.names, f"Missing _source_id in {part}"


# ---------------------------------------------------------------------------
# Test: round-trip matches ingest_catalog
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_source_ids_match_ingest_catalog(
        self, catalog_fits_files, shard_root, lake_root, tmp_path
    ):
        """_source_id values from shard ingest must match sequential ingest_catalog."""
        # Shard ingest
        _run_map(catalog_fits_files, shard_root)
        catalog_root_shard = _run_reduce(shard_root, lake_root)
        sids_shard = _collect_sids(catalog_root_shard)

        # Sequential ingest into separate tree
        seq_root = tmp_path / "lake_seq"
        for f in catalog_fits_files:
            ingest_catalog(
                source_path=f,
                output_root=seq_root,
                survey_name=_SURVEY,
                ra_col=_RA_COL,
                dec_col=_DEC_COL,
                norder=_NORDER,
                link_id_col=_LINK_ID_COL,
                tile_mode="append",
                defer_finalize=True,
            )
        sids_seq = _collect_sids(seq_root / "catalogs" / _SURVEY)

        assert sids_shard == sids_seq, (
            f"Shard IDs differ from sequential ingest. "
            f"Only in shard: {sids_shard - sids_seq!r}, "
            f"Only in seq: {sids_seq - sids_shard!r}."
        )

    def test_total_row_count(self, catalog_fits_files, shard_root, lake_root):
        n_input = sum(len(_make_catalog_table(*b)) for b in _RA_BANDS)
        _run_map(catalog_fits_files, shard_root)
        catalog_root = _run_reduce(shard_root, lake_root)

        n_output = sum(
            _read_tile(pq_file).num_rows
            for pq_file in sorted(catalog_root.rglob("Npix=*.parquet"))
        )
        assert n_output == n_input

    def test_catalog_info_json_exists(self, catalog_fits_files, shard_root, lake_root):
        _run_map(catalog_fits_files, shard_root)
        catalog_root = _run_reduce(shard_root, lake_root)
        info_path = catalog_root / "catalog_info.json"
        assert info_path.exists()
        info = json.loads(info_path.read_text())
        assert info.get("hats_order") == _NORDER


# ---------------------------------------------------------------------------
# Test: dedup in reduce
# ---------------------------------------------------------------------------


class TestReduceDedup:
    def _make_table_with_dup(self) -> pa.Table:
        sids = pa.array([1, 2, 2, 3], type=pa.int64())
        ra = pa.array([10.0, 20.0, 20.5, 30.0])
        return pa.table({LAKE_JOIN_ID_COLUMN: sids, "ra": ra})

    def test_dedup_skip_keeps_first(self):
        t = self._make_table_with_dup()
        out = _dedup_table(t, "skip")
        assert len(out) == 3
        sids = out.column(LAKE_JOIN_ID_COLUMN).to_pylist()
        assert sids == [1, 2, 3]
        # ra for sid=2 should be first occurrence
        ra_for_2 = out.column("ra")[1].as_py()
        assert ra_for_2 == pytest.approx(20.0)

    def test_dedup_last_keeps_last(self):
        t = self._make_table_with_dup()
        out = _dedup_table(t, "last")
        assert len(out) == 3
        sids = out.column(LAKE_JOIN_ID_COLUMN).to_pylist()
        assert sids == [1, 2, 3]
        ra_for_2 = out.column("ra")[1].as_py()
        assert ra_for_2 == pytest.approx(20.5)

    def test_dedup_error_raises(self):
        t = self._make_table_with_dup()
        with pytest.raises(ValueError, match="Duplicate _source_id"):
            _dedup_table(t, "error")

    def test_no_dups_passes_error(self):
        t = pa.table({LAKE_JOIN_ID_COLUMN: pa.array([1, 2, 3], type=pa.int64())})
        out = _dedup_table(t, "error")
        assert len(out) == 3

    def test_reduce_dedup_skip(self, catalog_fits_files, shard_root, lake_root, tmp_path):
        """Duplicate part file (simulated by copying) is deduped during reduce."""
        _run_map(catalog_fits_files, shard_root)

        # Duplicate every part file by copying with a different stem
        for part in list((shard_root / _SURVEY).rglob("part-*.parquet")):
            dup = part.parent / part.name.replace("part-", "part-dup-")
            import shutil
            shutil.copy2(part, dup)

        # Reduce should not raise and should dedup
        catalog_root = _run_reduce(shard_root, lake_root, on_duplicate_id="skip")
        # Row count should still be n_input (duplicates dropped)
        n_output = sum(
            _read_tile(f).num_rows
            for f in catalog_root.rglob("Npix=*.parquet")
        )
        n_input = sum(len(_make_catalog_table(*b)) for b in _RA_BANDS)
        assert n_output == n_input


# ---------------------------------------------------------------------------
# Test: _iter_shard_parts
# ---------------------------------------------------------------------------


class TestIterShardParts:
    def test_empty_shard_root(self, tmp_path):
        result = list(_iter_shard_parts(tmp_path / "shards", "s", 3))
        assert result == []

    def test_finds_parts(self, catalog_fits_files, shard_root):
        _run_map(catalog_fits_files, shard_root)
        groups = dict(_iter_shard_parts(shard_root, _SURVEY, _NORDER))
        assert len(groups) > 0
        for npix, parts in groups.items():
            assert isinstance(npix, int)
            assert all(p.suffix == ".parquet" for p in parts)


# ---------------------------------------------------------------------------
# Test: CLI integration (smoke test)
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_map_and_reduce(self, catalog_fits_files, tmp_path):
        from click.testing import CliRunner
        from data_lake.ingest.catalog_shard_ingest import cli

        shard_root = tmp_path / "shards"
        lake_root = tmp_path / "lake"
        fits_dir = catalog_fits_files[0].parent

        runner = CliRunner()

        # Map phase
        result = runner.invoke(cli, [
            "--phase", "map",
            "--input-dir", str(fits_dir),
            "--survey", _SURVEY,
            "--shard-root", str(shard_root),
            "--ra-col", _RA_COL,
            "--dec-col", _DEC_COL,
            "--link-id-col", _LINK_ID_COL,
            "--norder", str(_NORDER),
        ])
        assert result.exit_code == 0, result.output

        # Reduce phase
        result = runner.invoke(cli, [
            "--phase", "reduce",
            "--survey", _SURVEY,
            "--shard-root", str(shard_root),
            "--lake-root", str(lake_root),
            "--norder", str(_NORDER),
        ])
        assert result.exit_code == 0, result.output

        # Verify tiles were written
        catalog_root = lake_root / "catalogs" / _SURVEY
        tiles = list(catalog_root.rglob("Npix=*.parquet"))
        assert len(tiles) > 0

    def test_cli_map_no_input_exits_nonzero(self, tmp_path):
        from click.testing import CliRunner
        from data_lake.ingest.catalog_shard_ingest import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "--phase", "map",
            "--survey", _SURVEY,
            "--shard-root", str(tmp_path / "shards"),
            "--norder", str(_NORDER),
        ])
        # Missing --input-dir or --file-list: should show error
        assert result.exit_code != 0
