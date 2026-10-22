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

    def test_torn_line_does_not_hide_later_entries(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        _append_checkpoint(cp, "/data/a.fits")
        with open(cp, "a") as fh:
            fh.write('{"path": "/data/tor\n')
        _append_checkpoint(cp, "/data/b.fits")
        assert _load_checkpoint(cp) == {"/data/a.fits", "/data/b.fits"}


# ---------------------------------------------------------------------------
# Test: failures log
# ---------------------------------------------------------------------------


class TestFailuresLog:
    def test_map_records_bad_fits(self, catalog_fits_files, shard_root, tmp_path):
        bad = catalog_fits_files[0].parent / "broken.fits"
        bad.write_bytes(b"not a fits file")
        flog = tmp_path / "fail.jsonl"
        result = run_map_phase(
            source_paths=[*catalog_fits_files, bad],
            shard_root=shard_root,
            survey=_SURVEY,
            ra_col=_RA_COL,
            dec_col=_DEC_COL,
            norder=_NORDER,
            link_id_col=_LINK_ID_COL,
            failures_log=flog,
        )
        assert result["n_files_ok"] == len(catalog_fits_files)
        assert result["n_files_failed"] == 1
        entries = [json.loads(l) for l in flog.read_text().splitlines()]
        assert [e["path"] for e in entries] == [str(bad)]
        assert entries[0]["error"] and "Traceback" in entries[0]["traceback"]

    def test_reduce_records_bad_npix(self, catalog_fits_files, shard_root, lake_root, tmp_path):
        _run_map(catalog_fits_files, shard_root)
        npix, parts = next(_iter_shard_parts(shard_root, _SURVEY, _NORDER))
        parts[0].write_bytes(b"corrupt")
        flog = tmp_path / "fail.jsonl"
        catalog_root = lake_root / "catalogs" / _SURVEY
        result = run_reduce_phase(
            shard_root=shard_root,
            survey=_SURVEY,
            catalog_root=catalog_root,
            norder=_NORDER,
            failures_log=flog,
        )
        assert result["n_tiles_failed"] == 1
        assert result["finalized"] is False
        assert not (catalog_root / "catalog_info.json").exists()
        (entry,) = [json.loads(l) for l in flog.read_text().splitlines()]
        assert entry["npix"] == npix
        assert str(parts[0]) in entry["parts"]


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
# Test: reducing into an existing lake survey
# ---------------------------------------------------------------------------


def _map_reduce(files, shard_root, lake_root, *, tile_mode="append", norder=_NORDER,
                on_duplicate_id="skip", link_id_col=_LINK_ID_COL):
    run_map_phase(
        source_paths=files,
        shard_root=shard_root,
        survey=_SURVEY,
        ra_col=_RA_COL,
        dec_col=_DEC_COL,
        norder=norder,
        link_id_col=link_id_col,
    )
    catalog_root = lake_root / "catalogs" / _SURVEY
    run_reduce_phase(
        shard_root=shard_root,
        survey=_SURVEY,
        catalog_root=catalog_root,
        norder=norder,
        on_duplicate_id=on_duplicate_id,
        tile_mode=tile_mode,
    )
    return catalog_root


def _n_rows(catalog_root: Path) -> int:
    return sum(_read_tile(t).num_rows for t in catalog_root.rglob("Npix=*.parquet"))


class TestExistingTiles:
    def test_append_adds_new_files_to_existing_survey(
        self, catalog_fits_files, lake_root, tmp_path
    ):
        catalog_root = _map_reduce(catalog_fits_files[:2], tmp_path / "s1", lake_root)
        first = _n_rows(catalog_root)
        _map_reduce(catalog_fits_files[2:], tmp_path / "s2", lake_root)
        n_input = sum(len(_make_catalog_table(*b)) for b in _RA_BANDS)
        assert first < n_input
        assert _n_rows(catalog_root) == n_input

    def test_append_same_shards_twice_is_noop(
        self, catalog_fits_files, shard_root, lake_root
    ):
        catalog_root = _map_reduce(catalog_fits_files, shard_root, lake_root)
        before = {t: t.stat().st_mtime_ns for t in catalog_root.rglob("Npix=*.parquet")}
        _map_reduce(catalog_fits_files, shard_root, lake_root)
        after = {t: t.stat().st_mtime_ns for t in catalog_root.rglob("Npix=*.parquet")}
        assert before == after

    def test_append_error_on_overlap(self, catalog_fits_files, lake_root, tmp_path):
        catalog_root = _map_reduce(catalog_fits_files, tmp_path / "s1", lake_root)
        s2 = tmp_path / "s2"
        _run_map(catalog_fits_files, s2)
        result = run_reduce_phase(
            shard_root=s2,
            survey=_SURVEY,
            catalog_root=catalog_root,
            norder=_NORDER,
            on_duplicate_id="error",
        )
        assert result["n_tiles_failed"] == result["n_tiles"] > 0
        assert result["finalized"] is False

    def test_skip_leaves_existing_tiles(self, catalog_fits_files, lake_root, tmp_path):
        catalog_root = _map_reduce(catalog_fits_files[:1], tmp_path / "s1", lake_root)
        tiles_before = {t: _read_tile(t).num_rows for t in catalog_root.rglob("Npix=*.parquet")}
        _map_reduce(catalog_fits_files, tmp_path / "s2", lake_root, tile_mode="skip")
        for t, n in tiles_before.items():
            assert _read_tile(t).num_rows == n

    def test_overwrite_replaces_existing_tiles(
        self, catalog_fits_files, lake_root, tmp_path
    ):
        catalog_root = _map_reduce(catalog_fits_files, tmp_path / "s1", lake_root)
        s2 = tmp_path / "s2"
        _map_reduce(catalog_fits_files[:1], s2, lake_root, tile_mode="overwrite")
        groups = dict(_iter_shard_parts(s2, _SURVEY, _NORDER))
        assert groups
        for npix, parts in groups.items():
            tile = next(catalog_root.rglob(f"Npix={npix}.parquet"))
            assert _read_tile(tile).num_rows == sum(_read_tile(p).num_rows for p in parts)

    def test_catalog_info_records_shard_link_id_mode(
        self, catalog_fits_files, shard_root, lake_root
    ):
        catalog_root = _map_reduce(catalog_fits_files, shard_root, lake_root)
        shard_info = json.loads((shard_root / _SURVEY / "shard_info.json").read_text())
        info = json.loads((catalog_root / "catalog_info.json").read_text())
        assert info["link_id_mode"] == shard_info["link_id_mode"] != "sequential"


class TestGuards:
    def test_reduce_norder_mismatch_with_lake(
        self, catalog_fits_files, lake_root, tmp_path
    ):
        _map_reduce(catalog_fits_files, tmp_path / "s1", lake_root, norder=_NORDER)
        with pytest.raises(ValueError, match="hats_order"):
            _map_reduce(catalog_fits_files, tmp_path / "s2", lake_root, norder=_NORDER + 1)

    def test_reduce_link_id_mismatch_with_lake(
        self, catalog_fits_files, shard_root, lake_root
    ):
        catalog_root = _map_reduce(catalog_fits_files, shard_root, lake_root)
        info_path = catalog_root / "catalog_info.json"
        info = json.loads(info_path.read_text())
        info["link_id_mode"] = "column:SOMETHING_ELSE"
        info_path.write_text(json.dumps(info))
        with pytest.raises(ValueError, match="link_id_mode"):
            run_reduce_phase(
                shard_root=shard_root,
                survey=_SURVEY,
                catalog_root=catalog_root,
                norder=_NORDER,
            )

    def test_reduce_norder_mismatch_with_shards(
        self, catalog_fits_files, shard_root, lake_root
    ):
        _run_map(catalog_fits_files, shard_root)
        with pytest.raises(ValueError, match="does not match shard tree"):
            run_reduce_phase(
                shard_root=shard_root,
                survey=_SURVEY,
                catalog_root=lake_root / "catalogs" / _SURVEY,
                norder=_NORDER + 1,
            )

    def test_map_link_id_mismatch_with_shards(self, catalog_fits_files, shard_root):
        run_map_phase(
            source_paths=catalog_fits_files[:1],
            shard_root=shard_root,
            survey=_SURVEY,
            ra_col=_RA_COL,
            dec_col=_DEC_COL,
            norder=_NORDER,
            link_id_col=_LINK_ID_COL,
        )
        with pytest.raises(ValueError, match="link_id_mode"):
            run_map_phase(
                source_paths=catalog_fits_files[1:],
                shard_root=shard_root,
                survey=_SURVEY,
                ra_col=_RA_COL,
                dec_col=_DEC_COL,
                norder=_NORDER,
                link_id_col=f"{_LINK_ID_COL},dec",
            )


# ---------------------------------------------------------------------------
# Test: staging area
# ---------------------------------------------------------------------------


class TestStaging:
    def test_map_and_reduce_via_staging(
        self, catalog_fits_files, shard_root, lake_root, tmp_path
    ):
        staging = tmp_path / "ssd"
        run_map_phase(
            source_paths=catalog_fits_files,
            shard_root=shard_root,
            survey=_SURVEY,
            ra_col=_RA_COL,
            dec_col=_DEC_COL,
            norder=_NORDER,
            link_id_col=_LINK_ID_COL,
            staging_dir=staging,
        )
        catalog_root = lake_root / "catalogs" / _SURVEY
        run_reduce_phase(
            shard_root=shard_root,
            survey=_SURVEY,
            catalog_root=catalog_root,
            norder=_NORDER,
            ra_col=_RA_COL,
            dec_col=_DEC_COL,
            staging_dir=staging,
        )
        assert not [p for p in staging.rglob("*") if p.is_file()]
        n_input = sum(len(_make_catalog_table(*b)) for b in _RA_BANDS)
        tiles = list(catalog_root.rglob("Npix=*.parquet"))
        assert sum(_read_tile(t).num_rows for t in tiles) == n_input

    def test_tiles_have_no_partition_columns(
        self, catalog_fits_files, shard_root, lake_root
    ):
        _run_map(catalog_fits_files, shard_root)
        catalog_root = _run_reduce(shard_root, lake_root)
        for tile in catalog_root.rglob("Npix=*.parquet"):
            names = set(_read_tile(tile).schema.names)
            assert not names & {"Norder", "Dir", "Npix"}, tile

    def test_falls_back_when_staging_full(
        self, catalog_fits_files, shard_root, monkeypatch, tmp_path
    ):
        import data_lake.ingest.catalog_shard_ingest as mod

        monkeypatch.setattr(mod, "_staging_has_room", lambda *_a, **_k: False)
        staging = tmp_path / "ssd"
        run_map_phase(
            source_paths=catalog_fits_files,
            shard_root=shard_root,
            survey=_SURVEY,
            ra_col=_RA_COL,
            dec_col=_DEC_COL,
            norder=_NORDER,
            link_id_col=_LINK_ID_COL,
            staging_dir=staging,
        )
        assert list((shard_root / _SURVEY).rglob("part-*.parquet"))
        assert not staging.exists() or not any(staging.rglob("*.parquet"))

    def test_publish_cross_device(self, tmp_path, monkeypatch):
        import errno as _errno
        import data_lake.ingest.catalog_shard_ingest as mod

        staged = tmp_path / "a" / "x.parquet"
        staged.parent.mkdir()
        staged.write_bytes(b"data")
        final = tmp_path / "b" / "x.parquet"

        real_replace = mod.os.replace

        def fake_replace(src, dst):
            if Path(src) == staged:
                raise OSError(_errno.EXDEV, "cross-device")
            return real_replace(src, dst)

        monkeypatch.setattr(mod.os, "replace", fake_replace)
        mod._publish(staged, final)
        assert final.read_bytes() == b"data"
        assert not staged.exists()
        assert not final.with_name(final.name + ".tmp").exists()


# ---------------------------------------------------------------------------
# Test: CLI integration (smoke test)
# ---------------------------------------------------------------------------


_TOKEN = "shard-secret"


@pytest.fixture()
def deployment(tmp_path, monkeypatch):
    """Deployment config (lake root = tmp/cfg_lake) with an ingest token."""
    from data_lake.cli_utils import INGEST_TOKEN_ENV, write_ingest_token_hash

    dep = tmp_path / "dep"
    dep.mkdir()
    lake = tmp_path / "cfg_lake"
    cfg_path = dep / "lake_config.toml"
    cfg_path.write_text(
        f'schema_version = "1"\n[lake]\nname = "t"\nroot = "{lake.as_posix()}"\n'
        '[guardrails]\ningest_token_file = ".ingest_token_hash"\n'
    )
    write_ingest_token_hash(dep, _TOKEN)
    monkeypatch.setenv("DATA_LAKE_CONFIG", str(cfg_path))
    monkeypatch.setenv(INGEST_TOKEN_ENV, _TOKEN)
    monkeypatch.delenv("DATA_LAKE_STAGING_DIR", raising=False)
    return lake


def _invoke(args):
    from click.testing import CliRunner
    from data_lake.ingest.catalog_shard_ingest import cli

    return CliRunner().invoke(cli, args)


def _map_args(fits_dir: Path, shard_root: Path, *extra: str) -> list[str]:
    return [
        "--phase", "map",
        "--input-dir", str(fits_dir),
        "--survey", _SURVEY,
        "--shard-root", str(shard_root),
        "--ra-col", _RA_COL,
        "--dec-col", _DEC_COL,
        "--link-id-col", _LINK_ID_COL,
        "--norder", str(_NORDER),
        "--no-progress",
        *extra,
    ]


def _reduce_args(shard_root: Path, *extra: str) -> list[str]:
    return [
        "--phase", "reduce",
        "--survey", _SURVEY,
        "--shard-root", str(shard_root),
        "--no-progress",
        *extra,
    ]


class TestCLI:
    def test_map_and_reduce_parallel_positional_root(
        self, deployment, catalog_fits_files, tmp_path
    ):
        shard_root = tmp_path / "shards"
        lake = tmp_path / "lake"
        fits_dir = catalog_fits_files[0].parent

        result = _invoke(_map_args(fits_dir, shard_root, "--n-workers", "2", "--max-in-flight", "1"))
        assert result.exit_code == 0, result.output

        result = _invoke([str(lake), *_reduce_args(shard_root, "--n-workers", "2")])
        assert result.exit_code == 0, result.output
        catalog_root = lake / "catalogs" / _SURVEY
        n_input = sum(len(_make_catalog_table(*b)) for b in _RA_BANDS)
        assert _n_rows(catalog_root) == n_input
        assert json.loads((catalog_root / "catalog_info.json").read_text())["hats_order"] == _NORDER

    def test_reduce_defaults_to_config_lake_and_shard_norder(
        self, deployment, catalog_fits_files, tmp_path
    ):
        shard_root = tmp_path / "shards"
        assert _invoke(_map_args(catalog_fits_files[0].parent, shard_root)).exit_code == 0
        result = _invoke(_reduce_args(shard_root))  # no OUTPUT_ROOT, no --norder
        assert result.exit_code == 0, result.output
        assert list((deployment / "catalogs" / _SURVEY).rglob("Npix=*.parquet"))

    def test_lake_root_alias_still_works(self, deployment, catalog_fits_files, tmp_path):
        shard_root = tmp_path / "shards"
        lake = tmp_path / "legacy_lake"
        assert _invoke(_map_args(catalog_fits_files[0].parent, shard_root)).exit_code == 0
        result = _invoke(_reduce_args(shard_root, "--lake-root", str(lake)))
        assert result.exit_code == 0, result.output
        assert list((lake / "catalogs" / _SURVEY).rglob("Npix=*.parquet"))

    def test_lake_root_alias_conflict(self, deployment, tmp_path):
        result = _invoke([
            str(tmp_path / "a"),
            *_reduce_args(tmp_path / "shards", "--lake-root", str(tmp_path / "b")),
        ])
        assert result.exit_code != 0
        assert "disagree" in result.output

    @pytest.mark.parametrize("phase_args", ["map", "reduce"])
    def test_token_required_in_both_phases(
        self, deployment, catalog_fits_files, tmp_path, monkeypatch, phase_args
    ):
        from data_lake.cli_utils import INGEST_TOKEN_ENV

        monkeypatch.delenv(INGEST_TOKEN_ENV)
        shard_root = tmp_path / "shards"
        args = (
            _map_args(catalog_fits_files[0].parent, shard_root)
            if phase_args == "map"
            else _reduce_args(shard_root)
        )
        result = _invoke(args)
        assert result.exit_code != 0
        assert "ingest token" in result.output
        assert not shard_root.exists()

    def test_map_requires_link_id_col(self, deployment, catalog_fits_files, tmp_path):
        args = _map_args(catalog_fits_files[0].parent, tmp_path / "shards")
        i = args.index("--link-id-col")
        del args[i:i + 2]
        result = _invoke(args)
        assert result.exit_code != 0
        assert "--link-id-col is required" in result.output

    def test_map_failure_exits_1(self, deployment, catalog_fits_files, tmp_path):
        (catalog_fits_files[0].parent / "broken.fits").write_bytes(b"nope")
        result = _invoke(_map_args(catalog_fits_files[0].parent, tmp_path / "shards"))
        assert result.exit_code == 1

    def test_guard_mismatch_is_clean_error(self, deployment, catalog_fits_files, tmp_path):
        shard_root = tmp_path / "shards"
        assert _invoke(_map_args(catalog_fits_files[0].parent, shard_root)).exit_code == 0
        result = _invoke(_reduce_args(shard_root, "--norder", str(_NORDER + 1)))
        assert result.exit_code == 1
        assert "does not match shard tree" in result.output
        assert not isinstance(result.exception, ValueError)

    def test_log_file(self, deployment, catalog_fits_files, tmp_path):
        log_file = tmp_path / "logs" / "map.log"
        result = _invoke(_map_args(
            catalog_fits_files[0].parent, tmp_path / "shards",
            "--quiet", "--log-file", str(log_file),
        ))
        assert result.exit_code == 0, result.output
        assert "Map phase done" in log_file.read_text()

    def test_quiet_and_verbose_conflict(self, tmp_path):
        result = _invoke([*_reduce_args(tmp_path / "shards"), "-q", "-v"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_map_no_input_exits_nonzero(self, deployment, tmp_path):
        result = _invoke([
            "--phase", "map",
            "--survey", _SURVEY,
            "--shard-root", str(tmp_path / "shards"),
            "--link-id-col", _LINK_ID_COL,
        ])
        assert result.exit_code != 0
        assert "--input-dir or --file-list" in result.output
