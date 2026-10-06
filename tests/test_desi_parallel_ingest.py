"""
Tests for data_lake.ingest.desi_parallel_ingest.

These tests do NOT spawn ``desispec`` or real ProcessPool workers — they
inject a synthetic decoder and use ``ThreadPoolExecutor`` (same Future API)
so the orchestrator's writer logic, checkpointing, and failure handling
are exercised end-to-end without external deps.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from data_lake.ingest.checkpoint_sidecars import paths_from_file_list_file
from data_lake.ingest.desi_parallel_ingest import (
    TileBatch,
    WorkerResult,
    _atomic_write_json,
    _canonical_fits_path,
    _checkpoint_jsonl_path,
    _load_checkpoint,
    _recover_stale_parallel_commit,
    _truncate_spectrum_tile_row_arrays,
    ingest_spectra_parallel,
)
from data_lake.ingest.fits_to_spectra_zarr import _META_DTYPE, _meta_to_bytes, _open_or_create_spectrum_tile


N_PIX = 16


# ---------------------------------------------------------------------------
# Synthetic decoder & fixtures
# ---------------------------------------------------------------------------


def _synthetic_payload(path_str: str, sources: list[tuple[int, int, float]]):
    """Build a WorkerResult covering the given (source_id, npix, z) records.

    Flux/ivar/mask encode the source_id so round-trip checks are trivial.
    """
    by_pix: dict[int, list[tuple[int, float]]] = {}
    for sid, npix, z in sources:
        by_pix.setdefault(npix, []).append((sid, z))

    batches: list[TileBatch] = []
    for pix, items in by_pix.items():
        n = len(items)
        flux = np.stack([np.full(N_PIX, sid, dtype=np.float32) for sid, _ in items])
        ivar = np.stack([np.full(N_PIX, 1.0 / (sid + 1), dtype=np.float32) for sid, _ in items])
        mask = np.stack([np.full(N_PIX, sid % 5, dtype=np.uint8) for sid, _ in items])
        sids = np.array([sid for sid, _ in items], dtype=np.int64)
        meta_bytes = b"".join(
            _meta_to_bytes({
                "z": z, "z_err": 0.001, "snr": 5.0,
                "exptime": 1000.0, "R": 3000.0, "instr": "TEST",
            })
            for _, z in items
        )
        batches.append(TileBatch(
            npix=pix, flux=flux, ivar=ivar, mask=mask,
            source_ids=sids, meta_bytes=meta_bytes,
        ))

    wavelength = np.linspace(3600.0, 9800.0, N_PIX, dtype=np.float64)
    wcs = {
        "ctype": "WAVE", "crval": float(wavelength[0]),
        "cdelt": float(wavelength[1] - wavelength[0]), "crpix": 1.0,
        "unit": "Angstrom", "air_or_vacuum": "vacuum", "n_pix": N_PIX,
    }
    return WorkerResult(
        path=path_str, ok=True, batches=batches,
        wavelength=wavelength, wcs_attrs=wcs,
        n_pix=N_PIX, n_spectra=sum(len(v) for v in by_pix.values()),
        elapsed_s=0.0,
    )


# Each file maps to a known set of (source_id, npix, z) records.
# Two of the files share tile 100 → exercises append/merge across files.
FILE_MAP: dict[str, list[tuple[int, int, float]]] = {
    "fileA": [(1001, 100, 0.10), (1002, 100, 0.11), (1003, 200, 0.20)],
    "fileB": [(2001, 100, 1.10), (2002, 300, 2.20)],
    "fileC": [(3001, 200, 0.50)],
}


def _fake_decoder(path_str: str, norder: int) -> WorkerResult:
    """Lookup the file label and synthesise its payload."""
    label = Path(path_str).stem
    if label == "BOOM":
        raise RuntimeError("synthetic decoder failure")
    sources = FILE_MAP.get(label)
    if sources is None:
        return WorkerResult(path=path_str, ok=False, error=f"unknown label {label!r}")
    return _synthetic_payload(path_str, sources)


def _thread_executor(n_workers: int) -> ThreadPoolExecutor:
    """ThreadPool with same Future API → exercises orchestrator without subprocesses."""
    return ThreadPoolExecutor(max_workers=n_workers)


@pytest.fixture
def fake_files(tmp_path: Path) -> list[Path]:
    """Create empty placeholder files matching FILE_MAP labels."""
    paths = []
    for label in FILE_MAP:
        p = tmp_path / f"{label}.fits"
        p.touch()
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParallelIngestBasic:
    def test_happy_path_writes_all_tiles(self, tmp_path: Path, fake_files):
        """All synthetic spectra are appended to the correct tile zarrs."""
        import zarr

        out = tmp_path / "lake"
        result = ingest_spectra_parallel(
            file_paths=fake_files,
            output_root=out,
            survey_name="syn",
            n_workers=2,
            norder=5,
            checkpoint_path=tmp_path / "ckpt.json",
            failures_log=tmp_path / "fail.jsonl",
            show_progress=False,
            decoder=_fake_decoder,
            executor_factory=_thread_executor,
            track_index_map=True,
        )

        assert result["n_files_succeeded"] == 3
        assert result["n_files_failed"] == 0
        assert result["n_spectra"] == 6
        assert result["n_tiles"] == 3
        assert set(result["index_map"]) == {1001, 1002, 1003, 2001, 2002, 3001}

        survey_root = out / "spectra" / "syn"
        info = json.loads((survey_root / "spectrum_info.json").read_text())
        assert info["wavelength_mode"] == "shared"
        assert info["n_pix"] == N_PIX

        # Verify tile 100 has 3 sources (1001, 1002 from fileA + 2001 from fileB)
        tile_100_zarr_paths = list(survey_root.rglob("Npix=100.zarr"))
        assert len(tile_100_zarr_paths) == 1
        root = zarr.open_group(
            store=zarr.storage.LocalStore(str(tile_100_zarr_paths[0])),
            mode="r", zarr_format=3,
        )
        from data_lake.ingest.zarr_ids import zarr_join_array

        sids_100 = np.asarray(zarr_join_array(root)[:])
        flux_100 = np.asarray(root["flux"][:])
        assert root["flux"].shape == (3, N_PIX)
        assert set(sids_100.tolist()) == {1001, 1002, 2001}
        for i, sid in enumerate(sids_100.tolist()):
            assert np.all(flux_100[i] == sid), f"flux mismatch for sid {sid}"

    def test_max_open_tiles_eviction_still_writes(self, tmp_path: Path, fake_files):
        """Writer closes evicted tiles and re-opens them on later writes."""
        result = ingest_spectra_parallel(
            file_paths=fake_files,
            output_root=tmp_path / "lake",
            survey_name="syn",
            n_workers=2,
            max_open_tiles=1,
            show_progress=False,
            decoder=_fake_decoder,
            executor_factory=_thread_executor,
        )
        assert result["n_files_succeeded"] == 3
        assert result["n_spectra"] == 6
        assert result["n_tiles"] == 3

    def test_failure_isolated_and_logged(self, tmp_path: Path):
        """A failing file logs to failures.jsonl but does not abort the run."""
        files = [tmp_path / f"{n}.fits" for n in ("fileA", "BOOM", "fileC")]
        for p in files:
            p.touch()

        failures_log = tmp_path / "fail.jsonl"
        result = ingest_spectra_parallel(
            file_paths=files,
            output_root=tmp_path / "lake",
            survey_name="syn",
            n_workers=2,
            checkpoint_path=tmp_path / "ckpt.json",
            failures_log=failures_log,
            show_progress=False,
            decoder=_fake_decoder,
            executor_factory=_thread_executor,
        )

        assert result["n_files_succeeded"] == 2
        assert result["n_files_failed"] == 1
        assert failures_log.exists()
        entries = [json.loads(l) for l in failures_log.read_text().splitlines() if l.strip()]
        assert len(entries) == 1
        assert "BOOM" in entries[0]["path"]
        assert "synthetic decoder failure" in entries[0]["error"]


class TestCheckpoint:
    def test_resume_skips_completed(self, tmp_path: Path, fake_files):
        """Files already in the checkpoint are not re-decoded on resume."""
        ckpt = tmp_path / "ckpt.json"

        # First run: only fileA
        first = ingest_spectra_parallel(
            file_paths=[fake_files[0]],
            output_root=tmp_path / "lake",
            survey_name="syn",
            n_workers=1,
            checkpoint_path=ckpt,
            failures_log=tmp_path / "fail.jsonl",
            show_progress=False,
            decoder=_fake_decoder,
            executor_factory=_thread_executor,
        )
        assert first["n_files_succeeded"] == 1

        # Track which files the decoder is invoked for in the second run
        seen: list[str] = []

        def tracking_decoder(path_str, norder):
            seen.append(path_str)
            return _fake_decoder(path_str, norder)

        # Second run: all three files; A should be skipped via checkpoint
        second = ingest_spectra_parallel(
            file_paths=fake_files,
            output_root=tmp_path / "lake",
            survey_name="syn",
            n_workers=2,
            checkpoint_path=ckpt,
            failures_log=tmp_path / "fail.jsonl",
            show_progress=False,
            decoder=tracking_decoder,
            executor_factory=_thread_executor,
        )

        assert second["n_files_skipped"] == 1
        assert second["n_files_processed"] == 2
        assert all("fileA" not in s for s in seen)
        assert any("fileB" in s for s in seen)
        assert any("fileC" in s for s in seen)

        # Final checkpoint records all three files (append-only JSONL)
        assert len(_load_checkpoint(ckpt)) == 3
        jsonl = _checkpoint_jsonl_path(ckpt)
        assert jsonl.exists()
        assert len(jsonl.read_text().strip().splitlines()) >= 2

        inflight = tmp_path / "lake" / "spectra" / "syn" / ".ingest_inflight.json"
        assert inflight.exists()
        assert json.loads(inflight.read_text()).get("commit") is None

    def test_atomic_write_json_round_trip(self, tmp_path: Path):
        target = tmp_path / "nested" / "ckpt.json"
        _atomic_write_json(target, {"completed": ["/a", "/b"]})
        assert target.exists()
        assert _load_checkpoint(target) == {"/a", "/b"}

    def test_load_checkpoint_missing_returns_empty(self, tmp_path: Path):
        assert _load_checkpoint(tmp_path / "nope.json") == set()
        assert _load_checkpoint(None) == set()

    def test_load_checkpoint_normalizes_paths(self, tmp_path: Path):
        """Equivalent path spellings in JSON match the canonical pending path."""
        f = tmp_path / "d" / "x.fits"
        f.parent.mkdir(parents=True)
        f.touch()
        ckpt = tmp_path / "ckpt.json"
        redundant = str(tmp_path / "d" / "." / "x.fits")
        _atomic_write_json(ckpt, {"completed": [redundant]})
        assert _load_checkpoint(ckpt) == {_canonical_fits_path(f)}

    def test_load_checkpoint_jsonl(self, tmp_path: Path):
        """Append-only JSONL sidecar is merged with any legacy JSON checkpoint."""
        f = tmp_path / "a.fits"
        f.touch()
        ckpt = tmp_path / "ckpt.json"
        canon = _canonical_fits_path(f)
        _checkpoint_jsonl_path(ckpt).write_text(canon + "\n")
        assert _load_checkpoint(ckpt) == {canon}

    def test_paths_from_file_list_relative_to_list_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Relative lines resolve vs the list file's parent, not process cwd."""
        lists = tmp_path / "lists"
        lists.mkdir()
        data = tmp_path / "data"
        data.mkdir()
        coadd = data / "coadd.fits"
        coadd.touch()
        flist = lists / "files.txt"
        flist.write_text("../data/coadd.fits\n")

        monkeypatch.chdir(tmp_path)
        paths_a = paths_from_file_list_file(flist)
        monkeypatch.chdir("/")
        paths_b = paths_from_file_list_file(flist)

        assert len(paths_a) == 1 and len(paths_b) == 1
        assert _canonical_fits_path(paths_a[0]) == _canonical_fits_path(paths_b[0])
        assert _canonical_fits_path(paths_a[0]) == _canonical_fits_path(coadd)


class TestInflightJournal:
    def test_truncate_spectrum_tile_row_arrays(self, tmp_path: Path) -> None:
        import zarr

        wave = np.linspace(3600.0, 9800.0, N_PIX, dtype=np.float64)
        wcs = {
            "ctype": "WAVE", "crval": float(wave[0]),
            "cdelt": float(wave[1] - wave[0]), "crpix": 1.0,
            "unit": "Angstrom", "air_or_vacuum": "vacuum", "n_pix": N_PIX,
        }
        tile_path = tmp_path / "Npix=1.zarr"
        root = _open_or_create_spectrum_tile(
            tile_path, N_PIX, "shared", np.dtype(np.uint8), wcs,
        )
        mb_one = _meta_to_bytes({
            "z": 0.5, "z_err": 0.01, "snr": 5.0,
            "exptime": 100.0, "R": 3000.0, "instr": "TEST",
        })
        meta_dtype = "|V" + str(_META_DTYPE.itemsize)
        meta_arr = np.frombuffer(mb_one * 5, dtype=meta_dtype)
        root["flux"].append(np.ones((5, N_PIX), dtype=np.float32))
        root["ivar"].append(np.ones((5, N_PIX), dtype=np.float32))
        root["mask"].append(np.zeros((5, N_PIX), dtype=np.uint8))
        from data_lake.ingest.zarr_ids import zarr_join_array

        zarr_join_array(root).append(np.arange(5, dtype=np.int64))
        root["meta"].append(meta_arr)
        assert root["flux"].shape[0] == 5

        _truncate_spectrum_tile_row_arrays(root, 2)
        assert root["flux"].shape[0] == 2
        root_r = zarr.open_group(
            store=zarr.storage.LocalStore(str(tile_path)), mode="r", zarr_format=3,
        )
        assert root_r["meta"].shape[0] == 2

    def test_recovery_truncates_orphan_commit(self, tmp_path: Path, fake_files) -> None:
        """Stale inflight for a file not in checkpoint rewinds Zarr row counts."""
        import zarr

        ckpt = tmp_path / "ckpt.json"
        lake = tmp_path / "lake"
        survey_root = lake / "spectra" / "syn"
        inflight = survey_root / ".ingest_inflight.json"

        first = ingest_spectra_parallel(
            file_paths=[fake_files[0]],
            output_root=lake,
            survey_name="syn",
            n_workers=1,
            norder=5,
            checkpoint_path=ckpt,
            failures_log=tmp_path / "fail.jsonl",
            show_progress=False,
            decoder=_fake_decoder,
            executor_factory=_thread_executor,
        )
        assert first["n_files_succeeded"] == 1

        p_a = _canonical_fits_path(fake_files[0])
        _atomic_write_json(ckpt, {"completed": []})
        jsonl = _checkpoint_jsonl_path(ckpt)
        if jsonl.exists():
            jsonl.unlink()
        _atomic_write_json(
            inflight,
            {
                "commit": {
                    "path": p_a,
                    "tiles": {"100": 0, "200": 0},
                    "norder": 5,
                },
            },
        )
        _recover_stale_parallel_commit(survey_root, 5, inflight, completed=set())
        assert json.loads(inflight.read_text()).get("commit") is None

        t100 = next(survey_root.rglob("Npix=100.zarr"))
        r100 = zarr.open_group(
            store=zarr.storage.LocalStore(str(t100)), mode="r", zarr_format=3,
        )
        assert r100["flux"].shape[0] == 0

        second = ingest_spectra_parallel(
            file_paths=[fake_files[0]],
            output_root=lake,
            survey_name="syn",
            n_workers=1,
            norder=5,
            checkpoint_path=ckpt,
            failures_log=tmp_path / "fail.jsonl",
            show_progress=False,
            decoder=_fake_decoder,
            executor_factory=_thread_executor,
        )
        assert second["n_files_succeeded"] == 1
        r100b = zarr.open_group(
            store=zarr.storage.LocalStore(str(t100)), mode="r", zarr_format=3,
        )
        assert r100b["flux"].shape[0] == 2

    def test_recovery_clears_inflight_when_path_already_completed(
        self, tmp_path: Path, fake_files,
    ) -> None:
        import zarr

        ckpt = tmp_path / "ckpt.json"
        lake = tmp_path / "lake"
        survey_root = lake / "spectra" / "syn"
        inflight = survey_root / ".ingest_inflight.json"

        ingest_spectra_parallel(
            file_paths=[fake_files[0]],
            output_root=lake,
            survey_name="syn",
            n_workers=1,
            checkpoint_path=ckpt,
            failures_log=tmp_path / "fail.jsonl",
            show_progress=False,
            decoder=_fake_decoder,
            executor_factory=_thread_executor,
        )
        p_a = _canonical_fits_path(fake_files[0])
        _atomic_write_json(
            inflight,
            {"commit": {"path": p_a, "tiles": {"100": 0}, "norder": 5}},
        )
        _recover_stale_parallel_commit(survey_root, 5, inflight, completed={p_a})

        assert json.loads(inflight.read_text()).get("commit") is None
        t100 = next(survey_root.rglob("Npix=100.zarr"))
        r100 = zarr.open_group(
            store=zarr.storage.LocalStore(str(t100)), mode="r", zarr_format=3,
        )
        assert r100["flux"].shape[0] == 2


class TestInconsistentNPix:
    def test_npix_mismatch_rejected_per_file(self, tmp_path: Path):
        """A second file with different N_pix is rejected without aborting."""
        files = [tmp_path / "fileA.fits", tmp_path / "fileMISMATCH.fits"]
        for p in files:
            p.touch()

        def mismatch_decoder(path_str, norder):
            if "MISMATCH" in path_str:
                wavelength = np.linspace(3600.0, 9800.0, N_PIX * 2)
                flux = np.zeros((1, N_PIX * 2), dtype=np.float32)
                ivar = np.zeros((1, N_PIX * 2), dtype=np.float32)
                mask = np.zeros((1, N_PIX * 2), dtype=np.uint8)
                wcs = {
                    "ctype": "WAVE", "crval": 3600.0, "cdelt": 1.0, "crpix": 1.0,
                    "unit": "Angstrom", "air_or_vacuum": "vacuum", "n_pix": N_PIX * 2,
                }
                return WorkerResult(
                    path=path_str, ok=True,
                    batches=[TileBatch(
                        npix=400, flux=flux, ivar=ivar, mask=mask,
                        source_ids=np.array([42], dtype=np.int64),
                        meta_bytes=_meta_to_bytes({"z": 0, "instr": "X"}),
                    )],
                    wavelength=wavelength, wcs_attrs=wcs, n_pix=N_PIX * 2, n_spectra=1,
                )
            return _fake_decoder(path_str, norder)

        result = ingest_spectra_parallel(
            file_paths=files,
            output_root=tmp_path / "lake",
            survey_name="syn",
            n_workers=1,    # serial → deterministic that fileA fixes N_pix first
            checkpoint_path=tmp_path / "ckpt.json",
            failures_log=tmp_path / "fail.jsonl",
            show_progress=False,
            decoder=mismatch_decoder,
            executor_factory=_thread_executor,
        )
        assert result["n_files_succeeded"] == 1
        assert result["n_files_failed"] == 1
        assert any("n_pix=" in f["error"] for f in result["failures"])


class TestCLIInputValidation:
    def test_both_inputs_rejected(self, tmp_path: Path):
        """Passing both --file-list and --coadd-root must error."""
        from click.testing import CliRunner

        from data_lake.ingest.desi_parallel_ingest import cli

        flist = tmp_path / "files.txt"
        flist.write_text("")
        cadr = tmp_path / "coadd_root"
        cadr.mkdir()

        runner = CliRunner()
        res = runner.invoke(cli, [
            "--survey", "x",
            "--file-list", str(flist),
            "--coadd-root", str(cadr),
            "--n-workers", "1",
            str(tmp_path),
        ])
        assert res.exit_code != 0
        assert "exactly one of --file-list or --coadd-root" in res.output

    def test_neither_input_rejected(self, tmp_path: Path):
        from click.testing import CliRunner

        from data_lake.ingest.desi_parallel_ingest import cli

        runner = CliRunner()
        res = runner.invoke(cli, [
            "--survey", "x",
            "--n-workers", "1",
            str(tmp_path),
        ])
        assert res.exit_code != 0
        assert "exactly one of --file-list or --coadd-root" in res.output

    def test_zero_workers_rejected(self, tmp_path: Path):
        from click.testing import CliRunner

        from data_lake.ingest.desi_parallel_ingest import cli

        flist = tmp_path / "files.txt"
        flist.write_text("/tmp/nonexistent.fits\n")

        runner = CliRunner()
        res = runner.invoke(cli, [
            "--survey", "x",
            "--file-list", str(flist),
            "--n-workers", "0",
            str(tmp_path),
        ])
        assert res.exit_code != 0
        assert "n-workers" in res.output.lower()

    def test_link_id_col_in_help(self):
        from click.testing import CliRunner

        from data_lake.ingest.desi_parallel_ingest import cli

        res = CliRunner().invoke(cli, ["--help"])
        assert res.exit_code == 0
        assert "--link-id-col" in res.output
        assert "TARGETID,SURVEY,PROGRAM" in res.output


def test_default_decoder_binds_link_id_col(monkeypatch, tmp_path: Path):
    """ingest_spectra_parallel passes link_id_col into the default worker decoder."""
    from concurrent.futures import ThreadPoolExecutor

    from data_lake.ingest import desi_parallel_ingest as dpi

    seen: dict[str, object] = {}

    def fake_decode(path_str: str, norder: int, link_id_col: str | None = None):
        seen["link_id_col"] = link_id_col
        seen["path"] = path_str
        return dpi.WorkerResult(path=path_str, ok=True, elapsed_s=0.0)

    monkeypatch.setattr(dpi, "_decode_one_coadd_safe", fake_decode)

    fake = tmp_path / "coadd.fits"
    fake.write_bytes(b"")

    result = dpi.ingest_spectra_parallel(
        file_paths=[fake],
        output_root=tmp_path / "lake",
        survey_name="DESI_T",
        n_workers=1,
        checkpoint_path=None,
        failures_log=None,
        show_progress=False,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
        link_id_col="TARGETID",
    )
    assert seen.get("link_id_col") == "TARGETID"
    assert result["n_files_succeeded"] == 1
