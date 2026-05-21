"""
catalog_parallel_ingest – parallel file-list catalog ingest (Option B).

Workers read/decode one catalog file at a time (whole file in RAM).  A single
main-thread writer is the only code that read–merges–writes ``Npix=*.parquet``
tiles, so ``--tile-mode append`` is safe for overlapping HEALPix pixels.

Does **not** support ``--streaming`` (use sequential ``dl-ingest-catalog-from-list``).
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import time
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import pyarrow as pa

from data_lake.ingest.fits_to_parquet import (
    CatalogParquetOptions,
    DuplicateIdMode,
    TileMode,
    _finalize_catalog_writes,
    _remove_stale_parquet_tmp_files,
    _resolve_tile_mode,
    _write_tile_for_mode,
    decode_catalog_file_to_batches,
    healpix_dir,
    normalize_catalog_table_types,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogDecodeConfig:
    """Picklable worker arguments."""

    norder: int
    ra_col: str
    dec_col: str
    source_id_col: str | None
    columns: tuple[str, ...] | None


@dataclass
class CatalogTileBatch:
    npix: int
    table: pa.Table


@dataclass
class CatalogTileBatchRef:
    """Lightweight IPC handle: tile Parquet written in the worker temp dir."""

    npix: int
    spool_path: str


@dataclass
class CatalogWorkerResult:
    path: str
    ok: bool
    batches: list[CatalogTileBatch] = field(default_factory=list)
    batch_refs: list[CatalogTileBatchRef] = field(default_factory=list)
    spool_dir: str | None = None
    n_rows: int = 0
    source_id_mode: str = "sequential"
    error: str | None = None
    tb: str | None = None
    elapsed_s: float = 0.0


def _canonical_catalog_path(p: Path | str) -> str:
    return str(Path(p).expanduser().resolve())


def _decode_one_catalog(path_str: str, config: CatalogDecodeConfig) -> CatalogWorkerResult:
    from data_lake.cli_utils import apply_parallel_worker_logging_after_heavy_imports

    apply_parallel_worker_logging_after_heavy_imports()
    t0 = time.perf_counter()
    cols = list(config.columns) if config.columns else None
    batches_raw, sid_mode, n_rows = decode_catalog_file_to_batches(
        path_str,
        ra_col=config.ra_col,
        dec_col=config.dec_col,
        norder=config.norder,
        source_id_col=config.source_id_col,
        columns=cols,
    )
    # Do not pickle pa.Table across processes (ALLWISE/Gaia-scale tables OOM the
    # worker or parent). Spool per-tile fragments in the worker; writer reads them.
    import pyarrow.parquet as pq

    spool_dir = tempfile.mkdtemp(prefix="dl_catalog_spool_")
    batch_refs: list[CatalogTileBatchRef] = []
    try:
        for npix, tbl in batches_raw:
            spool_path = Path(spool_dir) / f"Npix={npix}.parquet"
            pq.write_table(
                normalize_catalog_table_types(tbl),
                spool_path,
                compression="zstd",
                compression_level=3,
            )
            batch_refs.append(CatalogTileBatchRef(npix=npix, spool_path=str(spool_path)))
    except Exception:
        shutil.rmtree(spool_dir, ignore_errors=True)
        raise
    del batches_raw

    return CatalogWorkerResult(
        path=path_str,
        ok=True,
        batch_refs=batch_refs,
        spool_dir=spool_dir,
        n_rows=n_rows,
        source_id_mode=sid_mode,
        elapsed_s=time.perf_counter() - t0,
    )


def _decode_one_catalog_safe(
    path_str: str,
    config: CatalogDecodeConfig,
) -> CatalogWorkerResult:
    try:
        return _decode_one_catalog(path_str, config)
    except Exception as exc:
        return CatalogWorkerResult(
            path=path_str,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            tb=traceback.format_exc(),
        )


def ingest_catalogs_parallel(
    file_paths: Sequence[Path | str],
    *,
    output_root: Path | str,
    survey_name: str,
    n_workers: int,
    ra_col: str = "ra",
    dec_col: str = "dec",
    norder: int = 5,
    source_id_col: str | None = None,
    columns: Sequence[str] | None = None,
    tile_mode: TileMode | None = None,
    on_duplicate_id: DuplicateIdMode = "skip",
    overwrite: bool = False,
    parquet_options: CatalogParquetOptions | None = None,
    compact: bool = False,
    checkpoint_path: Path | str | None = None,
    failures_log: Path | str | None = None,
    show_progress: bool = True,
    skip_completed: bool = True,
    max_in_flight: int | None = None,
    executor_factory: Callable[[int], Executor] | None = None,
    decoder: Callable[[str, CatalogDecodeConfig], CatalogWorkerResult] | None = None,
) -> dict:
    """Ingest many catalog files with parallel decode and a single-thread writer.

    Returns a summary dict (counts, failures, elapsed_s).
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")

    output_root = Path(output_root)
    catalog_root = output_root / "catalogs" / survey_name
    catalog_root.mkdir(parents=True, exist_ok=True)
    _remove_stale_parquet_tmp_files(catalog_root)

    resolved_tile_mode = _resolve_tile_mode(tile_mode, overwrite)
    if resolved_tile_mode != "append":
        log.warning(
            "Parallel catalog ingest is intended for --tile-mode append; got %r.",
            resolved_tile_mode,
        )

    pq_opts = (
        CatalogParquetOptions.compact()
        if compact
        else (parquet_options or CatalogParquetOptions())
    )

    decode_cfg = CatalogDecodeConfig(
        norder=norder,
        ra_col=ra_col,
        dec_col=dec_col,
        source_id_col=source_id_col,
        columns=tuple(columns) if columns else None,
    )
    decoder = decoder or _decode_one_catalog_safe

    from data_lake.ingest.file_list_ingest import _append_checkpoint, _load_completed

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    failures_log = Path(failures_log) if failures_log else None
    completed: set[str] = (
        _load_completed(checkpoint_path) if skip_completed and checkpoint_path else set()
    )

    requested = [_canonical_catalog_path(p) for p in file_paths]
    pending = sorted(p for p in requested if p not in completed)
    n_skipped_ckpt = len(requested) - len(pending)

    if not pending:
        return {
            "n_files_requested": len(requested),
            "n_files_processed": 0,
            "n_files_skipped": n_skipped_ckpt,
            "n_files_succeeded": 0,
            "n_files_failed": 0,
            "n_rows": 0,
            "n_tiles_touched": 0,
            "failures": [],
            "elapsed_s": 0.0,
        }

    if max_in_flight is None:
        # Keep low: writer reads one spooled file at a time; extra in-flight
        # futures only add worker RAM (full decode still in each child).
        max_in_flight = n_workers

    if executor_factory is None:
        from data_lake.cli_utils import init_parallel_ingest_subprocess as _init

        def _default_executor(nw: int) -> Executor:
            return ProcessPoolExecutor(max_workers=nw, initializer=_init)

        executor_factory = _default_executor

    failures: list[dict] = []
    n_files_ok = n_files_fail = 0
    n_rows = 0
    tiles_touched: set[int] = set()
    sid_mode: str | None = None
    fallback_n_cols = 0

    if failures_log is not None:
        failures_log.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    work_queue: deque[str] = deque(sorted(pending))
    in_flight: dict[Future, str] = {}
    pool_recoveries = 0
    # Cap recoveries so a pathological loop cannot run forever.
    max_pool_recoveries = max(len(pending) * 8, 512)

    pool_cm = executor_factory(n_workers)
    pool: Executor = pool_cm.__enter__()

    def _recycle_pool_after_broken() -> None:
        """Worker died (often OOM): shut down pool, re-create, clear in_flight futures."""
        nonlocal pool, pool_cm, pool_recoveries
        pool_recoveries += 1
        if pool_recoveries > max_pool_recoveries:
            raise RuntimeError(
                f"Process pool broke more than {max_pool_recoveries} times; aborting. "
                "Try --n-workers 1–2, lower --max-in-flight, and/or --columns to cut RAM."
            ) from None
        log.warning(
            "Process pool broke (worker terminated abruptly — often OOM while decoding "
            "a large FITS). Recreating pool (recovery #%d / %d). "
            "Re-queuing %d in-flight file(s). Consider fewer --n-workers, "
            "smaller --max-in-flight, or --columns.",
            pool_recoveries,
            max_pool_recoveries,
            len(in_flight),
        )
        try:
            pool_cm.__exit__(None, None, None)
        except Exception:
            log.exception("While shutting down broken process pool")
        pool_cm = executor_factory(n_workers)
        pool = pool_cm.__enter__()
        in_flight.clear()

    def _requeue_in_flight_paths() -> None:
        for _fut, path_str in list(in_flight.items()):
            work_queue.appendleft(path_str)

    def _write_batch(batch: CatalogTileBatch) -> None:
        out_dir = catalog_root / healpix_dir(norder, batch.npix)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"Npix={batch.npix}.parquet"
        _write_tile_for_mode(
            out_file,
            batch.table,
            tile_mode=resolved_tile_mode,
            on_duplicate_id=on_duplicate_id,
            source_id_col=source_id_col,
            parquet_options=pq_opts,
        )
        tiles_touched.add(batch.npix)

    def _process_result(res: CatalogWorkerResult) -> None:
        nonlocal n_files_ok, n_files_fail, n_rows, sid_mode, fallback_n_cols

        if not res.ok:
            n_files_fail += 1
            fail_entry = {
                "path": res.path,
                "error": res.error,
                "traceback": res.tb,
            }
            failures.append(fail_entry)
            log.warning("FAIL %s: %s", res.path, res.error)
            if failures_log is not None:
                with failures_log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(fail_entry, ensure_ascii=False) + "\n")
            return

        if sid_mode is None:
            sid_mode = res.source_id_mode
        try:
            if res.batch_refs:
                import pyarrow.parquet as pq

                if res.batches:
                    fallback_n_cols = len(res.batches[0].table.schema)
                elif res.batch_refs:
                    fallback_n_cols = len(
                        pq.read_schema(res.batch_refs[0].spool_path)
                    )
                for ref in res.batch_refs:
                    _write_batch(
                        CatalogTileBatch(npix=ref.npix, table=pq.read_table(ref.spool_path))
                    )
            else:
                if res.batches:
                    fallback_n_cols = len(res.batches[0].table.schema)
                for batch in res.batches:
                    _write_batch(batch)
        finally:
            if res.spool_dir:
                shutil.rmtree(res.spool_dir, ignore_errors=True)

        n_rows += res.n_rows
        n_files_ok += 1
        if checkpoint_path is not None:
            _append_checkpoint(checkpoint_path, res.path)

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    pbar = (
        tqdm(total=len(pending), disable=not show_progress, unit="file", desc="ingest")
        if tqdm is not None
        else None
    )
    try:
        while work_queue or in_flight:
            submit_broken = False
            while len(in_flight) < max_in_flight and work_queue:
                p = work_queue.popleft()
                try:
                    in_flight[pool.submit(decoder, p, decode_cfg)] = p
                except BrokenProcessPool:
                    work_queue.appendleft(p)
                    _requeue_in_flight_paths()
                    _recycle_pool_after_broken()
                    submit_broken = True
                    break
            if submit_broken:
                continue

            if not in_flight:
                break

            done_set, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            result_broken = False
            for fut in done_set:
                path_str = in_flight.pop(fut)
                try:
                    res = fut.result()
                except BrokenProcessPool:
                    work_queue.appendleft(path_str)
                    _requeue_in_flight_paths()
                    _recycle_pool_after_broken()
                    result_broken = True
                    break
                except Exception as exc:
                    res = CatalogWorkerResult(
                        path=path_str,
                        ok=False,
                        error=f"executor: {type(exc).__name__}: {exc}",
                        tb=traceback.format_exc(),
                    )
                _process_result(res)
                if pbar is not None:
                    pbar.update(1)
            if result_broken:
                continue
    finally:
        if pbar is not None:
            pbar.close()
        try:
            pool_cm.__exit__(None, None, None)
        except Exception:
            log.exception("While shutting down catalog ingest process pool")

    if sid_mode is None:
        sid_mode = "sequential"

    _finalize_catalog_writes(
        catalog_root,
        survey_name,
        norder,
        ra_col=ra_col,
        dec_col=dec_col,
        source_id_mode=sid_mode,
        streaming=False,
        fallback_n_cols=fallback_n_cols,
    )

    elapsed = time.perf_counter() - t_start
    log.info(
        "Parallel catalog ingest: ok=%d fail=%d skip_ckpt=%d rows=%d tiles=%d in %.1fs",
        n_files_ok,
        n_files_fail,
        n_skipped_ckpt,
        n_rows,
        len(tiles_touched),
        elapsed,
    )

    return {
        "n_files_requested": len(requested),
        "n_files_processed": len(pending),
        "n_files_skipped": n_skipped_ckpt,
        "n_files_succeeded": n_files_ok,
        "n_files_failed": n_files_fail,
        "n_rows": n_rows,
        "n_tiles_touched": len(tiles_touched),
        "failures": failures,
        "elapsed_s": elapsed,
    }


try:
    import click

    from ..cli_utils import (
        config_option,
        configure_warning_filters,
        ingest_token_option,
        load_optional_config,
        pick,
        require_ingest_permission,
        require_output_root,
    )
    from data_lake.ingest.checkpoint_sidecars import paths_from_file_list_file

    @click.command("dl-ingest-catalog-batch")
    @click.argument("paths_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @ingest_token_option
    @click.option("--survey", "survey_name", required=True)
    @click.option("--ra-col", default="ra", show_default=True)
    @click.option("--dec-col", default="dec", show_default=True)
    @click.option("--norder", default=None, type=int)
    @click.option("--source-id-col", default=None)
    @click.option(
        "--tile-mode",
        type=click.Choice(["skip", "overwrite", "append"], case_sensitive=False),
        default="append",
        show_default=True,
        help="Use append for multi-file ingest (recommended).",
    )
    @click.option(
        "--on-duplicate-id",
        type=click.Choice(["skip", "error", "last"], case_sensitive=False),
        default="skip",
        show_default=True,
    )
    @click.option("--overwrite", is_flag=True, help="Deprecated: use --tile-mode overwrite.")
    @click.option(
        "--columns",
        default=None,
        help="Comma-separated columns to keep (plus required sky/ID/index cols).",
    )
    @click.option(
        "--compact", is_flag=True,
        help="Smaller Parquet tiles (ZSTD-9, no stats/dictionary).",
    )
    @click.option(
        "--n-workers", default=4, show_default=True, type=int,
        help="Parallel decode workers; one writer thread commits Parquet tiles.",
    )
    @click.option(
        "--max-in-flight",
        default=None,
        type=int,
        help="Max decoded files buffered (default: n_workers + 2).",
    )
    @click.option(
        "--checkpoint",
        type=click.Path(path_type=Path),
        default=None,
    )
    @click.option("--failures-log", type=click.Path(path_type=Path), default=None)
    @click.option("--no-progress", is_flag=True)
    @click.option(
        "--no-skip-completed",
        is_flag=True,
        help="Ignore checkpoint when deciding which files to run.",
    )
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        paths_file: Path,
        output_root: Path | None,
        config_path: Path | None,
        ingest_token: str | None,
        survey_name: str,
        ra_col: str,
        dec_col: str,
        norder: int | None,
        source_id_col: str | None,
        tile_mode: str,
        on_duplicate_id: str,
        overwrite: bool,
        columns: str | None,
        compact: bool,
        n_workers: int,
        max_in_flight: int | None,
        checkpoint: Path | None,
        failures_log: Path | None,
        no_progress: bool,
        no_skip_completed: bool,
        verbose: bool,
    ) -> None:
        """Parallel catalog ingest from a file list (parallel decode, single writer)."""
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        configure_warning_filters()
        cfg = load_optional_config(config_path)
        require_ingest_permission(cfg, ingest_token)
        lake = require_output_root(output_root, cfg, kind="catalogs")
        n = pick(norder, cfg.partitioning.hats_order if cfg else None, 5)
        col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        default_ck = lake / "catalogs" / survey_name / ".ingest_checkpoint.json"

        paths = paths_from_file_list_file(paths_file)
        result = ingest_catalogs_parallel(
            paths,
            output_root=lake,
            survey_name=survey_name,
            n_workers=n_workers,
            ra_col=ra_col,
            dec_col=dec_col,
            norder=n,
            source_id_col=source_id_col,
            columns=col_list,
            tile_mode=tile_mode.lower(),  # type: ignore[arg-type]
            on_duplicate_id=on_duplicate_id.lower(),  # type: ignore[arg-type]
            overwrite=overwrite,
            compact=compact,
            checkpoint_path=checkpoint or default_ck,
            failures_log=failures_log,
            show_progress=not no_progress,
            skip_completed=not no_skip_completed,
            max_in_flight=max_in_flight,
        )
        sys.exit(0 if result["n_files_failed"] == 0 else 1)

except ImportError:
    cli = None  # type: ignore[misc, assignment]
