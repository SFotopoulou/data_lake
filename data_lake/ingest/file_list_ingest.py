"""
Checkpointed ingest from a text file list (one path per line).

Complements ``dl-ingest-spectra-batch-desi-coadds`` (DESI parallel) with a **sequential**
pattern suitable for catalog FITS and cutout FITS batches: one process,
optional tqdm progress, JSON checkpoint under the survey directory, optional
JSONL failure log.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Callable

from data_lake.ingest.checkpoint_sidecars import paths_from_file_list_file

log = logging.getLogger(__name__)


def _canonical_path(p: Path) -> str:
    return str(p.expanduser().resolve())


def _load_completed(checkpoint: Path) -> set[str]:
    if not checkpoint.exists():
        return set()
    try:
        data = json.loads(checkpoint.read_text())
    except Exception:
        return set()
    out: set[str] = set()
    for item in data.get("completed", []):
        if isinstance(item, str) and item:
            try:
                out.add(_canonical_path(Path(item)))
            except OSError:
                out.add(item)
    return out


def _append_checkpoint(checkpoint: Path, path_done: str) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"completed": []}
    if checkpoint.exists():
        try:
            data = json.loads(checkpoint.read_text())
        except Exception:
            pass
    comp = [str(x) for x in data.get("completed", []) if x]
    if path_done not in comp:
        comp.append(path_done)
    data["completed"] = comp
    tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, checkpoint)


def _run_file_list(
    *,
    paths_file: Path,
    survey_name: str,
    lake_root: Path,
    checkpoint: Path | None,
    failures_log: Path | None,
    show_progress: bool,
    skip_completed: bool,
    ingest_one: Callable[[Path], None],
    default_checkpoint: Path,
) -> int:
    paths = paths_from_file_list_file(paths_file)
    ck = checkpoint or default_checkpoint
    done = _load_completed(ck) if skip_completed else set()
    fail_fh = open(failures_log, "a", encoding="utf-8") if failures_log else None
    n_ok = n_fail = n_skip = 0

    iterator = paths
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(paths, desc="ingest", unit="file")
        except ImportError:
            pass

    for src in iterator:
        key = _canonical_path(src)
        if skip_completed and key in done:
            n_skip += 1
            continue
        try:
            ingest_one(src)
        except Exception as exc:
            n_fail += 1
            log.exception("Failed ingest: %s", src)
            if fail_fh is not None:
                fail_fh.write(
                    json.dumps(
                        {
                            "path": str(src),
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            continue
        n_ok += 1
        if skip_completed:
            _append_checkpoint(ck, key)

    if fail_fh is not None:
        fail_fh.close()

    log.info(
        "File-list ingest finished: ok=%d fail=%d skip=%d checkpoint=%s",
        n_ok, n_fail, n_skip, ck,
    )
    return 0 if n_fail == 0 else 1


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
    from data_lake.ingest.fits_to_parquet import ingest_catalog
    from data_lake.ingest.fits_to_zarr import ingest_cutouts_from_fits

    @click.command("dl-ingest-catalog-from-list")
    @click.argument("paths_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @ingest_token_option
    @click.option("--survey", "survey_name", required=True)
    @click.option("--ra-col", default="ra", show_default=True)
    @click.option("--dec-col", default="dec", show_default=True)
    @click.option("--norder", default=None, type=int)
    @click.option("--link-id-col", default=None)
    @click.option(
        "--tile-mode",
        type=click.Choice(["skip", "overwrite", "append"], case_sensitive=False),
        default=None,
        help="Existing Npix tile: skip (default), replace, or append (multi-FITS).",
    )
    @click.option(
        "--on-duplicate-id",
        type=click.Choice(["skip", "error", "last"], case_sensitive=False),
        default="skip",
        show_default=True,
        help="When --tile-mode=append and an ID column exists.",
    )
    @click.option(
        "--streaming/--no-streaming", default=False, show_default=True,
        help="FITS-only streaming path (passed through to ingest_catalog).",
    )
    @click.option(
        "--checkpoint",
        type=click.Path(path_type=Path),
        default=None,
        help="Checkpoint JSON (default: catalogs/<survey>/.ingest_checkpoint.json).",
    )
    @click.option(
        "--failures-log",
        type=click.Path(path_type=Path),
        default=None,
        help="Append-only JSONL for per-file failures.",
    )
    @click.option("--no-progress", is_flag=True, help="Disable tqdm progress bar.")
    @click.option(
        "--no-skip-completed",
        is_flag=True,
        help="Ignore checkpoint when deciding which files to run.",
    )
    @click.option(
        "--columns",
        default=None,
        help="Comma-separated columns to keep (parallel path only when --n-workers > 1).",
    )
    @click.option(
        "--compact", is_flag=True,
        help="Smaller Parquet tiles (parallel path when --n-workers > 1).",
    )
    @click.option(
        "--n-workers",
        default=1,
        show_default=True,
        type=int,
        help="Decode workers; >1 uses parallel decode + single writer (no --streaming).",
    )
    @click.option(
        "--max-in-flight",
        default=None,
        type=int,
        help="Max decoded files buffered when --n-workers > 1 (default: n_workers + 2).",
    )
    @click.option("-v", "--verbose", is_flag=True)
    def cli_catalog_list(
        paths_file: Path,
        output_root: Path | None,
        config_path: Path | None,
        ingest_token: str | None,
        survey_name: str,
        ra_col: str,
        dec_col: str,
        norder: int | None,
        link_id_col: str | None,
        tile_mode: str | None,
        on_duplicate_id: str,
        streaming: bool,
        checkpoint: Path | None,
        failures_log: Path | None,
        no_progress: bool,
        no_skip_completed: bool,
        columns: str | None,
        compact: bool,
        n_workers: int,
        max_in_flight: int | None,
        verbose: bool,
    ) -> None:
        """Catalog ingest from a text file list (sequential or parallel decode)."""
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        configure_warning_filters()
        cfg = load_optional_config(config_path)
        require_ingest_permission(cfg, ingest_token)
        lake = require_output_root(output_root, cfg, kind="catalogs")
        n = pick(norder, cfg.partitioning.hats_order if cfg else None, 5)
        default_ck = lake / "catalogs" / survey_name / ".ingest_checkpoint.json"
        col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None

        if n_workers < 1:
            raise click.UsageError("--n-workers must be >= 1")
        if n_workers > 1 and streaming:
            raise click.UsageError(
                "--streaming is not supported with --n-workers > 1; "
                "use sequential ingest or dl-ingest-catalog-batch."
            )

        if n_workers > 1:
            from data_lake.ingest.catalog_parallel_ingest import ingest_catalogs_parallel

            paths = paths_from_file_list_file(paths_file)
            result = ingest_catalogs_parallel(
                paths,
                output_root=lake,
                survey_name=survey_name,
                n_workers=n_workers,
                ra_col=ra_col,
                dec_col=dec_col,
                norder=n,
                link_id_col=link_id_col,
                columns=col_list,
                tile_mode=(tile_mode or "append").lower(),  # type: ignore[arg-type]
                on_duplicate_id=on_duplicate_id.lower(),  # type: ignore[arg-type]
                compact=compact,
                checkpoint_path=checkpoint or default_ck,
                failures_log=failures_log,
                show_progress=not no_progress,
                skip_completed=not no_skip_completed,
                max_in_flight=max_in_flight,
            )
            sys.exit(0 if result["n_files_failed"] == 0 else 1)

        def one(p: Path) -> None:
            ingest_catalog(
                source_path=p,
                output_root=lake,
                survey_name=survey_name,
                ra_col=ra_col,
                dec_col=dec_col,
                norder=n,
                link_id_col=link_id_col,
                tile_mode=tile_mode.lower() if tile_mode else None,  # type: ignore[arg-type]
                on_duplicate_id=on_duplicate_id.lower(),  # type: ignore[arg-type]
                streaming=streaming,
                columns=col_list,
                compact=compact,
            )

        code = _run_file_list(
            paths_file=paths_file,
            survey_name=survey_name,
            lake_root=lake,
            checkpoint=checkpoint,
            failures_log=failures_log,
            show_progress=not no_progress,
            skip_completed=not no_skip_completed,
            ingest_one=one,
            default_checkpoint=default_ck,
        )
        sys.exit(code)

    @click.command("dl-ingest-cutouts-from-list")
    @click.argument("paths_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @ingest_token_option
    @click.option("--survey", "survey_name", required=True)
    @click.option("--ra-col", default="RA", show_default=True)
    @click.option("--dec-col", default="DEC", show_default=True)
    @click.option(
        "--link-id-col",
        default=None,
        help="FITS header keyword for object ID (e.g. TARGETID); must match catalog.",
    )
    @click.option("--image-hdu", "image_hdu_index", default=0, type=int, show_default=True)
    @click.option("--band-axis", default=None, type=int)
    @click.option("--norder", default=None, type=int)
    @click.option(
        "--band-names",
        default=None,
        help="Comma-separated band names (stored as Zarr attrs).",
    )
    @click.option(
        "--dtype",
        default="float32",
        show_default=True,
        help="NumPy dtype name for image storage.",
    )
    @click.option(
        "--on-duplicate",
        type=click.Choice(["append", "error", "skip"]),
        default="skip",
        show_default=True,
        help="How to handle source_id already present in a tile Zarr.",
    )
    @click.option("--checkpoint", type=click.Path(path_type=Path), default=None)
    @click.option("--failures-log", type=click.Path(path_type=Path), default=None)
    @click.option("--no-progress", is_flag=True)
    @click.option("--no-skip-completed", is_flag=True)
    @click.option(
        "--update-catalog/--no-update-catalog", default=True, show_default=True,
        help="Patch _cutout_index in the Parquet catalog after ingest "
             "(skipped silently if no catalog exists for this survey).",
    )
    @click.option("-v", "--verbose", is_flag=True)
    def cli_cutout_list(
        paths_file: Path,
        output_root: Path | None,
        config_path: Path | None,
        ingest_token: str | None,
        survey_name: str,
        ra_col: str,
        dec_col: str,
        link_id_col: str | None,
        image_hdu_index: int,
        band_axis: int | None,
        norder: int | None,
        band_names: str | None,
        dtype: str,
        on_duplicate: str,
        checkpoint: Path | None,
        failures_log: Path | None,
        no_progress: bool,
        no_skip_completed: bool,
        update_catalog: bool,
        verbose: bool,
    ) -> None:
        """Sequential cutout ingest from a text file list (one FITS path per line)."""
        import numpy as np

        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        configure_warning_filters()
        cfg = load_optional_config(config_path)
        require_ingest_permission(cfg, ingest_token)
        lake = require_output_root(output_root, cfg, kind="cutouts")
        n = pick(norder, cfg.partitioning.hats_order if cfg else None, 5)
        bn = [x.strip() for x in band_names.split(",")] if band_names else None
        default_ck = lake / "cutouts" / survey_name / ".ingest_checkpoint.json"

        # Accumulate index_map across all files so we patch the catalog once.
        total_index_map: dict[int, int] = {}

        def one(p: Path) -> None:
            m = ingest_cutouts_from_fits(
                source_path=p,
                output_root=lake,
                survey_name=survey_name,
                ra_col=ra_col,
                dec_col=dec_col,
                link_id_col=link_id_col,
                image_hdu_index=image_hdu_index,
                band_axis=band_axis,
                band_names=bn,
                norder=n,
                dtype=np.dtype(dtype),
                on_duplicate_source_id=on_duplicate,  # type: ignore[arg-type]
            )
            total_index_map.update(m)

        code = _run_file_list(
            paths_file=paths_file,
            survey_name=survey_name,
            lake_root=lake,
            checkpoint=checkpoint,
            failures_log=failures_log,
            show_progress=not no_progress,
            skip_completed=not no_skip_completed,
            ingest_one=one,
            default_checkpoint=default_ck,
        )

        if update_catalog and total_index_map:
            try:
                from data_lake.ingest.update_catalog_indices import update_index_column
                n_modified = update_index_column(
                    lake_root=lake,
                    survey_name=survey_name,
                    source_id_to_index=total_index_map,
                    kind="cutout",
                    norder=n,
                    link_id_col=None,
                )
                click.echo(f"Patched _cutout_index in {n_modified} catalog tile(s).")
            except FileNotFoundError:
                log.info(
                    "No catalog found for survey %r — skipping _cutout_index patch.",
                    survey_name,
                )

        sys.exit(code)

    @click.command("dl-ingest-spectra-from-list")
    @click.argument("paths_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @ingest_token_option
    @click.option("--survey", "survey_name", required=True)
    @click.option("--ra-col", default="RA", show_default=True)
    @click.option("--dec-col", default="DEC", show_default=True)
    @click.option("--link-id-col", default=None)
    @click.option("--norder", default=None, type=int)
    @click.option(
        "--fmt",
        default=None,
        type=click.Choice(
            [
                "sdss_boss",
                "sdss_spplate",
                "desi_coadd",
                "generic",
                "2df",
                "gama",
                "6df",
                "wig",
                "ozdes",
                "zcosmos",
                "vandels",
                "vipers",
                "vuds",
                "vvds",
            ],
        ),
        help="Force FITS format (default: auto-detect).",
    )
    @click.option(
        "--specobj-lookup",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Parquet/CSV sidecar for spPlate (survey, PLATE, MJD, FIBERID, SPECOBJID).",
    )
    @click.option(
        "--specobj-lookup-from-catalog/--no-specobj-lookup-from-catalog",
        default=False,
        show_default=True,
    )
    @click.option("--specobj-lookup-survey", default=None)
    @click.option(
        "--specobj-lookup-from-plate/--no-specobj-lookup-from-plate",
        default=False,
        show_default=True,
        help="Synthesize SPECOBJID from spPlate header (no sidecar).",
    )
    @click.option(
        "--specobj-id-layout",
        type=click.Choice(["auto", "dr7", "dr8plus"], case_sensitive=False),
        default="auto",
        show_default=True,
        help="specObjID packing for --specobj-lookup-from-plate (DR7 vs DR8+).",
    )
    @click.option(
        "--on-duplicate",
        type=click.Choice(["append", "error", "skip"]),
        default="skip",
        show_default=True,
        help="If source_id already exists in a tile Zarr, append, raise, or skip.",
    )
    @click.option(
        "--wavelength-mode",
        type=click.Choice(["shared", "per_source"]),
        default=None,
        help="Wavelength storage (overrides config; SDSS auto-uses per_source).",
    )
    @click.option(
        "--on-length-mismatch",
        type=click.Choice(["error", "pad", "truncate"]),
        default="error",
        show_default=True,
        help="When pixel count differs from tile n_pix: error, pad, or truncate "
             "(SDSS ingest defaults to pad when left at error).",
    )
    @click.option("--checkpoint", type=click.Path(path_type=Path), default=None)
    @click.option("--failures-log", type=click.Path(path_type=Path), default=None)
    @click.option("--no-progress", is_flag=True)
    @click.option("--no-skip-completed", is_flag=True)
    @click.option(
        "--update-catalog/--no-update-catalog", default=True, show_default=True,
        help="Patch _spectrum_index in the Parquet catalog after ingest.",
    )
    @click.option("-v", "--verbose", is_flag=True)
    def cli_spectra_list(
        paths_file: Path,
        output_root: Path | None,
        config_path: Path | None,
        ingest_token: str | None,
        survey_name: str,
        ra_col: str,
        dec_col: str,
        link_id_col: str | None,
        norder: int | None,
        fmt: str | None,
        specobj_lookup: Path | None,
        specobj_lookup_from_catalog: bool,
        specobj_lookup_survey: str | None,
        specobj_lookup_from_plate: bool,
        specobj_id_layout: str,
        on_duplicate: str,
        wavelength_mode: str | None,
        on_length_mismatch: str,
        checkpoint: Path | None,
        failures_log: Path | None,
        no_progress: bool,
        no_skip_completed: bool,
        update_catalog: bool,
        verbose: bool,
    ) -> None:
        """Sequential spectrum ingest from a text file list (one FITS path per line)."""
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        configure_warning_filters()
        cfg = load_optional_config(config_path)
        require_ingest_permission(cfg, ingest_token)
        lake = require_output_root(output_root, cfg, kind="spectra")
        n = pick(norder, cfg.partitioning.hats_order if cfg else None, 5)
        default_ck = lake / "spectra" / survey_name / ".ingest_checkpoint.json"

        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits

        total_index_map: dict[int, int] = {}

        def one(p: Path) -> None:
            m = ingest_spectra_from_fits(
                source_path=p,
                output_root=lake,
                survey_name=survey_name,
                ra_col=ra_col,
                dec_col=dec_col,
                link_id_col=link_id_col,
                norder=n,
                fmt=fmt,
                wavelength_mode=pick(
                    wavelength_mode,
                    cfg.defaults.wavelength_mode if cfg else None,
                    "shared",
                ),
                on_length_mismatch=on_length_mismatch,
                on_duplicate_source_id=on_duplicate,  # type: ignore[arg-type]
                specobj_lookup=specobj_lookup,
                specobj_lookup_from_catalog=specobj_lookup_from_catalog,
                specobj_lookup_survey=specobj_lookup_survey,
                specobj_lookup_from_plate=specobj_lookup_from_plate,
                specobj_id_layout=specobj_id_layout.lower(),
            )
            total_index_map.update(m)

        code = _run_file_list(
            paths_file=paths_file,
            survey_name=survey_name,
            lake_root=lake,
            checkpoint=checkpoint,
            failures_log=failures_log,
            show_progress=not no_progress,
            skip_completed=not no_skip_completed,
            ingest_one=one,
            default_checkpoint=default_ck,
        )

        if update_catalog and total_index_map:
            try:
                from data_lake.ingest.update_catalog_indices import update_index_column
                n_modified = update_index_column(
                    lake_root=lake,
                    survey_name=survey_name,
                    source_id_to_index=total_index_map,
                    kind="spectrum",
                    norder=n,
                    link_id_col=None,
                )
                click.echo(f"Patched _spectrum_index in {n_modified} catalog tile(s).")
            except FileNotFoundError:
                log.info(
                    "No catalog found for survey %r — skipping _spectrum_index patch.",
                    survey_name,
                )

        sys.exit(code)

except ImportError:
    cli_catalog_list = None  # type: ignore[assignment,misc]
    cli_cutout_list = None  # type: ignore[assignment,misc]
    cli_spectra_list = None  # type: ignore[assignment,misc]
