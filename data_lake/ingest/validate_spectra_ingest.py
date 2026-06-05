"""
Validate spectrum Zarr tiles and ingest sidecar files under ``spectra/<survey>/``.

Checks ``spectrum_info.json``, optional checkpoint / inflight / file-list
coverage, and per-tile array shape consistency (flux / ivar / mask /
source_id / meta, and wavelength for shared mode).

``spectrum_info.json`` ``n_pix`` is the survey-wide maximum pixel width.
Individual tiles may be narrower until ingest widens them on append; those
tiles emit a warning (an error with ``--strict``).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from data_lake.ingest.checkpoint_sidecars import (
    paths_from_file_list_file,
    validate_ingest_sidecars,
)

_ROW_ARRAYS = ("flux", "ivar", "mask", "_source_id", "meta")


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def ok(self, *, strict: bool) -> bool:
        if self.errors:
            return False
        if strict and self.warnings:
            return False
        return True


def _iter_tile_zarrs(survey_root: Path) -> Iterator[Path]:
    yield from sorted(survey_root.rglob("Npix=*.zarr"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def validate_spectrum_info(survey_root: Path, rep: ValidationReport) -> dict[str, Any] | None:
    info_path = survey_root / "spectrum_info.json"
    if not info_path.exists():
        rep.errors.append(f"Missing {info_path}")
        return None
    try:
        info = _load_json(info_path)
    except Exception as exc:
        rep.errors.append(f"Invalid JSON in {info_path}: {exc}")
        return None
    for key in ("n_pix", "hats_order", "wavelength_mode", "wcs"):
        if key not in info:
            rep.errors.append(f"spectrum_info.json missing key {key!r}")
    if rep.errors:
        return None
    return info


def validate_sidecars(
    survey_root: Path,
    *,
    checkpoint_path: Path | None,
    inflight_path: Path | None,
    file_list: Path | None,
    rep: ValidationReport,
) -> None:
    validate_ingest_sidecars(
        survey_root,
        rep,
        checkpoint_path=checkpoint_path,
        inflight_path=inflight_path,
        file_list=file_list,
    )


def validate_tile(tile_path: Path, info: dict[str, Any], rep: ValidationReport) -> None:
    import zarr

    zjson = tile_path / "zarr.json"
    if not zjson.exists():
        rep.errors.append(f"Not a Zarr v3 directory (no zarr.json): {tile_path}")
        return

    survey_n_pix = int(info["n_pix"])
    wave_mode = str(info.get("wavelength_mode", "shared"))

    try:
        root = zarr.open_group(
            store=zarr.storage.LocalStore(str(tile_path)),
            mode="r",
            zarr_format=3,
        )
    except Exception as exc:
        rep.errors.append(f"Cannot open Zarr group {tile_path}: {exc}")
        return

    for name in _ROW_ARRAYS:
        if name not in root:
            rep.errors.append(f"{tile_path}: missing array {name!r}")
            return

    flux = root["flux"]
    if len(flux.shape) != 2:
        rep.errors.append(f"{tile_path}: flux must be 2-D, got shape {flux.shape}")
        return
    n_row = int(flux.shape[0])
    tile_n_pix = int(flux.shape[1])
    if tile_n_pix > survey_n_pix:
        rep.errors.append(
            f"{tile_path}: flux shape {flux.shape} exceeds survey n_pix {survey_n_pix} "
            f"in spectrum_info.json (stale or corrupt metadata)"
        )
        return
    if tile_n_pix < survey_n_pix:
        rep.warnings.append(
            f"{tile_path}: flux width {tile_n_pix} < survey n_pix {survey_n_pix} "
            f"(tile not widened; ingest widens on append or re-ingest with pad)"
        )

    for name in _ROW_ARRAYS:
        arr = root[name]
        if int(arr.shape[0]) != n_row:
            rep.errors.append(
                f"{tile_path}: {name} row count {arr.shape[0]} != flux rows {n_row}"
            )
        if name in ("flux", "ivar", "mask") and len(arr.shape) == 2:
            if int(arr.shape[1]) != tile_n_pix:
                rep.errors.append(
                    f"{tile_path}: {name} second dim {arr.shape[1]} != tile n_pix "
                    f"{tile_n_pix}"
                )

    if wave_mode == "shared":
        if "wavelength" not in root:
            rep.errors.append(f"{tile_path}: missing wavelength (shared mode)")
        else:
            w = root["wavelength"]
            if tuple(w.shape) != (tile_n_pix,):
                rep.errors.append(
                    f"{tile_path}: wavelength shape {w.shape} "
                    f"(expected ({tile_n_pix},) for this tile)"
                )
    elif wave_mode == "per_source" and "wavelength" in root:
        w = root["wavelength"]
        if len(w.shape) == 2 and int(w.shape[0]) == n_row:
            if int(w.shape[1]) != tile_n_pix:
                rep.errors.append(
                    f"{tile_path}: wavelength shape {w.shape} "
                    f"(expected ({n_row}, {tile_n_pix}))"
                )
    if info.get("has_resolution") and "resolution" in root:
        res = root["resolution"]
        if int(res.shape[0]) != n_row:
            rep.errors.append(
                f"{tile_path}: resolution row count {res.shape[0]} != flux {n_row}"
            )
        if len(res.shape) == 3 and int(res.shape[2]) != tile_n_pix:
            rep.errors.append(
                f"{tile_path}: resolution pixel dim {res.shape[2]} != tile n_pix "
                f"{tile_n_pix}"
            )

    from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN
    from data_lake.ingest.zarr_ids import zarr_join_array

    if n_row > 0 and LAKE_JOIN_ID_COLUMN in root:
        sids = zarr_join_array(root)[:]
        if len(sids) != n_row:
            rep.errors.append(f"{tile_path}: source_id length mismatch")
        else:
            import numpy as np

            if len(np.unique(sids)) != len(sids):
                rep.warnings.append(
                    f"{tile_path}: duplicate source_id values within tile ({n_row} rows)"
                )


def run_validation(
    lake_root: Path,
    survey: str,
    *,
    file_list: Path | None = None,
    checkpoint_path: Path | None = None,
    inflight_path: Path | None = None,
    max_tiles: int | None = None,
) -> ValidationReport:
    survey_root = lake_root / "spectra" / survey
    rep = ValidationReport()
    if not survey_root.is_dir():
        rep.errors.append(f"Survey directory does not exist: {survey_root}")
        return rep

    validate_sidecars(
        survey_root,
        checkpoint_path=checkpoint_path,
        inflight_path=inflight_path,
        file_list=file_list,
        rep=rep,
    )

    info = validate_spectrum_info(survey_root, rep)
    if info is None:
        return rep

    tiles = list(_iter_tile_zarrs(survey_root))
    if not tiles:
        rep.warnings.append(f"No Npix=*.zarr tiles under {survey_root}")

    if max_tiles is not None:
        tiles = tiles[: max(0, max_tiles)]

    for tp in tiles:
        validate_tile(tp, info, rep)

    return rep


try:
    import click

    from ..cli_utils import (
        config_option,
        configure_cli_logging,
        load_optional_config,
        require_output_root,
    )
    from .validate_cli import (
        discover_spectra_ingest_surveys,
        echo_multi_survey_footer,
        echo_survey_banner,
        print_ingest_validation_messages,
        resolve_validation_survey_names,
        validation_survey_options,
    )

    @click.command("dl-validate-spectra-ingest")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @validation_survey_options
    @click.option(
        "--file-list",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Optional ingest list: warn on entries missing from checkpoint.",
    )
    @click.option(
        "--checkpoint",
        type=click.Path(path_type=Path),
        default=None,
        help="Override checkpoint path (default: survey/.ingest_checkpoint.json).",
    )
    @click.option(
        "--inflight",
        type=click.Path(path_type=Path),
        default=None,
        help="Override inflight journal path (default: survey/.ingest_inflight.json).",
    )
    @click.option(
        "--max-tiles",
        type=int,
        default=None,
        help="Validate only the first N tiles (sorted path order); for quick smoke tests.",
    )
    @click.option(
        "--strict",
        is_flag=True,
        help="Treat warnings (e.g. stale inflight, missing checkpoint files) as errors.",
    )
    @click.option("-q", "--quiet", is_flag=True, default=False,
                  help="Suppress INFO log messages.")
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        surveys: tuple[str, ...],
        validate_all: bool,
        file_list: Path | None,
        checkpoint: Path | None,
        inflight: Path | None,
        max_tiles: int | None,
        strict: bool,
        quiet: bool,
    ) -> None:
        """Validate spectrum Zarr layout and ingest sidecars."""
        import logging as _logging
        configure_cli_logging(level=_logging.WARNING if quiet else _logging.INFO, quiet=quiet)
        cfg = load_optional_config(config_path)
        lake = require_output_root(output_root, cfg, kind="spectra")

        names = resolve_validation_survey_names(
            surveys=surveys,
            validate_all=validate_all,
            discovered=discover_spectra_ingest_surveys(lake),
            empty_message="No spectrum surveys found under spectra/.",
        )

        all_ok = True
        for i, survey_name in enumerate(names):
            echo_survey_banner(i, survey_name, total=len(names))

            rep = run_validation(
                lake,
                survey_name,
                file_list=file_list,
                checkpoint_path=checkpoint,
                inflight_path=inflight,
                max_tiles=max_tiles,
            )
            survey_root = lake / "spectra" / survey_name
            n_tiles = len(list(_iter_tile_zarrs(survey_root)))
            if print_ingest_validation_messages(rep, strict=strict):
                click.echo(
                    f"OK: survey {survey_name!r} under {survey_root} ({n_tiles} tile(s))."
                )
            else:
                click.echo(f"Validation failed for {survey_name!r}.", err=True)
                all_ok = False

        echo_multi_survey_footer(
            all_ok=all_ok,
            n_surveys=len(names),
            ok_message=f"OK: spectrum ingest for all {len(names)} survey(s).",
        )
        sys.exit(0 if all_ok else 1)

except ImportError:
    cli = None  # type: ignore[assignment]
