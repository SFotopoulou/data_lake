"""
Widen spectrum Zarr tiles to the survey ``n_pix`` in ``spectrum_info.json``.

Use after ingest when some tiles stayed at an older pixel width because no
longer spectrum was appended to them.  Ingest widens tiles on append; this
command applies the same ``widen_spectrum_tile`` migration to every tile under
``spectra/<survey>/``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class WidenSurveyResult:
    """Outcome of widening one survey's spectrum tiles."""

    survey: str
    target_n_pix: int
    tiles_scanned: int = 0
    tiles_widened: int = 0
    tiles_already_ok: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _iter_spectrum_tile_zarrs(survey_root: Path) -> Iterator[Path]:
    yield from sorted(survey_root.rglob("Npix=*.zarr"))


def _load_spectrum_info(survey_root: Path) -> dict[str, Any]:
    info_path = survey_root / "spectrum_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _resolution_args(info: dict[str, Any]) -> tuple[int | None, np.ndarray | None]:
    if not info.get("has_resolution"):
        return None, None
    n_diag = info.get("resolution_n_diag")
    offsets = info.get("resolution_offsets")
    if n_diag is None or offsets is None:
        return None, None
    return int(n_diag), np.asarray(offsets, dtype=np.int64)


def widen_survey_tiles(
    lake_root: Path,
    survey: str,
    *,
    target_n_pix: int | None = None,
    dry_run: bool = False,
    max_tiles: int | None = None,
) -> WidenSurveyResult:
    """Widen every tile under ``spectra/<survey>/`` to the target pixel width."""
    from data_lake.ingest.fits_to_spectra_zarr import widen_spectrum_tile

    survey_root = Path(lake_root) / "spectra" / survey
    if not survey_root.is_dir():
        return WidenSurveyResult(
            survey=survey,
            target_n_pix=target_n_pix or 0,
            errors=[f"Survey directory does not exist: {survey_root}"],
        )

    try:
        info = _load_spectrum_info(survey_root)
    except FileNotFoundError as exc:
        return WidenSurveyResult(
            survey=survey,
            target_n_pix=target_n_pix or 0,
            errors=[str(exc)],
        )

    if "n_pix" not in info:
        return WidenSurveyResult(
            survey=survey,
            target_n_pix=target_n_pix or 0,
            errors=[f"{survey_root / 'spectrum_info.json'}: missing key 'n_pix'"],
        )

    survey_n_pix = int(target_n_pix if target_n_pix is not None else info["n_pix"])
    wavelength_mode = str(info.get("wavelength_mode", "shared"))
    mask_dtype = np.dtype(info.get("mask_dtype", np.uint8))
    wcs_attrs = dict(info.get("wcs") or {})
    wcs_attrs.setdefault("n_pix", survey_n_pix)
    n_diag, res_offsets = _resolution_args(info)

    result = WidenSurveyResult(survey=survey, target_n_pix=survey_n_pix)
    tiles = list(_iter_spectrum_tile_zarrs(survey_root))
    if max_tiles is not None:
        tiles = tiles[: max(0, max_tiles)]

    for tile_path in tiles:
        result.tiles_scanned += 1
        zjson = tile_path / "zarr.json"
        if not zjson.is_file():
            result.errors.append(f"Not a Zarr v3 directory (no zarr.json): {tile_path}")
            continue

        try:
            import zarr

            root = zarr.open_group(
                store=zarr.storage.LocalStore(str(tile_path)),
                mode="r",
                zarr_format=3,
            )
            if "flux" not in root or len(root["flux"].shape) != 2:
                result.errors.append(
                    f"{tile_path}: flux missing or not 2-D (shape "
                    f"{getattr(root.get('flux'), 'shape', None)})"
                )
                continue
            tile_n_pix = int(root["flux"].shape[1])
        except Exception as exc:
            result.errors.append(f"Cannot read {tile_path}: {exc}")
            continue

        if tile_n_pix > survey_n_pix:
            result.errors.append(
                f"{tile_path}: flux width {tile_n_pix} > survey n_pix {survey_n_pix} "
                f"(update spectrum_info.json or truncate before widening)"
            )
            continue
        if tile_n_pix >= survey_n_pix:
            result.tiles_already_ok += 1
            continue

        if dry_run:
            log.info(
                "Would widen %s: n_pix %d → %d",
                tile_path.name,
                tile_n_pix,
                survey_n_pix,
            )
            result.tiles_widened += 1
            continue

        try:
            widen_spectrum_tile(
                tile_path,
                survey_n_pix,
                wavelength_mode=wavelength_mode,
                mask_dtype=mask_dtype,
                wcs_attrs=wcs_attrs,
                n_diag=n_diag,
                resolution_offsets=res_offsets,
            )
            result.tiles_widened += 1
        except Exception as exc:
            result.errors.append(f"{tile_path}: widen failed: {exc}")

    return result


try:
    import sys

    import click

    from ..cli_utils import (
        config_option,
        configure_cli_logging,
        load_optional_config,
        logging_options,
        require_output_root,
        resolve_log_level,
        validate_quiet_verbose,
    )
    from .validate_cli import (
        discover_spectra_ingest_surveys,
        echo_survey_banner,
        resolve_validation_survey_names,
        validation_survey_options,
    )

    def _report_widen_result(result: WidenSurveyResult, *, dry_run: bool) -> None:
        prefix = "Would widen" if dry_run else "Widened"
        click.echo(
            f"{result.survey}: scanned {result.tiles_scanned} tile(s); "
            f"{prefix.lower()} {result.tiles_widened}; "
            f"already at n_pix={result.target_n_pix}: {result.tiles_already_ok}"
        )
        for msg in result.errors:
            click.echo(f"ERROR:   {msg}", err=True)

    @click.command("dl-widen-spectrum-tiles")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @validation_survey_options
    @logging_options
    @click.option(
        "--n-pix",
        type=int,
        default=None,
        help="Target width (default: n_pix from spectrum_info.json per survey).",
    )
    @click.option(
        "--dry-run",
        is_flag=True,
        help="Report tiles that would be widened without modifying Zarr stores.",
    )
    @click.option(
        "--max-tiles",
        type=int,
        default=None,
        help="Process only the first N tiles per survey (sorted path order).",
    )
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        surveys: tuple[str, ...],
        validate_all: bool,
        n_pix: int | None,
        dry_run: bool,
        max_tiles: int | None,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Widen spectrum Zarr tiles to the survey pixel width in spectrum_info.json."""
        validate_quiet_verbose(quiet, verbose)
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(quiet=quiet, verbose=verbose,
                                    config_level=cfg.ingest.log_level if cfg else None),
            quiet=quiet,
        )
        lake = require_output_root(output_root, cfg, kind="spectra")

        names = resolve_validation_survey_names(
            surveys=surveys,
            validate_all=validate_all,
            discovered=discover_spectra_ingest_surveys(lake),
            empty_message="No spectrum surveys found under spectra/.",
        )

        all_ok = True
        total_widened = 0
        for i, survey_name in enumerate(names):
            echo_survey_banner(i, survey_name, total=len(names))
            result = widen_survey_tiles(
                lake,
                survey_name,
                target_n_pix=n_pix,
                dry_run=dry_run,
                max_tiles=max_tiles,
            )
            _report_widen_result(result, dry_run=dry_run)
            total_widened += result.tiles_widened
            if not result.ok:
                all_ok = False

        if len(names) > 1:
            click.echo()
            verb = "Would widen" if dry_run else "Widened"
            if all_ok:
                click.echo(
                    f"{verb} {total_widened} tile(s) across {len(names)} survey(s)."
                )
            else:
                click.echo(
                    f"Widen failed for one or more of {len(names)} survey(s).",
                    err=True,
                )

        sys.exit(0 if all_ok else 1)

except ImportError:
    cli = None  # type: ignore[assignment,misc]
