"""
Validate cutout Zarr tiles and ingest sidecars under ``cutouts/<survey>/``.

Checks ``cutout_info.json``, optional checkpoint / inflight / file-list
coverage, and per-tile array shape consistency (images / source_id / wcs).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from data_lake.ingest.checkpoint_sidecars import validate_ingest_sidecars


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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _iter_cutout_tiles(survey_root: Path) -> Iterator[Path]:
    yield from sorted(survey_root.rglob("Npix=*.zarr"))


def validate_cutout_info(survey_root: Path, rep: ValidationReport) -> dict[str, Any] | None:
    info_path = survey_root / "cutout_info.json"
    if not info_path.exists():
        rep.errors.append(f"Missing {info_path}")
        return None
    try:
        info = _load_json(info_path)
    except Exception as exc:
        rep.errors.append(f"Invalid JSON in {info_path}: {exc}")
        return None
    for key in ("hats_order", "n_bands", "height", "width"):
        if key not in info:
            rep.errors.append(f"cutout_info.json missing key {key!r}")
    if rep.errors:
        return None
    return info


def validate_cutout_tile(tile_path: Path, info: dict[str, Any], rep: ValidationReport) -> None:
    import numpy as np
    import zarr

    zjson = tile_path / "zarr.json"
    if not zjson.exists():
        rep.errors.append(f"Not a Zarr v3 directory (no zarr.json): {tile_path}")
        return

    n_b = int(info["n_bands"])
    h = int(info["height"])
    w = int(info["width"])

    try:
        root = zarr.open_group(
            store=zarr.storage.LocalStore(str(tile_path)),
            mode="r",
            zarr_format=3,
        )
    except Exception as exc:
        rep.errors.append(f"Cannot open Zarr group {tile_path}: {exc}")
        return

    from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN

    for name in ("images", "wcs"):
        if name not in root:
            rep.errors.append(f"{tile_path}: missing array {name!r}")
            return
    if LAKE_JOIN_ID_COLUMN not in root:
        rep.errors.append(
            f"{tile_path}: missing join array {LAKE_JOIN_ID_COLUMN!r}"
        )
        return

    images = root["images"]
    if len(images.shape) != 4:
        rep.errors.append(f"{tile_path}: images must be 4-D, got {images.shape}")
        return
    n_row = int(images.shape[0])
    if tuple(images.shape[1:4]) != (n_b, h, w):
        rep.errors.append(
            f"{tile_path}: images shape {images.shape} "
            f"(expected (*, {n_b}, {h}, {w}) from cutout_info)"
        )

    from data_lake.ingest.zarr_ids import zarr_join_array

    sid = zarr_join_array(root)
    if int(sid.shape[0]) != n_row:
        rep.errors.append(f"{tile_path}: source_id length {sid.shape[0]} != images rows {n_row}")

    wcs = root["wcs"]
    if int(wcs.shape[0]) != n_row:
        rep.errors.append(f"{tile_path}: wcs length {wcs.shape[0]} != images rows {n_row}")

    if n_row > 0:
        sids = np.asarray(sid[:])
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
    survey_root = lake_root / "cutouts" / survey
    rep = ValidationReport()
    if not survey_root.is_dir():
        rep.errors.append(f"Survey directory does not exist: {survey_root}")
        return rep

    validate_ingest_sidecars(
        survey_root,
        rep,
        checkpoint_path=checkpoint_path,
        inflight_path=inflight_path,
        file_list=file_list,
    )

    info = validate_cutout_info(survey_root, rep)
    if info is None:
        return rep

    tiles = list(_iter_cutout_tiles(survey_root))
    if not tiles:
        rep.warnings.append(f"No Npix=*.zarr tiles under {survey_root}")

    if max_tiles is not None:
        tiles = tiles[: max(0, max_tiles)]

    for tp in tiles:
        validate_cutout_tile(tp, info, rep)

    return rep


try:
    import click

    from ..cli_utils import (
        config_option,
        load_optional_config,
        require_output_root,
    )

    @click.command("dl-validate-cutout-ingest")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option("--survey", "survey_name", required=True, help="Survey under cutouts/.")
    @click.option(
        "--file-list",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
    )
    @click.option("--checkpoint", type=click.Path(path_type=Path), default=None)
    @click.option("--inflight", type=click.Path(path_type=Path), default=None)
    @click.option("--max-tiles", type=int, default=None)
    @click.option("--strict", is_flag=True)
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        survey_name: str,
        file_list: Path | None,
        checkpoint: Path | None,
        inflight: Path | None,
        max_tiles: int | None,
        strict: bool,
    ) -> None:
        """Validate cutout Zarr layout and ingest sidecars for one survey."""
        cfg = load_optional_config(config_path)
        lake = require_output_root(output_root, cfg, kind="cutouts")

        rep = run_validation(
            lake,
            survey_name,
            file_list=file_list,
            checkpoint_path=checkpoint,
            inflight_path=inflight,
            max_tiles=max_tiles,
        )
        for msg in rep.errors:
            click.echo(f"ERROR:   {msg}", err=True)
        for msg in rep.warnings:
            click.echo(f"WARNING: {msg}", err=True)
        survey_root = lake / "cutouts" / survey_name
        n_tiles = len(list(_iter_cutout_tiles(survey_root)))
        if rep.ok(strict=strict):
            click.echo(
                f"OK: survey {survey_name!r} under {survey_root} ({n_tiles} tile(s))."
            )
            sys.exit(0)
        click.echo("Validation failed.", err=True)
        sys.exit(1)

except ImportError:
    cli = None  # type: ignore[assignment]
