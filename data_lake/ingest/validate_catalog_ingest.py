"""
Validate HATS-partitioned catalog Parquet under ``catalogs/<survey>/``.

Checks ``catalog_info.json``, optional checkpoint / inflight / file-list
coverage (same sidecar convention as spectra ingest), aggregate ``_metadata``,
and a sample of per-tile Parquet files for schema consistency.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from data_lake.ingest.checkpoint_sidecars import validate_ingest_sidecars
from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, LEGACY_JOIN_ID_COLUMN


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


def _iter_catalog_tiles(catalog_root: Path) -> Iterator[Path]:
    yield from sorted(catalog_root.rglob("Npix=*.parquet"))


def validate_catalog_info(catalog_root: Path, rep: ValidationReport) -> dict[str, Any] | None:
    info_path = catalog_root / "catalog_info.json"
    if not info_path.exists():
        rep.errors.append(f"Missing {info_path}")
        return None
    try:
        info = _load_json(info_path)
    except Exception as exc:
        rep.errors.append(f"Invalid JSON in {info_path}: {exc}")
        return None
    for key in ("hats_order", "total_rows", "total_columns", "ra_column", "dec_column"):
        if key not in info:
            rep.errors.append(f"catalog_info.json missing key {key!r}")
    if rep.errors:
        return None
    return info


def validate_catalog_tile(tile_path: Path, info: dict[str, Any], rep: ValidationReport) -> None:
    ra_col = str(info["ra_column"])
    dec_col = str(info["dec_column"])
    try:
        schema = pq.read_schema(str(tile_path))
    except Exception as exc:
        rep.errors.append(f"{tile_path}: cannot read Parquet schema: {exc}")
        return
    names = set(schema.names)
    join_col = str(info.get("source_id_column") or LAKE_JOIN_ID_COLUMN)
    for need in (ra_col, dec_col):
        if need not in names:
            rep.errors.append(f"{tile_path}: schema missing column {need!r}")
    if join_col not in names and LAKE_JOIN_ID_COLUMN not in names and LEGACY_JOIN_ID_COLUMN not in names:
        rep.errors.append(
            f"{tile_path}: schema missing join column {LAKE_JOIN_ID_COLUMN!r} "
            f"(or legacy {LEGACY_JOIN_ID_COLUMN!r})"
        )
    hp = f"_healpix_norder{int(info['hats_order'])}"
    if hp not in names:
        rep.warnings.append(f"{tile_path}: missing expected HEALPix column {hp!r}")


def run_validation(
    lake_root: Path,
    survey: str,
    *,
    file_list: Path | None = None,
    checkpoint_path: Path | None = None,
    inflight_path: Path | None = None,
    max_tiles: int | None = None,
) -> ValidationReport:
    catalog_root = lake_root / "catalogs" / survey
    rep = ValidationReport()
    if not catalog_root.is_dir():
        rep.errors.append(f"Catalog directory does not exist: {catalog_root}")
        return rep

    validate_ingest_sidecars(
        catalog_root,
        rep,
        checkpoint_path=checkpoint_path,
        inflight_path=inflight_path,
        file_list=file_list,
    )

    info = validate_catalog_info(catalog_root, rep)
    if info is None:
        return rep

    meta_path = catalog_root / "_metadata"
    if not meta_path.exists():
        rep.warnings.append(f"No aggregate Parquet footer at {meta_path}")

    tiles = list(_iter_catalog_tiles(catalog_root))
    if not tiles:
        rep.warnings.append(f"No Npix=*.parquet tiles under {catalog_root}")

    if max_tiles is not None:
        tiles = tiles[: max(0, max_tiles)]

    for tp in tiles:
        validate_catalog_tile(tp, info, rep)

    return rep


try:
    import click

    from ..cli_utils import (
        config_option,
        load_optional_config,
        require_output_root,
    )

    @click.command("dl-validate-catalog-ingest")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option("--survey", "survey_name", required=True, help="Survey under catalogs/.")
    @click.option(
        "--file-list",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
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
    )
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
        """Validate catalog Parquet layout and ingest sidecars for one survey."""
        cfg = load_optional_config(config_path)
        lake = require_output_root(output_root, cfg, kind="catalogs")

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
        survey_root = lake / "catalogs" / survey_name
        n_tiles = len(list(_iter_catalog_tiles(survey_root)))
        if rep.ok(strict=strict):
            click.echo(
                f"OK: survey {survey_name!r} under {survey_root} ({n_tiles} tile(s))."
            )
            sys.exit(0)
        click.echo("Validation failed.", err=True)
        sys.exit(1)

except ImportError:
    cli = None  # type: ignore[assignment]
