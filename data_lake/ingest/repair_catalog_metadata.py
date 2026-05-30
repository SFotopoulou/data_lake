"""
repair_catalog_metadata – rebuild catalog sidecars from on-disk Parquet tiles.

Fixes stale or incorrect ``catalog_info.json`` fields (especially
``source_id_column`` / ``source_id_mode``), aggregate ``_metadata``, and
``schema_manifest.json`` without re-ingesting FITS.

Console entry point: ``dl-repair-catalog-metadata``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import (
    LAKE_JOIN_ID_COLUMN,
    LEGACY_JOIN_ID_COLUMN,
    _iter_valid_parquet_tiles,
    finalize_catalog_survey,
    migrate_parquet_tile_join_column,
    native_id_column_from_mode,
    rebuild_parquet_tile_link_id,
    resolve_source_id_column,
)
from data_lake.lake_registry import iter_catalog_surveys

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepairCatalogMetadataResult:
    """Outcome of repairing one survey catalog."""

    survey: str
    catalog_root: Path
    ok: bool
    error: str | None = None
    source_id_column_before: str | None = None
    source_id_column_after: str | None = None
    source_id_mode_before: str | None = None
    source_id_mode_after: str | None = None
    total_rows: int | None = None
    parquet_tiles_renamed: int = 0
    parquet_tiles_rebuilt: int = 0


@dataclass(frozen=True)
class JoinColumnCheckResult:
    """Per-survey join-column validation."""

    survey: str
    modality: str
    root: Path
    ok: bool
    source_id_mode: str | None = None
    source_id_column: str | None = None
    native_id_column: str | None = None
    tiles_checked: int = 0
    tiles_missing_join: int = 0
    tiles_legacy_only: int = 0
    sample_columns: list[str] = field(default_factory=list)
    error: str | None = None


def migrate_zarr_tile_join_array(tile_path: Path | str) -> str:
    """Rename legacy Zarr ``source_id`` array to ``_source_id``. Returns status string."""
    import numpy as np
    import zarr

    tile_path = Path(tile_path)
    store = zarr.storage.LocalStore(str(tile_path))
    root = zarr.open_group(store=store, mode="r+")
    keys = list(root.array_keys()) if hasattr(root, "array_keys") else [
        k for k in root.keys() if k not in root.attrs and hasattr(root[k], "shape")
    ]
    if LAKE_JOIN_ID_COLUMN in keys:
        if LEGACY_JOIN_ID_COLUMN in keys:
            del root[LEGACY_JOIN_ID_COLUMN]
            return "renamed"
        return "ok"
    if LEGACY_JOIN_ID_COLUMN not in keys:
        return "missing"
    old = root[LEGACY_JOIN_ID_COLUMN]
    data = np.asarray(old[:])
    chunks = getattr(old, "chunks", None) or (min(4096, max(1, len(data))),)
    root.create_array(
        LAKE_JOIN_ID_COLUMN,
        shape=data.shape,
        chunks=chunks,
        dtype=np.int64,
    )
    root[LAKE_JOIN_ID_COLUMN][:] = data
    del root[LEGACY_JOIN_ID_COLUMN]
    return "renamed"


def _read_catalog_info(catalog_root: Path) -> dict:
    info_path = catalog_root / "catalog_info.json"
    if not info_path.is_file():
        return {}
    with open(info_path) as fh:
        return json.load(fh)


def check_catalog_join_column(catalog_root: Path | str, survey: str) -> JoinColumnCheckResult:
    """Validate that every Parquet tile has the lake join column."""
    catalog_root = Path(catalog_root)
    info = _read_catalog_info(catalog_root)
    tiles = _iter_valid_parquet_tiles(catalog_root)
    if not tiles:
        return JoinColumnCheckResult(
            survey=survey,
            modality="catalog",
            root=catalog_root,
            ok=False,
            error="no valid Parquet tiles",
        )
    missing = 0
    legacy_only = 0
    sample: list[str] = []
    for path in tiles:
        names = pq.read_schema(str(path)).names
        if not sample:
            sample = list(names)[:20]
        has_join = LAKE_JOIN_ID_COLUMN in names
        has_legacy = LEGACY_JOIN_ID_COLUMN in names
        if not has_join and has_legacy:
            legacy_only += 1
        if not has_join and not has_legacy:
            missing += 1
    ok = missing == 0 and legacy_only == 0
    return JoinColumnCheckResult(
        survey=survey,
        modality="catalog",
        root=catalog_root,
        ok=ok,
        source_id_mode=info.get("source_id_mode"),
        source_id_column=info.get("source_id_column"),
        native_id_column=info.get("native_id_column"),
        tiles_checked=len(tiles),
        tiles_missing_join=missing,
        tiles_legacy_only=legacy_only,
        sample_columns=sample,
        error=None if ok else (
            f"{missing} tile(s) missing join column; "
            f"{legacy_only} tile(s) have only legacy {LEGACY_JOIN_ID_COLUMN!r}"
        ),
    )


def migrate_catalog_join_columns(
    catalog_root: Path | str,
    *,
    native_col: str | None = None,
) -> tuple[int, int, int]:
    """Rename legacy join column in all catalog tiles. Returns (renamed, ok, missing)."""
    info = _read_catalog_info(Path(catalog_root))
    native = native_col or info.get("native_id_column") or native_id_column_from_mode(
        str(info.get("source_id_mode", "sequential"))
    )
    renamed = ok = missing = 0
    for path in _iter_valid_parquet_tiles(Path(catalog_root)):
        status = migrate_parquet_tile_join_column(path, native_col=native)
        if status == "renamed":
            renamed += 1
        elif status == "ok":
            ok += 1
        else:
            missing += 1
    return renamed, ok, missing


def rebuild_catalog_link_ids(
    catalog_root: Path | str,
    link_col: str,
) -> tuple[int, str]:
    """Recompute ``_source_id`` from *link_col* on every catalog tile.

    Returns ``(n_rebuilt, source_id_mode)``.
    """
    catalog_root = Path(catalog_root)
    rebuilt = 0
    mode = "sequential"
    for path in _iter_valid_parquet_tiles(catalog_root):
        _, mode = rebuild_parquet_tile_link_id(path, link_col)
        rebuilt += 1
    return rebuilt, mode


def migrate_zarr_survey_join_columns(survey_root: Path | str) -> tuple[int, int, int]:
    """Rename legacy join array in all Zarr tiles under a survey."""
    renamed = ok = missing = 0
    survey_root = Path(survey_root)
    for path in sorted(survey_root.rglob("Npix=*.zarr")):
        status = migrate_zarr_tile_join_array(path)
        if status == "renamed":
            renamed += 1
        elif status == "ok":
            ok += 1
        else:
            missing += 1
    return renamed, ok, missing


def repair_catalog_metadata(
    catalog_root: Path | str,
    survey_name: str,
    *,
    norder: int | None = None,
    ra_col: str | None = None,
    dec_col: str | None = None,
    migrate_join_column: bool = False,
    rebuild_link_id: str | None = None,
) -> RepairCatalogMetadataResult:
    """
    Rebuild metadata sidecars for one ingested catalog from its Parquet tiles.

    Returns a result with ``ok=False`` when no valid tiles exist under
    *catalog_root*.
    """
    catalog_root = Path(catalog_root)
    if migrate_join_column and rebuild_link_id:
        raise ValueError(
            "Use either migrate_join_column or rebuild_link_id, not both."
        )
    before = _read_catalog_info(catalog_root)
    hats_order = int(norder if norder is not None else before.get("hats_order", 5))
    ra = ra_col or str(before.get("ra_column", "ra"))
    dec = dec_col or str(before.get("dec_column", "dec"))
    n_renamed = 0
    n_rebuilt = 0
    rebuilt_mode: str | None = None

    try:
        if migrate_join_column:
            n_renamed, _, _ = migrate_catalog_join_columns(catalog_root)
        if rebuild_link_id:
            n_rebuilt, rebuilt_mode = rebuild_catalog_link_ids(
                catalog_root, rebuild_link_id,
            )
        ok = finalize_catalog_survey(
            catalog_root,
            survey_name,
            hats_order,
            ra_col=ra,
            dec_col=dec,
            source_id_mode=rebuilt_mode,
        )
    except Exception as exc:
        log.exception("Repair failed for %s", survey_name)
        return RepairCatalogMetadataResult(
            survey=survey_name,
            catalog_root=catalog_root,
            ok=False,
            error=str(exc),
            source_id_column_before=before.get("source_id_column"),
            source_id_mode_before=before.get("source_id_mode"),
        )

    if not ok:
        return RepairCatalogMetadataResult(
            survey=survey_name,
            catalog_root=catalog_root,
            ok=False,
            error="no valid Parquet tiles",
            source_id_column_before=before.get("source_id_column"),
            source_id_mode_before=before.get("source_id_mode"),
        )

    after = _read_catalog_info(catalog_root)
    return RepairCatalogMetadataResult(
        survey=survey_name,
        catalog_root=catalog_root,
        ok=True,
        source_id_column_before=before.get("source_id_column"),
        source_id_column_after=after.get("source_id_column"),
        source_id_mode_before=before.get("source_id_mode"),
        source_id_mode_after=after.get("source_id_mode"),
        total_rows=after.get("total_rows"),
        parquet_tiles_renamed=n_renamed,
        parquet_tiles_rebuilt=n_rebuilt,
    )


def repair_catalogs_under_lake(
    lake_root: Path | str,
    survey_names: list[str],
    *,
    norder: int | None = None,
    migrate_join_column: bool = False,
    rebuild_link_id: str | None = None,
) -> list[RepairCatalogMetadataResult]:
    """Repair metadata for each named survey under ``<lake_root>/catalogs/``."""
    lake_root = Path(lake_root)
    catalogs_root = lake_root / "catalogs"
    results: list[RepairCatalogMetadataResult] = []
    for name in survey_names:
        catalog_root = catalogs_root / name
        if not catalog_root.is_dir():
            results.append(
                RepairCatalogMetadataResult(
                    survey=name,
                    catalog_root=catalog_root,
                    ok=False,
                    error=f"catalog directory not found: {catalog_root}",
                )
            )
            continue
        results.append(
            repair_catalog_metadata(
                catalog_root,
                name,
                norder=norder,
                migrate_join_column=migrate_join_column,
                rebuild_link_id=rebuild_link_id,
            )
        )
    return results


def check_lake_join_columns(
    lake_root: Path | str,
    survey_names: list[str],
    *,
    include_spectra: bool = True,
    include_cutouts: bool = True,
) -> list[JoinColumnCheckResult]:
    """Validate join columns for catalogs (and optionally Zarr modalities)."""
    lake_root = Path(lake_root)
    out: list[JoinColumnCheckResult] = []
    for name in survey_names:
        cat_root = lake_root / "catalogs" / name
        if cat_root.is_dir():
            out.append(check_catalog_join_column(cat_root, name))
        if include_spectra:
            spec_root = lake_root / "spectra" / name
            if spec_root.is_dir():
                chk = check_catalog_join_column(spec_root, name)
                tiles = list(spec_root.rglob("Npix=*.zarr"))
                legacy = renamed = 0
                for zp in tiles:
                    import zarr
                    root = zarr.open_group(
                        store=zarr.storage.LocalStore(str(zp)), mode="r",
                    )
                    keys = list(root.array_keys()) if hasattr(root, "array_keys") else []
                    if LAKE_JOIN_ID_COLUMN not in keys and LEGACY_JOIN_ID_COLUMN in keys:
                        legacy += 1
                out.append(
                    JoinColumnCheckResult(
                        survey=name,
                        modality="spectra",
                        root=spec_root,
                        ok=legacy == 0 and len(tiles) > 0,
                        tiles_checked=len(tiles),
                        tiles_legacy_only=legacy,
                        error=None if legacy == 0 else f"{legacy} Zarr tile(s) need --migrate-join-column",
                    )
                )
    return out


try:
    import click

    from data_lake.cli_utils import (
        config_option,
        configure_warning_filters,
        load_optional_config,
        require_output_root,
    )

    @click.command("dl-repair-catalog-metadata")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option(
        "--survey",
        "surveys",
        multiple=True,
        help="Survey name(s) under catalogs/. Repeat for multiple surveys.",
    )
    @click.option(
        "--all",
        "repair_all",
        is_flag=True,
        help="Repair every catalog under catalogs/ (except crossmatch/).",
    )
    @click.option(
        "--norder",
        default=None,
        type=int,
        help="HEALPix order override when catalog_info.json is missing.",
    )
    @click.option(
        "--check-only",
        is_flag=True,
        help="Report join-column status; exit 1 if any tile is missing _source_id.",
    )
    @click.option(
        "--migrate-join-column",
        is_flag=True,
        help="Rename legacy source_id → _source_id in Parquet (and Zarr when --spectra/--cutouts).",
    )
    @click.option(
        "--rebuild-link-id",
        default=None,
        metavar="COL",
        help=(
            "Recompute catalog _source_id from column COL on every tile (catalog only). "
            "Resets _spectrum_index and _cutout_index; run dl-rebuild-catalog-indices afterward."
        ),
    )
    @click.option("--spectra", "migrate_spectra", is_flag=True, help="With --migrate-join-column, migrate spectra Zarr tiles.")
    @click.option("--cutouts", "migrate_cutouts", is_flag=True, help="With --migrate-join-column, migrate cutout Zarr tiles.")
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        surveys: tuple[str, ...],
        repair_all: bool,
        norder: int | None,
        check_only: bool,
        migrate_join_column: bool,
        rebuild_link_id: str | None,
        migrate_spectra: bool,
        migrate_cutouts: bool,
        verbose: bool,
    ) -> None:
        """Rebuild catalog metadata from Parquet tiles (no FITS re-ingest).

        Refreshes ``catalog_info.json`` (``source_id_column`` is always ``_source_id``),
        aggregate ``_metadata``, and ``schema_manifest.json``.

        Examples::

            dl-repair-catalog-metadata /data/lake --survey ultraVISTA_DR6 --check-only
            dl-repair-catalog-metadata /data/lake --survey ultraVISTA_DR6 --migrate-join-column
            dl-repair-catalog-metadata /data/lake --survey zCOSMOS_DR3 --rebuild-link-id filename
            dl-repair-catalog-metadata /data/lake --all --migrate-join-column --spectra
        """
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        configure_warning_filters()
        cfg = load_optional_config(config_path)
        lake = require_output_root(output_root, cfg, kind="catalogs")

        if migrate_join_column and rebuild_link_id:
            raise click.ClickException(
                "Use either --migrate-join-column or --rebuild-link-id, not both."
            )

        if repair_all and surveys:
            raise click.ClickException("Use either --survey or --all, not both.")
        if repair_all:
            names = [name for name, _ in iter_catalog_surveys(lake / "catalogs")]
        elif surveys:
            names = list(surveys)
        else:
            raise click.ClickException("Provide --survey NAME (repeatable) or --all.")

        if not names:
            raise click.ClickException(f"No catalogs found under {lake / 'catalogs'}")

        if check_only:
            failed = False
            for name in names:
                chk = check_catalog_join_column(lake / "catalogs" / name, name)
                click.echo(
                    f"{name} [catalog]: mode={chk.source_id_mode!r} "
                    f"join_col={chk.source_id_column!r} native={chk.native_id_column!r} "
                    f"tiles={chk.tiles_checked} missing={chk.tiles_missing_join} "
                    f"legacy_only={chk.tiles_legacy_only}"
                )
                if chk.sample_columns:
                    click.echo(f"  sample columns: {chk.sample_columns[:15]}")
                if not chk.ok:
                    failed = True
                    click.echo(f"  → {chk.error}", err=True)
                try:
                    resolved = resolve_source_id_column(lake / "catalogs" / name)
                    click.echo(f"  resolve_source_id_column → {resolved!r}")
                except KeyError as exc:
                    failed = True
                    click.echo(f"  resolve failed: {exc}", err=True)
            if failed:
                raise SystemExit(1)
            click.echo("All checked catalogs have _source_id on every tile.")
            return

        results = repair_catalogs_under_lake(
            lake,
            names,
            norder=norder,
            migrate_join_column=migrate_join_column,
            rebuild_link_id=rebuild_link_id,
        )
        n_ok = 0
        n_fail = 0
        for res in results:
            if not res.ok:
                n_fail += 1
                click.echo(f"{res.survey}: FAILED — {res.error}", err=True)
                continue
            n_ok += 1
            sid_before = res.source_id_column_before or "—"
            sid_after = res.source_id_column_after or "—"
            mode_before = res.source_id_mode_before or "—"
            mode_after = res.source_id_mode_after or "—"
            changed = sid_before != sid_after or mode_before != mode_after
            suffix = ""
            if changed:
                suffix = (
                    f" (source_id_column {sid_before!r} → {sid_after!r}; "
                    f"source_id_mode {mode_before!r} → {mode_after!r})"
                )
            if res.parquet_tiles_renamed:
                suffix += f"; {res.parquet_tiles_renamed} Parquet tile(s) migrated"
            if res.parquet_tiles_rebuilt:
                suffix += f"; {res.parquet_tiles_rebuilt} Parquet tile(s) link-id rebuilt"
            rows = res.total_rows
            row_txt = f", {rows:,} rows" if rows is not None else ""
            click.echo(f"{res.survey}: OK → {res.catalog_root}{row_txt}{suffix}")
            if res.parquet_tiles_rebuilt:
                click.echo(
                    f"  → run dl-rebuild-catalog-indices --survey {res.survey} "
                    f"--kind spectrum (and dl-validate-catalog-spectra-link)"
                )

        if migrate_join_column and (migrate_spectra or migrate_cutouts):
            for name in names:
                if migrate_spectra:
                    r, o, m = migrate_zarr_survey_join_columns(lake / "spectra" / name)
                    click.echo(f"{name} [spectra zarr]: renamed={r} ok={o} missing={m}")
                if migrate_cutouts:
                    r, o, m = migrate_zarr_survey_join_columns(lake / "cutouts" / name)
                    click.echo(f"{name} [cutouts zarr]: renamed={r} ok={o} missing={m}")

        if n_fail:
            raise SystemExit(1)
        click.echo(f"Repaired {n_ok} catalog(s).")

except ImportError:
    cli = None  # type: ignore[assignment,misc]


def main() -> None:
    """Console entry point for ``dl-repair-catalog-metadata``."""
    if cli is None:
        raise SystemExit(
            "dl-repair-catalog-metadata requires 'click'. "
            "Install the package in this environment: pip install -e ."
        )
    cli()


if __name__ == "__main__":
    main()
