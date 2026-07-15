"""
repair_catalog_metadata – rebuild catalog sidecars from on-disk Parquet tiles.

Fixes stale or incorrect ``catalog_info.json`` fields (especially
``link_id_column`` / ``link_id_mode``), aggregate ``_metadata``, and
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
    _iter_valid_parquet_tiles,
    finalize_catalog_survey,
    has_padded_column_names,
    rebuild_parquet_tile_link_id,
    reconcile_catalog_column_names,
    resolve_link_id_column,
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
    link_id_column_before: str | None = None
    link_id_column_after: str | None = None
    link_id_mode_before: str | None = None
    link_id_mode_after: str | None = None
    total_rows: int | None = None
    parquet_tiles_rebuilt: int = 0
    tiles_column_renamed: int = 0


@dataclass(frozen=True)
class JoinColumnCheckResult:
    """Per-survey join-column validation."""

    survey: str
    modality: str
    root: Path
    ok: bool
    link_id_mode: str | None = None
    link_id_column: str | None = None
    native_id_column: str | None = None
    tiles_checked: int = 0
    tiles_missing_join: int = 0
    sample_columns: list[str] = field(default_factory=list)
    error: str | None = None


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
    sample: list[str] = []
    for path in tiles:
        names = pq.read_schema(str(path)).names
        if not sample:
            sample = list(names)[:20]
        if LAKE_JOIN_ID_COLUMN not in names:
            missing += 1
    ok = missing == 0
    return JoinColumnCheckResult(
        survey=survey,
        modality="catalog",
        root=catalog_root,
        ok=ok,
        link_id_mode=info.get("link_id_mode"),
        link_id_column=info.get("link_id_column"),
        native_id_column=info.get("native_id_column"),
        tiles_checked=len(tiles),
        tiles_missing_join=missing,
        sample_columns=sample,
        error=None if ok else f"{missing} tile(s) missing {LAKE_JOIN_ID_COLUMN!r}",
    )


def rebuild_catalog_link_ids(
    catalog_root: Path | str,
    link_col: str,
    *,
    allow_incomplete_link_id: bool | None = None,
) -> tuple[int, str]:
    """Recompute ``_source_id`` from *link_col* on every catalog tile.

    Returns ``(n_rebuilt, link_id_mode)``.
    """
    catalog_root = Path(catalog_root)
    if allow_incomplete_link_id is None:
        info_path = catalog_root / "catalog_info.json"
        allow_incomplete_link_id = False
        if info_path.is_file():
            with open(info_path) as fh:
                allow_incomplete_link_id = bool(
                    json.load(fh).get("allow_incomplete_link_id", False),
                )
    rebuilt = 0
    mode = "sequential"
    for path in _iter_valid_parquet_tiles(catalog_root):
        _, mode = rebuild_parquet_tile_link_id(
            path,
            link_col,
            allow_incomplete_link_id=bool(allow_incomplete_link_id),
        )
        rebuilt += 1
    return rebuilt, mode


def repair_catalog_metadata(
    catalog_root: Path | str,
    survey_name: str,
    *,
    norder: int | None = None,
    ra_col: str | None = None,
    dec_col: str | None = None,
    rebuild_link_id: str | None = None,
    allow_incomplete_link_id: bool | None = None,
    normalize_column_names: bool = False,
    show_progress: bool = False,
) -> RepairCatalogMetadataResult:
    """Rebuild metadata sidecars for one ingested catalog from its Parquet tiles."""
    catalog_root = Path(catalog_root)
    before = _read_catalog_info(catalog_root)
    hats_order = int(norder if norder is not None else before.get("hats_order", 5))
    ra = ra_col or str(before.get("ra_column", "ra"))
    dec = dec_col or str(before.get("dec_column", "dec"))
    n_rebuilt = 0
    n_col_renamed = 0
    rebuilt_mode: str | None = None

    try:
        if normalize_column_names:
            n_col_renamed = reconcile_catalog_column_names(catalog_root, show_progress=show_progress)
            if n_col_renamed:
                # After renaming, ra/dec stored in catalog_info.json may now match
                # the stripped names, so re-read the resolved names from Parquet.
                ra = ra.strip()
                dec = dec.strip()
        if rebuild_link_id:
            n_rebuilt, rebuilt_mode = rebuild_catalog_link_ids(
                catalog_root,
                rebuild_link_id,
                allow_incomplete_link_id=allow_incomplete_link_id,
            )
        ok = finalize_catalog_survey(
            catalog_root,
            survey_name,
            hats_order,
            ra_col=ra,
            dec_col=dec,
            link_id_mode=rebuilt_mode,
            allow_incomplete_link_id=allow_incomplete_link_id,
        )
    except Exception as exc:
        log.exception("Repair failed for %s", survey_name)
        return RepairCatalogMetadataResult(
            survey=survey_name,
            catalog_root=catalog_root,
            ok=False,
            error=str(exc),
            link_id_column_before=before.get("link_id_column"),
            link_id_mode_before=before.get("link_id_mode"),
        )

    if not ok:
        return RepairCatalogMetadataResult(
            survey=survey_name,
            catalog_root=catalog_root,
            ok=False,
            error="no valid Parquet tiles",
            link_id_column_before=before.get("link_id_column"),
            link_id_mode_before=before.get("link_id_mode"),
        )

    after = _read_catalog_info(catalog_root)
    return RepairCatalogMetadataResult(
        survey=survey_name,
        catalog_root=catalog_root,
        ok=True,
        link_id_column_before=before.get("link_id_column"),
        link_id_column_after=after.get("link_id_column"),
        link_id_mode_before=before.get("link_id_mode"),
        link_id_mode_after=after.get("link_id_mode"),
        total_rows=after.get("total_rows"),
        parquet_tiles_rebuilt=n_rebuilt,
        tiles_column_renamed=n_col_renamed,
    )


def repair_catalogs_under_lake(
    lake_root: Path | str,
    survey_names: list[str],
    *,
    norder: int | None = None,
    rebuild_link_id: str | None = None,
    allow_incomplete_link_id: bool | None = None,
    normalize_column_names: bool = False,
    show_progress: bool = False,
) -> list[RepairCatalogMetadataResult]:
    """Repair metadata for each named survey under ``<lake_root>/catalogs/``."""
    lake_root = Path(lake_root)
    catalogs_root = lake_root / "catalogs"
    results: list[RepairCatalogMetadataResult] = []

    # Survey-level progress bar (useful when repairing many surveys with --all)
    survey_iter: object
    survey_pbar = None
    if show_progress and len(survey_names) > 1:
        try:
            from tqdm.auto import tqdm as _tqdm
            survey_pbar = _tqdm(survey_names, unit="survey", desc="repair surveys")
            survey_iter = survey_pbar
        except ImportError:
            survey_iter = survey_names
    else:
        survey_iter = survey_names

    try:
        for name in survey_iter:  # type: ignore[union-attr]
            if survey_pbar is not None:
                survey_pbar.set_description(f"repair {name}")
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
                    rebuild_link_id=rebuild_link_id,
                    allow_incomplete_link_id=allow_incomplete_link_id,
                    normalize_column_names=normalize_column_names,
                    show_progress=show_progress,
                )
            )
    finally:
        if survey_pbar is not None:
            survey_pbar.close()

    return results


def check_zarr_join_arrays(survey_root: Path | str) -> JoinColumnCheckResult:
    """Validate that every Zarr tile has ``_source_id``."""
    import zarr

    survey_root = Path(survey_root)
    tiles = list(survey_root.rglob("Npix=*.zarr"))
    missing = 0
    for zp in tiles:
        root = zarr.open_group(
            store=zarr.storage.LocalStore(str(zp)), mode="r",
        )
        keys = list(root.array_keys()) if hasattr(root, "array_keys") else []
        if LAKE_JOIN_ID_COLUMN not in keys:
            missing += 1
    ok = len(tiles) > 0 and missing == 0
    return JoinColumnCheckResult(
        survey=survey_root.name,
        modality="zarr",
        root=survey_root,
        ok=ok,
        tiles_checked=len(tiles),
        tiles_missing_join=missing,
        error=None if ok else f"{missing} Zarr tile(s) missing {LAKE_JOIN_ID_COLUMN!r}",
    )


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
                chk = check_zarr_join_arrays(spec_root)
                out.append(
                    JoinColumnCheckResult(
                        survey=name,
                        modality="spectra",
                        root=spec_root,
                        ok=chk.ok,
                        tiles_checked=chk.tiles_checked,
                        tiles_missing_join=chk.tiles_missing_join,
                        error=chk.error,
                    )
                )
        if include_cutouts:
            cut_root = lake_root / "cutouts" / name
            if cut_root.is_dir():
                chk = check_zarr_join_arrays(cut_root)
                out.append(
                    JoinColumnCheckResult(
                        survey=name,
                        modality="cutouts",
                        root=cut_root,
                        ok=chk.ok,
                        tiles_checked=chk.tiles_checked,
                        tiles_missing_join=chk.tiles_missing_join,
                        error=chk.error,
                    )
                )
    return out


try:
    import click

    from data_lake.cli_utils import (
        config_option,
        configure_cli_logging,
        load_optional_config,
        logging_options,
        progress_option,
        require_output_root,
        resolve_log_level,
        validate_quiet_verbose,
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
        help="Repair every catalog under catalogs/.",
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
        "--rebuild-link-id",
        default=None,
        metavar="COL",
        help=(
            "Recompute catalog _source_id from column COL on every tile (catalog only). "
            "Resets _spectrum_index and _cutout_index; run dl-rebuild-catalog-indices afterward."
        ),
    )
    @click.option(
        "--allow-incomplete-link-id",
        is_flag=True,
        default=None,
        help="With --rebuild-link-id: null _source_id when link parts missing "
        "(default: read catalog_info.json).",
    )
    @click.option("--spectra", "check_spectra", is_flag=True, help="With --check-only, include spectra Zarr tiles.")
    @click.option("--cutouts", "check_cutouts", is_flag=True, help="With --check-only, include cutout Zarr tiles.")
    @click.option(
        "--normalize-column-names",
        is_flag=True,
        help=(
            "Strip leading/trailing whitespace from every Parquet column name. "
            "Fixes FITS TTYPE padding (e.g. ' dec' → 'dec') so DuckDB queries and "
            "crossmatch work without re-ingesting."
        ),
    )
    @click.option(
        "--check-padded-columns",
        is_flag=True,
        help=(
            "Report any padded column names (name != name.strip()) without rewriting tiles. "
            "Implies --check-only behaviour for column names."
        ),
    )
    @progress_option
    @logging_options
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        surveys: tuple[str, ...],
        repair_all: bool,
        norder: int | None,
        check_only: bool,
        rebuild_link_id: str | None,
        allow_incomplete_link_id: bool | None,
        check_spectra: bool,
        check_cutouts: bool,
        normalize_column_names: bool,
        check_padded_columns: bool,
        show_progress: bool,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Rebuild catalog metadata from Parquet tiles (no FITS re-ingest).

        Refreshes ``catalog_info.json`` (``link_id_column`` is always ``_source_id``),
        aggregate ``_metadata``, and ``schema_manifest.json``.

        Examples::

            dl-repair-catalog-metadata /data/lake --survey ultraVISTA_DR6 --check-only
            dl-repair-catalog-metadata /data/lake --survey zCOSMOS_DR3 --rebuild-link-id filename
            dl-repair-catalog-metadata /data/lake --all --check-only --spectra
            dl-repair-catalog-metadata /data/lake --survey ALLWISE --check-padded-columns
            dl-repair-catalog-metadata /data/lake --survey ALLWISE --normalize-column-names
        """
        validate_quiet_verbose(quiet, verbose)
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(quiet=quiet, verbose=verbose,
                                    config_level=cfg.ingest.log_level if cfg else None),
            quiet=quiet,
        )
        lake = require_output_root(output_root, cfg, kind="catalogs")

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

        if check_padded_columns:
            any_padded = False
            for name in names:
                catalog_root = lake / "catalogs" / name
                padded = has_padded_column_names(catalog_root)
                if padded:
                    any_padded = True
                    click.echo(
                        f"{name}: {len(padded)} padded column name(s): {padded!r}  "
                        "→ re-run with --normalize-column-names to fix"
                    )
                else:
                    click.echo(f"{name}: no padded column names")
            if any_padded:
                raise SystemExit(1)
            return

        if check_only:
            failed = False
            checks = check_lake_join_columns(
                lake, names,
                include_spectra=check_spectra,
                include_cutouts=check_cutouts,
            )
            for chk in checks:
                click.echo(
                    f"{chk.survey} [{chk.modality}]: join_col={chk.link_id_column!r} "
                    f"tiles={chk.tiles_checked} missing={chk.tiles_missing_join}"
                )
                if chk.sample_columns:
                    click.echo(f"  sample columns: {chk.sample_columns[:15]}")
                if not chk.ok:
                    failed = True
                    click.echo(f"  → {chk.error}", err=True)
            for name in names:
                try:
                    resolved = resolve_link_id_column(lake / "catalogs" / name)
                    click.echo(f"{name} [catalog]: resolve_link_id_column → {resolved!r}")
                except KeyError as exc:
                    failed = True
                    click.echo(f"{name} [catalog]: resolve failed: {exc}", err=True)
            if failed:
                raise SystemExit(1)
            click.echo("All checked tiles have _source_id.")
            return

        results = repair_catalogs_under_lake(
            lake,
            names,
            norder=norder,
            rebuild_link_id=rebuild_link_id,
            allow_incomplete_link_id=allow_incomplete_link_id,
            normalize_column_names=normalize_column_names,
            show_progress=show_progress and not quiet,
        )
        n_ok = n_fail = 0
        for res in results:
            if not res.ok:
                n_fail += 1
                click.echo(f"{res.survey}: FAILED — {res.error}", err=True)
                continue
            n_ok += 1
            sid_before = res.link_id_column_before or "—"
            sid_after = res.link_id_column_after or "—"
            mode_before = res.link_id_mode_before or "—"
            mode_after = res.link_id_mode_after or "—"
            changed = sid_before != sid_after or mode_before != mode_after
            suffix = ""
            if changed:
                suffix = (
                    f" (link_id_column {sid_before!r} → {sid_after!r}; "
                    f"link_id_mode {mode_before!r} → {mode_after!r})"
                )
            if res.parquet_tiles_rebuilt:
                suffix += f"; {res.parquet_tiles_rebuilt} Parquet tile(s) link-id rebuilt"
            if res.tiles_column_renamed:
                suffix += f"; {res.tiles_column_renamed} tile(s) column names normalized"
            rows = res.total_rows
            row_txt = f", {rows:,} rows" if rows is not None else ""
            click.echo(f"{res.survey}: OK → {res.catalog_root}{row_txt}{suffix}")
            if res.parquet_tiles_rebuilt:
                click.echo(
                    f"  → run dl-rebuild-catalog-indices --survey {res.survey} "
                    f"--kind spectrum (and dl-validate-catalog-spectra-link)"
                )

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
