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
from dataclasses import dataclass
from pathlib import Path

from data_lake.ingest.fits_to_parquet import finalize_catalog_survey
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


def _read_catalog_info(catalog_root: Path) -> dict:
    info_path = catalog_root / "catalog_info.json"
    if not info_path.is_file():
        return {}
    with open(info_path) as fh:
        return json.load(fh)


def repair_catalog_metadata(
    catalog_root: Path | str,
    survey_name: str,
    *,
    norder: int | None = None,
    ra_col: str | None = None,
    dec_col: str | None = None,
) -> RepairCatalogMetadataResult:
    """
    Rebuild metadata sidecars for one ingested catalog from its Parquet tiles.

    Returns a result with ``ok=False`` when no valid tiles exist under
    *catalog_root*.
    """
    catalog_root = Path(catalog_root)
    before = _read_catalog_info(catalog_root)
    hats_order = int(norder if norder is not None else before.get("hats_order", 5))
    ra = ra_col or str(before.get("ra_column", "ra"))
    dec = dec_col or str(before.get("dec_column", "dec"))

    try:
        ok = finalize_catalog_survey(
            catalog_root,
            survey_name,
            hats_order,
            ra_col=ra,
            dec_col=dec,
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
    )


def repair_catalogs_under_lake(
    lake_root: Path | str,
    survey_names: list[str],
    *,
    norder: int | None = None,
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
            repair_catalog_metadata(catalog_root, name, norder=norder)
        )
    return results


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
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        surveys: tuple[str, ...],
        repair_all: bool,
        norder: int | None,
        verbose: bool,
    ) -> None:
        """Rebuild catalog metadata from Parquet tiles (no FITS re-ingest).

        Refreshes ``catalog_info.json`` (including ``source_id_column``),
        aggregate ``_metadata``, and ``schema_manifest.json``.  Use after
        fixing ID-column logic or when cross-match reports a missing
        ``source_id`` column.

        Examples::

            dl-repair-catalog-metadata /data/lake --survey ultraVISTA_DR6
            dl-repair-catalog-metadata /data/lake --survey A --survey B
            dl-repair-catalog-metadata /data/lake --all
        """
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        configure_warning_filters()
        cfg = load_optional_config(config_path)
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

        results = repair_catalogs_under_lake(lake, names, norder=norder)
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
            changed = (
                sid_before != sid_after
                or mode_before != mode_after
            )
            suffix = ""
            if changed:
                suffix = (
                    f" (source_id_column {sid_before!r} → {sid_after!r}; "
                    f"source_id_mode {mode_before!r} → {mode_after!r})"
                )
            rows = res.total_rows
            row_txt = f", {rows:,} rows" if rows is not None else ""
            click.echo(f"{res.survey}: OK → {res.catalog_root}{row_txt}{suffix}")

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
