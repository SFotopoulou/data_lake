"""
Shared helpers for ``dl-validate-*`` Click commands.

Survey selection is consistent across validators: repeatable ``--survey``,
or ``--all`` to discover every survey under the relevant lake modality tree.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from data_lake.lake_registry import iter_catalog_surveys, iter_modality_surveys

F = TypeVar("F", bound=Callable[..., Any])


def discover_catalog_ingest_surveys(lake_root: Path | str) -> list[str]:
    """Survey names under ``catalogs/`` with ``catalog_info.json`` or HATS tiles."""
    lake_root = Path(lake_root)
    return sorted(name for name, _ in iter_catalog_surveys(lake_root / "catalogs"))


def discover_spectra_ingest_surveys(lake_root: Path | str) -> list[str]:
    """Survey names under ``spectra/`` with ``spectrum_info.json`` or HATS tiles."""
    lake_root = Path(lake_root)
    return sorted(
        name
        for name, _ in iter_modality_surveys(lake_root / "spectra", "spectrum_info.json")
    )


def discover_cutout_ingest_surveys(lake_root: Path | str) -> list[str]:
    """Survey names under ``cutouts/`` with ``cutout_info.json`` or HATS tiles."""
    lake_root = Path(lake_root)
    return sorted(
        name
        for name, _ in iter_modality_surveys(lake_root / "cutouts", "cutout_info.json")
    )


def discover_catalog_spectra_link_surveys(lake_root: Path | str) -> list[str]:
    """Surveys that have both a catalog tree and a spectrum store."""
    lake_root = Path(lake_root)
    catalogs = set(discover_catalog_ingest_surveys(lake_root))
    spectra = set(discover_spectra_ingest_surveys(lake_root))
    return sorted(catalogs & spectra)


def resolve_validation_survey_names(
    *,
    surveys: tuple[str, ...],
    validate_all: bool,
    discovered: list[str],
    empty_message: str,
) -> list[str]:
    """Resolve ``--survey`` / ``--all`` into an ordered survey name list."""
    import click

    if validate_all and surveys:
        raise click.ClickException("Use either --survey or --all, not both.")
    if validate_all:
        names = list(discovered)
    elif surveys:
        names = list(surveys)
    else:
        raise click.ClickException("Provide --survey NAME (repeatable) or --all.")
    if not names:
        raise click.ClickException(empty_message)
    return names


def validation_survey_options(f: F) -> F:
    """Add repeatable ``--survey`` and ``--all`` to a validation CLI command."""
    import click

    f = click.option(
        "--survey",
        "surveys",
        multiple=True,
        help="Survey name(s). Repeat for multiple surveys.",
    )(f)
    f = click.option(
        "--all",
        "validate_all",
        is_flag=True,
        help="Validate every survey discovered under the lake modality tree.",
    )(f)
    return f


def echo_survey_banner(index: int, survey_name: str, *, total: int) -> None:
    """Print a section header when validating more than one survey."""
    import click

    if total <= 1:
        return
    if index > 0:
        click.echo()
    click.echo(f"=== {survey_name} ===")


def print_ingest_validation_messages(
    rep: Any,
    *,
    strict: bool,
) -> bool:
    """Echo ERROR/WARNING lines from a catalog/spectra/cutout validation report."""
    import click

    for msg in rep.errors:
        click.echo(f"ERROR:   {msg}", err=True)
    for msg in rep.warnings:
        click.echo(f"WARNING: {msg}", err=True)
    return bool(rep.ok(strict=strict))


def echo_multi_survey_footer(*, all_ok: bool, n_surveys: int, ok_message: str) -> None:
    """Print a summary line after a multi-survey validation run."""
    import click

    if n_surveys <= 1:
        return
    click.echo()
    if all_ok:
        click.echo(ok_message)
    else:
        click.echo(
            f"Validation failed for one or more of {n_surveys} survey(s).",
            err=True,
        )
