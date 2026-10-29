"""
check_fits_table_format – header-only FITS catalog layout probe.

Reports whether ``dl-ingest-catalog*`` will use the fast standard BINTABLE
memmap path or the slow GALEX-style packed-vector (``NAXIS2=1``) reader.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from data_lake.ingest.checkpoint_sidecars import paths_from_file_list_file
from data_lake.ingest.fits_to_parquet import (
    _is_packed_vector_bintable,
    check_hdu_columns_match,
    estimate_bintable_source_count,
    inspect_joinable_table_hdus,
    is_catalog_fits_path,
)


@dataclass(frozen=True)
class FitsTableFormatReport:
    path: str
    ok: bool
    format: str
    file_size_bytes: int | None = None
    #: First selected table HDU (all selected ones are in ``hdu_indices``).
    hdu_index: int | None = None
    hdu_indices: tuple[int, ...] | None = None
    #: Table HDUs in the file (selected or not); >1 means multi-extension.
    n_table_hdus: int | None = None
    naxis2: int | None = None
    n_columns: int | None = None
    est_source_count: int | None = None
    ingest_path: str | None = None
    hint: str | None = None
    #: Astropy FITS warnings for this file (truncation, and similar), one string each.
    warnings: tuple[str, ...] = ()
    error: str | None = None


def _human_size(n_bytes: int | None) -> str:
    if n_bytes is None:
        return "?"
    if n_bytes >= 1024 ** 3:
        return f"{n_bytes / (1024 ** 3):.2f} GiB"
    if n_bytes >= 1024 ** 2:
        return f"{n_bytes / (1024 ** 2):.1f} MiB"
    if n_bytes >= 1024:
        return f"{n_bytes / 1024:.1f} KiB"
    return f"{n_bytes} B"


def _unique_warning_texts(caught: Sequence[warnings.WarningMessage]) -> tuple[str, ...]:
    seen: list[str] = []
    for item in caught:
        text = str(item.message).strip()
        if text and text not in seen:
            seen.append(text)
    return tuple(seen)


def _truncation_warning(path: Path, hdul, on_disk: int) -> str | None:
    """Header-claimed file size vs bytes on disk.

    Astropy only warns when something seeks past EOF, which a header-only
    probe often never does. Compare ``datLoc + datSpan`` ourselves so a short
    file is reported with its path.
    """
    expected = 0
    for hdu in hdul:
        info = hdu.fileinfo()
        if not info:
            continue
        end = int(info.get("datLoc") or 0) + int(info.get("datSpan") or 0)
        expected = max(expected, end)
    if expected > on_disk:
        return (
            "File may have been truncated: actual file length "
            f"({on_disk}) is smaller than the expected size ({expected}) ({path})"
        )
    return None


def inspect_fits_table_format(
    path: Path | str,
    *,
    fits_hdu: str | None = None,
) -> FitsTableFormatReport:
    """Classify one catalog FITS file using headers only (no column reads).

    *fits_hdu* selects table HDU(s) exactly like catalog ingest ``--hdu``;
    counts are summed over the selection.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        return FitsTableFormatReport(
            path=str(path),
            ok=False,
            format="error",
            error=f"cannot stat file: {exc}",
        )

    if not is_catalog_fits_path(path):
        return FitsTableFormatReport(
            path=str(path),
            ok=False,
            format="not-catalog-fits",
            file_size_bytes=size,
            error="suffix is not a catalog FITS type (.fits, .fit, .fz, .fits.gz)",
        )

    try:
        from astropy.io.fits.hdu.table import _TableLikeHDU

        from data_lake.io.fits_read import FitsReadPolicy, open_fits, resolve_table_hdus

        policy = FitsReadPolicy.for_sniff(fits_hdu)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with open_fits(path, policy) as hdul:
                indices = resolve_table_hdus(hdul, policy.table_hdu, source=path.name)
                join_hint = inspect_joinable_table_hdus(hdul)
                try:
                    check_hdu_columns_match(hdul, indices, source=path.name)
                except ValueError as exc:
                    if join_hint:
                        raise ValueError(f"{exc} {join_hint}") from None
                    raise
                n_tables = sum(1 for h in hdul if isinstance(h, _TableLikeHDU))
                hdu = hdul[indices[0]]
                packed = any(_is_packed_vector_bintable(hdul[i]) for i in indices)
                naxis2 = sum(int(hdul[i].header.get("NAXIS2", 0)) for i in indices)
                n_columns = len(hdu.columns) if hdu.columns is not None else 0
                counts = [
                    estimate_bintable_source_count(
                        hdul[i], packed=_is_packed_vector_bintable(hdul[i]),
                    )
                    for i in indices
                ]
                est_sources = None if any(c is None for c in counts) else sum(counts)
                fits_warnings = _unique_warning_texts(caught)
                short = _truncation_warning(path, hdul, size)
                if short is not None and not any("truncated" in w.lower() for w in fits_warnings):
                    fits_warnings = (*fits_warnings, short)
                label = str(path)
                fits_warnings = tuple(
                    w if label in w else f"{w} ({label})" for w in fits_warnings
                )
            if packed:
                fmt = "packed-vector"
                ingest_path = "fitsio column-by-column (slow; all columns read)"
                hint = (
                    "Prefer one-time conversion to row-oriented "
                    "Parquet, or expect long ingest times. Parallel ingest loads the full "
                    "table per worker."
                )
            else:
                fmt = "standard-bintable"
                ingest_path = "astropy memmap / --streaming (fast)"
                hint = (
                    "Normal one-row-per-source BINTABLE. Suitable for parallel ingest "
                    "(within DATA_LAKE_PARALLEL_CATALOG_MAX_BYTES) and --streaming."
                )
            if join_hint:
                hint = join_hint + " " + hint
            elif n_tables > len(indices):
                hint = (
                    f"File has {n_tables} table HDUs; ingest reads HDU(s) "
                    f"{list(indices)}. Use --hdu NAME|INDEX|all to choose. " + hint
                )
            return FitsTableFormatReport(
                path=str(path),
                ok=True,
                format=fmt,
                file_size_bytes=size,
                hdu_index=indices[0],
                hdu_indices=tuple(indices),
                n_table_hdus=n_tables,
                naxis2=naxis2,
                n_columns=n_columns,
                est_source_count=est_sources,
                ingest_path=ingest_path,
                hint=hint,
                warnings=fits_warnings,
            )
    except Exception as exc:
        return FitsTableFormatReport(
            path=str(path),
            ok=False,
            format="error",
            file_size_bytes=size,
            error=f"{type(exc).__name__}: {exc}",
        )


def resolve_inspect_paths(
    paths: Sequence[Path | str],
    *,
    file_list: Path | str | None = None,
) -> list[Path]:
    out: list[Path] = [Path(p) for p in paths]
    if file_list is not None:
        out.extend(paths_from_file_list_file(Path(file_list)))
    seen: set[str] = set()
    unique: list[Path] = []
    for p in out:
        key = str(p.expanduser().resolve())
        if key not in seen:
            seen.add(key)
            unique.append(Path(key))
    return unique


def format_report_text(rep: FitsTableFormatReport) -> str:
    lines = [rep.path]
    if not rep.ok:
        if rep.file_size_bytes is not None:
            lines.append(f"  file size:  {_human_size(rep.file_size_bytes)}")
        lines.append(f"  error:      {rep.error}")
        return "\n".join(lines)
    lines.extend(
        [
            f"  format:     {rep.format}",
            f"  file size:  {_human_size(rep.file_size_bytes)}",
            "  table HDU:  "
            + (
                ", ".join(str(i) for i in rep.hdu_indices)
                if rep.hdu_indices else str(rep.hdu_index)
            )
            + (f" (of {rep.n_table_hdus} table HDUs)" if (rep.n_table_hdus or 0) > 1 else ""),
            f"  NAXIS2:     {rep.naxis2}",
            f"  columns:    {rep.n_columns}",
        ]
    )
    if rep.est_source_count is not None:
        lines.append(f"  sources:    {rep.est_source_count:,}")
    if rep.ingest_path:
        lines.append(f"  ingest:     {rep.ingest_path}")
    if rep.hint:
        lines.append(f"  hint:       {rep.hint}")
    for warning in rep.warnings:
        lines.append(f"  warning:    {warning}")
    return "\n".join(lines)


def inspect_many(
    paths: Sequence[Path | str],
    *,
    fits_hdu: str | None = None,
) -> list[FitsTableFormatReport]:
    return [inspect_fits_table_format(p, fits_hdu=fits_hdu) for p in paths]


def format_summary_text(reports: Sequence[FitsTableFormatReport]) -> str:
    """Aggregate file counts, total on-disk size, and estimated source rows."""
    packed = sum(1 for r in reports if r.format == "packed-vector")
    standard = sum(1 for r in reports if r.format == "standard-bintable")
    errors = sum(1 for r in reports if not r.ok)
    n_warnings = sum(1 for r in reports if r.warnings)
    total_bytes = sum(r.file_size_bytes for r in reports if r.file_size_bytes is not None)
    sized = sum(1 for r in reports if r.file_size_bytes is not None)
    sources_known = [r for r in reports if r.est_source_count is not None]
    total_sources = sum(r.est_source_count for r in sources_known)

    parts = [
        f"Summary: {len(reports)} file(s); "
        f"{standard} standard-bintable, {packed} packed-vector, {errors} error(s), "
        f"{n_warnings} with FITS warning(s).",
    ]
    if n_warnings:
        parts.append(
            "Header source counts for warned files can be larger than the bytes on disk."
        )
    if sized:
        parts.append(f"Total size: {_human_size(total_bytes)} ({sized} file(s)).")
    if sources_known:
        parts.append(
            f"Total sources (est.): {total_sources:,} "
            f"({len(sources_known)} file(s) with header estimate)."
        )
    unknown_sources = len(reports) - len(sources_known) - errors
    if unknown_sources > 0:
        parts.append(f"Sources unknown for {unknown_sources} file(s) (header had no repeat/TDIM).")
    return "\n".join(parts)


try:
    import click

    from data_lake.cli_utils import hdu_option

    @click.command("dl-check-fits-table-format")
    @click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
    @click.option(
        "--file-list",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Text file with one FITS path per line.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Emit JSON lines (one object per file).")
    @click.option(
        "--summary/--no-summary",
        default=True,
        show_default=True,
        help="Print aggregate counts when inspecting multiple files.",
    )
    @hdu_option
    def cli(
        paths: tuple[Path, ...],
        file_list: Path | None,
        as_json: bool,
        summary: bool,
        fits_hdu: str | None,
    ) -> None:
        """Report FITS BINTABLE layout (standard vs vector)."""
        resolved = resolve_inspect_paths(paths, file_list=file_list)
        if not resolved:
            raise click.UsageError("Provide FITS path(s) and/or --file-list.")

        reports = inspect_many(resolved, fits_hdu=fits_hdu)
        if as_json:
            for rep in reports:
                click.echo(json.dumps(asdict(rep), ensure_ascii=False))
        else:
            for rep in reports:
                click.echo(format_report_text(rep))
                click.echo()
            if summary and len(reports) > 1:
                click.echo(format_summary_text(reports))

        sys.exit(0 if all(r.ok and not r.warnings for r in reports) else 1)

except ImportError:
    cli = None  # type: ignore[misc, assignment]
