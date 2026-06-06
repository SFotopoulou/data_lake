"""
check_fits_table_format – header-only FITS catalog layout probe.

Reports whether ``dl-ingest-catalog*`` will use the fast standard BINTABLE
memmap path or the slow GALEX-style packed-vector (``NAXIS2=1``) reader.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from data_lake.ingest.checkpoint_sidecars import paths_from_file_list_file
from data_lake.ingest.fits_to_parquet import (
    _bintable_hdu_index,
    _is_packed_vector_bintable,
    estimate_bintable_source_count,
    is_catalog_fits_path,
)


@dataclass(frozen=True)
class FitsTableFormatReport:
    path: str
    ok: bool
    format: str
    file_size_bytes: int | None = None
    hdu_index: int | None = None
    naxis2: int | None = None
    n_columns: int | None = None
    est_source_count: int | None = None
    ingest_path: str | None = None
    hint: str | None = None
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


def inspect_fits_table_format(path: Path | str) -> FitsTableFormatReport:
    """Classify one catalog FITS file using headers only (no column reads)."""
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
        from data_lake.io.fits_read import FitsReadPolicy, open_fits

        with open_fits(path, FitsReadPolicy.for_sniff()) as hdul:
            idx = _bintable_hdu_index(hdul)
            hdu = hdul[idx]
            packed = _is_packed_vector_bintable(hdu)
            naxis2 = int(hdu.header.get("NAXIS2", 0))
            n_columns = len(hdu.columns) if hdu.columns is not None else 0
            est_sources = estimate_bintable_source_count(hdu, packed=packed)
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
            return FitsTableFormatReport(
                path=str(path),
                ok=True,
                format=fmt,
                file_size_bytes=size,
                hdu_index=idx,
                naxis2=naxis2,
                n_columns=n_columns,
                est_source_count=est_sources,
                ingest_path=ingest_path,
                hint=hint,
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
            f"  table HDU:  {rep.hdu_index}",
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
    return "\n".join(lines)


def inspect_many(paths: Sequence[Path | str]) -> list[FitsTableFormatReport]:
    return [inspect_fits_table_format(p) for p in paths]


def format_summary_text(reports: Sequence[FitsTableFormatReport]) -> str:
    """Aggregate file counts, total on-disk size, and estimated source rows."""
    packed = sum(1 for r in reports if r.format == "packed-vector")
    standard = sum(1 for r in reports if r.format == "standard-bintable")
    errors = sum(1 for r in reports if not r.ok)
    total_bytes = sum(r.file_size_bytes for r in reports if r.file_size_bytes is not None)
    sized = sum(1 for r in reports if r.file_size_bytes is not None)
    sources_known = [r for r in reports if r.est_source_count is not None]
    total_sources = sum(r.est_source_count for r in sources_known)

    parts = [
        f"Summary: {len(reports)} file(s); "
        f"{standard} standard-bintable, {packed} packed-vector, {errors} error(s).",
    ]
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
    def cli(
        paths: tuple[Path, ...],
        file_list: Path | None,
        as_json: bool,
        summary: bool,
    ) -> None:
        """Report FITS BINTABLE layout (standard vs vector)."""
        resolved = resolve_inspect_paths(paths, file_list=file_list)
        if not resolved:
            raise click.UsageError("Provide FITS path(s) and/or --file-list.")

        reports = inspect_many(resolved)
        if as_json:
            for rep in reports:
                click.echo(json.dumps(asdict(rep), ensure_ascii=False))
        else:
            for rep in reports:
                click.echo(format_report_text(rep))
                click.echo()
            if summary and len(reports) > 1:
                click.echo(format_summary_text(reports))

        sys.exit(0 if all(r.ok for r in reports) else 1)

except ImportError:
    cli = None  # type: ignore[misc, assignment]
