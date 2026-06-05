"""
recommend_norder – quick pre-ingest scan to suggest HEALPix ``--norder``.

Reads only sky columns (and FITS headers for row counts) from a sample of catalog
files, estimates mean rows per occupied HEALPix pixel at candidate orders, and
picks an order near the README target (10⁴–10⁵ rows per non-empty tile).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from data_lake.ingest.fits_to_parquet import (
    assign_healpix,
    catalog_source_row_count,
    is_catalog_fits_path,
    read_catalog_sky_columns,
)
from data_lake.ingest.fits_to_parquet import (
    _bintable_hdu_index as _fits_bintable_hdu_index,
)
from data_lake.ingest.fits_to_parquet import (
    _is_packed_vector_bintable,
    _read_packed_vector_fits,
)

log = logging.getLogger(__name__)

# README guidance: aim for 10⁴–10⁵ rows per non-empty Parquet tile.
DEFAULT_TARGET_ROWS_PER_TILE = 50_000
DEFAULT_TARGET_MIN = 10_000
DEFAULT_TARGET_MAX = 100_000

# Same discovery patterns as catalog ingest (case variants included).
_DEFAULT_CATALOG_GLOBS = (
    "*.fits",
    "*.fit",
    "*.fz",
    "*.fits.gz",
    "*.FITS",
    "*.FIT",
    "*.FZ",
    "*.FITS.GZ",
)


@dataclass(frozen=True)
class NorderCandidate:
    norder: int
    n_pix_occupied: int
    est_rows_per_tile: float
    est_n_tiles: int
    pixel_sky_frac: float
    pixel_sky_deg2: float


@dataclass(frozen=True)
class NorderRecommendation:
    recommended: int
    total_rows: int
    n_files: int
    sample_rows: int
    ra_col: str
    dec_col: str
    target_rows_per_tile: int
    candidates: tuple[NorderCandidate, ...]
    note: str | None = None


def _bintable_column_names(path: Path) -> list[str]:
    """Return column names from the catalog BINTABLE HDU (for error hints)."""
    from data_lake.io.fits_read import open_fits

    with open_fits(path) as hdul:
        idx = _fits_bintable_hdu_index(hdul)
        hdu = hdul[idx]
        if _is_packed_vector_bintable(hdu):
            tbl = _read_packed_vector_fits(path, hdu_index=idx)
            return list(tbl.colnames)
        if hdu.columns is not None:
            return list(hdu.columns.names)
        data = hdu.data
        if data is not None and data.dtype.names is not None:
            return list(data.dtype.names)
    return []


def _collect_paths(
    paths: Sequence[Path | str],
    *,
    file_list: Path | None,
    glob_pattern: str | None,
) -> list[Path]:
    out: list[Path] = []
    if file_list is not None:
        text = file_list.read_text().splitlines()
        out.extend(Path(line.strip()) for line in text if line.strip() and not line.startswith("#"))
    for p in paths:
        p = Path(p)
        if p.is_dir():
            patterns = (glob_pattern,) if glob_pattern else _DEFAULT_CATALOG_GLOBS
            for pat in patterns:
                out.extend(sorted(p.rglob(pat)))
        else:
            out.append(p)
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen and p.is_file():
            seen.add(rp)
            unique.append(p)
    return unique


def _read_ra_dec_sample(
    path: Path,
    ra_col: str,
    dec_col: str,
    *,
    max_rows: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    ra, dec = read_catalog_sky_columns(path, ra_col, dec_col)
    n = ra.size
    if max_rows is not None and n > max_rows:
        pick = rng.choice(n, size=max_rows, replace=False)
        ra = ra[pick]
        dec = dec[pick]
    return ra, dec


def _pixel_sky_stats(norder: int, n_pix: int) -> tuple[float, float]:
    import healpy as hp

    nside = hp.order2nside(norder)
    n_full = hp.nside2npix(nside)
    area = hp.nside2pixarea(nside, degrees=True)
    return n_pix / n_full, n_pix * area


def recommend_catalog_norder(
    paths: Sequence[Path | str],
    *,
    ra_col: str = "ra",
    dec_col: str = "dec",
    file_list: Path | None = None,
    glob_pattern: str | None = None,
    max_files: int | None = 32,
    sample_rows: int = 500_000,
    norder_min: int = 3,
    norder_max: int = 8,
    target_rows_per_tile: int = DEFAULT_TARGET_ROWS_PER_TILE,
    target_min: int = DEFAULT_TARGET_MIN,
    target_max: int = DEFAULT_TARGET_MAX,
    seed: int = 0,
) -> NorderRecommendation:
    """
    Scan catalog FITS files and recommend ``--norder`` for ingest.

    Uses a random subsample of RA/Dec (``sample_rows`` cap) and total row counts
    from FITS headers to estimate mean rows per occupied HEALPix pixel.
    """
    import healpy as hp  # noqa: F401 — ensure available

    files = _collect_paths(paths, file_list=file_list, glob_pattern=glob_pattern)
    if not files:
        raise FileNotFoundError("No catalog files found to scan.")

    if max_files is not None and len(files) > max_files:
        rng_files = np.random.default_rng(seed)
        pick = rng_files.choice(len(files), size=max_files, replace=False)
        files = [files[i] for i in sorted(pick)]

    rng = np.random.default_rng(seed)
    total_rows = 0
    ra_parts: list[np.ndarray] = []
    dec_parts: list[np.ndarray] = []
    rows_left = sample_rows
    skip_reasons: list[str] = []

    for path in files:
        try:
            n = catalog_source_row_count(path)
        except Exception as exc:
            msg = f"{path.name}: row count — {exc}"
            log.warning("Skip %s", msg)
            skip_reasons.append(msg)
            continue
        total_rows += n
        if rows_left <= 0:
            continue
        take = min(rows_left, max(1, sample_rows // max(len(files), 1)))
        try:
            ra, dec = _read_ra_dec_sample(
                path, ra_col, dec_col, max_rows=take, rng=rng
            )
        except Exception as exc:
            msg = f"{path.name}: RA/Dec — {exc}"
            log.warning("Skip %s", msg)
            skip_reasons.append(msg)
            continue
        if ra.size == 0:
            skip_reasons.append(f"{path.name}: no rows after subsample")
            continue
        ra_parts.append(ra)
        dec_parts.append(dec)
        rows_left -= ra.size

    if not ra_parts:
        hint = ""
        fits_files = [p for p in files if is_catalog_fits_path(p)]
        probe = fits_files[0] if fits_files else (files[0] if files else None)
        if probe is not None:
            try:
                cols = _bintable_column_names(probe)
                if cols:
                    hint = (
                        f" First file ({probe.name}) columns include: "
                        f"{', '.join(cols[:15])}"
                        f"{'…' if len(cols) > 15 else ''}."
                    )
            except Exception:
                pass
        detail = ""
        if skip_reasons:
            detail = " Skips: " + "; ".join(skip_reasons[:5])
            if len(skip_reasons) > 5:
                detail += f" … (+{len(skip_reasons) - 5} more)"
        raise ValueError(
            f"Could not read RA/Dec from any of {len(files)} file(s) "
            f"(columns {ra_col!r}, {dec_col!r}).{hint}{detail}"
        )

    ra_all = np.concatenate(ra_parts)
    dec_all = np.concatenate(dec_parts)
    sample_n = int(ra_all.size)

    candidates: list[NorderCandidate] = []
    for norder in range(norder_min, norder_max + 1):
        pix = assign_healpix(ra_all, dec_all, norder)
        n_pix = int(np.unique(pix).size)
        if n_pix == 0:
            continue
        est_density = total_rows / n_pix
        frac, deg2 = _pixel_sky_stats(norder, n_pix)
        candidates.append(
            NorderCandidate(
                norder=norder,
                n_pix_occupied=n_pix,
                est_rows_per_tile=est_density,
                est_n_tiles=n_pix,
                pixel_sky_frac=frac,
                pixel_sky_deg2=deg2,
            )
        )

    if not candidates:
        raise ValueError("No HEALPix order candidates produced.")

    recommended = _pick_norder(
        candidates,
        target_rows_per_tile=target_rows_per_tile,
        target_min=target_min,
        target_max=target_max,
    )

    note = (
        f"Based on {sample_n:,} sampled RA/Dec rows from {len(files)} file(s) "
        f"({total_rows:,} total rows from headers). "
        "Subsample may underestimate occupied pixels on sparse footprints — "
        "validate with a larger --sample-rows if unsure."
    )

    return NorderRecommendation(
        recommended=recommended,
        total_rows=total_rows,
        n_files=len(files),
        sample_rows=sample_n,
        ra_col=ra_col,
        dec_col=dec_col,
        target_rows_per_tile=target_rows_per_tile,
        candidates=tuple(candidates),
        note=note,
    )


def _pick_norder(
    candidates: list[NorderCandidate],
    *,
    target_rows_per_tile: int,
    target_min: int,
    target_max: int,
) -> int:
    """Prefer order whose estimated density is closest to target, inside [min, max] if possible."""
    in_band = [
        c for c in candidates if target_min <= c.est_rows_per_tile <= target_max
    ]
    pool = in_band if in_band else candidates

    def score(c: NorderCandidate) -> float:
        return abs(
            math.log10(max(c.est_rows_per_tile, 1.0))
            - math.log10(max(target_rows_per_tile, 1.0))
        )

    return min(pool, key=score).norder


def format_recommendation_report(rec: NorderRecommendation) -> str:
    lines = [
        f"Recommended --norder: {rec.recommended}",
        f"Total rows (headers): {rec.total_rows:,}  |  Sampled RA/Dec: {rec.sample_rows:,}  "
        f"|  Files scanned: {rec.n_files}",
        f"Columns: {rec.ra_col!r}, {rec.dec_col!r}  |  "
        f"Target rows/tile: {rec.target_rows_per_tile:,} "
        f"(band {DEFAULT_TARGET_MIN:,}–{DEFAULT_TARGET_MAX:,})",
        "",
        f"{'norder':>6}  {'est_rows/tile':>14}  {'est_tiles':>10}  "
        f"{'pixel_sky_frac':>14}  {'pixel_sky_deg2':>14}",
    ]
    for c in rec.candidates:
        mark = " <--" if c.norder == rec.recommended else ""
        lines.append(
            f"{c.norder:>6}  {c.est_rows_per_tile:>14,.0f}  {c.est_n_tiles:>10,}  "
            f"{c.pixel_sky_frac:>14.4f}  {c.pixel_sky_deg2:>14,.0f}{mark}"
        )
    if rec.note:
        lines.extend(["", rec.note])
    return "\n".join(lines)


try:
    import click

    @click.command("dl-recommend-catalog-norder")
    @click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
    @click.option(
        "--file-list",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Text file with one catalog path per line.",
    )
    @click.option("--ra-col", default="ra", show_default=True)
    @click.option("--dec-col", default="dec", show_default=True)
    @click.option(
        "--glob",
        "glob_pattern",
        default=None,
        help="When PATH is a directory, match this glob (default: *.fits, *.fit, *.fz, …).",
    )
    @click.option("--max-files", default=32, show_default=True,
                  help="Cap number of files to scan (random subset).")
    @click.option("--sample-rows", default=500_000, show_default=True,
                  help="Max RA/Dec rows to load across all files.")
    @click.option("--norder-min", default=3, show_default=True)
    @click.option("--norder-max", default=8, show_default=True)
    @click.option(
        "--target-rows-per-tile",
        default=DEFAULT_TARGET_ROWS_PER_TILE,
        show_default=True,
        help="Aim for this mean rows per occupied HEALPix pixel.",
    )
    @click.option("--seed", default=0, show_default=True)
    @click.option("-v", "--verbose", is_flag=True)
    @click.option("-q", "--quiet", is_flag=True, default=False,
                  help="Suppress INFO output (default is already WARNING for this tool).")
    def cli(
        paths: tuple[Path, ...],
        file_list: Path | None,
        ra_col: str,
        dec_col: str,
        glob_pattern: str | None,
        max_files: int,
        sample_rows: int,
        norder_min: int,
        norder_max: int,
        target_rows_per_tile: int,
        seed: int,
        verbose: bool,
        quiet: bool,
    ) -> None:
        """Quick FITS scan: suggest ``--norder`` before ``dl-ingest-catalog*``."""
        import logging as _logging
        from data_lake.cli_utils import configure_cli_logging, validate_quiet_verbose
        validate_quiet_verbose(quiet, verbose)
        configure_cli_logging(
            level=_logging.DEBUG if verbose else _logging.WARNING,
            quiet=False,
        )
        rec = recommend_catalog_norder(
            paths,
            ra_col=ra_col,
            dec_col=dec_col,
            file_list=file_list,
            glob_pattern=glob_pattern,
            max_files=max_files,
            sample_rows=sample_rows,
            norder_min=norder_min,
            norder_max=norder_max,
            target_rows_per_tile=target_rows_per_tile,
            seed=seed,
        )
        click.echo(format_recommendation_report(rec))

except ImportError:
    cli = None  # type: ignore[misc, assignment]
