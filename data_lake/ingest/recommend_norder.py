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
    _bintable_hdu_index,
    _is_packed_vector_bintable,
    _read_packed_vector_fits,
    _squeeze_fits_vector_column,
    assign_healpix,
)

log = logging.getLogger(__name__)

# README guidance: aim for 10⁴–10⁵ rows per non-empty Parquet tile.
DEFAULT_TARGET_ROWS_PER_TILE = 50_000
DEFAULT_TARGET_MIN = 10_000
DEFAULT_TARGET_MAX = 100_000


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


def _packed_vector_length(path: Path) -> int:
    try:
        import fitsio

        with fitsio.FITS(str(path)) as fits:
            hdu = fits[_bintable_hdu_index_from_fitsio(fits)]
            for name in hdu.get_colnames():
                return len(_squeeze_fits_vector_column(hdu[name][:]))
    except ImportError:
        tbl = _read_packed_vector_fits(path)
        return len(tbl)


def _bintable_hdu_index_from_fitsio(fits) -> int:
    for i in range(len(fits)):
        if fits[i].get_colnames():
            return i
    return 1


def _bintable_nrows(hdu) -> int:
    """Row count for a FITS BINTABLE HDU."""
    if _is_packed_vector_bintable(hdu):
        raise ValueError("packed-vector table")
    data = hdu.data
    if data is not None:
        return len(data)
    return int(hdu.header.get("NAXIS2", 0))


def _fits_row_count(path: Path) -> int:
    from astropy.io import fits

    with fits.open(path, memmap=True, ignore_missing_simple=True) as hdul:
        idx = _bintable_hdu_index(hdul)
        hdu = hdul[idx]
        if _is_packed_vector_bintable(hdu):
            return _packed_vector_length(path)
        return _bintable_nrows(hdu)


def _resolve_column_name(available: Sequence[str], requested: str) -> str:
    """Match FITS column name exactly or case-insensitively."""
    names = list(available)
    if requested in names:
        return requested
    by_upper = {n.upper(): n for n in names}
    hit = by_upper.get(requested.upper())
    if hit is not None:
        return hit
    preview = ", ".join(names[:12])
    if len(names) > 12:
        preview += f", … (+{len(names) - 12} more)"
    raise KeyError(
        f"Column {requested!r} not in FITS BINTABLE. Available: {preview}"
    )


def _bintable_column_names(path: Path) -> list[str]:
    """Return column names from the catalog BINTABLE HDU (for error hints)."""
    from astropy.io import fits

    with fits.open(path, memmap=True, ignore_missing_simple=True) as hdul:
        idx = _bintable_hdu_index(hdul)
        hdu = hdul[idx]
        if _is_packed_vector_bintable(hdu):
            tbl = _read_packed_vector_fits(path, hdu_index=idx)
            return list(tbl.colnames)
        data = hdu.data
        if data is None or data.dtype.names is None:
            return []
        return list(data.dtype.names)


def _read_ra_dec_fits(
    path: Path,
    ra_col: str,
    dec_col: str,
    *,
    max_rows: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    from astropy.io import fits

    with fits.open(path, memmap=True, ignore_missing_simple=True) as hdul:
        idx = _bintable_hdu_index(hdul)
        hdu = hdul[idx]
        if _is_packed_vector_bintable(hdu):
            tbl = _read_packed_vector_fits(path, hdu_index=idx)
            ra_name = _resolve_column_name(tbl.colnames, ra_col)
            dec_name = _resolve_column_name(tbl.colnames, dec_col)
            ra = np.asarray(tbl[ra_name], dtype=np.float64)
            dec = np.asarray(tbl[dec_name], dtype=np.float64)
        else:
            data = hdu.data
            if data is None:
                raise ValueError(f"Empty BINTABLE in {path.name}")
            col_names = list(data.dtype.names or ())
            ra_name = _resolve_column_name(col_names, ra_col)
            dec_name = _resolve_column_name(col_names, dec_col)
            ra = np.ascontiguousarray(np.asarray(data[ra_name], dtype=np.float64))
            dec = np.ascontiguousarray(np.asarray(data[dec_name], dtype=np.float64))

    n = ra.size
    if max_rows is not None and n > max_rows:
        pick = rng.choice(n, size=max_rows, replace=False)
        ra = ra[pick]
        dec = dec[pick]
    return ra, dec


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
            out.extend(sorted(p.rglob(glob_pattern or "*.fits")))
        else:
            out.append(p)
    # Deduplicate, keep order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen and p.is_file():
            seen.add(rp)
            unique.append(p)
    return unique


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

    for path in files:
        try:
            n = _fits_row_count(path)
        except Exception as exc:
            log.warning("Skip %s (row count): %s", path, exc)
            continue
        total_rows += n
        if rows_left <= 0:
            continue
        take = min(rows_left, max(1, sample_rows // max(len(files), 1)))
        try:
            ra, dec = _read_ra_dec_fits(path, ra_col, dec_col, max_rows=take, rng=rng)
        except Exception as exc:
            log.warning("Skip %s (RA/Dec): %s", path, exc)
            continue
        ra_parts.append(ra)
        dec_parts.append(dec)
        rows_left -= ra.size

    if not ra_parts:
        hint = ""
        if files:
            try:
                cols = _bintable_column_names(files[0])
                if cols:
                    hint = (
                        f" First file ({files[0].name}) BINTABLE columns include: "
                        f"{', '.join(cols[:15])}"
                        f"{'…' if len(cols) > 15 else ''}."
                    )
            except Exception:
                pass
        raise ValueError(
            f"Could not read RA/Dec from any file (columns {ra_col!r}, {dec_col!r})."
            f"{hint} Use -v to see per-file skip reasons."
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
        # Rows per occupied pixel: use total row count / pixels seen in sample.
        # (Subsample can miss sparse pixels; we do not extrapolate n_pix upward.)
        est_n_pix = n_pix
        est_density = total_rows / est_n_pix
        frac, deg2 = _pixel_sky_stats(norder, est_n_pix)
        candidates.append(
            NorderCandidate(
                norder=norder,
                n_pix_occupied=est_n_pix,
                est_rows_per_tile=est_density,
                est_n_tiles=est_n_pix,
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
        return abs(math.log10(max(c.est_rows_per_tile, 1.0)) -
                   math.log10(max(target_rows_per_tile, 1.0)))

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
    @click.option("--glob", "glob_pattern", default="*.fits", show_default=True,
                  help="When PATH is a directory, match this glob.")
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
    def cli(
        paths: tuple[Path, ...],
        file_list: Path | None,
        ra_col: str,
        dec_col: str,
        glob_pattern: str,
        max_files: int,
        sample_rows: int,
        norder_min: int,
        norder_max: int,
        target_rows_per_tile: int,
        seed: int,
        verbose: bool,
    ) -> None:
        """Quick FITS scan: suggest ``--norder`` before ``dl-ingest-catalog*``."""
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.WARNING,
            format="%(levelname)s %(message)s",
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
