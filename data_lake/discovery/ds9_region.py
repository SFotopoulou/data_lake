"""Build DS9 region files from lake survey footprints and catalog sources."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence, TextIO

import numpy as np

from data_lake.discovery.region import Region
from data_lake.discovery.tile_index import survey_npix
from data_lake.schema_registry import MODALITY_CATALOG

log = logging.getLogger(__name__)

Ds9Mode = Literal["sources", "footprint", "both"]

_DEFAULT_SOURCE_RADIUS_ARCSEC = 0.5
_DEFAULT_COLOR_SOURCES = "green"
_DEFAULT_COLOR_FOOTPRINT = "cyan"


@dataclass(frozen=True)
class Ds9ExportResult:
    output: Path
    mode: Ds9Mode
    n_footprint: int
    n_sources: int


def healpix_tile_polygon_deg(nside: int, npix: int) -> list[tuple[float, float]]:
    """Return FK5 (RA, Dec) corners in degrees for a NESTED HEALPix pixel."""
    import healpy as hp

    verts = hp.boundaries(int(nside), int(npix), nest=True, step=1)
    # healpy returns shape (3, nverts); vec2ang expects (N, 3) or (3,)
    theta, phi = hp.vec2ang(np.ascontiguousarray(verts.T))
    ra = np.rad2deg(np.asarray(phi, dtype=np.float64))
    dec = 90.0 - np.rad2deg(np.asarray(theta, dtype=np.float64))
    return list(zip(ra.tolist(), dec.tolist()))


def format_ds9_polygon(vertices: Sequence[tuple[float, float]]) -> str:
    if len(vertices) < 3:
        raise ValueError("polygon needs at least 3 vertices")
    parts: list[str] = []
    for ra, dec in vertices:
        parts.append(f"{ra:.8f},{dec:.8f}")
    return "polygon(" + ",".join(parts) + ")"


def format_ds9_circle(
    ra: float,
    dec: float,
    radius_arcsec: float,
) -> str:
    if not np.isfinite(ra) or not np.isfinite(dec):
        raise ValueError("ra/dec must be finite")
    if radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive")
    # DS9: trailing " means arcseconds
    return f'circle({ra:.8f},{dec:.8f},{radius_arcsec:g}")'


def _write_header(fh: TextIO, *, color: str) -> None:
    fh.write("# Region file format: DS9 version 4.1\n")
    fh.write(
        f"global color={color} dashlist=8 3 width=1 "
        'font="helvetica 10 normal roman" select=1 highlite=1 '
        "dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
    )
    fh.write("fk5\n")


def footprint_npix_for_survey(
    lake_root: Path | str,
    survey: str,
    *,
    region: Region | None = None,
    modality: str = MODALITY_CATALOG,
) -> tuple[list[int], int]:
    """Populated tile npix (optionally region-clipped) and hats_order."""
    npix_set, hats_order = survey_npix(lake_root, survey, modality, allow_scan=True)
    if not npix_set:
        raise FileNotFoundError(
            f"No tiles found for {survey!r} modality {modality!r}"
        )
    if hats_order is None:
        from data_lake.io.catalog import CatalogAccessor

        with CatalogAccessor(lake_root, survey) as acc:
            hats_order = int(acc.norder)
    if region is not None:
        npix_set &= region.to_npix(int(hats_order))
    return sorted(int(p) for p in npix_set), int(hats_order)


def write_footprint_regions(
    fh: TextIO,
    npix: Iterable[int],
    *,
    norder: int,
    color: str = _DEFAULT_COLOR_FOOTPRINT,
    write_header: bool = True,
) -> int:
    """Write HEALPix tile polygons; return number of polygons written."""
    import healpy as hp

    nside = hp.order2nside(int(norder))
    if write_header:
        _write_header(fh, color=color)
    else:
        fh.write(f"\n# footprint color={color}\n")
        fh.write(f"global color={color}\n")

    n_written = 0
    for pix in npix:
        verts = healpix_tile_polygon_deg(nside, int(pix))
        fh.write(format_ds9_polygon(verts) + "\n")
        n_written += 1
    return n_written


def _quote_sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def iter_catalog_sky_rows(
    lake_root: Path | str,
    survey: str,
    *,
    where: str | None = None,
    limit: int | None = None,
    region: Region | None = None,
    ra_col: str | None = None,
    dec_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ra, dec)`` float64 arrays for selected catalog rows."""
    from data_lake.export.row_filter import compose_where_clause, format_limit, validate_predicate
    from data_lake.ingest.fits_to_parquet import match_schema_column
    from data_lake.io.catalog import CatalogAccessor

    if where is not None:
        where = validate_predicate(where)
    if limit is not None and int(limit) < 1:
        raise ValueError("--limit must be a positive integer")

    with CatalogAccessor(lake_root, survey) as acc:
        names = acc.columns
        info = acc.info or {}
        ra_req = ra_col or str(info.get("ra_column") or "ra")
        dec_req = dec_col or str(info.get("dec_column") or "dec")
        ra_name = match_schema_column(ra_req, names)
        dec_name = match_schema_column(dec_req, names)
        if ra_name is None or dec_name is None:
            raise ValueError(
                f"Cannot resolve sky columns {ra_req!r}/{dec_req!r} "
                f"in survey {survey!r}; available: {names[:30]}"
            )

        extra: list[str] = [
            f"{_quote_sql_ident(ra_name)} IS NOT NULL",
            f"{_quote_sql_ident(dec_name)} IS NOT NULL",
            f"isfinite({_quote_sql_ident(ra_name)})",
            f"isfinite({_quote_sql_ident(dec_name)})",
            f"abs({_quote_sql_ident(dec_name)}) <= 90",
        ]
        if region is not None:
            npix_list, _order = footprint_npix_for_survey(
                lake_root, survey, region=region,
            )
            if not npix_list:
                return (
                    np.asarray([], dtype=np.float64),
                    np.asarray([], dtype=np.float64),
                )
            hp_col = f"_healpix_norder{acc.norder}"
            if hp_col not in names:
                raise ValueError(
                    f"Region clip requires {hp_col!r} on catalog {survey!r}"
                )
            # Register npix list for a semi-join (avoids huge IN literals).
            import pyarrow as pa

            acc._con.register(
                "ds9_region_npix",
                pa.table({"npix": pa.array(npix_list, type=pa.int64())}),
            )
            extra.append(
                f"{_quote_sql_ident(hp_col)} IN "
                f"(SELECT npix FROM ds9_region_npix)"
            )

        sql = (
            f"SELECT {_quote_sql_ident(ra_name)} AS ra, "
            f"{_quote_sql_ident(dec_name)} AS dec FROM catalog"
            f"{compose_where_clause(where, extra)}"
            f"{format_limit(limit)}"
        )
        log.info("DS9 sources query: %s", sql)
        try:
            table = acc.query(sql, fmt="arrow")
        except Exception as exc:
            raise ValueError(f"invalid --where / region query: {exc}") from exc

    if table.num_rows == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
        )
    ra = np.asarray(table.column("ra").to_numpy(zero_copy_only=False), dtype=np.float64)
    dec = np.asarray(table.column("dec").to_numpy(zero_copy_only=False), dtype=np.float64)
    return ra, dec


def write_source_regions(
    fh: TextIO,
    ra: np.ndarray,
    dec: np.ndarray,
    *,
    radius_arcsec: float = _DEFAULT_SOURCE_RADIUS_ARCSEC,
    color: str = _DEFAULT_COLOR_SOURCES,
    write_header: bool = True,
) -> int:
    """Write source circles; return number of circles written."""
    if write_header:
        _write_header(fh, color=color)
    else:
        fh.write(f"\n# sources color={color}\n")
        fh.write(f"global color={color}\n")

    n_written = 0
    for r, d in zip(np.asarray(ra).ravel(), np.asarray(dec).ravel()):
        if not (np.isfinite(r) and np.isfinite(d)):
            continue
        fh.write(format_ds9_circle(float(r), float(d), float(radius_arcsec)) + "\n")
        n_written += 1
    return n_written


def export_ds9_region(
    lake_root: Path | str,
    survey: str,
    output: Path | str,
    *,
    mode: Ds9Mode = "both",
    where: str | None = None,
    limit: int | None = None,
    region: Region | None = None,
    radius_arcsec: float = _DEFAULT_SOURCE_RADIUS_ARCSEC,
    ra_col: str | None = None,
    dec_col: str | None = None,
    color_sources: str = _DEFAULT_COLOR_SOURCES,
    color_footprint: str = _DEFAULT_COLOR_FOOTPRINT,
    overwrite: bool = False,
) -> Ds9ExportResult:
    """Write a DS9 ``.reg`` for *survey* (footprint and/or sources)."""
    if mode not in ("sources", "footprint", "both"):
        raise ValueError(f"unsupported mode {mode!r}")
    if mode == "footprint" and (where is not None or limit is not None):
        raise ValueError("--where / --limit apply only to sources (or --mode both)")
    if radius_arcsec <= 0:
        raise ValueError("--radius-arcsec must be positive")

    out = Path(output)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)

    n_foot = 0
    n_src = 0
    with open(out, "w", encoding="utf-8") as fh:
        header_done = False
        if mode in ("footprint", "both"):
            npix_list, norder = footprint_npix_for_survey(
                lake_root, survey, region=region,
            )
            n_foot = write_footprint_regions(
                fh,
                npix_list,
                norder=norder,
                color=color_footprint,
                write_header=True,
            )
            header_done = True

        if mode in ("sources", "both"):
            ra, dec = iter_catalog_sky_rows(
                lake_root,
                survey,
                where=where,
                limit=limit,
                region=region,
                ra_col=ra_col,
                dec_col=dec_col,
            )
            n_src = write_source_regions(
                fh,
                ra,
                dec,
                radius_arcsec=radius_arcsec,
                color=color_sources,
                write_header=not header_done,
            )

    log.info(
        "Wrote DS9 region (%s): %d footprint polygon(s), %d source circle(s) → %s",
        mode,
        n_foot,
        n_src,
        out,
    )
    return Ds9ExportResult(
        output=out, mode=mode, n_footprint=n_foot, n_sources=n_src,
    )
