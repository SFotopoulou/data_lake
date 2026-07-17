"""``dl-gather`` – materialise a derived multi-survey catalog product.

Gather joins a base catalog to one or more partner catalogs **via existing
crossmatch trees**, restricted to a :class:`BaseSelection` (region / ids /
predicate). The result is a derived catalog stored under ``catalogs/<name>/``
with ``catalog_info.json`` marked ``kind: "product"`` plus full provenance, so
it is queryable like any catalog and filterable in ``dl-describe-lake --kind
product``.

Semantics:

- **nearest** (default): one partner match per base source (min separation);
  output is one wide row per base source.
- **all**: keep every partner match (fan-out); base rows repeat per match.
- Base-column predicates are applied during selection (filter-then-crossmatch);
  partner-column predicates (``where_joined``) are applied after the join.

Work is tile-bounded: only the selected base tiles (and the matching crossmatch
tiles) are read.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.discovery.partner_tile_cache import PartnerTileCache, PartnerTileCacheConfig
from data_lake.discovery.region import Region
from data_lake.discovery.selection import BaseSelection
from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN, healpix_dir
from data_lake.io.catalog import CatalogAccessor
from data_lake.io.crossmatch import (
    CROSSMATCH_HEALPIX_NPIX_B,
    resolve_column_crossmatch_root,
    resolve_crossmatch_root,
)
from data_lake.schema_registry import CATALOG_KIND_PRODUCT, MODALITY_CATALOG

log = logging.getLogger(__name__)

_MAX_WORKERS = 64
_ZSTD_LEVEL = 7


@dataclass
class PartnerSpec:
    survey: str
    radius_arcsec: float | None = None
    columns: list[str] = field(default_factory=list)
    match_mode: str = "sky"
    match_col_a: str | None = None
    match_col_b: str | None = None


@dataclass
class GatherResult:
    product: str
    output_root: Path
    n_tiles_written: int
    n_rows: int
    n_base_tiles: int
    multiplicity: str
    elapsed_s: float


def _prefixed(survey: str, col: str) -> str:
    return col if col.startswith(f"{survey}_") else f"{survey}_{col}"


def _partner_healpix_pixels_covering_base_tile(
    base_npix: int,
    base_order: int,
    partner_order: int,
) -> set[int]:
    """Partner HEALPix pixels that may hold matches for a base tile.

    Rescales the base pixel to the partner order, then adds a one-pixel neighbour
    ring so sources matched across tile edges are still found.
    """
    import healpy as hp

    from data_lake.discovery.region import rescale_npix_nested

    pixels = rescale_npix_nested([int(base_npix)], base_order, partner_order)
    nside = hp.order2nside(int(partner_order))
    expanded: set[int] = set(pixels)
    for npix in pixels:
        for neighbour in hp.get_all_neighbours(nside, int(npix), nest=True):
            if neighbour >= 0:
                expanded.add(int(neighbour))
    return expanded


def _apply_matches_only_filter(out, partners: list[PartnerSpec], include_sep: bool):
    """Drop base rows with no partner crossmatch (any partner)."""
    import polars as pl

    if not partners:
        return out
    checks = []
    for partner in partners:
        sep_name = f"{partner.survey}_sep_arcsec"
        if include_sep and sep_name in out.columns:
            checks.append(pl.col(sep_name).is_not_null())
        for c in partner.columns:
            col = _prefixed(partner.survey, c)
            if col in out.columns:
                checks.append(pl.col(col).is_not_null())
    if not checks:
        return out
    return out.filter(pl.any_horizontal(checks))


def _fetch_partner_catalog(
    matches,
    *,
    partner: PartnerSpec,
    pacc: CatalogAccessor,
    base_npix: int,
    base_order: int,
    cache: PartnerTileCache | None,
    fetch_cols: list[str],
):
    """Load partner catalog rows for crossmatch matches (exact or geometric tiles)."""
    import polars as pl

    sids_b = matches["source_id_b"].unique().to_list()
    if CROSSMATCH_HEALPIX_NPIX_B in matches.columns:
        partner_npix = [
            int(n) for n in matches[CROSSMATCH_HEALPIX_NPIX_B].unique().to_list()
        ]
    else:
        partner_npix = sorted(
            _partner_healpix_pixels_covering_base_tile(
                base_npix, base_order, pacc.norder,
            )
        )

    if cache is not None and cache.enabled:
        return cache.lookup_many(
            pacc, partner.survey, partner_npix, fetch_cols, sids_b,
        )
    return pacc.get_sources_by_id_in_healpix_pixels(
        sids_b, partner_npix, columns=fetch_cols, fmt="polars",
    )


def _gather_one_tile(
    npix: int,
    *,
    base_acc: CatalogAccessor,
    base_columns: list[str],
    partners: list[PartnerSpec],
    partner_acc: dict[str, CatalogAccessor],
    xm_roots: dict[str, Path],
    base_order: int,
    multiplicity: str,
    include_sep: bool,
    selection_ids: set[int] | None,
    keep_all: bool = True,
    partner_cache: PartnerTileCache | None = None,
):
    """Build the joined polars DataFrame for one base tile (or None if empty)."""
    import polars as pl

    base_id = base_acc.link_id_column
    cols = [base_id] + [c for c in base_columns if c != base_id]
    base_df = base_acc.sources_in_tile(npix, columns=cols, fmt="polars")
    if base_df.is_empty():
        return None
    if selection_ids is not None:
        base_df = base_df.filter(pl.col(base_id).is_in(list(selection_ids)))
        if base_df.is_empty():
            return None

    base_ids = base_df[base_id].to_list()
    out = base_df.rename({base_id: LAKE_JOIN_ID_COLUMN})

    for partner in partners:
        xm_tile = (
            xm_roots[partner.survey]
            / healpix_dir(base_order, npix)
            / f"Npix={npix}.parquet"
        )
        sep_name = f"{partner.survey}_sep_arcsec"
        if not xm_tile.is_file():
            out = _append_null_partner(out, partner, sep_name, include_sep)
            continue
        matches = pl.from_arrow(pq.read_table(str(xm_tile)))
        matches = matches.filter(pl.col("source_id_a").is_in(base_ids))
        if matches.is_empty():
            out = _append_null_partner(out, partner, sep_name, include_sep)
            continue
        if multiplicity == "nearest":
            matches = (
                matches.sort("sep_arcsec")
                .unique(subset=["source_id_a"], keep="first")
            )

        pacc = partner_acc[partner.survey]
        pcols = partner.columns or []
        pid = pacc.link_id_column
        fetch_cols = [pid] + [c for c in pcols if c != pid]
        pdf = _fetch_partner_catalog(
            matches,
            partner=partner,
            pacc=pacc,
            base_npix=npix,
            base_order=base_order,
            cache=partner_cache,
            fetch_cols=fetch_cols,
        )
        rename = {c: _prefixed(partner.survey, c) for c in pcols if c != pid}
        pdf = pdf.rename(rename)
        joined = matches.join(
            pdf, left_on="source_id_b", right_on=pid, how="left"
        )
        keep = ["source_id_a"] + [rename[c] for c in pcols if c != pid]
        if include_sep:
            joined = joined.rename({"sep_arcsec": sep_name})
            keep.append(sep_name)
        joined = joined.select(keep)
        out = out.join(
            joined, left_on=LAKE_JOIN_ID_COLUMN, right_on="source_id_a", how="left"
        )

    out = out.with_columns(
        pl.lit(npix).cast(pl.Int64).alias(f"_healpix_norder{base_order}")
    )
    if not keep_all:
        out = _apply_matches_only_filter(out, partners, include_sep)
        if out.is_empty():
            return None
    return out


def _append_null_partner(out, partner: PartnerSpec, sep_name: str, include_sep: bool):
    import polars as pl

    additions = [
        pl.lit(None, dtype=pl.Float64).alias(_prefixed(partner.survey, c))
        for c in partner.columns
    ]
    if include_sep:
        additions.append(pl.lit(None, dtype=pl.Float32).alias(sep_name))
    return out.with_columns(additions) if additions else out


def gather_product(
    lake_root: Path | str,
    base: str,
    partners: Sequence[PartnerSpec],
    selection: BaseSelection,
    *,
    base_columns: Sequence[str] | None = None,
    multiplicity: str = "nearest",
    include_sep: bool = True,
    materialize_as: str,
    where_joined: str | None = None,
    keep_all: bool = True,
    partner_cache: PartnerTileCache | None = None,
    overwrite: bool = False,
    show_progress: bool = False,
) -> GatherResult:
    """Materialise a derived product catalog joining base x partners over a selection."""
    if multiplicity not in ("nearest", "all"):
        raise ValueError("multiplicity must be 'nearest' or 'all'")
    lake_root = Path(lake_root)
    partners = list(partners)
    out_root = lake_root / "catalogs" / materialize_as
    if out_root.exists() and not overwrite:
        raise FileExistsError(
            f"product catalog already exists: {out_root} (use overwrite=True)"
        )

    base_order = selection.norder
    npix_list = sorted(selection.npix)
    selection_ids = (
        set(selection.source_ids) if selection.source_ids is not None else None
    )

    xm_roots: dict[str, Path] = {}
    for p in partners:
        if p.match_mode == "column":
            if not p.match_col_a or not p.match_col_b:
                raise ValueError(
                    f"column partner {p.survey!r} missing match_col_a/match_col_b; "
                    "set them in crossmatch_plan or PartnerSpec"
                )
            root = resolve_column_crossmatch_root(
                lake_root, base, p.survey, p.match_col_a, p.match_col_b,
            )
        else:
            if p.radius_arcsec is None:
                raise ValueError(
                    f"sky partner {p.survey!r} missing radius_arcsec"
                )
            root = resolve_crossmatch_root(
                lake_root, base, p.survey, p.radius_arcsec,
            )
        xm_roots[p.survey] = root
    for p in partners:
        if not xm_roots[p.survey].is_dir():
            if p.match_mode == "column":
                detail = f"col_a={p.match_col_a!r}, col_b={p.match_col_b!r}"
            else:
                detail = f"r={p.radius_arcsec}"
            raise FileNotFoundError(
                f"crossmatch tree missing for {base} x {p.survey} "
                f"({detail}); run dl-crossmatch first: {xm_roots[p.survey]}"
            )

    t0 = time.perf_counter()
    n_rows = 0
    n_tiles_written = 0
    product_n_columns: int | None = None

    iterator: Any = npix_list
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(npix_list, unit="tile", desc=f"gather {materialize_as}")
        except ImportError:
            pass

    base_acc = CatalogAccessor(lake_root, base, norder=base_order)
    partner_acc = {p.survey: CatalogAccessor(lake_root, p.survey) for p in partners}
    if partner_cache is None:
        partner_cache = PartnerTileCache()
    try:
        import polars as pl

        for npix in iterator:
            df = _gather_one_tile(
                npix,
                base_acc=base_acc,
                base_columns=list(base_columns or []),
                partners=partners,
                partner_acc=partner_acc,
                xm_roots=xm_roots,
                base_order=base_order,
                multiplicity=multiplicity,
                include_sep=include_sep,
                selection_ids=selection_ids,
                keep_all=keep_all,
                partner_cache=partner_cache,
            )
            if df is None or df.is_empty():
                continue
            if where_joined:
                import duckdb

                con = duckdb.connect(":memory:")
                try:
                    con.register("t", df.to_arrow())
                    filtered = con.execute(
                        f"SELECT * FROM t WHERE {where_joined}"
                    ).arrow()
                finally:
                    con.close()
                df = pl.from_arrow(filtered)
                if df.is_empty():
                    continue
            tile_dir = out_root / healpix_dir(base_order, npix)
            tile_dir.mkdir(parents=True, exist_ok=True)
            table = df.to_arrow()
            pq.write_table(
                table,
                str(tile_dir / f"Npix={npix}.parquet"),
                compression="zstd",
                compression_level=_ZSTD_LEVEL,
            )
            n_rows += table.num_rows
            n_tiles_written += 1
            if product_n_columns is None:
                product_n_columns = table.num_columns
    finally:
        base_acc.close()
        for acc in partner_acc.values():
            acc.close()

    _write_product_info(
        out_root,
        materialize_as,
        lake_root=lake_root,
        base=base,
        base_order=base_order,
        partners=partners,
        selection=selection,
        base_columns=list(base_columns or []),
        multiplicity=multiplicity,
        include_sep=include_sep,
        where_joined=where_joined,
        keep_all=keep_all,
        n_rows=n_rows,
        n_columns=product_n_columns,
    )

    return GatherResult(
        product=materialize_as,
        output_root=out_root,
        n_tiles_written=n_tiles_written,
        n_rows=n_rows,
        n_base_tiles=len(npix_list),
        multiplicity=multiplicity,
        elapsed_s=time.perf_counter() - t0,
    )


def _base_catalog_info(lake_root: Path, base: str) -> dict[str, Any]:
    info_path = lake_root / "catalogs" / base / "catalog_info.json"
    if not info_path.is_file():
        return {}
    with open(info_path) as fh:
        return json.load(fh)


def _write_product_info(
    out_root: Path,
    name: str,
    *,
    lake_root: Path,
    base: str,
    base_order: int,
    partners: list[PartnerSpec],
    selection: BaseSelection,
    base_columns: list[str],
    multiplicity: str,
    include_sep: bool,
    where_joined: str | None,
    keep_all: bool,
    n_rows: int,
    n_columns: int | None = None,
) -> None:
    base_info = _base_catalog_info(lake_root, base)
    info = {
        "catalog_name": name,
        "kind": CATALOG_KIND_PRODUCT,
        "modality": MODALITY_CATALOG,
        "hats_order": base_order,
        "link_id_mode": "column:" + LAKE_JOIN_ID_COLUMN,
        "link_id_column": LAKE_JOIN_ID_COLUMN,
        "ra_column": base_info.get("ra_column"),
        "dec_column": base_info.get("dec_column"),
        "total_rows": n_rows,
        "total_columns": n_columns,
        "n_columns": n_columns,
        "provenance": {
            "base_catalog": base,
            "base_columns": base_columns,
            "partners": [
                {
                    "survey": p.survey,
                    "match_mode": p.match_mode,
                    "radius_arcsec": p.radius_arcsec,
                    "match_col_a": p.match_col_a,
                    "match_col_b": p.match_col_b,
                    "columns": p.columns,
                }
                for p in partners
            ],
            "selection": {
                "base_survey": selection.base_survey,
                "norder": selection.norder,
                "n_npix": len(selection.npix),
                "n_source_ids": (
                    None if selection.source_ids is None else len(selection.source_ids)
                ),
            },
            "multiplicity": multiplicity,
            "include_sep": include_sep,
            "where_joined": where_joined,
            "keep_all": keep_all,
        },
        "schema_version": "1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "catalog_info.json", "w") as fh:
        json.dump(info, fh, indent=2)


def _product_source_ids(lake_root: Path, product: str) -> list[int]:
    with CatalogAccessor(lake_root, product) as acc:
        tbl = acc.query(
            f"SELECT DISTINCT {LAKE_JOIN_ID_COLUMN} FROM catalog", fmt="arrow"
        )
    if tbl.num_rows == 0:
        return []
    return [int(v) for v in tbl.column(LAKE_JOIN_ID_COLUMN).to_pylist() if v is not None]


def _extract_cutouts_to_fits(
    lake_root: Path,
    survey: str,
    source_ids: Sequence[int],
    out_dir: Path,
    *,
    missing: str = "skip",
) -> dict[str, Any]:
    from astropy.io import fits

    from data_lake.io.cutouts import CutoutAccessor

    out_dir.mkdir(parents=True, exist_ok=True)
    acc = CutoutAccessor(lake_root, survey)
    written = 0
    missing_ids: list[int] = []
    for sid in source_ids:
        try:
            image, wcs = acc.get_cutout(int(sid))
        except (KeyError, FileNotFoundError):
            missing_ids.append(int(sid))
            if missing == "error":
                raise
            continue
        header = wcs.to_fits_header() if hasattr(wcs, "to_fits_header") else None
        fits.PrimaryHDU(data=image, header=header).writeto(
            out_dir / f"cutout_{sid}.fits", overwrite=True
        )
        written += 1
    return {"survey": survey, "n_written": written, "missing": len(missing_ids)}


def extract_modalities_for_product(
    lake_root: Path | str,
    product: str,
    modalities: Sequence[str],
    output_dir: Path | str,
    *,
    survey: str | None = None,
    missing: str = "skip",
) -> dict[str, Any]:
    """Extract non-catalog modalities for a product's sources into a portable bundle.

    The product catalog drives the source-id list (its ``_source_id`` column).
    Heavy data (spectra/cutouts) is written **outside** the lake under
    ``output_dir`` and never duplicated in-lake. ``survey`` defaults to the
    product's base catalog.
    """
    lake_root = Path(lake_root)
    output_dir = Path(output_dir)
    info_path = lake_root / "catalogs" / product / "catalog_info.json"
    base_survey = survey
    if base_survey is None:
        try:
            with open(info_path) as fh:
                base_survey = json.load(fh).get("provenance", {}).get("base_catalog")
        except (OSError, json.JSONDecodeError):
            base_survey = None
    if not base_survey:
        raise ValueError(
            "could not determine source survey for modality extraction; pass survey="
        )

    source_ids = _product_source_ids(lake_root, product)
    results: dict[str, Any] = {"survey": base_survey, "n_sources": len(source_ids)}
    if not source_ids:
        return results

    import numpy as np

    for modality in modalities:
        if modality == "spectra":
            from data_lake.io.spectra import SpectrumAccessor

            acc = SpectrumAccessor(lake_root=lake_root, survey_name=base_survey)
            out = output_dir / f"spectra_{base_survey}.zarr"
            res = acc.extract_subset(
                source_ids=np.asarray(source_ids, dtype=np.int64),
                output=out,
                fmt="zarr",
                missing=missing,
                show_progress=False,
                overwrite=True,
            )
            results["spectra"] = {
                "output": str(res.get("output", out)),
                "n_written": res.get("n_written"),
            }
        elif modality == "cutout":
            res = _extract_cutouts_to_fits(
                lake_root,
                base_survey,
                source_ids,
                output_dir / f"cutouts_{base_survey}",
                missing=missing,
            )
            results["cutout"] = res
        else:
            raise ValueError(f"cannot extract modality {modality!r}; expected spectra|cutout")
    return results


def partners_from_columns(
    columns: dict[str, list[str]],
    radii: dict[str, float],
    base: str,
    *,
    column_partners: dict[str, dict] | None = None,
) -> list[PartnerSpec]:
    """Build PartnerSpec list from a ``{survey: [cols]}`` mapping (excludes base).

    *radii* supplies sky partners.  *column_partners* maps survey → column-mode
    plan fields (``match_col_a``, ``match_col_b``).  A survey must appear in
    exactly one of the two maps.
    """
    column_partners = column_partners or {}
    specs: list[PartnerSpec] = []
    for survey, cols in columns.items():
        if survey == base:
            continue
        if survey in column_partners:
            meta = column_partners[survey]
            specs.append(PartnerSpec(
                survey=survey,
                radius_arcsec=None,
                columns=list(cols),
                match_mode="column",
                match_col_a=str(meta.get("match_col_a", "")),
                match_col_b=str(meta.get("match_col_b", "")),
            ))
            continue
        radius = radii.get(survey)
        if radius is None:
            raise ValueError(
                f"no crossmatch radius or column match given for partner {survey!r}"
            )
        specs.append(PartnerSpec(
            survey=survey,
            radius_arcsec=float(radius),
            columns=list(cols),
            match_mode="sky",
        ))
    return specs


def partners_meta_from_crossmatch_plan(
    plan: dict | None,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Split a ``crossmatch_plan`` into sky radii and column-partner metadata."""
    from data_lake.discovery.area_plan import partner_match_mode

    radii: dict[str, float] = {}
    column_partners: dict[str, dict] = {}
    for p in (plan or {}).get("partners", []):
        survey = p.get("survey")
        if not survey:
            continue
        if partner_match_mode(p) == "column":
            column_partners[survey] = {
                "match_col_a": p.get("match_col_a"),
                "match_col_b": p.get("match_col_b"),
            }
        elif p.get("radius_arcsec") is not None:
            radii[survey] = float(p["radius_arcsec"])
    return radii, column_partners
