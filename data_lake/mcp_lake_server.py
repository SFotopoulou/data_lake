"""
Read-only MCP server exposing lake discovery, provenance, query building,
QA, cost advice, and ingest recommendations.

Run via ``dl-mcp-lake`` (stdio transport). Requires ``uv sync --extra mcp``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from data_lake.discovery.areas import list_areas, load_area
from data_lake.discovery.engine import DiscoveryRow, resolve_region, round_count
from data_lake.discovery.region import Region, parse_npix_arg
from data_lake.ingest_advisor import recommend_ingest
from data_lake.mcp_common import json_dumps, resolve_lake_root
from data_lake.mcp_inventory import (
    tool_describe_lake,
    tool_describe_survey,
    tool_list_products,
)
from data_lake.schema_registry import (
    CATALOG_KIND_PRODUCT,
    MODALITY_CATALOG,
    MODALITY_CUTOUT,
    MODALITY_SPECTRA,
    resolve_catalog_root,
)

_DEFAULT_MODALITIES = (MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT)

# Rough seconds-per-tile heuristics (order-of-magnitude only).
_SEC_PER_TILE = {
    "crossmatch": 30.0,
    "gather": 10.0,
    "ingest": 120.0,
}


def _region_from_params(
    lake_root: Path,
    *,
    from_area: str | None = None,
    npix: str | None = None,
    norder: int | None = None,
    cone_ra: float | None = None,
    cone_dec: float | None = None,
    radius_arcsec: float | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    moc: str | None = None,
) -> Region:
    if from_area is not None:
        return load_area(lake_root, from_area).region
    if npix is not None:
        if norder is None:
            raise ValueError("npix requires norder (source HEALPix order)")
        return Region.from_npix(parse_npix_arg(npix), norder)
    if cone_ra is not None and cone_dec is not None:
        if radius_arcsec is None:
            raise ValueError("cone requires radius_arcsec")
        return Region.cone(cone_ra, cone_dec, radius_arcsec)
    if bbox is not None:
        return Region.bbox(*bbox)
    if moc is not None:
        return Region.from_moc(path=moc)
    raise ValueError(
        "Provide exactly one region selector: from_area, npix+norder, "
        "cone_ra/cone_dec+radius_arcsec, bbox, or moc"
    )


def _discovery_rows_to_dict(rows: list[DiscoveryRow], *, exact: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        rows_display = r.exact_rows if exact and r.exact_rows is not None else r.est_rows
        out.append({
            "survey": r.survey,
            "modality": r.modality,
            "hats_order": r.hats_order,
            "n_tiles_overlap": r.n_tiles_overlap,
            "rows": rows_display,
            "rows_display": (
                f"{r.exact_rows:,}" if exact and r.exact_rows is not None
                else round_count(r.est_rows)
            ),
            "path": r.path,
        })
    return out


def tool_discover_region(
    lake_root: str | None = None,
    *,
    from_area: str | None = None,
    npix: str | None = None,
    norder: int | None = None,
    cone_ra: float | None = None,
    cone_dec: float | None = None,
    radius_arcsec: float | None = None,
    bbox_ra_min: float | None = None,
    bbox_ra_max: float | None = None,
    bbox_dec_min: float | None = None,
    bbox_dec_max: float | None = None,
    moc: str | None = None,
    surveys: str = "all",
    modalities: list[str] | None = None,
    count: bool = False,
) -> dict[str, Any]:
    root = resolve_lake_root(lake_root)
    bbox = None
    if bbox_ra_min is not None:
        bbox = (bbox_ra_min, bbox_ra_max, bbox_dec_min, bbox_dec_max)
    region = _region_from_params(
        root,
        from_area=from_area,
        npix=npix,
        norder=norder,
        cone_ra=cone_ra,
        cone_dec=cone_dec,
        radius_arcsec=radius_arcsec,
        bbox=bbox,
        moc=moc,
    )
    mods = tuple(modalities) if modalities else _DEFAULT_MODALITIES
    survey_arg: str | list[str] = surveys
    if surveys != "all":
        survey_arg = [s.strip() for s in surveys.split(",") if s.strip()]
    rows = resolve_region(
        root, region, surveys=survey_arg, modalities=mods, count=count,
    )
    return {
        "region": region.to_dict(),
        "exact_counts": count,
        "entries": _discovery_rows_to_dict(rows, exact=count),
    }


def tool_list_areas(lake_root: str | None = None) -> dict[str, Any]:
    root = resolve_lake_root(lake_root)
    return {"area_ids": list_areas(root)}


def tool_get_area(lake_root: str | None = None, area_id: str = "") -> dict[str, Any]:
    root = resolve_lake_root(lake_root)
    area = load_area(root, area_id)
    return {
        "area_id": area.area_id,
        "region": area.region.to_dict(),
        "discover": area.data.get("discover"),
        "crossmatch_plan": area.crossmatch_plan,
        "gather": area.gather,
        "homogenize": area.homogenize,
    }


def tool_list_crossmatches(
    lake_root: str | None = None,
    survey_a: str | None = None,
    survey_b: str | None = None,
) -> dict[str, Any]:
    from data_lake.io.crossmatch import list_crossmatch_descriptions

    root = resolve_lake_root(lake_root)
    entries = list_crossmatch_descriptions(
        root, survey_a, survey_b, recount=False, include_columns=False
    )
    return {"crossmatches": entries}


def tool_describe_crossmatch(
    lake_root: str | None = None,
    survey_a: str | None = None,
    survey_b: str | None = None,
    *,
    name: str | None = None,
    radius_arcsec: float | None = None,
    match_mode: str | None = None,
    match_col_a: str | None = None,
    match_col_b: str | None = None,
    recount: bool = False,
) -> dict[str, Any]:
    """Describe one crossmatch tree (same payload as ``dl-describe-crossmatch --json``)."""
    from data_lake.io.crossmatch import (
        describe_crossmatch_tree,
        resolve_crossmatch_describe_target,
    )

    root = resolve_lake_root(lake_root)
    target = resolve_crossmatch_describe_target(
        root,
        name=name,
        survey_a=survey_a,
        survey_b=survey_b,
        radius_arcsec=radius_arcsec,
        match_mode=match_mode,
        match_col_a=match_col_a,
        match_col_b=match_col_b,
    )
    return describe_crossmatch_tree(target, recount=recount, include_columns=True)


def tool_describe_product(
    lake_root: str | None = None,
    name: str = "",
) -> dict[str, Any]:
    from data_lake.io.crossmatch import find_crossmatch_roots

    root = resolve_lake_root(lake_root)
    catalog_root = resolve_catalog_root(root, name)
    info_path = catalog_root / "catalog_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"No catalog_info.json for {name!r} "
            f"(checked catalogs/{name} and products/{name})"
        )
    with open(info_path) as fh:
        info = json.load(fh)
    if info.get("kind") != CATALOG_KIND_PRODUCT:
        raise ValueError(f"{catalog_root.name} is not a product (kind={info.get('kind')!r})")

    provenance = info.get("provenance") or {}
    base = provenance.get("base_catalog") or provenance.get("base")
    partners = provenance.get("partners") or []
    radii = provenance.get("radii") or {}

    crossmatch_trees: list[dict[str, Any]] = []
    if base:
        for partner in partners:
            survey = partner if isinstance(partner, str) else partner.get("survey")
            if not survey:
                continue
            radius = radii.get(survey) if isinstance(radii, dict) else None
            matches = find_crossmatch_roots(root, base, survey)
            for a, b, r, path in matches:
                if radius is not None and abs(r - float(radius)) > 1e-6:
                    continue
                crossmatch_trees.append({
                    "survey_a": a,
                    "survey_b": b,
                    "radius_arcsec": r,
                    "path": str(path),
                })

    # Include columns summary from schema registry if available
    columns_summary: list[dict[str, Any]] = []
    try:
        from data_lake.schema_registry import get_survey_manifest

        manifest = get_survey_manifest(
            root, name, "catalog", rebuild=False, apply_overlay=False
        )
        cols = manifest.get("columns") or []
        columns_summary = [
            {"name": c.get("name"), "dtype": c.get("dtype")} for c in cols[:20]
        ]
        if len(cols) > 20:
            columns_summary.append({"name": f"... +{len(cols) - 20} more", "dtype": None})
    except Exception:
        pass

    return {
        "name": name,
        "kind": info.get("kind"),
        "product_subtype": info.get("product_subtype"),
        "total_rows": info.get("total_rows"),
        "provenance": provenance,
        "homogenize_provenance": info.get("homogenize_provenance"),
        "crossmatch_trees": crossmatch_trees,
        "columns_summary": columns_summary,
        "catalog_root": str(catalog_root),
    }


def tool_build_query(
    lake_root: str | None = None,
    master: str = "",
    primary_survey: str = "",
    columns: list[str] | None = None,
) -> dict[str, Any]:
    from data_lake.query_from_master import build_select_from_master, parse_column_picks

    root = resolve_lake_root(lake_root)
    if not columns:
        raise ValueError("columns required: list of SURVEY:col1,col2 picks")
    col_map = parse_column_picks(columns)
    plan = build_select_from_master(
        root,
        master,
        primary_survey,
        col_map,
        include_views=True,
        validate_columns=False,
    )
    return plan.to_dict()


def tool_validate_survey(
    lake_root: str | None = None,
    survey: str = "",
    modality: str = MODALITY_CATALOG,
    *,
    max_tiles: int = 5,
) -> dict[str, Any]:
    root = resolve_lake_root(lake_root)
    if modality == MODALITY_CATALOG:
        from data_lake.ingest.validate_catalog_ingest import run_validation

        rep = run_validation(root, survey, max_tiles=max_tiles)
    elif modality == MODALITY_SPECTRA:
        from data_lake.ingest.validate_spectra_ingest import run_validation

        rep = run_validation(root, survey, max_tiles=max_tiles)
    elif modality == MODALITY_CUTOUT:
        from data_lake.ingest.validate_cutout_ingest import run_validation

        rep = run_validation(root, survey, max_tiles=max_tiles)
    else:
        raise ValueError(f"Unsupported modality for validation: {modality}")

    return {
        "survey": survey,
        "modality": modality,
        "ok": rep.ok(strict=False),
        "errors": list(rep.errors),
        "warnings": list(rep.warnings),
    }


def tool_lake_health(lake_root: str | None = None) -> dict[str, Any]:
    """Summarize lake health with entry counts, unfinalized surveys, and missing indices."""
    import os
    import time

    from data_lake.discovery import tile_index as ti
    from data_lake.lake_registry import (
        coalesce_registry_kinds,
        load_lake_registry,
        registry_path,
    )
    from data_lake.schema_registry import (
        CATALOG_KIND_INGESTED,
        CATALOG_KIND_PRODUCT,
        MODALITY_CROSSMATCH,
    )

    root = resolve_lake_root(lake_root)
    rpath = registry_path(root)

    if not rpath.is_file():
        return {
            "registry_present": False,
            "unfinalized_live": [],
            "missing_tile_index": [],
            "notes": ["Run dl-refresh-lake-registry to build the registry."],
        }

    # Stale hint: registry older than 24 h
    age_h = (time.time() - os.path.getmtime(rpath)) / 3600
    stale_hint = (
        f"Registry is {age_h:.0f} h old — consider dl-refresh-lake-registry."
        if age_h > 24
        else None
    )

    table = coalesce_registry_kinds(load_lake_registry(root))
    entries = table.to_pylist()

    unfinalized: list[dict[str, str]] = []
    missing_index: list[dict[str, str]] = []

    # Counts by kind and modality
    counts_by_kind: dict[str, int] = {}
    counts_by_modality: dict[str, int] = {}

    for row in entries:
        kind = row.get("kind", CATALOG_KIND_INGESTED)
        modality = row.get("modality", MODALITY_CATALOG)
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
        counts_by_modality[modality] = counts_by_modality.get(modality, 0) + 1

        if row.get("lifecycle") == "live" and row.get("finalized") is False:
            unfinalized.append({
                "survey": row.get("survey", ""),
                "modality": modality,
            })

        survey = row.get("survey")
        if survey and modality in (MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT):
            idx_path = ti.tile_index_path(root, survey, modality)
            if not idx_path.is_file():
                missing_index.append({"survey": survey, "modality": modality})

    notes: list[str] = []
    if stale_hint:
        notes.append(stale_hint)
    if unfinalized:
        notes.append(
            f"{len(unfinalized)} live survey(s) unfinalized — run dl-finalize-catalog."
        )
    if missing_index:
        notes.append(
            f"{len(missing_index)} survey(s) missing tile index — run dl-refresh-tile-index."
        )

    n_ingested = counts_by_kind.get(CATALOG_KIND_INGESTED, 0)
    n_products = counts_by_kind.get(CATALOG_KIND_PRODUCT, 0)
    n_crossmatch = counts_by_kind.get(MODALITY_CROSSMATCH, 0)

    return {
        "registry_present": True,
        "n_entries": len(entries),
        "counts_by_kind": {
            "ingested": n_ingested,
            "product": n_products,
            "crossmatch": n_crossmatch,
        },
        "counts_by_modality": counts_by_modality,
        "unfinalized_live": unfinalized,
        "missing_tile_index": missing_index,
        "registry_age_hours": round(age_h, 1),
        "notes": notes,
    }


def tool_recommend_norder(
    paths: list[str] | None = None,
    *,
    file_list: str | None = None,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> dict[str, Any]:
    from data_lake.ingest.recommend_norder import (
        format_recommendation_report,
        recommend_catalog_norder,
    )

    path_objs = [Path(p) for p in (paths or [])]
    if not path_objs and not file_list:
        raise ValueError("Provide paths (sample files) or file_list")
    rec = recommend_catalog_norder(
        path_objs,
        file_list=Path(file_list) if file_list else None,
        ra_col=ra_col,
        dec_col=dec_col,
    )
    return {
        "recommended_norder": rec.recommended,
        "report": format_recommendation_report(rec),
        "details": asdict(rec),
    }


def tool_list_homogenize_recipes(lake_root: str | None = None) -> dict[str, Any]:
    """List per-survey homogenization recipes (read-only).

    Returns which surveys have a recipe (lake override or bundled default) and
    which modalities are covered. To run homogenization use ``dl-homogenize``;
    to validate use ``dl-validate-homogenization``.
    """
    from data_lake.homogenize.survey_registry import (
        _PKG_SURVEYS,
        surveys_homogenize_dir,
    )

    root = resolve_lake_root(lake_root)
    lake_dir = surveys_homogenize_dir(root)
    bundled_dir = _PKG_SURVEYS

    # Collect all known survey names from both locations
    lake_paths: dict[str, Path] = {}
    bundled_paths: dict[str, Path] = {}

    if lake_dir.is_dir():
        for p in sorted(lake_dir.glob("*.json")):
            lake_paths[p.stem] = p
    if bundled_dir.is_dir():
        for p in sorted(bundled_dir.glob("*.json")):
            bundled_paths[p.stem] = p

    all_surveys = sorted(set(lake_paths) | set(bundled_paths))

    recipes: list[dict[str, Any]] = []
    for survey in all_surveys:
        path = lake_paths.get(survey) or bundled_paths.get(survey)
        source = "lake_override" if survey in lake_paths else "bundled"
        try:
            import json as _json

            with open(path, encoding="utf-8") as fh:  # type: ignore[arg-type]
                data = _json.load(fh)
            modalities = [m for m in ("catalog", "spectra", "cutout") if m in data]
        except Exception:
            modalities = []
        recipes.append({
            "survey": survey,
            "source": source,
            "modalities": modalities,
            "path": str(path),
        })

    return {
        "n_recipes": len(recipes),
        "recipes": recipes,
        "hint": (
            "Use dl-homogenize --survey <name> to apply a recipe. "
            "Use dl-validate-homogenization to lint and check coverage."
        ),
    }


def tool_estimate_operation_cost(
    lake_root: str | None = None,
    operation: str = "crossmatch",
    *,
    from_area: str | None = None,
    npix: str | None = None,
    norder: int | None = None,
    cone_ra: float | None = None,
    cone_dec: float | None = None,
    radius_arcsec: float | None = None,
    bbox_ra_min: float | None = None,
    bbox_ra_max: float | None = None,
    bbox_dec_min: float | None = None,
    bbox_dec_max: float | None = None,
    moc: str | None = None,
) -> dict[str, Any]:
    root = resolve_lake_root(lake_root)
    if operation not in _SEC_PER_TILE:
        raise ValueError(f"operation must be one of: {', '.join(_SEC_PER_TILE)}")

    bbox = None
    if bbox_ra_min is not None:
        bbox = (bbox_ra_min, bbox_ra_max, bbox_dec_min, bbox_dec_max)
    region = _region_from_params(
        root,
        from_area=from_area,
        npix=npix,
        norder=norder,
        cone_ra=cone_ra,
        cone_dec=cone_dec,
        radius_arcsec=radius_arcsec,
        bbox=bbox,
        moc=moc,
    )
    rows = resolve_region(root, region, surveys="all", modalities=(MODALITY_CATALOG,))
    total_tiles = sum(r.n_tiles_overlap for r in rows)
    sec_per = _SEC_PER_TILE[operation]
    est_seconds = total_tiles * sec_per

    def _fmt_duration(seconds: float) -> str:
        if seconds < 3600:
            return f"~{seconds / 60:.0f} min"
        return f"~{seconds / 3600:.1f} h"

    return {
        "operation": operation,
        "n_overlap_tiles": total_tiles,
        "sec_per_tile_assumed": sec_per,
        "estimated_duration": _fmt_duration(est_seconds),
        "estimated_seconds": est_seconds,
        "disclaimer": "Order-of-magnitude estimate only; actual runtime varies by hardware and row density.",
        "surveys": _discovery_rows_to_dict(rows, exact=False),
    }


def tool_recommend_ingest(
    modality: str,
    survey: str,
    *,
    file_list: str | None = None,
    sample_file: str | None = None,
    fmt: str | None = None,
    lifecycle: str = "static",
    n_files: int = 1,
    total_size_gb: float | None = None,
    streaming: bool = False,
) -> dict[str, Any]:
    rec = recommend_ingest(
        modality,  # type: ignore[arg-type]
        survey,
        file_list=Path(file_list) if file_list else None,
        sample_file=Path(sample_file) if sample_file else None,
        fmt=fmt,
        lifecycle=lifecycle,  # type: ignore[arg-type]
        n_files=n_files,
        total_size_gb=total_size_gb,
        streaming=streaming,
    )
    return rec.to_dict()


def create_lake_mcp_app() -> Any:
    """Build the FastMCP application (requires ``mcp`` package)."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("data-lake-explorer")

    @mcp.tool()
    def discover_region(
        lake_root: str | None = None,
        from_area: str | None = None,
        npix: str | None = None,
        norder: int | None = None,
        cone_ra: float | None = None,
        cone_dec: float | None = None,
        radius_arcsec: float | None = None,
        bbox_ra_min: float | None = None,
        bbox_ra_max: float | None = None,
        bbox_dec_min: float | None = None,
        bbox_dec_max: float | None = None,
        moc: str | None = None,
        surveys: str = "all",
        modalities: list[str] | None = None,
        count: bool = False,
    ) -> str:
        """Discover survey x modality data overlapping a sky region."""
        return json_dumps(tool_discover_region(
            lake_root,
            from_area=from_area,
            npix=npix,
            norder=norder,
            cone_ra=cone_ra,
            cone_dec=cone_dec,
            radius_arcsec=radius_arcsec,
            bbox_ra_min=bbox_ra_min,
            bbox_ra_max=bbox_ra_max,
            bbox_dec_min=bbox_dec_min,
            bbox_dec_max=bbox_dec_max,
            moc=moc,
            surveys=surveys,
            modalities=modalities,
            count=count,
        ))

    @mcp.tool()
    def list_areas(lake_root: str | None = None) -> str:
        """List saved area IDs (areas/*.json)."""
        return json_dumps(tool_list_areas(lake_root))

    @mcp.tool()
    def get_area(lake_root: str | None = None, area_id: str = "") -> str:
        """Load a saved area definition including crossmatch_plan and gather blocks."""
        return json_dumps(tool_get_area(lake_root, area_id))

    @mcp.tool()
    def list_crossmatches(
        lake_root: str | None = None,
        survey_a: str | None = None,
        survey_b: str | None = None,
    ) -> str:
        """List crossmatch trees (sky and column) with metadata."""
        return json_dumps(tool_list_crossmatches(lake_root, survey_a, survey_b))

    @mcp.tool()
    def describe_crossmatch(
        lake_root: str | None = None,
        survey_a: str | None = None,
        survey_b: str | None = None,
        name: str | None = None,
        radius_arcsec: float | None = None,
        match_mode: str | None = None,
        match_col_a: str | None = None,
        match_col_b: str | None = None,
        recount: bool = False,
    ) -> str:
        """Describe one crossmatch tree (dl-describe-crossmatch --json)."""
        return json_dumps(tool_describe_crossmatch(
            lake_root,
            survey_a,
            survey_b,
            name=name,
            radius_arcsec=radius_arcsec,
            match_mode=match_mode,
            match_col_a=match_col_a,
            match_col_b=match_col_b,
            recount=recount,
        ))

    @mcp.tool()
    def describe_product(lake_root: str | None = None, name: str = "") -> str:
        """Describe a derived product catalog and its crossmatch lineage."""
        return json_dumps(tool_describe_product(lake_root, name))

    @mcp.tool()
    def build_query(
        lake_root: str | None = None,
        master: str = "",
        primary_survey: str = "",
        columns: list[str] | None = None,
    ) -> str:
        """Build DuckDB SQL from a master association file (does not execute)."""
        return json_dumps(tool_build_query(
            lake_root, master, primary_survey, columns,
        ))

    @mcp.tool()
    def validate_survey(
        lake_root: str | None = None,
        survey: str = "",
        modality: str = "catalog",
        max_tiles: int = 5,
    ) -> str:
        """Run ingest validation checks on a survey (sample of tiles)."""
        return json_dumps(tool_validate_survey(
            lake_root, survey, modality, max_tiles=max_tiles,
        ))

    @mcp.tool()
    def lake_health(lake_root: str | None = None) -> str:
        """Summarize lake health: unfinalized live catalogs, missing tile indices."""
        return json_dumps(tool_lake_health(lake_root))

    @mcp.tool()
    def recommend_norder(
        paths: list[str] | None = None,
        file_list: str | None = None,
        ra_col: str = "ra",
        dec_col: str = "dec",
    ) -> str:
        """Recommend HEALPix norder for catalog ingest from sample files."""
        return json_dumps(tool_recommend_norder(
            paths, file_list=file_list, ra_col=ra_col, dec_col=dec_col,
        ))

    @mcp.tool()
    def estimate_operation_cost(
        lake_root: str | None = None,
        operation: str = "crossmatch",
        from_area: str | None = None,
        npix: str | None = None,
        norder: int | None = None,
        cone_ra: float | None = None,
        cone_dec: float | None = None,
        radius_arcsec: float | None = None,
        bbox_ra_min: float | None = None,
        bbox_ra_max: float | None = None,
        bbox_dec_min: float | None = None,
        bbox_dec_max: float | None = None,
        moc: str | None = None,
    ) -> str:
        """Rough runtime estimate for crossmatch/gather/ingest over a region."""
        return json_dumps(tool_estimate_operation_cost(
            lake_root,
            operation,
            from_area=from_area,
            npix=npix,
            norder=norder,
            cone_ra=cone_ra,
            cone_dec=cone_dec,
            radius_arcsec=radius_arcsec,
            bbox_ra_min=bbox_ra_min,
            bbox_ra_max=bbox_ra_max,
            bbox_dec_min=bbox_dec_min,
            bbox_dec_max=bbox_dec_max,
            moc=moc,
        ))

    @mcp.tool()
    def recommend_ingest(
        modality: str,
        survey: str,
        file_list: str | None = None,
        sample_file: str | None = None,
        fmt: str | None = None,
        lifecycle: str = "static",
        n_files: int = 1,
        total_size_gb: float | None = None,
        streaming: bool = False,
    ) -> str:
        """Recommend dl-* ingest command and Slurm script for a survey."""
        return json_dumps(tool_recommend_ingest(
            modality,
            survey,
            file_list=file_list,
            sample_file=sample_file,
            fmt=fmt,
            lifecycle=lifecycle,
            n_files=n_files,
            total_size_gb=total_size_gb,
            streaming=streaming,
        ))

    # --- Inventory tools (mirrored from dl-mcp-docs so the explorer alone suffices) ---

    @mcp.tool()
    def describe_lake(
        lake_root: str | None = None,
        kind: str | None = None,
        refresh: bool = False,
        count_total: bool = True,
        modality: str | None = None,
    ) -> str:
        """Summarize all surveys and modalities in the lake registry.

        kind: filter by 'ingested', 'product', or 'crossmatch' (default: all).
        Use list_products for a focused product listing.
        """
        return json_dumps(tool_describe_lake(
            lake_root,
            kind=kind,
            refresh=refresh,
            count_total=count_total,
            modality=modality,
        ))

    @mcp.tool()
    def describe_survey(
        survey: str,
        lake_root: str | None = None,
        modality: str = "catalog",
        rebuild: bool = False,
    ) -> str:
        """Return column manifest for a survey (dl-describe-survey --json)."""
        return json_dumps(tool_describe_survey(
            survey, lake_root=lake_root, modality=modality, rebuild=rebuild,
        ))

    @mcp.tool()
    def list_products(lake_root: str | None = None) -> str:
        """List all product catalogs with name, subtype, and row count.

        Use describe_product(name=...) for provenance and crossmatch lineage.
        """
        return json_dumps(tool_list_products(lake_root))

    @mcp.tool()
    def list_homogenize_recipes(lake_root: str | None = None) -> str:
        """List per-survey homogenization recipes (read-only).

        Shows which surveys have a recipe and which modalities are covered.
        Use dl-homogenize to apply; dl-validate-homogenization to lint.
        """
        return json_dumps(tool_list_homogenize_recipes(lake_root))

    return mcp


def main() -> None:
    """Entry point for dl-mcp-lake."""
    mcp = create_lake_mcp_app()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
