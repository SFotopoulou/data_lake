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

from data_lake.discovery.areas import iter_areas, list_areas, load_area
from data_lake.discovery.engine import DiscoveryRow, resolve_region, round_count
from data_lake.discovery.region import Region, parse_npix_arg
from data_lake.ingest_advisor import recommend_ingest
from data_lake.mcp_common import json_dumps, resolve_lake_root
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
    }


def tool_list_crossmatches(
    lake_root: str | None = None,
    survey_a: str | None = None,
    survey_b: str | None = None,
) -> dict[str, Any]:
    from data_lake.io.crossmatch import CROSSMATCH_INFO_FILENAME, find_crossmatch_roots

    root = resolve_lake_root(lake_root)
    entries: list[dict[str, Any]] = []
    for a, b, radius, path in find_crossmatch_roots(root, survey_a, survey_b):
        info: dict[str, Any] = {}
        info_path = path / CROSSMATCH_INFO_FILENAME
        if info_path.is_file():
            with open(info_path) as fh:
                info = json.load(fh)
        entries.append({
            "survey_a": a,
            "survey_b": b,
            "radius_arcsec": radius,
            "path": str(path),
            "info": info,
        })
    return {"crossmatches": entries}


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

    return {
        "name": name,
        "kind": info.get("kind"),
        "provenance": provenance,
        "crossmatch_trees": crossmatch_trees,
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
    from data_lake.discovery import tile_index as ti
    from data_lake.lake_registry import load_lake_registry, registry_path

    root = resolve_lake_root(lake_root)
    if not registry_path(root).is_file():
        return {
            "registry_present": False,
            "unfinalized_live": [],
            "missing_tile_index": [],
            "notes": ["Run dl-refresh-lake-registry to build the registry."],
        }

    table = load_lake_registry(root)
    entries = table.to_pylist()
    unfinalized: list[dict[str, str]] = []
    missing_index: list[dict[str, str]] = []

    for row in entries:
        if row.get("lifecycle") == "live" and row.get("finalized") is False:
            unfinalized.append({
                "survey": row["survey"],
                "modality": row.get("modality", MODALITY_CATALOG),
            })
        survey = row.get("survey")
        modality = row.get("modality", MODALITY_CATALOG)
        if survey and modality in (MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT):
            idx_path = ti.tile_index_path(root, survey, modality)
            if not idx_path.is_file():
                missing_index.append({"survey": survey, "modality": modality})

    return {
        "registry_present": True,
        "n_entries": len(entries),
        "unfinalized_live": unfinalized,
        "missing_tile_index": missing_index,
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
) -> dict[str, Any]:
    root = resolve_lake_root(lake_root)
    if operation not in _SEC_PER_TILE:
        raise ValueError(f"operation must be one of: {', '.join(_SEC_PER_TILE)}")

    region = _region_from_params(
        root,
        from_area=from_area,
        npix=npix,
        norder=norder,
        cone_ra=cone_ra,
        cone_dec=cone_dec,
        radius_arcsec=radius_arcsec,
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
        """List crossmatch trees and crossmatch_info.json metadata."""
        return json_dumps(tool_list_crossmatches(lake_root, survey_a, survey_b))

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

    return mcp


def main() -> None:
    """Entry point for dl-mcp-lake."""
    mcp = create_lake_mcp_app()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
