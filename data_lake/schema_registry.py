"""
schema_registry – per-survey column manifests for analyst discovery.

``schema_manifest.json`` is written at ingest finalize for catalogs (Parquet),
spectra (Zarr), and cutouts (Zarr).  Browsed via ``dl-describe-survey``; optional
column overlays live under ``shared/registry/overlays/*.json``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

MANIFEST_FILENAME = "schema_manifest.json"
MANIFEST_VERSION = "1"

# Heuristic role tags (for filtering in dl-describe-survey).
ROLE_ID = "id"
ROLE_SKY = "sky"
ROLE_REDSHIFT = "redshift"
ROLE_HEALPIX = "healpix"
ROLE_INDEX = "index"
ROLE_PHOTOMETRY = "photometry"
ROLE_CLASSIFICATION = "classification"
ROLE_SCIENCE = "science"
ROLE_FLUX = "flux"
ROLE_IVAR = "ivar"
ROLE_MASK = "mask"
ROLE_WAVELENGTH = "wavelength"
ROLE_IMAGE = "image"
ROLE_METADATA = "metadata"
ROLE_RESOLUTION = "resolution"

MODALITY_CATALOG = "catalog"
MODALITY_SPECTRA = "spectra"
MODALITY_CUTOUT = "cutout"

_LAYER_DIRS: dict[str, str] = {
    MODALITY_CATALOG: "catalogs",
    MODALITY_SPECTRA: "spectra",
    MODALITY_CUTOUT: "cutouts",
}

_PHOTOMETRY_MARKERS = ("MAG", "FLUX", "MPRO", "MMOD", "AB", "VEGA")
_CLASSIFICATION_NAMES = frozenset(
    n.upper()
    for n in (
        "SPECTYPE",
        "CLASS",
        "TYPE",
        "MORPHTYPE",
        "SUBTYPE",
        "CLASS_STAR",
    )
)


def _dtype_to_str(dtype: pa.DataType) -> str:
    if pa.types.is_fixed_size_list(dtype):
        inner = _dtype_to_str(dtype.value_type)
        return f"fixed_size_list<{inner}>[{dtype.list_size}]"
    return str(dtype)


def infer_column_role(
    name: str,
    *,
    source_id_column: str,
    ra_column: str,
    dec_column: str,
    redshift_column: str | None,
) -> str:
    upper = name.upper()
    if name == source_id_column:
        return ROLE_ID
    if name in (ra_column, dec_column) or upper in ("RA", "DEC", "TARGET_RA", "TARGET_DEC"):
        return ROLE_SKY
    if redshift_column and name == redshift_column:
        return ROLE_REDSHIFT
    if name.startswith("_healpix"):
        return ROLE_HEALPIX
    if name in ("_cutout_index", "_spectrum_index"):
        return ROLE_INDEX
    if upper in _CLASSIFICATION_NAMES or upper.endswith("TYPE"):
        return ROLE_CLASSIFICATION
    if any(m in upper for m in _PHOTOMETRY_MARKERS):
        return ROLE_PHOTOMETRY
    return ROLE_SCIENCE


def infer_column_groups(column_names: list[str]) -> dict[str, list[str]]:
    """Group columns that share a common prefix (e.g. MAG_G, MAG_R → MAG)."""
    buckets: dict[str, list[str]] = {}
    for name in column_names:
        if "_" not in name:
            continue
        prefix = name.split("_", 1)[0]
        if len(prefix) < 2:
            continue
        buckets.setdefault(prefix, []).append(name)
    return {k: sorted(v) for k, v in buckets.items() if len(v) >= 2}


def read_catalog_arrow_schema(catalog_root: Path) -> pa.Schema:
    """Load aggregate or single-tile Parquet schema for a survey catalog."""
    catalog_root = Path(catalog_root)
    meta_path = catalog_root / "_metadata"
    if meta_path.exists():
        return pq.read_metadata(str(meta_path)).schema.to_arrow_schema()
    first_tile = next(catalog_root.rglob("Npix=*.parquet"), None)
    if first_tile is None:
        raise FileNotFoundError(f"No Parquet tiles under {catalog_root}")
    return pq.read_schema(str(first_tile))


def build_catalog_schema_manifest(
    catalog_root: Path | str,
    survey_name: str,
    *,
    hats_order: int,
    ra_column: str,
    dec_column: str,
    source_id_mode: str,
    total_rows: int | None = None,
    schema: pa.Schema | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable manifest from on-disk catalog Parquet."""
    from data_lake.ingest.fits_to_parquet import resolve_source_id_column
    from data_lake.io.catalog import resolve_redshift_column

    catalog_root = Path(catalog_root)
    if schema is None:
        schema = read_catalog_arrow_schema(catalog_root)

    names = list(schema.names)
    source_id_column = resolve_source_id_column(
        catalog_root,
        schema_names=names,
    )
    redshift_column = resolve_redshift_column(names)

    columns = []
    for field in schema:
        columns.append(
            {
                "name": field.name,
                "dtype": _dtype_to_str(field.type),
                "nullable": field.nullable,
                "role": infer_column_role(
                    field.name,
                    source_id_column=source_id_column,
                    ra_column=ra_column,
                    dec_column=dec_column,
                    redshift_column=redshift_column,
                ),
            }
        )

    info_path = catalog_root / "catalog_info.json"
    if total_rows is None and info_path.exists():
        with open(info_path) as fh:
            total_rows = json.load(fh).get("total_rows")

    return {
        "manifest_version": MANIFEST_VERSION,
        "survey": survey_name,
        "modality": "catalog",
        "hats_order": hats_order,
        "source_id_column": source_id_column,
        "ra_column": ra_column,
        "dec_column": dec_column,
        "redshift_column": redshift_column,
        "source_id_mode": source_id_mode,
        "n_columns": len(columns),
        "total_rows": total_rows,
        "columns": columns,
        "column_groups": infer_column_groups(names),
        "catalog_root": str(catalog_root),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_catalog_schema_manifest(
    catalog_root: Path | str,
    survey_name: str,
    *,
    hats_order: int,
    ra_column: str,
    dec_column: str,
    source_id_mode: str,
    total_rows: int | None = None,
) -> Path:
    """Write ``schema_manifest.json`` under the survey catalog directory."""
    catalog_root = Path(catalog_root)
    manifest = build_catalog_schema_manifest(
        catalog_root,
        survey_name,
        hats_order=hats_order,
        ra_column=ra_column,
        dec_column=dec_column,
        source_id_mode=source_id_mode,
        total_rows=total_rows,
    )
    out_path = catalog_root / MANIFEST_FILENAME
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.debug("Wrote %s (%d columns)", out_path, manifest["n_columns"])
    return out_path


def survey_layer_root(lake_root: Path | str, survey: str, modality: str) -> Path:
    """Return ``catalogs|spectra|cutouts``/<survey> path."""
    if modality not in _LAYER_DIRS:
        raise ValueError(f"Unknown modality {modality!r}; use catalog, spectra, cutout.")
    return Path(lake_root) / _LAYER_DIRS[modality] / survey


def overlay_search_paths(
    lake_root: Path | str,
    survey: str,
    modality: str,
) -> list[Path]:
    """Optional analyst-authored metadata: ``shared/registry/overlays/<survey>.<modality>.json``."""
    base = Path(lake_root) / "shared" / "registry" / "overlays"
    return [
        base / f"{survey}.{modality}.json",
        base / f"{survey}.json",
    ]


def load_column_overlay(
    lake_root: Path | str,
    survey: str,
    modality: str,
) -> dict[str, dict[str, Any]]:
    """Load optional per-column overlay (unit, description, homogenization hints)."""
    for path in overlay_search_paths(lake_root, survey, modality):
        if path.is_file():
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            cols = data.get("columns", data)
            if isinstance(cols, dict):
                return cols
    return {}


def merge_column_overlays(manifest: dict[str, Any], overlay: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not overlay:
        return manifest
    out = dict(manifest)
    merged_cols = []
    for col in manifest.get("columns", []):
        c = dict(col)
        extra = overlay.get(col["name"])
        if extra:
            c["unit"] = extra.get("unit")
            c["description"] = extra.get("description")
            if "homogenized_ab_offset" in extra:
                c["homogenized_ab_offset"] = extra["homogenized_ab_offset"]
        merged_cols.append(c)
    out["columns"] = merged_cols
    out["has_column_overlay"] = True
    return out


def load_schema_manifest(survey_root: Path | str) -> dict[str, Any]:
    path = Path(survey_root) / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No {MANIFEST_FILENAME} under {survey_root}. "
            "Run ingest finalize or: dl-describe-survey <name> --rebuild"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_catalog_schema_manifest(catalog_root: Path | str) -> dict[str, Any]:
    return load_schema_manifest(catalog_root)


def _write_schema_manifest_file(survey_root: Path, manifest: dict[str, Any]) -> Path:
    survey_root = Path(survey_root)
    out_path = survey_root / MANIFEST_FILENAME
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.debug("Wrote %s", out_path)
    return out_path


def build_spectra_schema_manifest(spectra_root: Path | str, survey_name: str) -> dict[str, Any]:
    """Build manifest from ``spectrum_info.json`` (Zarr array layout)."""
    spectra_root = Path(spectra_root)
    info_path = spectra_root / "spectrum_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"No spectrum_info.json under {spectra_root}")
    with open(info_path) as fh:
        info = json.load(fh)

    n_pix = int(info["n_pix"])
    wl_mode = info.get("wavelength_mode", "shared")
    from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN

    columns: list[dict[str, Any]] = [
        {"name": LAKE_JOIN_ID_COLUMN, "dtype": "int64", "nullable": False, "role": ROLE_ID},
        {"name": "flux", "dtype": f"float32[{n_pix}]", "nullable": True, "role": ROLE_FLUX},
        {"name": "ivar", "dtype": f"float32[{n_pix}]", "nullable": True, "role": ROLE_IVAR},
        {"name": "mask", "dtype": f"uint8[{n_pix}]", "nullable": True, "role": ROLE_MASK},
    ]
    if wl_mode == "shared":
        columns.append(
            {
                "name": "wavelength",
                "dtype": f"float64[{n_pix}]",
                "nullable": False,
                "role": ROLE_WAVELENGTH,
            }
        )
    else:
        columns.append(
            {
                "name": "wavelength",
                "dtype": f"float32[{n_pix}] per source",
                "nullable": True,
                "role": ROLE_WAVELENGTH,
            }
        )
    for field in info.get("meta_fields", []):
        columns.append(
            {
                "name": f"meta.{field}",
                "dtype": "see spectrum_info meta_fields",
                "nullable": True,
                "role": ROLE_METADATA,
            }
        )
    if info.get("has_resolution"):
        n_diag = info.get("resolution_n_diag")
        columns.append(
            {
                "name": "resolution",
                "dtype": f"float32[{n_diag}×{n_pix}] sparse" if n_diag else "float32 sparse",
                "nullable": True,
                "role": ROLE_RESOLUTION,
            }
        )

    return {
        "manifest_version": MANIFEST_VERSION,
        "survey": survey_name,
        "modality": MODALITY_SPECTRA,
        "hats_order": info.get("hats_order"),
        "source_id_column": LAKE_JOIN_ID_COLUMN,
        "ra_column": None,
        "dec_column": None,
        "redshift_column": None,
        "n_columns": len(columns),
        "n_pix": n_pix,
        "wavelength_mode": wl_mode,
        "has_resolution": bool(info.get("has_resolution")),
        "mask_bits": info.get("mask_bits"),
        "meta_fields": info.get("meta_fields"),
        "columns": columns,
        "column_groups": {"meta": [c["name"] for c in columns if c["name"].startswith("meta.")]},
        "survey_root": str(spectra_root),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_spectra_schema_manifest(spectra_root: Path | str, survey_name: str) -> Path:
    manifest = build_spectra_schema_manifest(spectra_root, survey_name)
    return _write_schema_manifest_file(Path(spectra_root), manifest)


def build_cutout_schema_manifest(cutout_root: Path | str, survey_name: str) -> dict[str, Any]:
    """Build manifest from ``cutout_info.json`` (Zarr image stacks)."""
    cutout_root = Path(cutout_root)
    info_path = cutout_root / "cutout_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"No cutout_info.json under {cutout_root}")
    with open(info_path) as fh:
        info = json.load(fh)

    n_b = int(info["n_bands"])
    h, w = int(info["height"]), int(info["width"])
    band_names = list(info.get("band_names", []))
    from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN

    columns: list[dict[str, Any]] = [
        {"name": LAKE_JOIN_ID_COLUMN, "dtype": "int64", "nullable": False, "role": ROLE_ID},
        {
            "name": "images",
            "dtype": f"{info.get('dtype', 'float32')}[{n_b},{h},{w}]",
            "nullable": True,
            "role": ROLE_IMAGE,
        },
    ]
    for i, band in enumerate(band_names):
        columns.append(
            {
                "name": f"band.{band}",
                "dtype": f"slice {i} of images",
                "nullable": True,
                "role": ROLE_IMAGE,
            }
        )

    return {
        "manifest_version": MANIFEST_VERSION,
        "survey": survey_name,
        "modality": MODALITY_CUTOUT,
        "hats_order": info.get("hats_order"),
        "source_id_column": LAKE_JOIN_ID_COLUMN,
        "ra_column": None,
        "dec_column": None,
        "redshift_column": None,
        "n_columns": len(columns),
        "n_bands": n_b,
        "height": h,
        "width": w,
        "band_names": band_names,
        "columns": columns,
        "column_groups": {"band": [c["name"] for c in columns if c["name"].startswith("band.")]},
        "survey_root": str(cutout_root),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_cutout_schema_manifest(cutout_root: Path | str, survey_name: str) -> Path:
    manifest = build_cutout_schema_manifest(cutout_root, survey_name)
    return _write_schema_manifest_file(Path(cutout_root), manifest)


def rebuild_schema_manifest(
    lake_root: Path | str,
    survey: str,
    modality: str,
) -> dict[str, Any]:
    """Regenerate ``schema_manifest.json`` for one survey layer."""
    root = survey_layer_root(lake_root, survey, modality)
    if modality == MODALITY_CATALOG:
        info_path = root / "catalog_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Catalog not found: {root}")
        with open(info_path) as fh:
            info = json.load(fh)
        write_catalog_schema_manifest(
            root,
            survey,
            hats_order=int(info.get("hats_order", 5)),
            ra_column=str(info.get("ra_column", "ra")),
            dec_column=str(info.get("dec_column", "dec")),
            source_id_mode=str(info.get("source_id_mode", "sequential")),
            total_rows=info.get("total_rows"),
        )
    elif modality == MODALITY_SPECTRA:
        write_spectra_schema_manifest(root, survey)
    elif modality == MODALITY_CUTOUT:
        write_cutout_schema_manifest(root, survey)
    return load_schema_manifest(root)


def get_survey_manifest(
    lake_root: Path | str,
    survey: str,
    modality: str = MODALITY_CATALOG,
    *,
    rebuild: bool = False,
    apply_overlay: bool = True,
) -> dict[str, Any]:
    """Load (or rebuild) manifest and merge optional column overlays."""
    root = survey_layer_root(lake_root, survey, modality)
    if rebuild or not (root / MANIFEST_FILENAME).is_file():
        rebuild_schema_manifest(lake_root, survey, modality)
    manifest = load_schema_manifest(root)
    if apply_overlay:
        overlay = load_column_overlay(lake_root, survey, modality)
        manifest = merge_column_overlays(manifest, overlay)
    return manifest


def format_manifest_table(
    manifest: dict[str, Any],
    *,
    role: str | None = None,
    limit: int | None = None,
) -> str:
    """Human-readable column listing for CLI / notebooks."""
    cols = manifest.get("columns", [])
    if role:
        cols = [c for c in cols if c.get("role") == role]

    modality = manifest.get("modality", MODALITY_CATALOG)
    lines = [
        f"Survey: {manifest.get('survey')}  modality: {modality}",
        f"HEALPix order: {manifest.get('hats_order')}  columns: {manifest.get('n_columns')}",
    ]
    if modality == MODALITY_CATALOG:
        lines.append(
            f"rows: {manifest.get('total_rows')}  join: {manifest.get('source_id_column')!r}  "
            f"sky: {manifest.get('ra_column')!r}, {manifest.get('dec_column')!r}  "
            f"redshift: {manifest.get('redshift_column')!r}"
        )
    elif modality == MODALITY_SPECTRA:
        lines.append(
            f"n_pix: {manifest.get('n_pix')}  wavelength: {manifest.get('wavelength_mode')}  "
            f"resolution: {manifest.get('has_resolution')}  meta: {manifest.get('meta_fields')}"
        )
    elif modality == MODALITY_CUTOUT:
        lines.append(
            f"bands: {manifest.get('band_names')}  shape: "
            f"{manifest.get('n_bands')}×{manifest.get('height')}×{manifest.get('width')}"
        )
    has_overlay = any(c.get("unit") or c.get("description") for c in cols)
    if has_overlay:
        lines.append("(column overlay applied)")
    lines.extend(["", f"{'name':<28} {'dtype':<32} {'role':<12} notes", "-" * 84])
    shown = cols if limit is None else cols[:limit]
    for c in shown:
        notes = c.get("description") or c.get("unit") or ""
        if c.get("homogenized_ab_offset") is not None:
            notes = (notes + " " if notes else "") + f"AB offset {c['homogenized_ab_offset']}"
        lines.append(
            f"{c['name']:<28} {c['dtype']:<32} {c.get('role', ''):<12} {notes}"
        )
    if limit is not None and len(cols) > limit:
        lines.append(f"... ({len(cols) - limit} more columns; omit --limit to show all)")
    groups = manifest.get("column_groups") or {}
    if groups and not role:
        lines.extend(["", "Column groups (prefix → members):"])
        for prefix in sorted(groups)[:20]:
            members = groups[prefix]
            preview = ", ".join(members[:6])
            if len(members) > 6:
                preview += f", … (+{len(members) - 6})"
            lines.append(f"  {prefix}: {preview}")
        if len(groups) > 20:
            lines.append(f"  … ({len(groups) - 20} more groups)")
    return "\n".join(lines)


try:
    import click

    from data_lake.cli_utils import config_option, load_optional_config
    from data_lake.config import LakeConfig

    @click.command("dl-describe-survey")
    @click.argument("survey", type=str)
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option(
        "--role",
        default=None,
        help=f"Filter columns by role ({ROLE_ID}, {ROLE_SKY}, {ROLE_PHOTOMETRY}, …).",
    )
    @click.option("--limit", default=None, type=int, help="Max columns to print.")
    @click.option(
        "--modality",
        "modality",
        default=MODALITY_CATALOG,
        type=click.Choice([MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT]),
        show_default=True,
        help="Lake layer: catalog (Parquet), spectra (Zarr), or cutout (Zarr).",
    )
    @click.option(
        "--rebuild",
        is_flag=True,
        help="Regenerate schema_manifest.json from on-disk data before printing.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Emit full manifest as JSON.")
    def cli(
        survey: str,
        output_root: Path | None,
        config_path: Path | None,
        role: str | None,
        limit: int | None,
        modality: str,
        rebuild: bool,
        as_json: bool,
    ) -> None:
        """List columns, types, and join keys for a survey layer."""
        cfg = load_optional_config(config_path)
        if cfg is None and output_root is None:
            raise click.UsageError(
                "Provide OUTPUT_ROOT or set DATA_LAKE_CONFIG / lake_config.toml."
            )
        lake_root = Path(output_root) if output_root is not None else cfg.lake.root
        layer_root = survey_layer_root(lake_root, survey, modality)
        if not layer_root.is_dir():
            raise click.ClickException(f"Survey layer not found: {layer_root}")

        try:
            manifest = get_survey_manifest(
                lake_root, survey, modality, rebuild=rebuild, apply_overlay=True
            )
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc)) from exc

        if as_json:
            click.echo(json.dumps(manifest, indent=2))
        else:
            click.echo(format_manifest_table(manifest, role=role, limit=limit))

except ImportError:
    cli = None  # type: ignore[misc, assignment]
