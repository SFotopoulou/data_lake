"""
schema_registry – per-survey column manifests for analyst discovery.

Written at catalog ingest finalize (``schema_manifest.json``).  Browsed via
``dl-describe-survey``.
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


def load_catalog_schema_manifest(catalog_root: Path | str) -> dict[str, Any]:
    path = Path(catalog_root) / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No {MANIFEST_FILENAME} under {catalog_root}. "
            "Re-run catalog ingest finalize or dl-rebuild-catalog-indices after upgrading."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


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

    lines = [
        f"Survey: {manifest.get('survey')}  modality: {manifest.get('modality')}",
        f"HEALPix order: {manifest.get('hats_order')}  columns: {manifest.get('n_columns')}  "
        f"rows: {manifest.get('total_rows')}",
        f"Join: {manifest.get('source_id_column')!r}  sky: "
        f"{manifest.get('ra_column')!r}, {manifest.get('dec_column')!r}  "
        f"redshift: {manifest.get('redshift_column')!r}",
        "",
        f"{'name':<32} {'dtype':<36} {'role':<14}",
        "-" * 84,
    ]
    shown = cols if limit is None else cols[:limit]
    for c in shown:
        lines.append(
            f"{c['name']:<32} {c['dtype']:<36} {c.get('role', ''):<14}"
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
        "--rebuild",
        is_flag=True,
        help="Regenerate schema_manifest.json from Parquet on disk before printing.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Emit full manifest as JSON.")
    def cli(
        survey: str,
        output_root: Path | None,
        config_path: Path | None,
        role: str | None,
        limit: int | None,
        rebuild: bool,
        as_json: bool,
    ) -> None:
        """List catalog columns, types, and join keys for a survey."""
        cfg = load_optional_config(config_path)
        if cfg is None and output_root is None:
            raise click.UsageError(
                "Provide OUTPUT_ROOT or set DATA_LAKE_CONFIG / lake_config.toml."
            )
        lake_root = Path(output_root) if output_root is not None else cfg.lake.root
        catalog_root = lake_root / "catalogs" / survey

        if rebuild or not (catalog_root / MANIFEST_FILENAME).is_file():
            info_path = catalog_root / "catalog_info.json"
            if not info_path.is_file():
                raise click.ClickException(f"Catalog not found: {catalog_root}")
            with open(info_path) as fh:
                info = json.load(fh)
            write_catalog_schema_manifest(
                catalog_root,
                survey,
                hats_order=int(info.get("hats_order", 5)),
                ra_column=str(info.get("ra_column", "ra")),
                dec_column=str(info.get("dec_column", "dec")),
                source_id_mode=str(info.get("source_id_mode", "sequential")),
                total_rows=info.get("total_rows"),
            )

        manifest = load_catalog_schema_manifest(catalog_root)
        if as_json:
            click.echo(json.dumps(manifest, indent=2))
        else:
            click.echo(format_manifest_table(manifest, role=role, limit=limit))

except ImportError:
    cli = None  # type: ignore[misc, assignment]
