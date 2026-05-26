"""
lake_registry – lake-wide survey index and master-association metadata.

* ``shared/registry/surveys.parquet`` — one row per survey × modality
* ``<master>.meta.json`` — maps master Parquet ID columns → catalog join keys
* ``dl-describe-lake``, ``dl-describe-master``, ``dl-refresh-lake-registry``
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.schema_registry import (
    MANIFEST_FILENAME,
    MODALITY_CUTOUT,
    MODALITY_SPECTRA,
    format_manifest_table,
    load_catalog_schema_manifest,
    load_schema_manifest,
    write_catalog_schema_manifest,
    write_cutout_schema_manifest,
    write_spectra_schema_manifest,
)

log = logging.getLogger(__name__)

MASTER_META_VERSION = "1"
REGISTRY_DIRNAME = "registry"
REGISTRY_FILENAME = "surveys.parquet"

_MASTER_QA_COLUMNS = frozenset(
    {
        "sep_arcsec",
        "sep_arcsec_1",
        "sep_arcsec_2",
        "match_rank",
        "rank",
        "healpix_npix",
        "npix",
        "norder",
        "primary_survey",
    }
)


def registry_root(lake_root: Path | str, *, shared_subdir: str = REGISTRY_DIRNAME) -> Path:
    return Path(lake_root) / "shared" / shared_subdir


def registry_path(lake_root: Path | str) -> Path:
    return registry_root(lake_root) / REGISTRY_FILENAME


def master_meta_path(master_parquet: Path | str) -> Path:
    """Sidecar path: ``master.parquet`` → ``master.meta.json``."""
    p = Path(master_parquet)
    return p.with_suffix(".meta.json")


def read_parquet_column_names(path: Path | str) -> list[str]:
    return list(pq.read_schema(str(path)).names)


def _is_likely_id_column(name: str) -> bool:
    lower = name.lower()
    if lower in _MASTER_QA_COLUMNS:
        return False
    if lower.endswith("_zarr_row") or (lower.endswith("_row") and "id" not in lower):
        return False
    if lower.endswith("_index"):
        return False
    return (
        lower.endswith("_id")
        or lower.endswith("_targetid")
        or lower.endswith("_source_id")
        or lower in ("source_id", "targetid", "objid")
    )


def iter_catalog_surveys(catalogs_root: Path) -> Iterator[tuple[str, Path]]:
    if not catalogs_root.is_dir():
        return
    for p in sorted(catalogs_root.iterdir()):
        if not p.is_dir() or p.name == "crossmatch":
            continue
        if (p / "catalog_info.json").is_file() or any(p.glob("Norder=*")):
            yield p.name, p


def iter_modality_surveys(layer_root: Path, info_name: str) -> Iterator[tuple[str, Path]]:
    if not layer_root.is_dir():
        return
    for p in sorted(layer_root.iterdir()):
        if p.is_dir() and ((p / info_name).is_file() or any(p.glob("Norder=*"))):
            yield p.name, p


def _iter_zarr_tiles(survey_root: Path) -> Iterator[Path]:
    yield from sorted(survey_root.rglob("Npix=*.zarr"))


def _zarr_tile_n_sources(tile_path: Path) -> int | None:
    """Row count for one spectrum/cutout tile (reads ``source_id`` array metadata only)."""
    meta_path = tile_path / "source_id" / "zarr.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            shape = meta.get("shape")
            if shape:
                return int(shape[0])
        except Exception:
            pass
    try:
        import zarr

        root = zarr.open_group(
            store=zarr.storage.LocalStore(str(tile_path)),
            mode="r",
            zarr_format=3,
        )
        return int(root["source_id"].shape[0])
    except Exception:
        log.debug("Could not count sources in Zarr tile %s", tile_path, exc_info=True)
        return None


def _count_zarr_sources(survey_root: Path) -> int | None:
    """Sum ``source_id`` lengths across all ``Npix=*.zarr`` tiles under a survey."""
    total = 0
    n_tiles = 0
    for tile in _iter_zarr_tiles(survey_root):
        n = _zarr_tile_n_sources(tile)
        if n is None:
            log.warning("Skipping unreadable Zarr tile for row count: %s", tile)
            continue
        total += n
        n_tiles += 1
    return total if n_tiles else None


def _load_catalog_manifest_or_none(catalog_root: Path, survey: str) -> dict[str, Any] | None:
    path = catalog_root / MANIFEST_FILENAME
    if path.is_file():
        return load_catalog_schema_manifest(catalog_root)
    info_path = catalog_root / "catalog_info.json"
    if not info_path.is_file():
        return None
    with open(info_path) as fh:
        info = json.load(fh)
    try:
        write_catalog_schema_manifest(
            catalog_root,
            survey,
            hats_order=int(info.get("hats_order", 5)),
            ra_column=str(info.get("ra_column", "ra")),
            dec_column=str(info.get("dec_column", "dec")),
            source_id_mode=str(info.get("source_id_mode", "sequential")),
            total_rows=info.get("total_rows"),
        )
        return load_catalog_schema_manifest(catalog_root)
    except Exception as exc:
        log.warning("Could not build manifest for %s: %s", survey, exc)
        return None


def guess_master_meta(
    master_parquet: Path | str,
    lake_root: Path | str,
    *,
    primary_survey: str | None = None,
) -> dict[str, Any]:
    """Infer partner mapping from master column names + on-disk catalog manifests."""
    master_parquet = Path(master_parquet)
    lake_root = Path(lake_root)
    columns = read_parquet_column_names(master_parquet)

    manifests: dict[str, dict[str, Any]] = {}
    for survey, root in iter_catalog_surveys(lake_root / "catalogs"):
        m = _load_catalog_manifest_or_none(root, survey)
        if m is not None:
            manifests[survey] = m

    partners: list[dict[str, str]] = []
    used_surveys: set[str] = set()

    for col in columns:
        if not _is_likely_id_column(col):
            continue
        col_lower = col.lower().replace("-", "_")
        best: tuple[str, str] | None = None

        for survey, manifest in manifests.items():
            if survey in used_surveys:
                continue
            cat_id = manifest["source_id_column"]
            cat_lower = cat_id.lower()
            survey_key = survey.lower().replace("-", "_")

            if col_lower == cat_lower or col == cat_id:
                best = (survey, cat_id)
                break
            if col_lower.endswith(f"_{cat_lower}") or col_lower == f"{survey_key}_{cat_lower}":
                best = (survey, cat_id)
                break
            tokens = [t for t in survey_key.split("_") if len(t) > 2]
            if tokens and all(t in col_lower for t in tokens[:2]):
                best = (survey, cat_id)
                break
            if any(t in col_lower for t in tokens):
                best = (survey, cat_id)
                break

        if best is not None:
            used_surveys.add(best[0])
            partners.append(
                {
                    "survey": best[0],
                    "master_column": col,
                    "catalog_id_column": best[1],
                }
            )

    if primary_survey is None and partners:
        primary_survey = partners[0]["survey"]

    return {
        "meta_version": MASTER_META_VERSION,
        "master_parquet": str(master_parquet),
        "primary_survey": primary_survey,
        "partners": partners,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inferred": True,
    }


def load_master_meta(
    master_parquet: Path | str,
    lake_root: Path | str,
    *,
    allow_guess: bool = True,
) -> dict[str, Any]:
    master_parquet = Path(master_parquet)
    meta_path = master_meta_path(master_parquet)
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("inferred", False)
        return data
    if allow_guess:
        return guess_master_meta(master_parquet, lake_root)
    raise FileNotFoundError(
        f"No master metadata at {meta_path}. "
        "Create it manually or re-run with default guess enabled."
    )


def write_master_meta(master_parquet: Path | str, meta: dict[str, Any]) -> Path:
    master_parquet = Path(master_parquet)
    out = master_meta_path(master_parquet)
    meta = dict(meta)
    meta["meta_version"] = MASTER_META_VERSION
    meta["master_parquet"] = str(master_parquet)
    meta["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["inferred"] = False
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return out


def _catalog_registry_row(
    lake_root: Path,
    survey: str,
    survey_root: Path,
) -> dict[str, Any]:
    info_path = survey_root / "catalog_info.json"
    info: dict[str, Any] = {}
    if info_path.is_file():
        with open(info_path) as fh:
            info = json.load(fh)

    manifest_rel = None
    manifest_path = survey_root / MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest_rel = str(manifest_path.relative_to(lake_root))

    source_id = ra = dec = None
    if manifest_path.is_file():
        with open(manifest_path) as fh:
            m = json.load(fh)
        source_id = m.get("source_id_column")
        ra = m.get("ra_column")
        dec = m.get("dec_column")

    return {
        "survey": survey,
        "modality": "catalog",
        "path": str(survey_root.relative_to(lake_root)),
        "hats_order": info.get("hats_order"),
        "source_id_column": source_id,
        "ra_column": ra,
        "dec_column": dec,
        "n_columns": info.get("total_columns"),
        "total_rows": info.get("total_rows"),
        "manifest_path": manifest_rel,
        "has_schema_manifest": manifest_path.is_file(),
    }


def _load_layer_manifest_or_none(
    survey_root: Path,
    survey: str,
    modality: str,
) -> dict[str, Any] | None:
    path = survey_root / MANIFEST_FILENAME
    if path.is_file():
        return load_schema_manifest(survey_root)
    try:
        if modality == MODALITY_SPECTRA:
            write_spectra_schema_manifest(survey_root, survey)
        elif modality == MODALITY_CUTOUT:
            write_cutout_schema_manifest(survey_root, survey)
        else:
            return None
        return load_schema_manifest(survey_root)
    except Exception as exc:
        log.warning("Could not build %s manifest for %s: %s", modality, survey, exc)
        return None


def _info_registry_row(
    lake_root: Path,
    survey: str,
    survey_root: Path,
    modality: str,
    info_name: str,
) -> dict[str, Any] | None:
    info_path = survey_root / info_name
    if not info_path.is_file():
        return None
    with open(info_path) as fh:
        info = json.load(fh)

    manifest_rel = None
    n_columns = None
    source_id = "source_id"
    manifest_path = survey_root / MANIFEST_FILENAME
    manifest = _load_layer_manifest_or_none(survey_root, survey, modality)
    if manifest is not None:
        manifest_rel = str(manifest_path.relative_to(lake_root))
        n_columns = manifest.get("n_columns")
        source_id = manifest.get("source_id_column") or source_id

    total_rows: int | None = None
    if modality == MODALITY_SPECTRA:
        counted = _count_zarr_sources(survey_root)
        if counted is not None:
            total_rows = counted
        else:
            raw = info.get("total_spectra", info.get("total_rows"))
            total_rows = int(raw) if raw is not None else None

    return {
        "survey": survey,
        "modality": modality,
        "path": str(survey_root.relative_to(lake_root)),
        "hats_order": info.get("hats_order"),
        "source_id_column": source_id,
        "ra_column": info.get("ra_column"),
        "dec_column": info.get("dec_column"),
        "n_columns": n_columns,
        "total_rows": total_rows,
        "manifest_path": manifest_rel or str(info_path.relative_to(lake_root)),
        "has_schema_manifest": manifest_path.is_file(),
    }


def build_lake_registry_table(lake_root: Path | str) -> pa.Table:
    """Scan the lake and build registry rows for all discovered surveys."""
    lake_root = Path(lake_root)
    rows: list[dict[str, Any]] = []

    catalogs_root = lake_root / "catalogs"
    for survey, root in iter_catalog_surveys(catalogs_root):
        rows.append(_catalog_registry_row(lake_root, survey, root))

    for survey, root in iter_modality_surveys(lake_root / "spectra", "spectrum_info.json"):
        row = _info_registry_row(lake_root, survey, root, "spectra", "spectrum_info.json")
        if row:
            rows.append(row)

    for survey, root in iter_modality_surveys(lake_root / "cutouts", "cutout_info.json"):
        row = _info_registry_row(lake_root, survey, root, "cutout", "cutout_info.json")
        if row:
            rows.append(row)

    if not rows:
        return pa.table(
            {
                "survey": pa.array([], type=pa.string()),
                "modality": pa.array([], type=pa.string()),
                "path": pa.array([], type=pa.string()),
            }
        )

    return pa.Table.from_pylist(rows)


def refresh_lake_registry(lake_root: Path | str) -> Path:
    """Write ``shared/registry/surveys.parquet``."""
    lake_root = Path(lake_root)
    out_dir = registry_root(lake_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = registry_path(lake_root)
    table = build_lake_registry_table(lake_root)
    pq.write_table(table, str(out_path), compression="zstd")
    log.info("Wrote lake registry (%d rows) → %s", table.num_rows, out_path)
    return out_path


def load_lake_registry(lake_root: Path | str) -> pa.Table:
    path = registry_path(lake_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"No lake registry at {path}. Run: dl-refresh-lake-registry"
        )
    return pq.read_table(str(path))


def format_lake_registry_table(table: pa.Table) -> str:
    import polars as pl

    df = pl.from_arrow(table).sort(["modality", "survey"])
    lines = [
        f"{'survey':<24} {'modality':<10} {'hats':>4} {'cols':>6} {'rows':>12}  "
        f"{'id_col':<16} manifest",
        "-" * 90,
    ]
    for row in df.iter_rows(named=True):
        rows_s = f"{row['total_rows']:,}" if row.get("total_rows") is not None else "—"
        cols_s = str(row["n_columns"]) if row.get("n_columns") is not None else "—"
        hats = row.get("hats_order")
        hats_s = str(hats) if hats is not None else "—"
        manifest = "yes" if row.get("has_schema_manifest") else "no"
        lines.append(
            f"{row['survey']:<24} {row['modality']:<10} {hats_s:>4} {cols_s:>6} {rows_s:>12}  "
            f"{str(row.get('source_id_column') or '—'):<16} {manifest}"
        )
    return "\n".join(lines)


def format_master_report(
    master_parquet: Path | str,
    meta: dict[str, Any],
    lake_root: Path | str,
    *,
    column_limit: int = 15,
) -> str:
    master_parquet = Path(master_parquet)
    lake_root = Path(lake_root)
    cols = read_parquet_column_names(master_parquet)
    meta_path = master_meta_path(master_parquet)

    lines = [
        f"Master Parquet: {master_parquet}",
        f"Master columns ({len(cols)}): {', '.join(cols[:column_limit])}"
        + (f" … (+{len(cols) - column_limit})" if len(cols) > column_limit else ""),
        f"Metadata: {meta_path} ({'found' if meta_path.is_file() else 'not on disk; ' + ('guessed' if meta.get('inferred') else 'missing')})",
        f"Primary survey: {meta.get('primary_survey')}",
        "",
        f"{'survey':<20} {'master_column':<24} {'catalog_id_column':<20} {'n_cols':>6}",
        "-" * 74,
    ]

    for p in meta.get("partners", []):
        survey = p["survey"]
        n_cols = "—"
        manifest_path = lake_root / "catalogs" / survey / MANIFEST_FILENAME
        if manifest_path.is_file():
            with open(manifest_path) as fh:
                n_cols = str(json.load(fh).get("n_columns", "—"))
        lines.append(
            f"{survey:<20} {p['master_column']:<24} {p['catalog_id_column']:<20} {n_cols:>6}"
        )

    lines.extend(
        [
            "",
            "Per-survey column manifests (dl-describe-survey <name>):",
        ]
    )
    for p in meta.get("partners", []):
        survey = p["survey"]
        root = lake_root / "catalogs" / survey
        try:
            manifest = load_catalog_schema_manifest(root)
            lines.append("")
            lines.append(format_manifest_table(manifest, limit=8))
        except FileNotFoundError:
            lines.append(f"\n  {survey}: no schema_manifest.json (run dl-describe-survey {survey} --rebuild)")

    return "\n".join(lines)


try:
    import click

    from data_lake.cli_utils import config_option, load_optional_config

    def _resolve_lake_root(output_root: Path | None, config_path: Path | None) -> Path:
        cfg = load_optional_config(config_path)
        if cfg is None and output_root is None:
            raise click.UsageError(
                "Provide OUTPUT_ROOT or set DATA_LAKE_CONFIG / lake_config.toml."
            )
        return Path(output_root) if output_root is not None else cfg.lake.root

    @click.command("dl-refresh-lake-registry")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    def cli_refresh(output_root: Path | None, config_path: Path | None) -> None:
        """Scan catalogs/spectra/cutouts and write shared/registry/surveys.parquet."""
        lake_root = _resolve_lake_root(output_root, config_path)
        out = refresh_lake_registry(lake_root)
        click.echo(f"Wrote {out} ({pq.read_metadata(str(out)).num_rows} surveys/modalities)")

    @click.command("dl-describe-lake")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option(
        "--refresh",
        is_flag=True,
        help="Rebuild shared/registry/surveys.parquet before printing.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Emit registry as JSON.")
    def cli_describe_lake(
        output_root: Path | None,
        config_path: Path | None,
        refresh: bool,
        as_json: bool,
    ) -> None:
        """List surveys and modalities on disk (registry index)."""
        lake_root = _resolve_lake_root(output_root, config_path)
        if refresh or not registry_path(lake_root).is_file():
            refresh_lake_registry(lake_root)
        table = load_lake_registry(lake_root)
        if as_json:
            click.echo(table.to_pandas().to_json(orient="records", indent=2))
        else:
            click.echo(format_lake_registry_table(table))

    @click.command("dl-describe-master")
    @click.argument("master_parquet", type=click.Path(exists=True, path_type=Path))
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option(
        "--write-meta",
        is_flag=True,
        help="Save guessed or loaded mapping to <master>.meta.json.",
    )
    @click.option(
        "--primary-survey",
        default=None,
        help="Primary survey name when guessing metadata.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Emit master metadata as JSON.")
    def cli_describe_master(
        master_parquet: Path,
        output_root: Path | None,
        config_path: Path | None,
        write_meta: bool,
        primary_survey: str | None,
        as_json: bool,
    ) -> None:
        """Show master association columns mapped to per-survey catalog schemas."""
        lake_root = _resolve_lake_root(output_root, config_path)
        meta = load_master_meta(master_parquet, lake_root, allow_guess=True)
        if primary_survey:
            meta["primary_survey"] = primary_survey
        if write_meta:
            write_master_meta(master_parquet, meta)
            meta["inferred"] = False
            click.echo(f"Wrote {master_meta_path(master_parquet)}", err=True)
        if as_json:
            click.echo(json.dumps(meta, indent=2))
        else:
            click.echo(format_master_report(master_parquet, meta, lake_root))

except ImportError:
    cli_refresh = None  # type: ignore[misc, assignment]
    cli_describe_lake = None
    cli_describe_master = None
