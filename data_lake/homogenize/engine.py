"""Materialise homogenized catalog products from native survey tiles."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from data_lake.discovery.selection import BaseSelection
from data_lake.homogenize.product import resolve_rules_for_product
from data_lake.homogenize.registry import load_transform
from data_lake.homogenize.transforms import (
    RuleResolution,
    TransformRule,
    apply_rules_to_frame,
    resolve_rules_from_manifest,
)
from data_lake.ingest.fits_to_parquet import (
    LAKE_JOIN_ID_COLUMN,
    canonical_arrow_type,
    finalize_catalog_survey,
    healpix_dir,
    normalize_catalog_table_types,
)
from data_lake.schema_registry import (
    CATALOG_KIND_PRODUCT,
    MODALITY_CATALOG,
    PRODUCT_SUBTYPE_HOMOGENIZED,
    resolve_catalog_root,
)

log = logging.getLogger(__name__)

_MAX_WORKERS = 64
_ZSTD_LEVEL = 3


def _storage_type(dtype: pa.DataType) -> pa.DataType:
    """Canonical Parquet storage type; promote null-only columns to float64."""
    if pa.types.is_null(dtype):
        return pa.float64()
    return canonical_arrow_type(dtype)


def _align_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Add missing columns as typed nulls and cast so every tile shares *schema*."""
    import pyarrow.compute as pc

    arrays: list[pa.Array] = []
    for field in schema:
        if field.name not in table.schema.names:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
            continue
        col = table.column(field.name).combine_chunks()
        if col.type.equals(field.type):
            arrays.append(col)
        elif pa.types.is_null(col.type):
            arrays.append(pa.nulls(len(col), type=field.type))
        else:
            try:
                arrays.append(pc.cast(col, field.type, safe=False))
            except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
                raise TypeError(
                    f"Cannot cast column {field.name!r} from {col.type} to "
                    f"{field.type}: {exc}"
                ) from exc
    return pa.Table.from_arrays(arrays, schema=schema)


def _homogenize_target_fields(rules: Sequence[TransformRule]) -> list[pa.Field]:
    """Homogenized photometry columns are always float64 (null when rule skipped)."""
    fields: list[pa.Field] = []
    seen: set[str] = set()
    for rule in rules:
        for name in (rule.target_column, rule.target_uncertainty_column):
            if name and name not in seen:
                fields.append(pa.field(name, pa.float64()))
                seen.add(name)
    return fields


def build_homogenize_catalog_schema(
    *,
    norder: int,
    ra_col: str,
    dec_col: str,
    rules: Sequence[TransformRule],
    passthrough: Sequence[str] = (),
) -> pa.Schema:
    """Canonical schema for ``homogenize_catalog`` output tiles."""
    fields: list[pa.Field] = [
        pa.field(LAKE_JOIN_ID_COLUMN, pa.int64()),
        pa.field(ra_col, pa.float64()),
        pa.field(dec_col, pa.float64()),
        pa.field(f"_healpix_norder{norder}", pa.int64()),
    ]
    seen = {f.name for f in fields}
    for col in passthrough:
        if col not in seen:
            fields.append(pa.field(col, pa.float64()))
            seen.add(col)
    for field in _homogenize_target_fields(rules):
        if field.name not in seen:
            fields.append(field)
            seen.add(field.name)
    return pa.schema(fields)


def build_homogenize_product_schema(
    source_schema: pa.Schema,
    rules: Sequence[TransformRule],
) -> pa.Schema:
    """Canonical schema for ``homogenize_product``: source columns + AB targets."""
    fields: list[pa.Field] = [
        pa.field(f.name, _storage_type(f.type), nullable=True)
        for f in source_schema
    ]
    seen = {f.name for f in fields}
    for field in _homogenize_target_fields(rules):
        if field.name not in seen:
            fields.append(field)
            seen.add(field.name)
    for rule in rules:
        for name in (rule.source_column, rule.uncertainty_column):
            if name and name not in seen:
                fields.append(pa.field(name, pa.float64()))
                seen.add(name)
    return pa.schema(fields)


def union_product_tile_schema(catalog_root: Path) -> pa.Schema:
    """Union column names/types across product tiles (handles sparse gather).

    Null-typed columns (all-unmatched partner fields in some tiles) must not be
    promoted to float64 until every tile has been seen — otherwise a later
    string/int column is locked into float64 and ``pc.cast`` fails with
    ``Failed to parse string`` on a column that is not in the recipe.
    """
    from data_lake.ingest.fits_to_parquet import _canonical_merge_types

    catalog_root = Path(catalog_root)
    types: dict[str, pa.DataType] = {}
    order: list[str] = []
    for path in sorted(catalog_root.rglob("Npix=*.parquet")):
        sch = pq.read_schema(str(path))
        for field in sch:
            if field.name not in types:
                # Keep null as null so a later concrete dtype can win.
                types[field.name] = field.type
                order.append(field.name)
            else:
                types[field.name] = _canonical_merge_types(types[field.name], field.type)
    if not order:
        raise FileNotFoundError(f"No Parquet tiles under {catalog_root}")
    # Remaining all-null columns → float64 (photometry partner convention).
    return pa.schema(
        [pa.field(n, _storage_type(types[n]), nullable=True) for n in order]
    )


def prepare_homogenize_tile(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Align and normalize a homogenized tile before Parquet write."""
    aligned = _align_table_to_schema(table, schema)
    return normalize_catalog_table_types(aligned)


@dataclass
class HomogenizeResult:
    product: str
    output_root: Path
    n_tiles_written: int
    n_rows: int
    transform_id: str
    resolution: dict[str, Any]
    lineage: list[dict[str, Any]]
    check_only: bool
    elapsed_s: float


def _read_source_info(lake_root: Path, survey: str) -> dict[str, Any]:
    catalog_root = resolve_catalog_root(lake_root, survey)
    path = catalog_root / "catalog_info.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No catalog_info.json for {survey!r} "
            f"(checked catalogs/{survey}/ and products/{survey}/)"
        )
    return json.loads(path.read_text())


def _homogenize_tile(
    npix: int,
    *,
    lake_root: Path,
    source_catalog: str,
    norder: int,
    rules: Sequence[TransformRule],
    ra_col: str,
    dec_col: str,
    selection_ids: set[int] | None,
    passthrough: Sequence[str],
    keep_native: bool = False,
) -> tuple[int, pa.Table | None, list[dict[str, Any]]]:
    import polars as pl

    tile_path = (
        resolve_catalog_root(lake_root, source_catalog) / healpix_dir(norder, npix) / f"Npix={npix}.parquet"
    )
    if not tile_path.is_file():
        return 0, None, []

    table = pq.read_table(tile_path)
    df = pl.from_arrow(table)
    if selection_ids is not None:
        if LAKE_JOIN_ID_COLUMN in df.columns:
            df = df.filter(pl.col(LAKE_JOIN_ID_COLUMN).is_in(list(selection_ids)))
        if df.is_empty():
            return 0, None, []

    transformed, lineage = apply_rules_to_frame(df, rules)
    hp_col = f"_healpix_norder{norder}"
    if keep_native:
        out = transformed
    else:
        keep = [LAKE_JOIN_ID_COLUMN, ra_col, dec_col]
        if hp_col in transformed.columns:
            keep.append(hp_col)
        for col in passthrough:
            if col in transformed.columns and col not in keep:
                keep.append(col)
        for rule in rules:
            if rule.target_column in transformed.columns:
                keep.append(rule.target_column)
            if (
                rule.target_uncertainty_column
                and rule.target_uncertainty_column in transformed.columns
            ):
                keep.append(rule.target_uncertainty_column)

        seen: set[str] = set()
        cols: list[str] = []
        for c in keep:
            if c in transformed.columns and c not in seen:
                cols.append(c)
                seen.add(c)
        out = transformed.select(cols)
    return out.height, out.to_arrow(), lineage


def _write_product_info(
    out_root: Path,
    *,
    name: str,
    source_survey: str | None,
    source_product: str | None,
    transform_id: str,
    transform_version: int,
    source_info: dict[str, Any],
    selection: BaseSelection,
    resolution: RuleResolution,
    lineage: list[dict[str, Any]],
    n_rows: int,
    norder: int,
) -> None:
    ra = source_info.get("ra_column") or "ra"
    dec = source_info.get("dec_column") or "dec"
    provenance: dict[str, Any] = {
        "transform_id": transform_id,
        "transform_version": transform_version,
        "selection": {
            "survey": selection.base_survey,
            "norder": selection.norder,
            "n_npix": len(selection.npix),
            "n_source_ids": (
                None if selection.source_ids is None else len(selection.source_ids)
            ),
        },
        "resolution": resolution.to_dict(),
        "column_lineage": lineage,
    }
    if source_product is not None:
        provenance["source_product"] = source_product
        provenance["source_provenance"] = source_info.get("provenance")
    else:
        provenance["source_survey"] = source_survey
    info = {
        "catalog_name": name,
        "kind": CATALOG_KIND_PRODUCT,
        "product_subtype": PRODUCT_SUBTYPE_HOMOGENIZED,
        "modality": MODALITY_CATALOG,
        "hats_order": norder,
        "ra_column": ra,
        "dec_column": dec,
        "link_id_mode": source_info.get("link_id_mode", f"column:{LAKE_JOIN_ID_COLUMN}"),
        "link_id_column": source_info.get("link_id_column", LAKE_JOIN_ID_COLUMN),
        "total_rows": n_rows,
        "provenance": provenance,
        "schema_version": "1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "catalog_info.json", "w") as fh:
        json.dump(info, fh, indent=2)


def homogenize_catalog(
    lake_root: Path | str,
    survey: str,
    transform_id: str,
    selection: BaseSelection,
    *,
    materialize_as: str,
    columns: Sequence[str] | None = None,
    passthrough: Sequence[str] | None = None,
    overwrite: bool = False,
    check_only: bool = False,
    n_workers: int = 8,
    show_progress: bool = False,
) -> HomogenizeResult:
    """Materialise a homogenized catalog product for one native survey + selection."""
    lake_root = Path(lake_root)
    if selection.base_survey != survey:
        raise ValueError(
            f"selection base_survey {selection.base_survey!r} != survey {survey!r}"
        )

    transform = load_transform(lake_root, transform_id)
    if transform.get("modality", "catalog") != "catalog":
        raise ValueError(
            f"Transform {transform_id!r} modality is {transform.get('modality')!r}; "
            "use --modality for spectra/cutout (Phase F)"
        )

    resolution = resolve_rules_from_manifest(
        lake_root, transform, survey, columns=columns,
    )
    if not resolution.applied:
        raise ValueError(
            f"No applicable rules for survey {survey!r} and transform {transform_id!r}: "
            f"{resolution.to_dict()}"
        )

    source_info = _read_source_info(lake_root, survey)
    ra_col = str(source_info.get("ra_column", "ra"))
    dec_col = str(source_info.get("dec_column", "dec"))
    from data_lake.schema_registry import catalog_write_root
    norder = selection.norder
    npix_list = sorted(selection.npix)
    selection_ids = set(selection.source_ids) if selection.source_ids is not None else None

    out_root = catalog_write_root(lake_root, materialize_as)
    if out_root.exists() and not overwrite and not check_only:
        raise FileExistsError(
            f"product catalog already exists: {out_root} (use --overwrite)"
        )

    if check_only:
        return HomogenizeResult(
            product=materialize_as,
            output_root=out_root,
            n_tiles_written=0,
            n_rows=0,
            transform_id=transform_id,
            resolution=resolution.to_dict(),
            lineage=[],
            check_only=True,
            elapsed_s=0.0,
        )

    passthrough_cols = list(passthrough or [])
    rules = resolution.applied
    out_schema = build_homogenize_catalog_schema(
        norder=norder,
        ra_col=ra_col,
        dec_col=dec_col,
        rules=rules,
        passthrough=passthrough_cols,
    )
    t0 = time.perf_counter()
    n_rows = 0
    n_tiles = 0
    all_lineage: list[dict[str, Any]] = []

    workers = max(1, min(n_workers, _MAX_WORKERS, len(npix_list) or 1))

    def _work(npix: int) -> tuple[int, pa.Table | None, list[dict[str, Any]]]:
        return _homogenize_tile(
            npix,
            lake_root=lake_root,
            source_catalog=survey,
            norder=norder,
            rules=rules,
            ra_col=ra_col,
            dec_col=dec_col,
            selection_ids=selection_ids,
            passthrough=passthrough_cols,
            keep_native=False,
        )

    results: list[tuple[int, pa.Table | None, list[dict[str, Any]]]] = []
    if workers <= 1 or len(npix_list) <= 1:
        tile_iter = npix_list
        if show_progress:
            try:
                from tqdm.auto import tqdm

                tile_iter = tqdm(npix_list, unit="tile", desc=f"homogenize {materialize_as}")
            except ImportError:
                pass
        for npix in tile_iter:
            results.append(_work(npix))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_work, npix) for npix in npix_list]
            if show_progress:
                try:
                    from tqdm.auto import tqdm

                    for fut in tqdm(
                        as_completed(futures), total=len(futures),
                        unit="tile", desc=f"homogenize {materialize_as}",
                    ):
                        results.append(fut.result())
                except ImportError:
                    for fut in as_completed(futures):
                        results.append(fut.result())
            else:
                for fut in as_completed(futures):
                    results.append(fut.result())

    for n, table, lin in results:
        if table is None or n == 0:
            continue
        table = prepare_homogenize_tile(table, out_schema)
        npix_val = int(table.column(f"_healpix_norder{norder}")[0].as_py())
        tile_dir = out_root / healpix_dir(norder, npix_val)
        tile_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table,
            tile_dir / f"Npix={npix_val}.parquet",
            compression="zstd",
            compression_level=_ZSTD_LEVEL,
        )
        n_rows += n
        n_tiles += 1
        if lin and not all_lineage:
            all_lineage = lin

    _write_product_info(
        out_root,
        name=materialize_as,
        source_survey=survey,
        source_product=None,
        transform_id=transform_id,
        transform_version=int(transform.get("version", 1)),
        source_info=source_info,
        selection=selection,
        resolution=resolution,
        lineage=all_lineage,
        n_rows=n_rows,
        norder=norder,
    )
    finalize_catalog_survey(
        out_root,
        materialize_as,
        norder,
        ra_col=ra_col,
        dec_col=dec_col,
    )

    return HomogenizeResult(
        product=materialize_as,
        output_root=out_root,
        n_tiles_written=n_tiles,
        n_rows=n_rows,
        transform_id=transform_id,
        resolution=resolution.to_dict(),
        lineage=all_lineage,
        check_only=False,
        elapsed_s=time.perf_counter() - t0,
    )


def homogenize_product(
    lake_root: Path | str,
    from_product: str,
    transform_id: str,
    selection: BaseSelection,
    *,
    materialize_as: str,
    columns: Sequence[str] | None = None,
    overwrite: bool = False,
    check_only: bool = False,
    n_workers: int = 8,
    show_progress: bool = False,
) -> HomogenizeResult:
    """Homogenize a gathered wide product catalog in place on disk → new product."""
    lake_root = Path(lake_root)
    if selection.base_survey != from_product:
        raise ValueError(
            f"selection base_survey {selection.base_survey!r} != from_product {from_product!r}"
        )

    transform = load_transform(lake_root, transform_id)
    if transform.get("modality", "catalog") != "catalog":
        raise ValueError(
            f"Transform {transform_id!r} modality is {transform.get('modality')!r}; "
            "catalog --from-product requires a catalog transform"
        )

    source_info = _read_source_info(lake_root, from_product)
    provenance = source_info.get("provenance") or {}
    product_root = resolve_catalog_root(lake_root, from_product)
    source_schema = union_product_tile_schema(product_root)
    available = set(source_schema.names)
    ra_col = str(source_info.get("ra_column") or "ra")
    dec_col = str(source_info.get("dec_column") or "dec")

    resolution = resolve_rules_for_product(
        lake_root, transform, provenance, available, columns=columns,
    )
    if not resolution.applied:
        raise ValueError(
            f"No applicable rules for product {from_product!r} and transform {transform_id!r}: "
            f"{resolution.to_dict()}"
        )

    from data_lake.schema_registry import catalog_write_root as _cwr
    norder = selection.norder
    npix_list = sorted(selection.npix)
    selection_ids = set(selection.source_ids) if selection.source_ids is not None else None
    out_root = _cwr(lake_root, materialize_as)
    if out_root.exists() and not overwrite and not check_only:
        raise FileExistsError(
            f"product catalog already exists: {out_root} (use --overwrite)"
        )

    if check_only:
        return HomogenizeResult(
            product=materialize_as,
            output_root=out_root,
            n_tiles_written=0,
            n_rows=0,
            transform_id=transform_id,
            resolution=resolution.to_dict(),
            lineage=[],
            check_only=True,
            elapsed_s=0.0,
        )

    rules = resolution.applied
    out_schema = build_homogenize_product_schema(source_schema, rules)
    t0 = time.perf_counter()
    n_rows = 0
    n_tiles = 0
    all_lineage: list[dict[str, Any]] = []
    workers = max(1, min(n_workers, _MAX_WORKERS, len(npix_list) or 1))

    def _work(npix: int) -> tuple[int, pa.Table | None, list[dict[str, Any]]]:
        return _homogenize_tile(
            npix,
            lake_root=lake_root,
            source_catalog=from_product,
            norder=norder,
            rules=rules,
            ra_col=ra_col,
            dec_col=dec_col,
            selection_ids=selection_ids,
            passthrough=[],
            keep_native=True,
        )

    results: list[tuple[int, pa.Table | None, list[dict[str, Any]]]] = []
    if workers <= 1 or len(npix_list) <= 1:
        tile_iter = npix_list
        if show_progress:
            try:
                from tqdm.auto import tqdm

                tile_iter = tqdm(npix_list, unit="tile", desc=f"homogenize {materialize_as}")
            except ImportError:
                pass
        for npix in tile_iter:
            results.append(_work(npix))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_work, npix) for npix in npix_list]
            if show_progress:
                try:
                    from tqdm.auto import tqdm

                    for fut in tqdm(
                        as_completed(futures), total=len(futures),
                        unit="tile", desc=f"homogenize {materialize_as}",
                    ):
                        results.append(fut.result())
                except ImportError:
                    for fut in as_completed(futures):
                        results.append(fut.result())
            else:
                for fut in as_completed(futures):
                    results.append(fut.result())

    for n, table, lin in results:
        if table is None or n == 0:
            continue
        table = prepare_homogenize_tile(table, out_schema)
        npix_val = int(table.column(f"_healpix_norder{norder}")[0].as_py())
        tile_dir = out_root / healpix_dir(norder, npix_val)
        tile_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table,
            tile_dir / f"Npix={npix_val}.parquet",
            compression="zstd",
            compression_level=_ZSTD_LEVEL,
        )
        n_rows += n
        n_tiles += 1
        if lin and not all_lineage:
            all_lineage = lin

    _write_product_info(
        out_root,
        name=materialize_as,
        source_survey=None,
        source_product=from_product,
        transform_id=transform_id,
        transform_version=int(transform.get("version", 1)),
        source_info=source_info,
        selection=selection,
        resolution=resolution,
        lineage=all_lineage,
        n_rows=n_rows,
        norder=norder,
    )
    finalize_catalog_survey(
        out_root,
        materialize_as,
        norder,
        ra_col=ra_col,
        dec_col=dec_col,
    )

    return HomogenizeResult(
        product=materialize_as,
        output_root=out_root,
        n_tiles_written=n_tiles,
        n_rows=n_rows,
        transform_id=transform_id,
        resolution=resolution.to_dict(),
        lineage=all_lineage,
        check_only=False,
        elapsed_s=time.perf_counter() - t0,
    )
