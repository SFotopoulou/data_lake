"""
Cross-validate catalog Parquet ↔ spectrum Zarr linkage for one survey.

Checks that ``_source_id``, ``_healpix_norder{N}``, and ``_spectrum_index`` on
each catalog row agree with the paired ``Npix=*.zarr`` tile's ``_source_id``
array (or legacy ``source_id``).
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import (
    LAKE_JOIN_ID_COLUMN,
    healpix_dir,
    normalize_object_id,
    resolve_link_id_column,
)
from data_lake.ingest.zarr_ids import zarr_join_array


@dataclass
class LinkValidationStats:
    n_tiles_checked: int = 0
    n_linked: int = 0
    n_stale_index: int = 0
    n_wrong_id: int = 0
    n_wrong_healpix: int = 0
    n_orphan_zarr: int = 0
    n_unpatched_catalog: int = 0
    n_missing_catalog_tile: int = 0
    n_empty_zarr: int = 0
    n_null_source_id: int = 0
    n_null_source_id_linked: int = 0


@dataclass
class LinkValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: LinkValidationStats = field(default_factory=LinkValidationStats)

    def ok(self, *, strict: bool) -> bool:
        if self.errors:
            return False
        if strict and self.warnings:
            return False
        return True


def _iter_spectrum_zarr_tiles(spectra_root: Path) -> Iterator[Path]:
    yield from sorted(spectra_root.rglob("Npix=*.zarr"))


def _npix_from_tile_name(name: str) -> int:
    return int(name.split("=")[-1].split(".")[0])


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_catalog_tile_columns(tile_path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    pf = pq.ParquetFile(tile_path)
    available = set(pf.schema_arrow.names)
    missing = [c for c in columns if c not in available]
    if missing:
        raise KeyError(
            f"{tile_path}: missing column(s) {missing!r} "
            f"(available: {sorted(available)[:30]})"
        )
    table = pf.read(columns=columns)
    out: dict[str, np.ndarray] = {}
    for name in columns:
        col = table.column(name).combine_chunks()
        if pa.types.is_dictionary(col.type):
            col = pc.cast(col, col.type.value_type)
        # Integer columns with nulls: to_numpy() yields float64 NaN and breaks ID coercion.
        if pa.types.is_integer(col.type) and col.null_count > 0:
            out[name] = np.array(col.to_pylist(), dtype=object)
        else:
            out[name] = np.asarray(col.to_numpy(zero_copy_only=False))
    return out


def _resolve_norder(
    catalog_root: Path,
    spectra_root: Path,
    norder: int | None,
    rep: LinkValidationReport,
) -> int | None:
    cat_order: int | None = norder
    spec_order: int | None = norder

    cat_info = catalog_root / "catalog_info.json"
    if cat_info.is_file():
        cat_order = int(_load_json(cat_info).get("hats_order", cat_order or 5))
    spec_info = spectra_root / "spectrum_info.json"
    if spec_info.is_file():
        spec_order = int(_load_json(spec_info).get("hats_order", spec_order or 5))

    if cat_order is None:
        cat_order = spec_order or 5
    if spec_order is None:
        spec_order = cat_order

    if cat_order != spec_order:
        rep.errors.append(
            f"hats_order mismatch: catalog {cat_order} vs spectra {spec_order}"
        )
        return None
    return cat_order


def _catalog_tile_path(catalog_root: Path, norder: int, npix: int) -> Path:
    return catalog_root / healpix_dir(norder, npix) / f"Npix={npix}.parquet"


def _optional_catalog_sid(value: object) -> int | None:
    """Return normalized int64 ID, or None for null / missing link parts."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return int(normalize_object_id(value))
    except (ValueError, TypeError):
        return None


def validate_tile_link(
    *,
    zarr_tile: Path,
    catalog_tile: Path | None,
    npix: int,
    norder: int,
    sid_col: str,
    rep: LinkValidationReport,
    sample: int | None = None,
    rng: random.Random | None = None,
) -> None:
    """Validate one paired Zarr / catalog HEALPix tile."""
    import zarr

    stats = rep.stats
    stats.n_tiles_checked += 1

    try:
        root = zarr.open_group(
            store=zarr.storage.LocalStore(str(zarr_tile)),
            mode="r",
            zarr_format=3,
        )
    except Exception as exc:
        rep.errors.append(f"Cannot open Zarr tile {zarr_tile}: {exc}")
        return

    if LAKE_JOIN_ID_COLUMN not in root:
        rep.errors.append(f"{zarr_tile}: missing join array")
        return

    zarr_ids = np.asarray(zarr_join_array(root)[:], dtype=np.int64)
    n_zarr = int(zarr_ids.shape[0])
    if n_zarr == 0:
        stats.n_empty_zarr += 1
        return

    if catalog_tile is None or not catalog_tile.is_file():
        stats.n_missing_catalog_tile += 1
        rep.warnings.append(
            f"{zarr_tile.name}: no catalog tile at expected path "
            f"(Npix={npix}); {n_zarr} Zarr row(s) treated as orphan"
        )
        stats.n_orphan_zarr += n_zarr
        return

    hp_col = f"_healpix_norder{norder}"
    index_col = "_spectrum_index"
    try:
        cols = _read_catalog_tile_columns(
            catalog_tile,
            [sid_col, hp_col, index_col],
        )
    except KeyError as exc:
        rep.errors.append(f"{catalog_tile}: {exc}")
        return

    sid_raw = cols[sid_col]
    sid_iter = sid_raw.tolist() if isinstance(sid_raw, np.ndarray) else list(sid_raw)
    cat_sids: list[int | None] = [_optional_catalog_sid(x) for x in sid_iter]
    cat_hp = np.asarray(cols[hp_col], dtype=np.int64)
    cat_idx = np.asarray(cols[index_col], dtype=np.int64)

    linked_rows = np.nonzero(cat_idx >= 0)[0]
    if sample is not None and sample > 0 and linked_rows.size > sample:
        pick = rng or random.Random(0)
        linked_rows = np.array(
            pick.sample(linked_rows.tolist(), sample),
            dtype=np.int64,
        )

    for row_i in linked_rows.tolist():
        idx = int(cat_idx[row_i])
        sid_opt = cat_sids[row_i]
        if sid_opt is None:
            stats.n_null_source_id_linked += 1
            rep.errors.append(
                f"{catalog_tile.name} row {row_i}: {index_col}={idx} but "
                f"{sid_col} is null (cannot verify Zarr linkage)"
            )
            continue
        sid = sid_opt
        hp = int(cat_hp[row_i])

        if hp != npix:
            stats.n_wrong_healpix += 1
            rep.errors.append(
                f"{catalog_tile.name} row {row_i}: {hp_col}={hp} != tile Npix={npix}"
            )

        if idx < 0 or idx >= n_zarr:
            stats.n_stale_index += 1
            rep.errors.append(
                f"{catalog_tile.name} row {row_i}: {index_col}={idx} out of range "
                f"(Zarr rows={n_zarr})"
            )
            continue

        zarr_sid = int(normalize_object_id(int(zarr_ids[idx])))
        if zarr_sid != sid:
            stats.n_wrong_id += 1
            rep.errors.append(
                f"{catalog_tile.name} row {row_i}: {sid_col}={sid} but "
                f"Zarr {index_col}={idx} has _source_id={zarr_sid}"
            )
        else:
            stats.n_linked += 1

    # Catalog rows keyed by _source_id for reverse lookup (skip null IDs).
    rows_by_sid: dict[int, list[tuple[int, int]]] = {}
    for row_i, sid_opt in enumerate(cat_sids):
        if sid_opt is None:
            stats.n_null_source_id += 1
            continue
        rows_by_sid.setdefault(sid_opt, []).append((row_i, int(cat_idx[row_i])))

    for j, sid_raw in enumerate(zarr_ids.tolist()):
        sid = int(normalize_object_id(int(sid_raw)))
        matches = rows_by_sid.get(sid, [])
        if not matches:
            stats.n_orphan_zarr += 1
            rep.warnings.append(
                f"{zarr_tile.name} row {j}: _source_id={sid} not in catalog tile"
            )
            continue

        if all(idx < 0 for _, idx in matches):
            stats.n_unpatched_catalog += 1
            rep.warnings.append(
                f"{zarr_tile.name} row {j}: _source_id={sid} in catalog but "
                f"{index_col}=-1 (run dl-rebuild-catalog-indices)"
            )
            continue

        if not any(idx == j for _, idx in matches):
            stats.n_orphan_zarr += 1
            rep.warnings.append(
                f"{zarr_tile.name} row {j}: _source_id={sid} not linked at index {j} "
                f"(catalog claims {index_col} in {[idx for _, idx in matches]})"
            )


def run_validation(
    lake_root: Path,
    survey: str,
    *,
    norder: int | None = None,
    link_id_col: str | None = None,
    max_tiles: int | None = None,
    sample: int | None = None,
    seed: int = 0,
) -> LinkValidationReport:
    """Cross-check catalog ``_spectrum_index`` against spectrum Zarr tiles."""
    lake_root = Path(lake_root)
    catalog_root = lake_root / "catalogs" / survey
    spectra_root = lake_root / "spectra" / survey
    rep = LinkValidationReport()

    if not catalog_root.is_dir():
        rep.errors.append(f"Catalog not found: {catalog_root}")
        return rep
    if not spectra_root.is_dir():
        rep.errors.append(f"Spectrum store not found: {spectra_root}")
        return rep

    order = _resolve_norder(catalog_root, spectra_root, norder, rep)
    if order is None:
        return rep

    schema_names: list[str] | None = None
    sample_parquet = next(catalog_root.rglob("Npix=*.parquet"), None)
    if sample_parquet is not None:
        schema_names = pq.read_schema(str(sample_parquet)).names
    try:
        sid_col = resolve_link_id_column(
            catalog_root,
            schema_names=schema_names,
            override=link_id_col,
        )
    except Exception as exc:
        rep.errors.append(f"Cannot resolve join ID column: {exc}")
        return rep

    if "_spectrum_index" not in (schema_names or []):
        rep.errors.append(
            f"Catalog {survey!r} has no _spectrum_index column "
            "(re-ingest catalog or run dl-rebuild-catalog-indices)"
        )
        return rep

    tiles = list(_iter_spectrum_zarr_tiles(spectra_root))
    if not tiles:
        rep.warnings.append(f"No Npix=*.zarr tiles under {spectra_root}")
        return rep

    if max_tiles is not None:
        tiles = tiles[: max(0, max_tiles)]

    rng = random.Random(seed) if sample else None
    for zarr_tile in tiles:
        npix = _npix_from_tile_name(zarr_tile.name)
        cat_tile = _catalog_tile_path(catalog_root, order, npix)
        if not cat_tile.is_file():
            cat_tile = next(catalog_root.rglob(f"Npix={npix}.parquet"), None)
        validate_tile_link(
            zarr_tile=zarr_tile,
            catalog_tile=cat_tile,
            npix=npix,
            norder=order,
            sid_col=sid_col,
            rep=rep,
            sample=sample,
            rng=rng,
        )

    return rep


try:
    import click

    from ..cli_utils import config_option, load_optional_config, require_output_root

    @click.command("dl-validate-catalog-spectra-link")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option("--survey", "survey_name", required=True, help="Survey name.")
    @click.option("--norder", type=int, default=None, help="HEALPix order (default: info JSON).")
    @click.option(
        "--link-id-col",
        default=None,
        help="Override catalog join column (default: resolve from catalog_info.json).",
    )
    @click.option(
        "--max-tiles",
        type=int,
        default=None,
        help="Validate only the first N Zarr tiles (sorted path order).",
    )
    @click.option(
        "--sample",
        type=int,
        default=None,
        help="Check at most N random catalog-linked rows per tile (smoke test).",
    )
    @click.option("--seed", type=int, default=0, show_default=True, help="RNG seed for --sample.")
    @click.option(
        "--strict",
        is_flag=True,
        help="Treat warnings (orphan Zarr, unpatched catalog) as errors.",
    )
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        survey_name: str,
        norder: int | None,
        link_id_col: str | None,
        max_tiles: int | None,
        sample: int | None,
        seed: int,
        strict: bool,
    ) -> None:
        """Verify catalog _spectrum_index matches spectrum Zarr _source_id tiles."""
        cfg = load_optional_config(config_path)
        lake = require_output_root(output_root, cfg, kind="spectra")

        rep = run_validation(
            lake,
            survey_name,
            norder=norder,
            link_id_col=link_id_col,
            max_tiles=max_tiles,
            sample=sample,
            seed=seed,
        )
        st = rep.stats
        for msg in rep.errors:
            click.echo(f"ERROR:   {msg}", err=True)
        for msg in rep.warnings:
            click.echo(f"WARNING: {msg}", err=True)

        click.echo(
            f"Tiles checked: {st.n_tiles_checked}  linked rows verified: {st.n_linked}  "
            f"stale index: {st.n_stale_index}  wrong id: {st.n_wrong_id}  "
            f"wrong healpix: {st.n_wrong_healpix}  orphan zarr: {st.n_orphan_zarr}  "
            f"unpatched catalog: {st.n_unpatched_catalog}  "
            f"missing catalog tile: {st.n_missing_catalog_tile}  "
            f"null source_id: {st.n_null_source_id}  "
            f"null source_id linked: {st.n_null_source_id_linked}"
        )

        if st.n_unpatched_catalog > 0:
            click.echo(
                f"Hint: run dl-rebuild-catalog-indices --survey {survey_name!r} "
                f"--kind spectrum (uses hats_order from catalog_info.json unless --norder is set)"
            )

        if rep.ok(strict=strict):
            if rep.warnings and not strict:
                click.echo(
                    f"OK (with {len(rep.warnings)} warning(s)): "
                    f"catalog ↔ spectra link for {survey_name!r}."
                )
            else:
                click.echo(f"OK: catalog ↔ spectra link for {survey_name!r}.")
            sys.exit(0)
        click.echo("Link validation failed.", err=True)
        sys.exit(1)

except ImportError:
    cli = None  # type: ignore[assignment]
