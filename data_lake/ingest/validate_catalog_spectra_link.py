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
        if strict:
            if self.warnings:
                return False
            st = self.stats
            if st.n_orphan_zarr > 0 or st.n_unpatched_catalog > 0:
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


def _resolve_norders(
    catalog_root: Path,
    spectra_root: Path,
    norder: int | None,
    rep: LinkValidationReport,
) -> tuple[int, int]:
    """Return ``(cat_order, spec_order)``; each defaults independently.

    Orders may differ when catalog and spectra were partitioned at different
    resolutions — this is now supported via ``_spectrum_npix`` on catalog rows.
    A warning (not an error) is emitted when orders differ.
    """
    cat_order: int | None = norder
    spec_order: int | None = norder

    cat_info = catalog_root / "catalog_info.json"
    if cat_info.is_file():
        cat_order = int(_load_json(cat_info).get("hats_order", cat_order or 5))
    spec_info = spectra_root / "spectrum_info.json"
    if spec_info.is_file():
        spec_order = int(_load_json(spec_info).get("hats_order", spec_order or 5))

    cat_order = cat_order or 5
    spec_order = spec_order or 5

    if cat_order != spec_order:
        rep.warnings.append(
            f"hats_order differs: catalog {cat_order} vs spectra {spec_order} "
            f"(OK when _spectrum_npix linkage is present)"
        )
    return cat_order, spec_order


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
    zarr_npix: int,
    cat_order: int,
    cat_tiles: list[Path],
    sid_col: str,
    rep: LinkValidationReport,
    has_npix_col: bool,
    sample: int | None = None,
    rng: random.Random | None = None,
    quiet: bool = False,
) -> None:
    """Validate one Zarr tile against all catalog tiles that reference it.

    When ``has_npix_col`` is True (new-format catalog), catalog rows are
    matched via ``_spectrum_npix == zarr_npix``.  In the legacy path
    (``has_npix_col`` is False), only the catalog tile with the same Npix
    is checked.
    """
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

    index_col = "_spectrum_index"
    npix_col = "_spectrum_npix"
    hp_col = f"_healpix_norder{cat_order}"

    # Track which Zarr row indices were correctly linked across all catalog tiles.
    zarr_idx_linked: set[int] = set()
    # For reverse check: which Zarr source_ids appear in any catalog tile.
    zarr_sid_in_catalog: set[int] = set()
    # Which Zarr source_ids appear with index=-1 (unpatched) in the catalog.
    zarr_sid_unpatched: set[int] = set()

    # In legacy mode (no _spectrum_npix), only check the same-Npix catalog tile.
    if not has_npix_col:
        candidate_tiles = [t for t in cat_tiles if f"Npix={zarr_npix}.parquet" in t.name]
        if not candidate_tiles:
            stats.n_missing_catalog_tile += 1
            rep.warnings.append(
                f"{zarr_tile.name}: no catalog tile Npix={zarr_npix} "
                f"(legacy mode, orders must match)"
            )
            stats.n_orphan_zarr += n_zarr
            return
    else:
        candidate_tiles = cat_tiles

    for cat_tile in candidate_tiles:
        if not cat_tile.is_file():
            continue

        read_cols = [sid_col, index_col, npix_col if has_npix_col else hp_col]
        try:
            cols = _read_catalog_tile_columns(cat_tile, read_cols)
        except KeyError as exc:
            rep.errors.append(f"{cat_tile}: {exc}")
            continue

        sid_raw = cols[sid_col]
        sid_iter = sid_raw.tolist() if isinstance(sid_raw, np.ndarray) else list(sid_raw)
        cat_sids: list[int | None] = [_optional_catalog_sid(x) for x in sid_iter]
        cat_idx = np.asarray(cols[index_col], dtype=np.int64)
        ref_col = npix_col if has_npix_col else hp_col
        cat_ref = np.asarray(cols[ref_col], dtype=np.int64)

        # Forward check: catalog rows claiming to link into this Zarr tile.
        if has_npix_col:
            candidate_rows = np.nonzero((cat_idx >= 0) & (cat_ref == zarr_npix))[0]
        else:
            candidate_rows = np.nonzero(cat_idx >= 0)[0]

        if sample is not None and sample > 0 and candidate_rows.size > sample:
            pick = rng or random.Random(0)
            candidate_rows = np.array(
                pick.sample(candidate_rows.tolist(), sample), dtype=np.int64,
            )

        for row_i in candidate_rows.tolist():
            idx = int(cat_idx[row_i])
            sid_opt = cat_sids[row_i]
            if sid_opt is None:
                stats.n_null_source_id_linked += 1
                rep.errors.append(
                    f"{cat_tile.name} row {row_i}: {index_col}={idx} but "
                    f"{sid_col} is null (cannot verify Zarr linkage)"
                )
                continue
            sid = sid_opt

            if not has_npix_col:
                hp = int(cat_ref[row_i])
                if hp != zarr_npix:
                    continue  # different healpix tile in legacy mode

            if idx < 0 or idx >= n_zarr:
                stats.n_stale_index += 1
                rep.errors.append(
                    f"{cat_tile.name} row {row_i}: {index_col}={idx} out of range "
                    f"(Zarr rows={n_zarr} in Npix={zarr_npix})"
                )
                continue

            zarr_sid = int(normalize_object_id(int(zarr_ids[idx])))
            if zarr_sid != sid:
                stats.n_wrong_id += 1
                rep.errors.append(
                    f"{cat_tile.name} row {row_i}: {sid_col}={sid} but "
                    f"Zarr {index_col}={idx} has _source_id={zarr_sid}"
                )
            else:
                stats.n_linked += 1
                zarr_idx_linked.add(idx)

        # Reverse: note catalog source_ids for orphan/unpatched detection.
        for row_i, sid_opt in enumerate(cat_sids):
            if sid_opt is None:
                stats.n_null_source_id += 1
                continue
            zarr_sid_in_catalog.add(sid_opt)
            if int(cat_idx[row_i]) < 0:
                zarr_sid_unpatched.add(sid_opt)

    # Reverse check: classify unlinked Zarr rows as "unpatched" or "orphan".
    for j, sid_raw in enumerate(zarr_ids.tolist()):
        if j in zarr_idx_linked:
            continue
        sid = int(normalize_object_id(int(sid_raw)))
        if sid in zarr_sid_unpatched:
            stats.n_unpatched_catalog += 1
            if not quiet:
                rep.warnings.append(
                    f"{zarr_tile.name} row {j}: _source_id={sid} in catalog but "
                    f"{index_col}=-1 (run dl-rebuild-catalog-indices)"
                )
        elif sid not in zarr_sid_in_catalog:
            stats.n_orphan_zarr += 1
            if not quiet:
                rep.warnings.append(
                    f"{zarr_tile.name} row {j}: _source_id={sid} not linked by any "
                    f"catalog row with {npix_col if has_npix_col else hp_col}={zarr_npix} "
                    f"(run dl-rebuild-catalog-indices)"
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
    quiet: bool = False,
) -> LinkValidationReport:
    """Cross-check catalog ``_spectrum_index`` / ``_spectrum_npix`` against Zarr tiles.

    Catalog and spectrum may use different HEALPix orders; linkage is validated
    via ``_spectrum_npix`` (new format) or by same-Npix pairing (legacy).
    """
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

    cat_order, _spec_order = _resolve_norders(catalog_root, spectra_root, norder, rep)

    all_parquet = sorted(catalog_root.rglob("Npix=*.parquet"))
    schema_names: list[str] | None = None
    if all_parquet:
        schema_names = pq.read_schema(str(all_parquet[0])).names
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

    has_npix_col = "_spectrum_npix" in (schema_names or [])

    tiles = list(_iter_spectrum_zarr_tiles(spectra_root))
    if not tiles:
        rep.warnings.append(f"No Npix=*.zarr tiles under {spectra_root}")
        return rep

    if max_tiles is not None:
        tiles = tiles[: max(0, max_tiles)]

    rng = random.Random(seed) if sample else None
    for zarr_tile in tiles:
        npix = _npix_from_tile_name(zarr_tile.name)
        validate_tile_link(
            zarr_tile=zarr_tile,
            zarr_npix=npix,
            cat_order=cat_order,
            cat_tiles=all_parquet,
            sid_col=sid_col,
            rep=rep,
            has_npix_col=has_npix_col,
            sample=sample,
            rng=rng,
            quiet=quiet,
        )

    return rep


def discover_surveys_for_spectra_link_validation(lake_root: Path | str) -> list[str]:
    """Survey names that have both a catalog tree and a spectrum store."""
    from data_lake.ingest.validate_cli import discover_catalog_spectra_link_surveys

    return discover_catalog_spectra_link_surveys(lake_root)


def _report_validation(
    rep: LinkValidationReport,
    survey_name: str,
    *,
    strict: bool,
    quiet: bool = False,
) -> bool:
    """Print one survey's report; return whether validation passed."""
    st = rep.stats
    for msg in rep.errors:
        click.echo(f"ERROR:   {msg}", err=True)
    if not quiet:
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
        n_warn = len(rep.warnings)
        if quiet and (st.n_orphan_zarr or st.n_unpatched_catalog):
            n_warn = st.n_orphan_zarr + st.n_unpatched_catalog
        if n_warn and not strict:
            click.echo(
                f"OK (with {n_warn} warning(s)): "
                f"catalog ↔ spectra link for {survey_name!r}."
            )
        else:
            click.echo(f"OK: catalog ↔ spectra link for {survey_name!r}.")
        return True

    click.echo(f"Link validation failed for {survey_name!r}.", err=True)
    return False


try:
    import click

    from ..cli_utils import (
        config_option,
        configure_cli_logging,
        load_optional_config,
        require_output_root,
    )
    from .validate_cli import (
        discover_catalog_spectra_link_surveys,
        echo_multi_survey_footer,
        echo_survey_banner,
        resolve_validation_survey_names,
        validation_survey_options,
    )

    @click.command("dl-validate-catalog-spectra-link")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @validation_survey_options
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
    @click.option(
        "-q",
        "--quiet",
        is_flag=True,
        default=False,
        help="Summary only: do not print per-row WARNING lines (use for surveys "
             "with many orphan spectra). Counts still appear in the stats line.",
    )
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        surveys: tuple[str, ...],
        validate_all: bool,
        norder: int | None,
        link_id_col: str | None,
        max_tiles: int | None,
        sample: int | None,
        seed: int,
        strict: bool,
        quiet: bool,
    ) -> None:
        """Verify catalog _spectrum_index matches spectrum Zarr _source_id tiles."""
        import logging as _logging

        configure_cli_logging(
            level=_logging.WARNING if quiet else _logging.INFO,
            quiet=quiet,
        )
        cfg = load_optional_config(config_path)
        lake = require_output_root(output_root, cfg, kind="spectra")

        names = resolve_validation_survey_names(
            surveys=surveys,
            validate_all=validate_all,
            discovered=discover_catalog_spectra_link_surveys(lake),
            empty_message=(
                "No surveys with both catalogs/ and spectra/ found under the lake root."
            ),
        )

        all_ok = True
        for i, survey_name in enumerate(names):
            echo_survey_banner(i, survey_name, total=len(names))

            rep = run_validation(
                lake,
                survey_name,
                norder=norder,
                link_id_col=link_id_col,
                max_tiles=max_tiles,
                sample=sample,
                seed=seed,
                quiet=quiet,
            )
            if not _report_validation(rep, survey_name, strict=strict, quiet=quiet):
                all_ok = False

        echo_multi_survey_footer(
            all_ok=all_ok,
            n_surveys=len(names),
            ok_message=f"OK: catalog ↔ spectra link for all {len(names)} survey(s).",
        )

        sys.exit(0 if all_ok else 1)

except ImportError:
    cli = None  # type: ignore[assignment]
