"""
sdss_specobj_lookup – resolve (survey, plate, mjd, fiber) → SPECOBJID for spPlate ingest.

spPlate FITS files do not carry SPECOBJID; ingest joins against a sidecar Parquet/CSV
or scans an ingested lake catalog under ``catalogs/<survey>/``.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import normalize_object_id

log = logging.getLogger(__name__)

_SURVEY_COL_ALIASES = ("SURVEY", "survey", "SURVEY_NAME", "survey_name")
_PLATE_COL_ALIASES = ("PLATE", "plate", "PLATEID", "plateid")
_MJD_COL_ALIASES = ("MJD", "mjd")
_FIBER_COL_ALIASES = ("FIBERID", "fiberid", "FIBER_ID", "fiber", "FIBER")
_SPECOBJID_COL_ALIASES = ("SPECOBJID", "specobjid", "SPEC_OBJID", "spec_objid")


def _resolve_column(names: Iterable[str], aliases: tuple[str, ...]) -> str | None:
    by_lower = {n.lower(): n for n in names}
    for cand in aliases:
        key = by_lower.get(cand.lower())
        if key is not None:
            return key
    return None


def load_lookup_table(path: Path | str) -> pa.Table:
    """Load a specObj lookup sidecar (Parquet or CSV); cached per resolved path."""
    resolved = str(Path(path).resolve())
    return _load_lookup_table_cached(resolved)


@lru_cache(maxsize=8)
def _load_lookup_table_cached(resolved_path: str) -> pa.Table:
    p = Path(resolved_path)
    if not p.is_file():
        raise FileNotFoundError(f"specObj lookup file not found: {p}")
    if p.suffix.lower() in (".parquet", ".pq"):
        return pq.read_table(p)
    if p.suffix.lower() in (".csv", ".tsv"):
        return pa.csv.read_csv(p)
    raise ValueError(
        f"Unsupported specObj lookup format {p.suffix!r} ({p}); use .parquet or .csv"
    )


def _filter_sidecar_table(
    table: pa.Table,
    *,
    survey_name: str,
    plate: int,
    mjd: int,
    lookup_survey: str | None,
) -> pa.Table:
    names = table.schema.names
    survey_col = _resolve_column(names, _SURVEY_COL_ALIASES)
    if survey_col is None:
        if lookup_survey is None:
            raise ValueError(
                "specObj lookup table has no survey column "
                f"(expected one of {_SURVEY_COL_ALIASES}). "
                "Add a SURVEY column to the sidecar or pass --specobj-lookup-survey to ingest."
            )
        if lookup_survey != survey_name:
            raise ValueError(
                f"lookup_survey={lookup_survey!r} does not match ingest survey_name={survey_name!r}"
            )
    else:
        survey_vals = table.column(survey_col).to_pylist()
        if not all(str(v) == survey_name for v in survey_vals):
            mask = pc.equal(
                table.column(survey_col).cast(pa.string()),
                pa.scalar(survey_name, type=pa.string()),
            )
            table = table.filter(mask)

    plate_col = _resolve_column(names, _PLATE_COL_ALIASES)
    mjd_col = _resolve_column(names, _MJD_COL_ALIASES)
    fiber_col = _resolve_column(names, _FIBER_COL_ALIASES)
    sid_col = _resolve_column(names, _SPECOBJID_COL_ALIASES)
    missing = [
        name
        for name, col in (
            ("PLATE", plate_col),
            ("MJD", mjd_col),
            ("FIBERID", fiber_col),
            ("SPECOBJID", sid_col),
        )
        if col is None
    ]
    if missing:
        raise ValueError(
            f"specObj lookup missing required column(s) {missing}; "
            f"available: {names[:30]}{'…' if len(names) > 30 else ''}"
        )

    plate_mask = pc.equal(table.column(plate_col).cast(pa.int64()), pa.scalar(int(plate), pa.int64()))
    mjd_mask = pc.equal(table.column(mjd_col).cast(pa.int64()), pa.scalar(int(mjd), pa.int64()))
    return table.filter(pc.and_(plate_mask, mjd_mask))


def _table_to_fiber_map(table: pa.Table) -> dict[int, int]:
    names = table.schema.names
    fiber_col = _resolve_column(names, _FIBER_COL_ALIASES)
    sid_col = _resolve_column(names, _SPECOBJID_COL_ALIASES)
    assert fiber_col is not None and sid_col is not None

    fibers = table.column(fiber_col).to_pylist()
    specobjids = table.column(sid_col).to_pylist()
    out: dict[int, int] = {}
    for fiber, sid in zip(fibers, specobjids):
        if fiber is None or sid is None:
            continue
        fid = int(fiber)
        if fid in out:
            log.warning(
                "Duplicate FIBERID %d in lookup (keeping first SPECOBJID)", fid,
            )
            continue
        out[fid] = normalize_object_id(sid)
    return out


def _build_from_sidecar(
    survey_name: str,
    plate: int,
    mjd: int,
    *,
    lookup_path: Path | str,
    lookup_survey: str | None,
) -> dict[int, int]:
    table = load_lookup_table(lookup_path)
    filtered = _filter_sidecar_table(
        table,
        survey_name=survey_name,
        plate=plate,
        mjd=mjd,
        lookup_survey=lookup_survey,
    )
    return _table_to_fiber_map(filtered)


def _build_from_catalog(
    survey_name: str,
    plate: int,
    mjd: int,
    *,
    catalog_root: Path | str,
) -> dict[int, int]:
    root = Path(catalog_root) / "catalogs" / survey_name
    if not root.is_dir():
        raise FileNotFoundError(f"Catalog not found for survey {survey_name!r}: {root}")

    plate_col: str | None = None
    mjd_col: str | None = None
    fiber_col: str | None = None
    sid_col: str | None = None
    out: dict[int, int] = {}

    for tile_path in sorted(root.rglob("Npix=*.parquet")):
        schema = pq.read_schema(str(tile_path))
        names = schema.names
        if plate_col is None:
            plate_col = _resolve_column(names, _PLATE_COL_ALIASES)
            mjd_col = _resolve_column(names, _MJD_COL_ALIASES)
            fiber_col = _resolve_column(names, _FIBER_COL_ALIASES)
            sid_col = _resolve_column(names, _SPECOBJID_COL_ALIASES)
            if not all((plate_col, mjd_col, fiber_col, sid_col)):
                raise ValueError(
                    f"Catalog {root} missing plate/mjd/fiber/SPECOBJID columns; "
                    f"found schema sample: {names[:25]}"
                )
        cols = [plate_col, mjd_col, fiber_col, sid_col]  # type: ignore[list-item]
        chunk = pq.read_table(str(tile_path), columns=cols)
        plates = np.asarray(chunk.column(plate_col).to_numpy(zero_copy_only=False))
        mjds = np.asarray(chunk.column(mjd_col).to_numpy(zero_copy_only=False))
        sel = (plates == int(plate)) & (mjds == int(mjd))
        if not np.any(sel):
            continue
        sub = chunk.filter(pa.array(sel))
        partial = _table_to_fiber_map(sub)
        for fid, sid in partial.items():
            if fid in out:
                log.warning(
                    "Duplicate FIBERID %d across catalog tiles (keeping first)", fid,
                )
                continue
            out[fid] = sid

    return out


def build_fiber_to_specobjid_map(
    survey_name: str,
    plate: int,
    mjd: int,
    *,
    lookup_path: Path | str | None = None,
    catalog_root: Path | str | None = None,
    lookup_survey: str | None = None,
) -> dict[int, int]:
    """
    Return ``{fiber_id: specobjid}`` for one plate–MJD within a survey.

    Provide exactly one of ``lookup_path`` (sidecar) or ``catalog_root`` (lake root).
    ``survey_name`` is always required and scopes sidecar rows and catalog directory.
    """
    if lookup_path is None and catalog_root is None:
        raise ValueError(
            "spPlate ingest requires specObj lookup: pass lookup_path= or catalog_root="
        )
    if lookup_path is not None and catalog_root is not None:
        raise ValueError("Pass only one of lookup_path= or catalog_root=, not both")

    if lookup_path is not None:
        return _build_from_sidecar(
            survey_name,
            int(plate),
            int(mjd),
            lookup_path=lookup_path,
            lookup_survey=lookup_survey,
        )
    return _build_from_catalog(
        survey_name,
        int(plate),
        int(mjd),
        catalog_root=catalog_root,  # type: ignore[arg-type]
    )


def spplate_plate_mjd_from_hdul(hdul, path: Path | None = None) -> tuple[int, int]:
    """Read plate and MJD from spPlate primary header or ``spPlate-PLATE-MJD`` filename."""
    phdr = hdul[0].header
    plate: int | None = None
    mjd: int | None = None
    for key in ("PLATEID", "PLATE"):
        if key in phdr:
            plate = int(phdr[key])
            break
    if "MJD" in phdr:
        mjd = int(phdr["MJD"])
    if path is not None and (plate is None or mjd is None):
        stem = path.stem
        if stem.lower().startswith("spplate-"):
            parts = stem.split("-")
            if len(parts) >= 3:
                try:
                    if plate is None:
                        plate = int(parts[1])
                    if mjd is None:
                        mjd = int(parts[2])
                except ValueError:
                    pass
    if plate is None or mjd is None:
        raise KeyError(
            "Could not resolve plate and MJD for spPlate "
            f"(header PLATEID/PLATE/MJD; filename spPlate-PLATE-MJD). path={path!r}"
        )
    return plate, mjd
