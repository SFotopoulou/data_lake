"""
sdss_specobj_lookup – resolve (survey, plate, mjd, fiber) → SPECOBJID for spPlate ingest.

spPlate FITS files do not carry SPECOBJID; ingest joins against a sidecar Parquet/CSV
or scans an ingested lake catalog under ``catalogs/<survey>/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal

SpecObjIdLayout = Literal["auto", "dr7", "dr8plus"]

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_lake.ingest.fits_to_parquet import normalize_object_id

log = logging.getLogger(__name__)

_SURVEY_COL_ALIASES = ("SURVEY", "survey", "SURVEY_NAME", "survey_name")
_PLATE_COL_ALIASES = ("PLATE", "plate", "PLATEID", "plateid")
_MJD_COL_ALIASES = ("MJD", "mjd")
_FIBER_COL_ALIASES = (
    "FIBERID",
    "fiberid",
    "FIBER_ID",
    "fiberID",
    "fiber",
    "FIBER",
)
# Do not alias ``objid`` / ``OBJID`` here — that is the photometric ID, not specObjID.
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


def _column_as_int64_numpy(column: pa.ChunkedArray) -> np.ndarray:
    """Cast a catalog column to int64 for plate/mjd/fiber comparisons."""
    return np.asarray(
        column.cast(pa.int64()).to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )


def _warn_catalog_lookup_miss(
    catalog_dir: Path,
    survey_name: str,
    plate: int,
    mjd: int,
    *,
    plate_col: str,
    mjd_col: str,
    fiber_col: str,
    sid_col: str,
    mjds_for_plate: list[int] | None,
    n_plate_rows: int,
) -> None:
    if n_plate_rows == 0:
        log.warning(
            "Catalog %s has no rows with %s=%d (checked all Parquet tiles). "
            "spPlate ingest needs a **specObj** catalog with plate/mjd/fiber columns — "
            "not a photo-only table. For BOSS plates without specObj rows, use "
            "--specobj-lookup-from-plate.",
            catalog_dir,
            plate_col,
            plate,
        )
        return
    if mjds_for_plate and int(mjd) not in mjds_for_plate:
        sample = mjds_for_plate[:12]
        extra = f" … (+{len(mjds_for_plate) - 12} more)" if len(mjds_for_plate) > 12 else ""
        mjd_list = ", ".join(str(x) for x in sample)
        log.warning(
            "Catalog %s has %d row(s) with %s=%d but none with %s=%d. "
            "MJDs present for this plate include: %s%s. "
            "Check spPlate header MJD vs catalog, or use --specobj-lookup-from-plate.",
            catalog_dir,
            n_plate_rows,
            plate_col,
            plate,
            mjd_col,
            mjd,
            mjd_list,
            extra,
        )
        return
    log.warning(
        "Catalog %s matched plate=%d mjd=%d using columns "
        "%s, %s, %s, %s but produced an empty fiber map "
        "(null IDs or no overlapping FIBERID values).",
        catalog_dir,
        plate,
        mjd,
        plate_col,
        mjd_col,
        fiber_col,
        sid_col,
    )


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


@dataclass
class CatalogLookupProbe:
    """Diagnostics from scanning ``catalogs/<survey>/`` for one plate–MJD."""

    catalog_dir: Path
    tile_count: int = 0
    sample_schema: list[str] = field(default_factory=list)
    plate_col: str | None = None
    mjd_col: str | None = None
    fiber_col: str | None = None
    sid_col: str | None = None
    n_plate_rows: int = 0
    n_plate_mjd_rows: int = 0
    mjds_for_plate: list[int] = field(default_factory=list)
    fiber_map: dict[int, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.fiber_map)

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_dir": str(self.catalog_dir),
            "tile_count": self.tile_count,
            "sample_schema": self.sample_schema,
            "columns": {
                "plate": self.plate_col,
                "mjd": self.mjd_col,
                "fiber": self.fiber_col,
                "specobjid": self.sid_col,
            },
            "n_plate_rows": self.n_plate_rows,
            "n_plate_mjd_rows": self.n_plate_mjd_rows,
            "mjds_for_plate": self.mjds_for_plate,
            "fiber_map_size": len(self.fiber_map),
            "error": self.error,
        }


def probe_catalog_specobj_lookup(
    survey_name: str,
    plate: int,
    mjd: int,
    *,
    catalog_root: Path | str,
) -> CatalogLookupProbe:
    """Scan lake catalog tiles and return join stats (same logic as ingest)."""
    root = Path(catalog_root) / "catalogs" / survey_name
    probe = CatalogLookupProbe(catalog_dir=root)
    if not root.is_dir():
        probe.error = f"Catalog directory not found: {root}"
        return probe

    tile_paths = sorted(root.rglob("Npix=*.parquet"))
    probe.tile_count = len(tile_paths)
    if not tile_paths:
        probe.error = f"No Parquet tiles under {root}"
        return probe

    plate_col: str | None = None
    mjd_col: str | None = None
    fiber_col: str | None = None
    sid_col: str | None = None
    mjds_for_plate: set[int] = set()

    try:
        for tile_path in tile_paths:
            schema = pq.read_schema(str(tile_path))
            names = schema.names
            if not probe.sample_schema:
                probe.sample_schema = list(names[:40])
            if plate_col is None:
                plate_col = _resolve_column(names, _PLATE_COL_ALIASES)
                mjd_col = _resolve_column(names, _MJD_COL_ALIASES)
                fiber_col = _resolve_column(names, _FIBER_COL_ALIASES)
                sid_col = _resolve_column(names, _SPECOBJID_COL_ALIASES)
                probe.plate_col = plate_col
                probe.mjd_col = mjd_col
                probe.fiber_col = fiber_col
                probe.sid_col = sid_col
                if not all((plate_col, mjd_col, fiber_col, sid_col)):
                    probe.error = (
                        "Missing plate/mjd/fiber/SPECOBJID columns "
                        "(photometric objid is not valid); "
                        f"sample schema: {names[:30]}"
                    )
                    return probe

            cols = [plate_col, mjd_col, fiber_col, sid_col]  # type: ignore[list-item]
            chunk = pq.read_table(str(tile_path), columns=cols)
            plates = _column_as_int64_numpy(chunk.column(plate_col))
            mjds = _column_as_int64_numpy(chunk.column(mjd_col))
            plate_sel = plates == int(plate)
            if not np.any(plate_sel):
                continue
            probe.n_plate_rows += int(np.count_nonzero(plate_sel))
            mjds_for_plate.update(int(x) for x in mjds[plate_sel].tolist())
            sel = plate_sel & (mjds == int(mjd))
            if not np.any(sel):
                continue
            probe.n_plate_mjd_rows += int(np.count_nonzero(sel))
            sub = chunk.filter(pa.array(sel))
            partial = _table_to_fiber_map(sub)
            for fid, sid in partial.items():
                if fid in probe.fiber_map:
                    log.warning(
                        "Duplicate FIBERID %d across catalog tiles (keeping first)", fid,
                    )
                    continue
                probe.fiber_map[fid] = sid
    except Exception as exc:
        probe.error = str(exc)
        return probe

    probe.mjds_for_plate = sorted(mjds_for_plate)
    if not probe.fiber_map and probe.error is None:
        assert plate_col and mjd_col and fiber_col and sid_col
        _warn_catalog_lookup_miss(
            root,
            survey_name,
            int(plate),
            int(mjd),
            plate_col=plate_col,
            mjd_col=mjd_col,
            fiber_col=fiber_col,
            sid_col=sid_col,
            mjds_for_plate=probe.mjds_for_plate,
            n_plate_rows=probe.n_plate_rows,
        )
    return probe


def _build_from_catalog(
    survey_name: str,
    plate: int,
    mjd: int,
    *,
    catalog_root: Path | str,
) -> dict[int, int]:
    probe = probe_catalog_specobj_lookup(
        survey_name, int(plate), int(mjd), catalog_root=catalog_root,
    )
    if probe.error and not probe.fiber_map:
        if probe.tile_count == 0 or not probe.catalog_dir.is_dir():
            raise FileNotFoundError(probe.error)
        raise ValueError(probe.error)
    if probe.fiber_map:
        log.info(
            "spPlate catalog lookup: plate=%d mjd=%d → %d fiber ID(s)",
            plate,
            mjd,
            len(probe.fiber_map),
        )
    return probe.fiber_map


def build_fiber_to_specobjid_map(
    survey_name: str,
    plate: int,
    mjd: int,
    *,
    lookup_path: Path | str | None = None,
    catalog_root: Path | str | None = None,
    lookup_survey: str | None = None,
    spplate_hdul=None,
    lookup_from_plate: bool = False,
    specobj_id_layout: SpecObjIdLayout = "auto",
) -> dict[int, int]:
    """
    Return ``{fiber_id: specobjid}`` for one plate–MJD within a survey.

    Provide exactly one of ``lookup_path`` (sidecar) or ``catalog_root`` (lake root).
    ``survey_name`` is always required and scopes sidecar rows and catalog directory.
    """
    if lookup_from_plate:
        if spplate_hdul is None:
            raise ValueError("lookup_from_plate=True requires spplate_hdul=")
        return build_fiber_to_specobjid_from_spplate(
            spplate_hdul,
            survey_name=survey_name,
            specobj_id_layout=specobj_id_layout,
        )

    if lookup_path is None and catalog_root is None:
        raise ValueError(
            "spPlate ingest requires specObj lookup: pass lookup_path=, catalog_root=, "
            "or lookup_from_plate=True"
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


def run2d_from_spplate_header(phdr) -> int | str | None:
    """Return ``RUN2D`` for specObjID encoding (never ``VERS2D`` / ``VERSCOMB``)."""
    if "RUN2D" not in phdr:
        return None
    run2d = phdr["RUN2D"]
    for key in ("VERS2D", "VERSCOMB"):
        if key not in phdr:
            continue
        other = phdr[key]
        if str(other).strip() != str(run2d).strip():
            log.warning(
                "spPlate header %s=%r ignored for specObjID; using RUN2D=%r only",
                key,
                other,
                run2d,
            )
    return run2d


def infer_specobjid_layout(
    phdr,
    survey_name: str | None = None,
    *,
    layout: SpecObjIdLayout = "auto",
) -> Literal["dr7", "dr8plus"]:
    """Choose DR7 vs DR8+ specObjID packing for spPlate synthesis.

    DR7 (SkyServer DR7): plate/mjd/fiber in low bits — see
    https://skyserver.sdss.org/dr7/en/help/docs/algorithm.asp?key=objID

    DR8+ (SDSS-III/IV, BOSS, eBOSS, DR17): plate@50, fiber@38, (mjd-50000)@24,
    run2d@10 — matches SciServer ``fSDSSfromSpecID``.
    """
    if layout in ("dr7", "dr8plus"):
        return layout

    survey_key = (survey_name or "").lower().replace("-", "").replace("_", "")
    if "dr7" in survey_key or survey_key.endswith("sdss7"):
        return "dr7"

    run2d = run2d_from_spplate_header(phdr)

    if isinstance(run2d, str) and run2d.strip().lower().startswith("v"):
        return "dr8plus"
    if run2d is not None:
        # Integer RUN2D (26, 103, 104, …) still uses the DR8+ 64-bit layout.
        return "dr8plus"

    mjd = int(phdr["MJD"]) if "MJD" in phdr else 0
    plate = None
    for key in ("PLATEID", "PLATE"):
        if key in phdr:
            plate = int(phdr[key])
            break
    if plate is not None and plate < 4000 and mjd < 53000:
        log.info(
            "spPlate specObjID: no RUN2D in header; using DR7 layout "
            "(plate=%s mjd=%s)",
            plate,
            mjd,
        )
        return "dr7"

    raise KeyError(
        "spPlate header missing RUN2D; cannot infer DR8+ specObjID. "
        "Use --specobj-id-layout dr7 for SDSS-II plates, or pass "
        "--specobj-lookup / --specobj-lookup-from-catalog."
    )


def sdss_specobjid_dr7_from_plate_fiber(
    plate: int,
    fiber: int,
    mjd: int,
    *,
    object_type: int = 0,
    line: int = 0,
) -> int:
    """DR7 CAS specObjID (plate @0, mjd @16, fiber @32, type @42, line @48)."""
    raw = (
        (int(plate) & 0xFFFF)
        | ((int(mjd) & 0xFFFF) << 16)
        | ((int(fiber) & 0x3FF) << 32)
        | ((int(object_type) & 0x3F) << 42)
        | ((int(line) & 0xFFFF) << 48)
    )
    return normalize_object_id(raw)


def encode_sdss_run2d(run2d: int | str) -> int:
    """Encode ``RUN2D`` into the 14-bit specObjID field.

    Accepts SDSS-II integer codes (e.g. ``26``), numeric strings (``"26"``), or
    SDSS-III/IV version strings (``v5_13_2``).
    """
    if isinstance(run2d, str):
        text = run2d.strip()
        if not text:
            raise ValueError("RUN2D is empty")
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        parts = text.lstrip("vV").split("_")
        if len(parts) != 3:
            raise ValueError(
                f"RUN2D must be an integer code or vN_M_P string, got {run2d!r}"
            )
        major, minor, patch = (int(parts[0]), int(parts[1]), int(parts[2]))
        return (major - 5) * 10000 + minor * 100 + patch
    return int(run2d)


def sdss_specobjid_dr8plus_from_plate_fiber(
    plate: int,
    fiber: int,
    mjd: int,
    run2d: int | str,
    *,
    line: int = 0,
) -> int:
    """DR8+ CAS specObjID from plate, fiber, MJD, and RUN2D (BOSS/DR17 layout)."""
    run2d_val = encode_sdss_run2d(run2d)
    mjd_val = int(mjd) - 50000
    if mjd_val < 0:
        raise ValueError(f"MJD must be >= 50000 for specObjID encoding, got {mjd}")
    raw = (
        (int(plate) << 50)
        | (int(fiber) << 38)
        | (mjd_val << 24)
        | (run2d_val << 10)
        | int(line)
    )
    return normalize_object_id(raw)


# Backward-compatible alias
sdss_specobjid_from_plate_fiber = sdss_specobjid_dr8plus_from_plate_fiber


def build_fiber_to_specobjid_from_spplate(
    hdul,
    path: Path | None = None,
    *,
    fiber_ids: Iterable[int] | None = None,
    survey_name: str | None = None,
    specobj_id_layout: SpecObjIdLayout = "auto",
) -> dict[int, int]:
    """Build ``{FIBERID: specObjID}`` from spPlate header + PLUGMAP (no specObj sidecar)."""
    phdr = hdul[0].header
    plate, mjd = spplate_plate_mjd_from_hdul(hdul, path)
    layout = infer_specobjid_layout(phdr, survey_name, layout=specobj_id_layout)
    log.info("spPlate specObjID synthesis: layout=%s survey=%r", layout, survey_name)

    run2d = None
    if layout == "dr8plus":
        run2d = run2d_from_spplate_header(phdr)
        if run2d is None:
            raise KeyError(
                "spPlate header missing RUN2D (required for DR8+ specObjID synthesis; "
                "VERS2D/VERSCOMB are not used)"
            )

    if fiber_ids is None:
        from data_lake.ingest.fits_to_spectra_zarr import _spplate_fiber_table_hdu

        ftable = _spplate_fiber_table_hdu(hdul)
        if ftable is None:
            raise ValueError("spPlate: no PLUGMAP / FIBERID BINTABLE for specObjID synthesis")
        fiber_ids = [int(x) for x in _fits_fiber_column(ftable.data)]

    out: dict[int, int] = {}
    for fiber in fiber_ids:
        fid = int(fiber)
        if fid in out:
            continue
        if layout == "dr7":
            out[fid] = sdss_specobjid_dr7_from_plate_fiber(plate, fid, mjd)
        else:
            out[fid] = sdss_specobjid_dr8plus_from_plate_fiber(plate, fid, mjd, run2d)
    return out


def _fits_fiber_column(fdata: np.ndarray) -> np.ndarray:
    names = fdata.dtype.names or ()
    by_lower = {n.lower(): n for n in names}
    for cand in ("fiberid", "fiber_id", "fiber"):
        key = by_lower.get(cand)
        if key is not None:
            return np.asarray(fdata[key])
    raise KeyError(f"No FIBERID column in plugmap; columns: {list(names)}")


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
