"""
Debug spPlate → specObjID lookup (sidecar, lake catalog, or plate synthesis).

``dl-debug-specobj-lookup`` prints FITS header facts, catalog column coverage,
and how many plugmap fibers resolve under each ingest lookup mode.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import click
from astropy.io import fits

from data_lake.cli_utils import config_option, load_optional_config, require_output_root
from data_lake.ingest.sdss_specobj_lookup import (
    SpecObjIdLayout,
    build_fiber_to_source_id_from_triplet,
    build_fiber_to_specobjid_from_spplate,
    build_fiber_to_specobjid_map,
    infer_specobjid_layout,
    probe_catalog_specobj_lookup,
    spplate_plate_mjd_from_hdul,
)

log = logging.getLogger(__name__)

_HEADER_KEYS = (
    "PLATE",
    "PLATEID",
    "MJD",
    "RUN2D",
    "VERS2D",
    "VERSCOMB",
    "NAXIS1",
    "NAXIS2",
    "OBJECT",
    "TELESCOP",
)


@dataclass
class LookupModeResult:
    name: str
    fiber_map: dict[int, int] = field(default_factory=dict)
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.fiber_map)


@dataclass
class SpecObjLookupDebugReport:
    spplate_path: Path
    survey_name: str
    plate: int
    mjd: int
    header: dict[str, Any]
    inferred_layout: str | None = None
    layout_error: str | None = None
    plugmap_fibers: list[int] = field(default_factory=list)
    plugmap_audit: dict[str, Any] = field(default_factory=dict)
    modes: list[LookupModeResult] = field(default_factory=list)

    def mode(self, name: str) -> LookupModeResult | None:
        for m in self.modes:
            if m.name == name:
                return m
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spplate": str(self.spplate_path),
            "survey": self.survey_name,
            "plate": self.plate,
            "mjd": self.mjd,
            "header": self.header,
            "inferred_layout": self.inferred_layout,
            "layout_error": self.layout_error,
            "plugmap_fiber_count": len(self.plugmap_fibers),
            "plugmap_audit": self.plugmap_audit,
            "modes": [
                {
                    "name": m.name,
                    "fiber_map_size": m.size,
                    "error": m.error,
                    **m.extra,
                }
                for m in self.modes
            ],
        }


_SPECOBJID_COLUMN_RE = re.compile(r"spec.*obj.*id|specobjid", re.I)
_OBJID_COLUMN_RE = re.compile(r"^obj.?id$|^objid$", re.I)
_PHOTO_ID_PARTS = frozenset({"RUN", "RERUN", "CAMCOL", "FIELD", "ID"})


def inspect_spplate_plugmap(hdul) -> dict[str, Any]:
    """Summarize PLUGMAP (HDU 5) columns and flag ID-like names.

    Official spPlate model:
    https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/PLATE4/spPlate.html
  """
    from data_lake.ingest.fits_to_spectra_zarr import _spplate_fiber_table_hdu

    audit: dict[str, Any] = {
        "hdu_name": None,
        "n_rows": 0,
        "columns": [],
        "specobjid_like": [],
        "objid_like": [],
        "photo_id_parts_present": [],
        "notes": [],
    }
    ftable = _spplate_fiber_table_hdu(hdul)
    if ftable is None:
        audit["notes"].append("No BINTABLE with FIBERID found (expected PLUGMAP).")
        return audit

    audit["hdu_name"] = (ftable.name or "").strip() or None
    names = list(ftable.data.dtype.names or ())
    audit["n_rows"] = len(ftable.data)
    row_i = min(500, max(0, len(ftable.data) - 1))

    for name in names:
        col = ftable.data[name]
        sample = col[row_i]
        if hasattr(sample, "tolist"):
            sample_repr = sample.tolist()
        else:
            sample_repr = sample
        entry: dict[str, Any] = {
            "name": name,
            "dtype": str(col.dtype),
            "sample": sample_repr,
        }
        upper = name.upper()
        if _SPECOBJID_COLUMN_RE.search(name):
            audit["specobjid_like"].append(name)
            entry["role"] = "possible_specobjid"
        elif _OBJID_COLUMN_RE.search(name) or upper == "OBJID":
            audit["objid_like"].append(name)
            entry["role"] = "photometric_objid_not_specobjid"
        elif upper == "FIBERID":
            entry["role"] = "fiber_join_key"
        elif upper in _PHOTO_ID_PARTS:
            audit["photo_id_parts_present"].append(name)
            entry["role"] = "photo_objid_component"
        audit["columns"].append(entry)

    if audit["specobjid_like"]:
        audit["notes"].append(
            "Column(s) matching specObjID name pattern found — verify against CAS "
            "before using as --link-id-col."
        )
    else:
        audit["notes"].append(
            "No SPECOBJID column in PLUGMAP (expected for spPlate). "
            "Use header PLATE/MJD + FIBERID + RUN2D synthesis or a SpecObj catalog."
        )

    if audit["objid_like"]:
        audit["notes"].append(
            "OBJID is the 5-part imaging ID (run/rerun/camcol/field/id), not specObjID."
        )

    parts = {n.upper() for n in audit["photo_id_parts_present"]}
    if parts >= {"RUN", "CAMCOL", "FIELD", "ID"} and "OBJID" in {c.upper() for c in audit["objid_like"]}:
        try:
            objid = ftable.data["OBJID"][row_i]
            run = int(ftable.data["RUN"][row_i]) if "RUN" in names else None
            field = int(ftable.data["FIELD"][row_i]) if "FIELD" in names else None
            if run is not None and int(objid[0]) == run:
                audit["notes"].append(
                    "OBJID[0] matches RUN — confirms imaging objID, not spectroscopic specObjID."
                )
        except Exception:
            pass

    return audit


def _plugmap_fiber_ids(hdul) -> list[int]:
    from data_lake.ingest.fits_to_spectra_zarr import _spplate_fiber_table_hdu
    from data_lake.ingest.sdss_specobj_lookup import _fits_fiber_column

    ftable = _spplate_fiber_table_hdu(hdul)
    if ftable is None:
        return []
    return sorted({int(x) for x in _fits_fiber_column(ftable.data)})


def _fiber_overlap(
    plugmap: list[int],
    fiber_map: dict[int, int],
) -> dict[str, Any]:
    plug_set = set(plugmap)
    mapped = set(fiber_map)
    matched = plug_set & mapped
    return {
        "plugmap_count": len(plug_set),
        "mapped_count": len(mapped),
        "matched_count": len(matched),
        "missing_from_lookup": sorted(plug_set - mapped)[:20],
        "extra_in_lookup": sorted(mapped - plug_set)[:20],
    }


def _sample_pairs(fiber_map: dict[int, int], n: int) -> list[dict[str, int]]:
    items = sorted(fiber_map.items())[:n]
    return [{"fiber": fid, "specobjid": sid} for fid, sid in items]


def debug_specobj_lookup(
    spplate_path: Path,
    survey_name: str,
    *,
    catalog_root: Path | str | None = None,
    lookup_path: Path | str | None = None,
    lookup_survey: str | None = None,
    try_triplet_hash: bool = False,
    try_from_plate: bool = True,
    try_from_catalog: bool = True,
    try_from_sidecar: bool = True,
    specobj_id_layout: SpecObjIdLayout = "auto",
    catalog_id_col: str | None = None,
) -> SpecObjLookupDebugReport:
    """Build a diagnostic report for one spPlate file."""
    path = Path(spplate_path)
    from data_lake.io.fits_read import open_fits

    with open_fits(path) as hdul:
        plate, mjd = spplate_plate_mjd_from_hdul(hdul, path)
        phdr = hdul[0].header
        header = {k: phdr[k] for k in _HEADER_KEYS if k in phdr}
        plugmap = _plugmap_fiber_ids(hdul)
        plugmap_audit = inspect_spplate_plugmap(hdul)

        report = SpecObjLookupDebugReport(
            spplate_path=path,
            survey_name=survey_name,
            plate=plate,
            mjd=mjd,
            header=header,
            plugmap_fibers=plugmap,
            plugmap_audit=plugmap_audit,
        )

        try:
            report.inferred_layout = infer_specobjid_layout(
                phdr, survey_name, layout=specobj_id_layout,
            )
        except Exception as exc:
            report.layout_error = str(exc)

        if try_triplet_hash:
            mode = LookupModeResult(name="triplet_hash")
            try:
                mode.fiber_map = build_fiber_to_source_id_from_triplet(
                    plate, mjd, plugmap or [],
                )
                mode.extra["label_format"] = f"{plate}|{mjd}|<fiberid>"
                mode.extra["note"] = (
                    "IDs match catalog only when ingested with "
                    "--link-id-col PLATE,MJD,FIBERID"
                )
            except Exception as exc:
                mode.error = str(exc)
            mode.extra["overlap"] = _fiber_overlap(plugmap, mode.fiber_map)
            report.modes.append(mode)

        if try_from_plate:
            mode = LookupModeResult(name="from_plate")
            try:
                mode.fiber_map = build_fiber_to_specobjid_from_spplate(
                    hdul,
                    path,
                    fiber_ids=plugmap or None,
                    survey_name=survey_name,
                    specobj_id_layout=specobj_id_layout,
                )
                mode.extra["layout"] = report.inferred_layout or specobj_id_layout
            except Exception as exc:
                mode.error = str(exc)
            mode.extra["overlap"] = _fiber_overlap(plugmap, mode.fiber_map)
            report.modes.append(mode)

        if try_from_sidecar and lookup_path is not None:
            mode = LookupModeResult(name="sidecar")
            try:
                mode.fiber_map = build_fiber_to_specobjid_map(
                    survey_name,
                    plate,
                    mjd,
                    lookup_path=lookup_path,
                    lookup_survey=lookup_survey,
                    catalog_id_col=catalog_id_col,
                )
            except Exception as exc:
                mode.error = str(exc)
            mode.extra["lookup_path"] = str(Path(lookup_path).resolve())
            mode.extra["overlap"] = _fiber_overlap(plugmap, mode.fiber_map)
            report.modes.append(mode)

        if try_from_catalog and catalog_root is not None:
            mode = LookupModeResult(name="catalog")
            probe = probe_catalog_specobj_lookup(
                survey_name,
                plate,
                mjd,
                catalog_root=catalog_root,
                catalog_id_col=catalog_id_col,
            )
            mode.fiber_map = dict(probe.fiber_map)
            mode.extra = probe.to_dict()
            if probe.error and not probe.fiber_map:
                mode.error = probe.error
            mode.extra["overlap"] = _fiber_overlap(plugmap, mode.fiber_map)
            report.modes.append(mode)

    return report


def format_debug_report(
    report: SpecObjLookupDebugReport,
    *,
    sample: int = 5,
) -> str:
    """Human-readable summary for terminal output."""
    lines: list[str] = []
    lines.append(f"spPlate: {report.spplate_path}")
    lines.append(f"survey:  {report.survey_name!r}")
    lines.append(f"plate:   {report.plate}  mjd: {report.mjd}")
    lines.append(f"plugmap: {len(report.plugmap_fibers)} distinct FIBERID(s)")
    if report.plugmap_fibers:
        preview = report.plugmap_fibers[:8]
        extra = (
            f" … +{len(report.plugmap_fibers) - 8}"
            if len(report.plugmap_fibers) > 8
            else ""
        )
        lines.append(f"  sample FIBERIDs: {preview}{extra}")

    audit = report.plugmap_audit
    if audit:
        lines.append("")
        lines.append("=== PLUGMAP columns (HDU 5) ===")
        hdu_name = audit.get("hdu_name") or "PLUGMAP"
        lines.append(f"  table: {hdu_name!r}  rows: {audit.get('n_rows', 0)}")
        for note in audit.get("notes") or []:
            lines.append(f"  • {note}")
        flagged = (audit.get("specobjid_like") or []) + (audit.get("objid_like") or [])
        if flagged:
            lines.append(f"  flagged ID-like columns: {', '.join(flagged)}")
        for col in audit.get("columns") or []:
            role = col.get("role")
            role_s = f" [{role}]" if role else ""
            col_sample = col.get("sample")
            sample_s = repr(col_sample)
            if len(sample_s) > 72:
                sample_s = sample_s[:69] + "…"
            lines.append(f"  {col['name']:22} {col['dtype']:12}{role_s}  sample={sample_s}")

    if report.header:
        lines.append("header:")
        for k, v in report.header.items():
            lines.append(f"  {k}: {v!r}")

    if report.inferred_layout:
        lines.append(f"inferred specObjID layout: {report.inferred_layout}")
    if report.header.get("RUN2D") is not None:
        lines.append(f"RUN2D used for synthesis: {report.header['RUN2D']!r}")
        for key in ("VERS2D", "VERSCOMB"):
            if key in report.header and str(report.header[key]) != str(report.header["RUN2D"]):
                lines.append(
                    f"  note: {key}={report.header[key]!r} differs — ignored for specObjID"
                )
    elif report.header.get("VERS2D") or report.header.get("VERSCOMB"):
        lines.append(
            "warning: VERS2D/VERSCOMB present but RUN2D missing — "
            "not used for specObjID synthesis"
        )
    if report.layout_error:
        lines.append(f"layout inference error: {report.layout_error}")

    for mode in report.modes:
        lines.append("")
        lines.append(f"=== {mode.name} ===")
        if mode.error:
            lines.append(f"  ERROR: {mode.error}")
        lines.append(f"  fiber mappings: {mode.size}")
        overlap = mode.extra.get("overlap")
        if overlap:
            lines.append(
                "  plugmap match: "
                f"{overlap['matched_count']}/{overlap['plugmap_count']} "
                f"(mapped={overlap['mapped_count']})"
            )
            miss = overlap.get("missing_from_lookup") or []
            if miss:
                lines.append(f"  first missing FIBERIDs: {miss}")
        if mode.name == "catalog":
            cols = mode.extra.get("columns") or {}
            lines.append(f"  catalog_dir: {mode.extra.get('catalog_dir')}")
            lines.append(f"  tiles: {mode.extra.get('tile_count', 0)}")
            lines.append(
                "  join: "
                f"plate={cols.get('plate')!r} mjd={cols.get('mjd')!r} "
                f"fiber={cols.get('fiber')!r}"
            )
            lines.append(f"  id column: {cols.get('source_id')!r}")
            lines.append(f"  rows with plate: {mode.extra.get('n_plate_rows', 0)}")
            lines.append(f"  rows with plate+mjd: {mode.extra.get('n_plate_mjd_rows', 0)}")
            mjds = mode.extra.get("mjds_for_plate") or []
            if mjds:
                sample_mjds = mjds[:12]
                extra = f" … (+{len(mjds) - 12})" if len(mjds) > 12 else ""
                lines.append(f"  MJDs for this plate in catalog: {sample_mjds}{extra}")
        if mode.name == "triplet_hash":
            if mode.extra.get("label_format"):
                lines.append(f"  label format: {mode.extra['label_format']}")
            if mode.extra.get("note"):
                lines.append(f"  note: {mode.extra['note']}")
        if mode.fiber_map and sample > 0:
            id_label = "source_id" if mode.name == "triplet_hash" else "specobjid"
            for row in _sample_pairs(mode.fiber_map, sample):
                lines.append(f"  fiber {row['fiber']} → {id_label} {row['specobjid']}")

    plate_mode = report.mode("from_plate")
    cat_mode = report.mode("catalog")
    if plate_mode and cat_mode and plate_mode.fiber_map and cat_mode.fiber_map:
        common = set(plate_mode.fiber_map) & set(cat_mode.fiber_map)
        mism = [
            fid
            for fid in sorted(common)[:20]
            if plate_mode.fiber_map[fid] != cat_mode.fiber_map[fid]
        ]
        if mism:
            lines.append("")
            lines.append("=== plate vs catalog ID mismatch (same fiber) ===")
            for fid in mism[:sample]:
                lines.append(
                    f"  fiber {fid}: plate={plate_mode.fiber_map[fid]} "
                    f"catalog={cat_mode.fiber_map[fid]}"
                )

    lines.append("")
    if not report.modes:
        lines.append("No lookup modes were run.")
    elif all(m.size == 0 for m in report.modes):
        lines.append(
            "RESULT: no fibers resolved — ingest would skip all spectra. "
            "Default (triplet hash) works without a catalog; for native specObjID "
            "use --specobj-lookup-from-plate or a sidecar lookup file."
        )
    else:
        best = max(report.modes, key=lambda m: m.size)
        lines.append(f"RESULT: best mode '{best.name}' maps {best.size} fiber(s).")

    return "\n".join(lines)


@click.command("dl-debug-specobj-lookup")
@click.argument("spplate", type=click.Path(exists=True, path_type=Path))
@click.option("--survey", "survey_name", required=True, help="Lake survey name (e.g. SDSS_DR17).")
@click.argument("output_root", type=click.Path(path_type=Path), required=False)
@config_option
@click.option(
    "--specobj-lookup",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Sidecar Parquet/CSV (same as dl-ingest-spectra --specobj-lookup).",
)
@click.option(
    "--specobj-lookup-survey",
    default=None,
    help="Survey column value when sidecar has no SURVEY column.",
)
@click.option(
    "--triplet-hash",
    is_flag=True,
    help=(
        "Show composite PLATE|MJD|FIBERID hash IDs (the default ingest mode). "
        "IDs match a catalog ingested with --link-id-col PLATE,MJD,FIBERID."
    ),
)
@click.option(
    "--no-catalog",
    is_flag=True,
    help="Skip scanning catalogs/<survey>/ under output_root.",
)
@click.option(
    "--no-plate",
    is_flag=True,
    help="Skip synthesizing IDs from spPlate header (RUN2D / DR7 layout).",
)
@click.option(
    "--specobj-id-layout",
    type=click.Choice(["auto", "dr7", "dr8plus"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="specObjID packing for plate synthesis.",
)
@click.option(
    "--link-id-col",
    default=None,
    help="Catalog/sidecar ID column after plate/mjd/fiber join (same as dl-ingest-spectra).",
)
@click.option("--sample", default=5, show_default=True, help="Sample mappings per mode.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.option("-v", "--verbose", is_flag=True)
def cli(
    spplate: Path,
    survey_name: str,
    output_root: Path | None,
    config_path: Path | None,
    specobj_lookup: Path | None,
    specobj_lookup_survey: str | None,
    triplet_hash: bool,
    no_catalog: bool,
    no_plate: bool,
    specobj_id_layout: str,
    link_id_col: str | None,
    sample: int,
    as_json: bool,
    verbose: bool,
) -> None:
    """Diagnose spPlate SPECOBJID lookup before ingest."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)

    catalog_root: Path | None = None
    if not no_catalog and output_root is not None:
        cfg = load_optional_config(config_path)
        catalog_root = require_output_root(output_root, cfg, kind="catalogs")

    nothing_to_probe = (
        not triplet_hash
        and no_plate
        and specobj_lookup is None
        and catalog_root is None
    )
    if nothing_to_probe:
        raise click.ClickException(
            "Nothing to probe: pass --triplet-hash, output_root (catalog), "
            "--specobj-lookup, or allow plate synthesis (default)."
        )

    report = debug_specobj_lookup(
        spplate,
        survey_name,
        catalog_root=catalog_root,
        lookup_path=specobj_lookup,
        lookup_survey=specobj_lookup_survey,
        try_triplet_hash=triplet_hash,
        try_from_plate=not no_plate,
        try_from_catalog=catalog_root is not None,
        try_from_sidecar=specobj_lookup is not None,
        specobj_id_layout=specobj_id_layout.lower(),  # type: ignore[arg-type]
        catalog_id_col=link_id_col,
    )

    if as_json:
        payload = report.to_dict()
        for mode in report.modes:
            if mode.fiber_map:
                key = f"{mode.name}_sample"
                payload[key] = _sample_pairs(mode.fiber_map, sample)
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(format_debug_report(report, sample=sample))
    if all(m.size == 0 for m in report.modes):
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
