"""Materialise homogenized spectra and cutout Zarr products."""

from __future__ import annotations

import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zarr

from data_lake.discovery.selection import BaseSelection
from data_lake.homogenize.registry import load_transform
from data_lake.ingest.fits_to_parquet import healpix_dir
from data_lake.schema_registry import MODALITY_CUTOUT, MODALITY_SPECTRA, PRODUCT_SUBTYPE_HOMOGENIZED

log = logging.getLogger(__name__)

_MAX_WORKERS = 64


@dataclass
class ZarrHomogenizeResult:
    product: str
    modality: str
    output_root: Path
    n_tiles_written: int
    n_sources: int
    transform_id: str
    resolution: dict[str, Any]
    check_only: bool
    elapsed_s: float


def _modality_root(lake_root: Path, modality: str) -> str:
    if modality == MODALITY_SPECTRA:
        return "spectra"
    if modality == MODALITY_CUTOUT:
        return "cutouts"
    raise ValueError(f"unsupported modality {modality!r}")


def _read_info(lake_root: Path, modality: str, survey: str) -> dict[str, Any]:
    info_name = "spectrum_info.json" if modality == MODALITY_SPECTRA else "cutout_info.json"
    path = lake_root / _modality_root(lake_root, modality) / survey / info_name
    if not path.is_file():
        raise FileNotFoundError(f"No {modality} survey at {path.parent}/")
    return json.loads(path.read_text())


def _rule_for_survey(transform: dict[str, Any], survey: str) -> dict[str, Any] | None:
    for raw in transform.get("rules") or []:
        if raw.get("survey") == survey:
            return raw
    return None


def _flux_scale_factor(rule: dict[str, Any]) -> float:
    t = rule.get("transform") or {}
    if t.get("type") != "flux_scale":
        raise ValueError(f"unsupported zarr transform type {t.get('type')!r}")
    return float(t.get("factor", 1.0))


def _homogenize_spectrum_tile(
    npix: int,
    *,
    lake_root: Path,
    survey: str,
    out_survey: str,
    norder: int,
    factor: float,
) -> int:
    src_root = lake_root / "spectra" / survey / healpix_dir(norder, npix) / f"Npix={npix}.zarr"
    if not src_root.is_dir():
        return 0
    dst_root = lake_root / "spectra" / out_survey / healpix_dir(norder, npix) / f"Npix={npix}.zarr"
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)

    store = zarr.storage.LocalStore(str(dst_root))
    root = zarr.open_group(store=store, mode="a", zarr_format=3)
    flux = np.array(root["flux"][:], dtype=np.float32)
    root["flux"][:] = flux * factor
    if "ivar" in root:
        ivar = np.array(root["ivar"][:], dtype=np.float32)
        safe = ivar > 0
        scaled = np.zeros_like(ivar)
        scaled[safe] = ivar[safe] / (factor * factor)
        root["ivar"][:] = scaled
    return int(root["flux"].shape[0])


def _homogenize_cutout_tile(
    npix: int,
    *,
    lake_root: Path,
    survey: str,
    out_survey: str,
    norder: int,
    factor: float,
) -> int:
    src_root = lake_root / "cutouts" / survey / healpix_dir(norder, npix) / f"Npix={npix}.zarr"
    if not src_root.is_dir():
        return 0
    dst_root = lake_root / "cutouts" / out_survey / healpix_dir(norder, npix) / f"Npix={npix}.zarr"
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)

    store = zarr.storage.LocalStore(str(dst_root))
    root = zarr.open_group(store=store, mode="a", zarr_format=3)
    images = np.array(root["images"][:], dtype=np.float32)
    root["images"][:] = images * factor
    return int(root["images"].shape[0])


def _write_zarr_product_info(
    out_root: Path,
    *,
    modality: str,
    name: str,
    source_survey: str,
    transform_id: str,
    transform_version: int,
    source_info: dict[str, Any],
    selection: BaseSelection,
    factor: float,
    n_sources: int,
) -> None:
    info_name = "spectrum_info.json" if modality == MODALITY_SPECTRA else "cutout_info.json"
    info = dict(source_info)
    info["survey_name"] = name
    info["kind"] = "product"
    info["product_subtype"] = PRODUCT_SUBTYPE_HOMOGENIZED
    info["provenance"] = {
        "source_survey": source_survey,
        "transform_id": transform_id,
        "transform_version": transform_version,
        "flux_scale_factor": factor,
        "selection": {
            "base_survey": selection.base_survey,
            "norder": selection.norder,
            "n_npix": len(selection.npix),
        },
        "n_sources": n_sources,
    }
    info["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / info_name, "w") as fh:
        json.dump(info, fh, indent=2)


def homogenize_zarr(
    lake_root: Path | str,
    modality: str,
    survey: str,
    transform_id: str,
    selection: BaseSelection,
    *,
    materialize_as: str,
    overwrite: bool = False,
    check_only: bool = False,
    n_workers: int = 8,
    show_progress: bool = False,
) -> ZarrHomogenizeResult:
    """Materialise a homogenized spectra or cutout survey tree."""
    lake_root = Path(lake_root)
    if modality not in (MODALITY_SPECTRA, MODALITY_CUTOUT):
        raise ValueError(f"modality must be {MODALITY_SPECTRA!r} or {MODALITY_CUTOUT!r}")

    transform = load_transform(lake_root, transform_id)
    if transform.get("modality") != modality:
        raise ValueError(
            f"Transform {transform_id!r} modality {transform.get('modality')!r} "
            f"!= requested {modality!r}"
        )

    rule = _rule_for_survey(transform, survey)
    if rule is None:
        raise ValueError(f"No {modality} rule for survey {survey!r} in {transform_id!r}")

    factor = _flux_scale_factor(rule)
    source_info = _read_info(lake_root, modality, survey)
    out_root = lake_root / _modality_root(lake_root, modality) / materialize_as
    if out_root.exists() and not overwrite and not check_only:
        raise FileExistsError(
            f"homogenized {modality} survey already exists: {out_root} (use --overwrite)"
        )

    resolution = {
        "n_applied": 1,
        "applied": [{"survey": survey, "factor": factor, "type": "flux_scale"}],
    }
    npix_list = sorted(selection.npix)
    if check_only:
        return ZarrHomogenizeResult(
            product=materialize_as,
            modality=modality,
            output_root=out_root,
            n_tiles_written=0,
            n_sources=0,
            transform_id=transform_id,
            resolution=resolution,
            check_only=True,
            elapsed_s=0.0,
        )

    norder = selection.norder
    tile_fn = _homogenize_spectrum_tile if modality == MODALITY_SPECTRA else _homogenize_cutout_tile
    t0 = time.perf_counter()
    n_sources = 0
    n_tiles = 0
    workers = max(1, min(n_workers, _MAX_WORKERS, len(npix_list) or 1))

    def _work(npix: int) -> int:
        return tile_fn(
            npix,
            lake_root=lake_root,
            survey=survey,
            out_survey=materialize_as,
            norder=norder,
            factor=factor,
        )

    counts: list[int] = []
    if workers <= 1 or len(npix_list) <= 1:
        tile_iter = npix_list
        if show_progress:
            try:
                from tqdm.auto import tqdm

                tile_iter = tqdm(npix_list, unit="tile", desc=f"homogenize {materialize_as}")
            except ImportError:
                pass
        for npix in tile_iter:
            counts.append(_work(npix))
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
                        counts.append(fut.result())
                except ImportError:
                    for fut in as_completed(futures):
                        counts.append(fut.result())
            else:
                for fut in as_completed(futures):
                    counts.append(fut.result())

    for c in counts:
        if c:
            n_sources += c
            n_tiles += 1

    _write_zarr_product_info(
        out_root,
        modality=modality,
        name=materialize_as,
        source_survey=survey,
        transform_id=transform_id,
        transform_version=int(transform.get("version", 1)),
        source_info=source_info,
        selection=selection,
        factor=factor,
        n_sources=n_sources,
    )

    return ZarrHomogenizeResult(
        product=materialize_as,
        modality=modality,
        output_root=out_root,
        n_tiles_written=n_tiles,
        n_sources=n_sources,
        transform_id=transform_id,
        resolution=resolution,
        check_only=False,
        elapsed_s=time.perf_counter() - t0,
    )
