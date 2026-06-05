"""
pack_tile – bundle a HEALPix tile's Parquet + Zarr into a shareable .tar archive.

Each archive contains
---------------------
  tile_<norder>_<npix>/
    catalog/<survey>/Norder=N/Dir=D/Npix=P.parquet  (one or more surveys)
    cutouts/<survey>/Norder=N/Npix=P.zarr/           (optional)
    spectra/<survey>/Norder=N/Npix=P.zarr/           (optional)
    catalog_info.json                                 (per survey)
    cutout_info.json                                  (per survey, if present)
    spectrum_info.json                                (per survey, if present)

Top-level MANIFEST.json lists all archives with:
  - tile pixel and order
  - included surveys
  - file sizes in bytes
  - SHA-256 checksums
  - n_spectra per tile

Usage
-----
    from data_lake.share.pack_tile import pack_tile, build_manifest

    pack_tile(
        lake_root="/data/lake",
        norder=5,
        npix=1234,
        surveys=["des_dr2", "kids_dr4"],
        output_dir="/data/share",
    )
    build_manifest("/data/share", "/data/share/MANIFEST.json")
"""

from __future__ import annotations

import hashlib
import json
import logging
import tarfile
import time
from pathlib import Path
from typing import Sequence

from data_lake.ingest.fits_to_parquet import healpix_dir

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core packing
# ---------------------------------------------------------------------------


def pack_tile(
    lake_root: Path | str,
    norder: int,
    npix: int,
    surveys: Sequence[str],
    output_dir: Path | str,
    include_cutouts: bool = True,
    include_spectra: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Bundle one HEALPix tile across the given surveys into a .tar archive.

    The archive is uncompressed because Parquet and Zarr data are already
    compressed internally.

    Parameters
    ----------
    lake_root:
        Data lake root.
    norder:
        HEALPix order.
    npix:
        HEALPix pixel index.
    surveys:
        Survey names to include.
    output_dir:
        Directory in which to write the archive.
    include_cutouts:
        If True (default), include the Zarr cutout store for each survey.
    include_spectra:
        If True (default), include the Zarr spectrum store for each survey.
    overwrite:
        If False (default), skip if archive already exists.

    Returns
    -------
    Path to the written .tar file.
    """
    lake_root = Path(lake_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_name = f"tile_{norder}_{npix}.tar"
    archive_path = output_dir / archive_name

    if archive_path.exists() and not overwrite:
        log.info("Archive already exists: %s (skip)", archive_path.name)
        return archive_path

    tile_dir_fragment = healpix_dir(norder, npix)

    with tarfile.open(archive_path, "w") as tar:
        for survey in surveys:
            prefix = f"tile_{norder}_{npix}"

            # ---- Catalog Parquet file ----
            parquet_file = (
                lake_root / "catalogs" / survey / tile_dir_fragment / f"Npix={npix}.parquet"
            )
            if parquet_file.exists():
                arcname = f"{prefix}/catalog/{survey}/{tile_dir_fragment}/Npix={npix}.parquet"
                tar.add(parquet_file, arcname=arcname)
                log.debug("Added %s", arcname)
            else:
                log.warning("Parquet tile not found for survey=%s npix=%d", survey, npix)

            # ---- catalog_info.json ----
            cat_info = lake_root / "catalogs" / survey / "catalog_info.json"
            if cat_info.exists():
                tar.add(cat_info, arcname=f"{prefix}/catalog/{survey}/catalog_info.json")

            # ---- Zarr cutout store ----
            if include_cutouts:
                zarr_store = (
                    lake_root / "cutouts" / survey / tile_dir_fragment / f"Npix={npix}.zarr"
                )
                if zarr_store.exists():
                    _add_directory_to_tar(tar, zarr_store,
                                          f"{prefix}/cutouts/{survey}/{tile_dir_fragment}/Npix={npix}.zarr")
                    log.debug("Added Zarr store for survey=%s npix=%d", survey, npix)

                cutout_info = lake_root / "cutouts" / survey / "cutout_info.json"
                if cutout_info.exists():
                    tar.add(cutout_info, arcname=f"{prefix}/cutouts/{survey}/cutout_info.json")

            # ---- Zarr spectrum store ----
            if include_spectra:
                spec_store = (
                    lake_root / "spectra" / survey / tile_dir_fragment / f"Npix={npix}.zarr"
                )
                if spec_store.exists():
                    _add_directory_to_tar(tar, spec_store,
                                          f"{prefix}/spectra/{survey}/{tile_dir_fragment}/Npix={npix}.zarr")
                    log.debug("Added spectrum store for survey=%s npix=%d", survey, npix)

                spec_info = lake_root / "spectra" / survey / "spectrum_info.json"
                if spec_info.exists():
                    tar.add(spec_info, arcname=f"{prefix}/spectra/{survey}/spectrum_info.json")

    log.info("Wrote %s (%.1f MB)", archive_path.name, archive_path.stat().st_size / 1e6)
    return archive_path


def _add_directory_to_tar(tar: tarfile.TarFile, src: Path, arcname_prefix: str) -> None:
    """Recursively add a directory into the tar archive."""
    for path in sorted(src.rglob("*")):
        relative = path.relative_to(src)
        arcname = f"{arcname_prefix}/{relative}"
        tar.add(path, arcname=arcname, recursive=False)


# ---------------------------------------------------------------------------
# Batch packing
# ---------------------------------------------------------------------------


def pack_tiles_batch(
    lake_root: Path | str,
    norder: int,
    surveys: Sequence[str],
    output_dir: Path | str,
    npix_list: Sequence[int] | None = None,
    include_cutouts: bool = True,
    include_spectra: bool = True,
    overwrite: bool = False,
) -> list[Path]:
    """
    Pack multiple tiles.  If ``npix_list`` is None, discovers all tiles that
    exist for the first survey listed in ``surveys``.
    """
    lake_root = Path(lake_root)
    if npix_list is None:
        survey = surveys[0]
        catalog_norder_dir = lake_root / "catalogs" / survey / f"Norder={norder}"
        npix_list = [
            int(p.stem.split("=")[-1])
            for p in catalog_norder_dir.rglob("Npix=*.parquet")
        ]

    archives = []
    for npix in npix_list:
        path = pack_tile(
            lake_root=lake_root,
            norder=norder,
            npix=npix,
            surveys=surveys,
            output_dir=output_dir,
            include_cutouts=include_cutouts,
            include_spectra=include_spectra,
            overwrite=overwrite,
        )
        archives.append(path)

    return archives


# ---------------------------------------------------------------------------
# MANIFEST.json
# ---------------------------------------------------------------------------


def _sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(block)
    return h.hexdigest()


def _parquet_row_count(path: Path) -> int | None:
    """Return row count from Parquet footer metadata (no full read)."""
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(str(path)).num_rows
    except Exception:
        return None


def _count_spectra_in_archive(arc: Path, norder: int, npix: int) -> int | None:
    """
    Peek inside the .tar archive to count spectra in the Zarr source_id array.

    Returns ``None`` if no spectrum store is present in the archive.
    Reads only the tiny ``source_id/.zarray`` and chunk metadata – not the data.
    """
    try:
        import tarfile as _tar
        import json as _json

        zarr_prefix = f"tile_{norder}_{npix}/spectra/"
        sid_array_meta = None

        with _tar.open(arc, "r") as t:
            for member in t.getmembers():
                # Look for a source_id zarr.json inside any spectra sub-tree
                if (zarr_prefix in member.name
                        and ("_source_id/zarr.json" in member.name
                             or "source_id/zarr.json" in member.name)):
                    fh = t.extractfile(member)
                    if fh:
                        sid_array_meta = _json.load(fh)
                        break

        if sid_array_meta is None:
            return None

        shape = sid_array_meta.get("shape", [])
        return int(shape[0]) if shape else None
    except Exception:
        return None


def build_manifest(
    share_dir: Path | str,
    output_path: Path | str | None = None,
) -> dict:
    """
    Build a ``MANIFEST.json`` listing all .tar archives in ``share_dir``.

    Each entry includes:
    - ``filename``
    - ``size_bytes``
    - ``sha256``
    - ``created_utc``
    - ``norder``, ``npix`` (parsed from filename ``tile_<norder>_<npix>.tar``)

    Parameters
    ----------
    share_dir:
        Directory containing the .tar archives.
    output_path:
        Where to write the MANIFEST.json.  Defaults to
        ``<share_dir>/MANIFEST.json``.

    Returns
    -------
    The manifest dict.
    """
    share_dir = Path(share_dir)
    output_path = Path(output_path) if output_path else share_dir / "MANIFEST.json"

    archives = sorted(share_dir.glob("tile_*.tar"))
    manifest: dict = {
        "version": "1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tiles": [],
    }

    for arc in archives:
        stem = arc.stem  # tile_<norder>_<npix>
        parts = stem.split("_")
        try:
            norder = int(parts[1])
            npix = int(parts[2])
        except (IndexError, ValueError):
            log.warning("Unexpected archive name: %s", arc.name)
            norder, npix = -1, -1

        log.info("Checksumming %s …", arc.name)
        entry = {
            "filename": arc.name,
            "size_bytes": arc.stat().st_size,
            "sha256": _sha256(arc),
            "norder": norder,
            "npix": npix,
            "n_spectra": _count_spectra_in_archive(arc, norder, npix),
            "created_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(arc.stat().st_mtime),
            ),
        }
        manifest["tiles"].append(entry)

    with open(output_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    log.info("MANIFEST.json written with %d entries → %s", len(archives), output_path)
    return manifest


def verify_manifest(
    share_dir: Path | str,
    manifest_path: Path | str | None = None,
) -> list[str]:
    """
    Verify SHA-256 checksums for all archives listed in the manifest.

    Returns a list of error messages (empty = all OK).
    """
    share_dir = Path(share_dir)
    manifest_path = Path(manifest_path) if manifest_path else share_dir / "MANIFEST.json"

    with open(manifest_path) as fh:
        manifest = json.load(fh)

    errors: list[str] = []
    for entry in manifest.get("tiles", []):
        arc = share_dir / entry["filename"]
        if not arc.exists():
            errors.append(f"Missing: {arc.name}")
            continue
        actual = _sha256(arc)
        if actual != entry["sha256"]:
            errors.append(
                f"Checksum mismatch for {arc.name}: "
                f"expected {entry['sha256'][:12]}… got {actual[:12]}…"
            )
    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    @click.command("dl-pack-tile")
    @click.argument("lake_root", type=click.Path(exists=True, path_type=Path))
    @click.argument("output_dir", type=click.Path(path_type=Path))
    @click.option("--norder", default=5, type=int, show_default=True)
    @click.option("--npix", type=int, default=None, help="Single tile to pack.")
    @click.option("--survey", "surveys", multiple=True, required=True, help="Survey name(s) to include.")
    @click.option("--no-cutouts", "include_cutouts", is_flag=True, default=True,
                  flag_value=False, help="Exclude Zarr cutout stores.")
    @click.option("--no-spectra", "include_spectra", is_flag=True, default=True,
                  flag_value=False, help="Exclude Zarr spectrum stores.")
    @click.option("--manifest", "write_manifest", is_flag=True, default=False,
                  help="(Re)generate MANIFEST.json after packing.")
    @click.option("--overwrite", is_flag=True)
    @click.option("-v", "--verbose", is_flag=True)
    @click.option("-q", "--quiet", is_flag=True, default=False,
                  help="Suppress INFO messages on the terminal.")
    def cli(
        lake_root: Path,
        output_dir: Path,
        norder: int,
        npix: int | None,
        surveys: tuple[str, ...],
        include_cutouts: bool,
        include_spectra: bool,
        write_manifest: bool,
        overwrite: bool,
        verbose: bool,
        quiet: bool,
    ) -> None:
        """Pack one or all tiles of LAKE_ROOT into OUTPUT_DIR as .tar archives."""
        import logging as _logging
        from data_lake.cli_utils import configure_cli_logging, validate_quiet_verbose
        validate_quiet_verbose(quiet, verbose)
        configure_cli_logging(
            level=_logging.DEBUG if verbose else (_logging.WARNING if quiet else _logging.INFO),
            quiet=quiet,
        )
        if npix is not None:
            pack_tile(lake_root, norder, npix, list(surveys), output_dir,
                      include_cutouts, include_spectra, overwrite)
        else:
            pack_tiles_batch(lake_root, norder, list(surveys), output_dir,
                             include_cutouts=include_cutouts,
                             include_spectra=include_spectra,
                             overwrite=overwrite)
        if write_manifest:
            build_manifest(output_dir)

except ImportError:
    cli = None  # type: ignore[assignment]
