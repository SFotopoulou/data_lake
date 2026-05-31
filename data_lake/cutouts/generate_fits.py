"""
generate_fits – cut fixed-size stamps from survey images into per-source FITS files.

Output format matches :func:`data_lake.ingest.fits_to_zarr.ingest_cutouts_from_fits`:

* One FITS file per catalog row
* Primary HDU data shape ``(N_bands, N_pix, N_pix)`` float32
* Header keywords for object ID, sky position, and 2-D TAN WCS

Example
-------
::

    dl-generate-cutout-fits targets.parquet /data/cutout_fits \\
        --images-file bands.txt \\
        --size 64 \\
        --id-col TARGETID --ra-col TARGET_RA --dec-col TARGET_DEC \\
        --id-hdu-key TARGETID --ra-hdu-key TARGET_RA --dec-hdu-key TARGET_DEC

    dl-ingest-cutouts-from-list cutout_paths.txt --survey desi_dr1 \\
        --link-id-col TARGETID --ra-col TARGET_RA --dec-col TARGET_DEC \\
        --band-names r,i,z
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy import units as u
from astropy.wcs import WCS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerateResult:
    """Summary of a stamp-generation run."""

    n_written: int
    n_skipped: int
    output_dir: Path


def _read_catalog_table(
    catalog_path: Path,
    id_col: str,
    ra_col: str,
    dec_col: str,
) -> Table:
    catalog_path = Path(catalog_path)
    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)

    tbl = Table.read(str(catalog_path))
    for col in (id_col, ra_col, dec_col):
        if col not in tbl.colnames:
            raise KeyError(
                f"Catalog column {col!r} not found in {catalog_path.name}. "
                f"Available: {tbl.colnames}"
            )
    return tbl


def _parse_image_paths(
    images: Sequence[str] | None,
    images_file: Path | str | None,
) -> list[Path]:
    paths: list[Path] = []
    if images:
        for item in images:
            for part in item.split(","):
                part = part.strip()
                if part:
                    paths.append(Path(part))
    if images_file:
        text = Path(images_file).read_text()
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                paths.append(Path(line))
    if not paths:
        raise ValueError("Provide at least one band image via --images or --images-file.")
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Band image not found: {p}")
    return paths


def _open_band_cube(path: Path, hdu_index: int) -> tuple[np.ndarray, WCS, fits.HDUList]:
    """Return memmapped data, WCS, and open HDUList (caller must close)."""
    hdul = fits.open(str(path), memmap=True, lazy_load_hdus=False)
    hdu = hdul[hdu_index]
    if hdu.data is None:
        hdul.close()
        raise ValueError(f"No image data in HDU {hdu_index} of {path}")
    wcs = WCS(hdu.header, naxis=2)
    if wcs.naxis != 2:
        hdul.close()
        raise ValueError(f"HDU {hdu_index} of {path} is not a 2-D image (WCS naxis={wcs.naxis})")
    return hdu.data, wcs, hdul


def _extract_one_band(
    data: np.ndarray,
    wcs: WCS,
    ra_deg: float,
    dec_deg: float,
    size_pix: int,
    *,
    fill_value: float = np.nan,
) -> Cutout2D:
    position = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    return Cutout2D(
        data,
        position,
        (size_pix, size_pix),
        wcs=wcs,
        mode="partial",
        fill_value=fill_value,
    )


def _stamp_cube_from_bands(
    band_data_wcs: list[tuple[np.ndarray, WCS]],
    ra_deg: float,
    dec_deg: float,
    size_pix: int,
    *,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, WCS]:
    """Stack bands in list order → ``(N_bands, size, size)`` and return reference WCS."""
    planes: list[np.ndarray] = []
    ref_wcs: WCS | None = None
    for data, wcs in band_data_wcs:
        cut = _extract_one_band(
            data, wcs, ra_deg, dec_deg, size_pix, fill_value=fill_value,
        )
        planes.append(np.asarray(cut.data, dtype=np.float32))
        if ref_wcs is None:
            ref_wcs = cut.wcs
    if ref_wcs is None:
        raise RuntimeError("No bands provided")
    cube = np.stack(planes, axis=0)
    return cube, ref_wcs


def _write_cutout_fits(
    out_path: Path,
    cube: np.ndarray,
    wcs: WCS,
    *,
    source_id: int,
    ra_deg: float,
    dec_deg: float,
    id_hdu_key: str,
    ra_hdu_key: str,
    dec_hdu_key: str,
    band_names: Sequence[str] | None = None,
) -> None:
    """Write one ingest-ready multi-band stamp FITS."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=cube.astype(np.float32, copy=False))
    hdr = hdu.header
    hdr[id_hdu_key] = int(source_id)
    hdr[ra_hdu_key] = float(ra_deg)
    hdr[dec_hdu_key] = float(dec_deg)
    hdr["ORIGIN"] = "data_lake.generate_fits"
    if band_names:
        hdr["NBANDS"] = len(band_names)
        hdr["BANDLIST"] = ",".join(band_names)[:68]
    hdr.update(wcs.to_header(relax=True))
    hdu.writeto(str(out_path), overwrite=True)


def generate_cutout_fits(
    catalog_path: Path | str,
    output_dir: Path | str,
    image_paths: Sequence[Path | str],
    *,
    size_pix: int,
    id_col: str = "source_id",
    ra_col: str = "ra",
    dec_col: str = "dec",
    id_hdu_key: str = "SOURCE_ID",
    ra_hdu_key: str = "RA",
    dec_hdu_key: str = "DEC",
    image_hdu_index: int = 0,
    fill_value: float = float("nan"),
    filename_template: str = "cutout_{source_id}.fits",
    band_names: Sequence[str] | None = None,
    max_sources: int | None = None,
    skip_existing: bool = False,
    show_progress: bool = True,
) -> GenerateResult:
    """Generate per-source multi-band cutout FITS files."""
    catalog_path = Path(catalog_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if size_pix < 1:
        raise ValueError(f"size_pix must be >= 1, got {size_pix}")

    paths = [Path(p) for p in image_paths]
    tbl = _read_catalog_table(catalog_path, id_col, ra_col, dec_col)
    n_rows = len(tbl)
    if max_sources is not None:
        n_rows = min(n_rows, max_sources)
        tbl = tbl[:n_rows]

    bands: list[tuple[np.ndarray, WCS]] = []
    open_hdus: list[fits.HDUList] = []
    try:
        for p in paths:
            data, wcs, hdul = _open_band_cube(p, image_hdu_index)
            open_hdus.append(hdul)
            bands.append((data, wcs))
        log.info(
            "Cutouts: %d sources × %d bands (%d×%d px) → %s",
            n_rows, len(bands), size_pix, size_pix, output_dir,
        )

        iterator = range(n_rows)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, desc="cutout FITS", unit="src")
            except ImportError:
                pass

        n_written = 0
        n_skipped = 0
        ids = np.asarray(tbl[id_col])
        ras = np.asarray(tbl[ra_col], dtype=np.float64)
        decs = np.asarray(tbl[dec_col], dtype=np.float64)

        for i in iterator:
            sid = int(ids[i])
            out_path = output_dir / filename_template.format(source_id=sid)
            if skip_existing and out_path.exists():
                n_skipped += 1
                continue
            cube, stamp_wcs = _stamp_cube_from_bands(
                bands,
                float(ras[i]),
                float(decs[i]),
                size_pix,
                fill_value=fill_value,
            )
            _write_cutout_fits(
                out_path,
                cube,
                stamp_wcs,
                source_id=sid,
                ra_deg=float(ras[i]),
                dec_deg=float(decs[i]),
                id_hdu_key=id_hdu_key,
                ra_hdu_key=ra_hdu_key,
                dec_hdu_key=dec_hdu_key,
                band_names=band_names,
            )
            n_written += 1
    finally:
        for hdul in open_hdus:
            hdul.close()

    return GenerateResult(n_written=n_written, n_skipped=n_skipped, output_dir=output_dir)


try:
    import click

    @click.command("dl-generate-cutout-fits")
    @click.argument("catalog_path", type=click.Path(exists=True, path_type=Path))
    @click.argument("output_dir", type=click.Path(path_type=Path))
    @click.option("--images", default=None)
    @click.option("--images-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
    @click.option("--size", "size_pix", required=True, type=int)
    @click.option("--id-col", default="source_id", show_default=True)
    @click.option("--ra-col", default="ra", show_default=True)
    @click.option("--dec-col", default="dec", show_default=True)
    @click.option("--id-hdu-key", default="SOURCE_ID", show_default=True)
    @click.option("--ra-hdu-key", default="RA", show_default=True)
    @click.option("--dec-hdu-key", default="DEC", show_default=True)
    @click.option("--image-hdu", "image_hdu_index", default=0, show_default=True)
    @click.option("--band-names", default=None)
    @click.option("--filename-template", default="cutout_{source_id}.fits", show_default=True)
    @click.option("--max-sources", default=None, type=int)
    @click.option("--skip-existing", is_flag=True)
    @click.option("--no-progress", is_flag=True)
    @click.option("-v", "--verbose", is_flag=True)
    def cli(
        catalog_path: Path,
        output_dir: Path,
        images: str | None,
        images_file: Path | None,
        size_pix: int,
        id_col: str,
        ra_col: str,
        dec_col: str,
        id_hdu_key: str,
        ra_hdu_key: str,
        dec_hdu_key: str,
        image_hdu_index: int,
        band_names: str | None,
        filename_template: str,
        max_sources: int | None,
        skip_existing: bool,
        no_progress: bool,
        verbose: bool,
    ) -> None:
        """Cut stamps from band images into per-source FITS for lake ingest."""
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        image_list = _parse_image_paths([images] if images else None, images_file)
        bn = [x.strip() for x in band_names.split(",") if x.strip()] if band_names else None
        if bn is not None and len(bn) != len(image_list):
            raise click.ClickException(
                f"--band-names has {len(bn)} entries but {len(image_list)} images were given."
            )
        result = generate_cutout_fits(
            catalog_path=catalog_path,
            output_dir=output_dir,
            image_paths=image_list,
            size_pix=size_pix,
            id_col=id_col,
            ra_col=ra_col,
            dec_col=dec_col,
            id_hdu_key=id_hdu_key,
            ra_hdu_key=ra_hdu_key,
            dec_hdu_key=dec_hdu_key,
            image_hdu_index=image_hdu_index,
            band_names=bn,
            filename_template=filename_template,
            max_sources=max_sources,
            skip_existing=skip_existing,
            show_progress=not no_progress,
        )
        click.echo(
            f"Wrote {result.n_written} cutout FITS to {result.output_dir} "
            f"(skipped {result.n_skipped} existing)."
        )

except ImportError:
    cli = None  # type: ignore[assignment,misc]
