"""
`dl-demo-lake` – build a small synthetic deployment for tutorials and tests.

The demo lake is produced by the same ingest code as a real lake, from
synthetic FITS files that are kept next to it (``raw/``) so notebooks can
also demonstrate ingest. Three surveys cover one 2°×2° sky patch:

``DEMO_PHOT``
    Imaging catalog with AB magnitudes in g, z, y, J, H, a source class and
    three-band image cutouts for the brightest sources.
``DEMO_SPEC``
    Spectroscopic catalog (redshift, class) with one 1-D spectrum per row,
    targeted at a subset of ``DEMO_PHOT`` sources.
``DEMO_IR``
    Mid-infrared catalog with W1/W2 Vega magnitudes, mostly overlapping
    ``DEMO_PHOT``.

Colours and spectra depend on redshift, so the lake supports the
crossmatch → gather → homogenize → photo-z workflow end to end. Values are
illustrative only; they are not a model of any real survey.

Example
-------
    dl-demo-lake ./demo_lake
    export DATA_LAKE_CONFIG=./demo_lake/lake_config.toml
    export LAKE_INGEST_TOKEN=demo
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PHOT_SURVEY = "DEMO_PHOT"
SPEC_SURVEY = "DEMO_SPEC"
IR_SURVEY = "DEMO_IR"
DEMO_SURVEYS = (PHOT_SURVEY, SPEC_SURVEY, IR_SURVEY)

DEMO_INGEST_TOKEN = "demo"
DEMO_DIR_ENV = "DATA_LAKE_DEMO_DIR"

CENTER_RA_DEG = 150.0
CENTER_DEC_DEG = 2.2
HALF_WIDTH_DEG = 1.0
NORDER = 5

PHOT_BANDS = ("g", "z", "y", "j", "h")
CUTOUT_BANDS = ("g", "z", "h")
CUTOUT_SIZE = 32
CUTOUT_PIXSCALE_ARCSEC = 0.2
AB_ZP_NJY = 31.4

WAVE_MIN_A = 3600.0
WAVE_MAX_A = 9800.0
WAVE_STEP_A = 4.0

# Vega → AB offsets for W1/W2 (same values as the bundled ALLWISE recipe).
W1_VEGA_TO_AB = 2.699
W2_VEGA_TO_AB = 3.339

_COMPLETE_MARKER = ".demo_complete"


@dataclass(frozen=True)
class DemoSizes:
    """Row counts for the generated surveys."""

    n_phot: int = 3000
    n_spec: int = 300
    n_cutouts: int = 150
    n_ir_only: int = 150

    def __post_init__(self) -> None:
        if self.n_phot < 10:
            raise ValueError("n_phot must be >= 10")
        if not 0 < self.n_spec <= self.n_phot:
            raise ValueError("n_spec must be in 1..n_phot")
        if not 0 < self.n_cutouts <= self.n_phot:
            raise ValueError("n_cutouts must be in 1..n_phot")
        if self.n_ir_only < 0:
            raise ValueError("n_ir_only must be >= 0")


@dataclass
class DemoInputs:
    """Paths of the synthetic raw files written by :func:`write_demo_inputs`."""

    root: Path
    phot_catalog: Path
    spec_catalog: Path
    ir_catalog: Path
    spectra: list[Path] = field(default_factory=list)
    cutouts: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic sky
# ---------------------------------------------------------------------------


def _sources(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Positions, classes, redshifts and AB magnitudes for *n* imaging sources."""
    ra = CENTER_RA_DEG + rng.uniform(-HALF_WIDTH_DEG, HALF_WIDTH_DEG, n) / np.cos(
        np.radians(CENTER_DEC_DEG)
    )
    dec = CENTER_DEC_DEG + rng.uniform(-HALF_WIDTH_DEG, HALF_WIDTH_DEG, n)

    cls = rng.choice(np.array(["GALAXY", "STAR", "QSO"]), size=n, p=[0.85, 0.10, 0.05])
    z = np.zeros(n)
    gal = cls == "GALAXY"
    qso = cls == "QSO"
    z[gal] = np.clip(rng.gamma(3.0, 0.15, gal.sum()), 0.02, 1.5)
    z[qso] = rng.uniform(0.5, 3.0, qso.sum())

    m_z = np.where(gal, 19.0 + 2.2 * np.sqrt(z) + rng.normal(0, 0.8, n), rng.uniform(17.0, 22.5, n))
    colour = {
        "g": np.where(gal, 0.4 + 1.6 * z, np.where(qso, 0.3, rng.uniform(0.2, 2.0, n))),
        "y": np.where(gal, -0.1 - 0.3 * z, np.where(qso, -0.1, -0.1)),
        "j": np.where(gal, -0.3 - 0.6 * z, np.where(qso, -0.3, -0.4)),
        "h": np.where(gal, -0.5 - 0.8 * z, np.where(qso, -0.5, -0.6)),
    }
    mags = {"z": m_z}
    for band, c in colour.items():
        mags[band] = m_z + c + rng.normal(0, 0.05, n)

    w1_ab = mags["h"] + np.where(qso, -0.8, np.where(gal, 0.3 - 0.2 * z, 0.6))
    w2_ab = w1_ab + np.where(qso, -0.4, np.where(gal, 0.1 + 0.2 * z, 0.5))
    return {
        "ra": ra, "dec": dec, "cls": cls, "z": z, "mags": mags,
        "w1_ab": w1_ab, "w2_ab": w2_ab,
    }


def _mag_err(rng: np.random.Generator, mag: np.ndarray, depth: float) -> tuple[np.ndarray, np.ndarray]:
    err = 0.01 + 0.2 * 10 ** (0.4 * (mag - depth))
    return mag + rng.normal(0, err), err


def _write_table(path: Path, columns: dict[str, np.ndarray]) -> Path:
    from astropy.table import Table

    Table(columns).write(path, format="fits", overwrite=True)
    return path


def _wavelength() -> np.ndarray:
    return np.arange(WAVE_MIN_A, WAVE_MAX_A + WAVE_STEP_A / 2, WAVE_STEP_A)


_GALAXY_LINES = ((3727.0, 4.0), (4861.0, 1.5), (5007.0, 3.0), (6563.0, 5.0))
_QSO_LINES = ((1216.0, 40.0), (1549.0, 20.0), (1909.0, 10.0), (2798.0, 12.0), (4861.0, 6.0), (6563.0, 15.0))
_STAR_ABSORPTION = (4861.0, 6563.0)


def _spectrum(
    rng: np.random.Generator, wave: np.ndarray, cls: str, z: float, m_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Flux (1e-17 erg/s/cm²/Å) and inverse variance for one source."""
    norm = 10 ** (-0.4 * (m_z - 20.0)) * 5.0
    x = wave / 5500.0
    if cls == "STAR":
        teff = rng.uniform(4000, 9000)
        bb = x ** -5 / np.expm1(2.6e3 / (x * teff))
        flux = norm * bb / bb[np.searchsorted(wave, 9000.0)]
        for line in _STAR_ABSORPTION:
            flux *= 1 - 0.3 * np.exp(-0.5 * ((wave - line) / 8.0) ** 2)
    elif cls == "QSO":
        flux = norm * x ** -0.5
        for rest, amp in _QSO_LINES:
            obs = rest * (1 + z)
            flux += norm * amp / 10 * np.exp(-0.5 * ((wave - obs) / (25.0 * (1 + z))) ** 2)
    else:
        flux = norm * x ** 0.8 * np.where(wave < 4000.0 * (1 + z), 0.55, 1.0)
        for rest, amp in _GALAXY_LINES:
            obs = rest * (1 + z)
            flux += norm * amp * np.exp(-0.5 * ((wave - obs) / (2.5 * (1 + z))) ** 2)
    sigma = 0.05 * norm + 0.1
    return (flux + rng.normal(0, sigma, wave.size)).astype(np.float32), np.full(
        wave.size, 1.0 / sigma**2, dtype=np.float32
    )


def _iter_spectrum_files(
    out_dir: Path, rng: np.random.Generator, ids, ra, dec, cls, z, m_z,
) -> Iterator[Path]:
    from astropy.io import fits

    wave = _wavelength()
    for i in range(len(ids)):
        flux, ivar = _spectrum(rng, wave, str(cls[i]), float(z[i]), float(m_z[i]))
        hdr = fits.Header()
        hdr.update({
            "TARGETID": int(ids[i]), "RA": float(ra[i]), "DEC": float(dec[i]),
            "Z": float(z[i]), "Z_ERR": 1e-4 * (1 + float(z[i])),
            "CRVAL1": float(wave[0]), "CDELT1": WAVE_STEP_A, "CRPIX1": 1.0,
            "CTYPE1": "WAVE", "CUNIT1": "Angstrom",
            "BUNIT": "1e-17 erg/(s cm2 Angstrom)", "INSTRUME": "DEMO",
        })
        path = out_dir / f"demospec_{int(ids[i])}.fits"
        fits.HDUList([
            fits.PrimaryHDU(flux, header=hdr),
            fits.ImageHDU(ivar, name="IVAR"),
        ]).writeto(path, overwrite=True)
        yield path


def _iter_cutout_files(
    out_dir: Path, rng: np.random.Generator, ids, ra, dec, mags: dict[str, np.ndarray],
) -> Iterator[Path]:
    from astropy.io import fits

    n = CUTOUT_SIZE
    yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2
    for i in range(len(ids)):
        size_pix = rng.uniform(1.2, 4.0)
        q = rng.uniform(0.4, 1.0)
        theta = rng.uniform(0, np.pi)
        xr = xx * np.cos(theta) + yy * np.sin(theta)
        yr = -xx * np.sin(theta) + yy * np.cos(theta)
        profile = np.exp(-0.5 * ((xr / size_pix) ** 2 + (yr / (q * size_pix)) ** 2))
        profile /= profile.sum()
        cube = np.empty((len(CUTOUT_BANDS), n, n), dtype=np.float32)
        for b, band in enumerate(CUTOUT_BANDS):
            total = 10 ** (-0.4 * (mags[band][i] - AB_ZP_NJY))
            cube[b] = total * profile + rng.normal(0, 0.02 * total / n + 5.0, (n, n))
        hdr = fits.Header()
        hdr.update({
            "OBJID": int(ids[i]), "RA": float(ra[i]), "DEC": float(dec[i]),
            "WCSAXES": 2, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
            "CRVAL1": float(ra[i]), "CRVAL2": float(dec[i]),
            "CRPIX1": (n + 1) / 2, "CRPIX2": (n + 1) / 2,
            "CD1_1": -CUTOUT_PIXSCALE_ARCSEC / 3600, "CD1_2": 0.0,
            "CD2_1": 0.0, "CD2_2": CUTOUT_PIXSCALE_ARCSEC / 3600,
            "BUNIT": "nJy",
        })
        path = out_dir / f"democut_{int(ids[i])}.fits"
        fits.PrimaryHDU(cube, header=hdr).writeto(path, overwrite=True)
        yield path


def write_demo_inputs(
    out_dir: Path | str, *, seed: int = 42, sizes: DemoSizes | None = None,
) -> DemoInputs:
    """Write the synthetic catalog, spectrum and cutout FITS files under *out_dir*."""
    sizes = sizes or DemoSizes()
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    (out / "spectra").mkdir(parents=True, exist_ok=True)
    (out / "cutouts").mkdir(parents=True, exist_ok=True)

    sky = _sources(rng, sizes.n_phot)
    n = sizes.n_phot
    phot_ids = np.arange(1_000_001, 1_000_001 + n, dtype=np.int64)
    phot_cols: dict[str, np.ndarray] = {
        "OBJECT_ID": phot_ids, "RA": sky["ra"], "DEC": sky["dec"], "CLASS": sky["cls"],
    }
    for band in PHOT_BANDS:
        mag, err = _mag_err(rng, sky["mags"][band], depth=24.5)
        phot_cols[f"MAG_{band.upper()}"] = mag.astype(np.float32)
        phot_cols[f"MAGERR_{band.upper()}"] = err.astype(np.float32)
    inputs = DemoInputs(
        root=out,
        phot_catalog=_write_table(out / f"{PHOT_SURVEY}.fits", phot_cols),
        spec_catalog=out / f"{SPEC_SURVEY}.fits",
        ir_catalog=out / f"{IR_SURVEY}.fits",
    )

    # Spectroscopic targets: every QSO, then the brightest remaining sources.
    order = np.lexsort((sky["mags"]["z"], sky["cls"] != "QSO"))
    sel = np.sort(order[: sizes.n_spec])
    jitter = 0.15 / 3600
    spec_ids = np.arange(39_000_000_001, 39_000_000_001 + sel.size, dtype=np.int64)
    spec_ra = sky["ra"][sel] + rng.normal(0, jitter, sel.size)
    spec_dec = sky["dec"][sel] + rng.normal(0, jitter, sel.size)
    _write_table(inputs.spec_catalog, {
        "TARGETID": spec_ids, "TARGET_RA": spec_ra, "TARGET_DEC": spec_dec,
        "Z": sky["z"][sel], "ZERR": 1e-4 * (1 + sky["z"][sel]),
        "SPECTYPE": sky["cls"][sel],
    })
    inputs.spectra = list(_iter_spectrum_files(
        out / "spectra", rng, spec_ids, spec_ra, spec_dec,
        sky["cls"][sel], sky["z"][sel], sky["mags"]["z"][sel],
    ))

    bright = np.sort(np.argsort(sky["mags"]["z"])[: sizes.n_cutouts])
    inputs.cutouts = list(_iter_cutout_files(
        out / "cutouts", rng, phot_ids[bright], sky["ra"][bright], sky["dec"][bright],
        {b: sky["mags"][b][bright] for b in CUTOUT_BANDS},
    ))

    detected = np.flatnonzero(sky["w1_ab"] < 21.0)
    ir_jitter = 0.5 / 3600
    extra = _sources(rng, sizes.n_ir_only) if sizes.n_ir_only else None
    ir_ra = sky["ra"][detected] + rng.normal(0, ir_jitter, detected.size)
    ir_dec = sky["dec"][detected] + rng.normal(0, ir_jitter, detected.size)
    w1_ab, w2_ab = sky["w1_ab"][detected], sky["w2_ab"][detected]
    if extra is not None:
        ir_ra = np.concatenate([ir_ra, extra["ra"]])
        ir_dec = np.concatenate([ir_dec, extra["dec"]])
        w1_ab = np.concatenate([w1_ab, rng.uniform(17.0, 20.5, sizes.n_ir_only)])
        w2_ab = np.concatenate([w2_ab, w1_ab[-sizes.n_ir_only:] - rng.uniform(0.0, 0.8, sizes.n_ir_only)])
    w1, w1_err = _mag_err(rng, w1_ab - W1_VEGA_TO_AB, depth=19.5)
    w2, w2_err = _mag_err(rng, w2_ab - W2_VEGA_TO_AB, depth=18.5)
    _write_table(inputs.ir_catalog, {
        "SOURCE_ID": np.arange(5_000_000_001, 5_000_000_001 + ir_ra.size, dtype=np.int64),
        "RA": ir_ra, "DEC": ir_dec,
        "W1MPRO": w1.astype(np.float32), "W1SIGMPRO": w1_err.astype(np.float32),
        "W2MPRO": w2.astype(np.float32), "W2SIGMPRO": w2_err.astype(np.float32),
    })
    return inputs


# ---------------------------------------------------------------------------
# Homogenization recipes
# ---------------------------------------------------------------------------


def _phot_rule(src: str, err: str, band: str, delta: float, system: str) -> dict:
    return {
        "source_column": src,
        "target_column": f"phot_ab_{band}",
        "uncertainty_column": err,
        "target_uncertainty_column": f"phot_ab_{band}_err",
        "native_system": system,
        "transform": {"type": "mag_offset", "delta": delta},
    }


def demo_recipes() -> dict[str, dict]:
    """Lake-local ``phot_ab_v1`` recipes for the demo surveys."""
    phot_band = {"g": "g", "z": "z", "y": "y", "j": "j", "h": "h"}
    return {
        PHOT_SURVEY: {
            "survey": PHOT_SURVEY, "version": 1,
            "catalog": {"phot_ab_v1": {"output_system": "AB_mag", "rules": [
                _phot_rule(f"MAG_{b.upper()}", f"MAGERR_{b.upper()}", phot_band[b], 0.0, "AB")
                for b in PHOT_BANDS
            ]}},
        },
        IR_SURVEY: {
            "survey": IR_SURVEY, "version": 1,
            "catalog": {"phot_ab_v1": {"output_system": "AB_mag", "rules": [
                _phot_rule("W1MPRO", "W1SIGMPRO", "w1", W1_VEGA_TO_AB, "Vega"),
                _phot_rule("W2MPRO", "W2SIGMPRO", "w2", W2_VEGA_TO_AB, "Vega"),
            ]}},
        },
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _ingest_modalities(lake: Path, inputs: DemoInputs) -> None:
    from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
    from data_lake.ingest.fits_to_zarr import ingest_cutouts_from_fits

    for p in inputs.spectra:
        ingest_spectra_from_fits(p, lake, SPEC_SURVEY, norder=NORDER, link_id_col="TARGETID", fmt="generic")
    for p in inputs.cutouts:
        ingest_cutouts_from_fits(
            p, lake, PHOT_SURVEY, norder=NORDER, link_id_col="OBJID",
            band_axis=0, band_names=list(CUTOUT_BANDS),
        )


def build_demo_lake(
    path: Path | str,
    *,
    seed: int = 42,
    sizes: DemoSizes | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a demo deployment at *path* and return its ``lake_config.toml``.

    Layout: ``<path>/lake_config.toml``, ``<path>/data/`` (the lake) and
    ``<path>/raw/`` (the synthetic FITS inputs).
    """
    from data_lake.admin.init_lake import init_lake
    from data_lake.config import LakeConfig
    from data_lake.ingest.fits_to_parquet import ingest_catalog
    from data_lake.ingest.update_catalog_indices import update_index_column_from_zarr_tiles
    from data_lake.lake_registry import refresh_lake_registry

    deployment = Path(path).expanduser().resolve()
    if deployment.exists():
        if not overwrite:
            raise FileExistsError(f"{deployment} exists (pass overwrite=True to rebuild)")
        shutil.rmtree(deployment)
    deployment.parent.mkdir(parents=True, exist_ok=True)
    init_lake(
        deployment.name, deployment.parent,
        description="Synthetic demo lake (dl-demo-lake)",
        norder=NORDER, ingest_token=DEMO_INGEST_TOKEN,
    )
    config_path = deployment / "lake_config.toml"
    lake = Path(LakeConfig.discover(config_path).lake.root)

    log.info("Writing synthetic inputs under %s", deployment / "raw")
    inputs = write_demo_inputs(deployment / "raw", seed=seed, sizes=sizes)

    log.info("Ingesting catalogs")
    for src, survey, ra, dec, link in (
        (inputs.phot_catalog, PHOT_SURVEY, "RA", "DEC", "OBJECT_ID"),
        (inputs.spec_catalog, SPEC_SURVEY, "TARGET_RA", "TARGET_DEC", "TARGETID"),
        (inputs.ir_catalog, IR_SURVEY, "RA", "DEC", "SOURCE_ID"),
    ):
        ingest_catalog(src, lake, survey, ra_col=ra, dec_col=dec, norder=NORDER, link_id_col=link)

    log.info("Ingesting %d spectra and %d cutouts", len(inputs.spectra), len(inputs.cutouts))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification.*")
        _ingest_modalities(lake, inputs)
    update_index_column_from_zarr_tiles(lake, SPEC_SURVEY, kind="spectrum", norder=NORDER)
    update_index_column_from_zarr_tiles(lake, PHOT_SURVEY, kind="cutout", norder=NORDER)

    recipe_dir = lake / "shared" / "registry" / "homogenize"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    for survey, recipe in demo_recipes().items():
        (recipe_dir / f"{survey}.json").write_text(json.dumps(recipe, indent=2) + "\n")

    refresh_lake_registry(lake)
    (deployment / _COMPLETE_MARKER).write_text(f"seed={seed}\n")
    log.info("Demo lake ready: %s", config_path)
    return config_path


def default_demo_dir() -> Path:
    """``$DATA_LAKE_DEMO_DIR`` when set, else ``./_demo_lake``."""
    return Path(os.environ.get(DEMO_DIR_ENV) or "_demo_lake").expanduser().resolve()


def ensure_demo_lake(path: Path | str | None = None, *, seed: int = 42) -> Path:
    """Return the demo ``lake_config.toml`` at *path*, building it on first use.

    A partially built directory (no completion marker) is rebuilt.
    """
    deployment = Path(path).expanduser().resolve() if path else default_demo_dir()
    config_path = deployment / "lake_config.toml"
    if (deployment / _COMPLETE_MARKER).is_file() and config_path.is_file():
        return config_path
    return build_demo_lake(deployment, seed=seed, overwrite=deployment.exists())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import click

    from data_lake.cli_utils import (
        configure_cli_logging,
        logging_options,
        resolve_log_level,
        validate_quiet_verbose,
    )

    @click.command("dl-demo-lake")
    @click.argument("path", type=click.Path(path_type=Path), required=False)
    @click.option("--overwrite", is_flag=True, help="Delete and rebuild an existing PATH.")
    @click.option("--seed", default=42, show_default=True, type=int, help="Random seed.")
    @click.option(
        "--n-sources", default=DemoSizes.n_phot, show_default=True, type=int,
        help=f"Rows in {PHOT_SURVEY}; the other surveys scale with it.",
    )
    @logging_options
    def cli(
        path: Path | None, overwrite: bool, seed: int, n_sources: int,
        quiet: bool, verbose: bool,
    ) -> None:
        """Build a small synthetic lake for tutorials (default PATH: ./_demo_lake)."""
        validate_quiet_verbose(quiet, verbose)
        configure_cli_logging(level=resolve_log_level(quiet=quiet, verbose=verbose))
        scale = n_sources / DemoSizes.n_phot
        try:
            sizes = DemoSizes(
                n_phot=n_sources,
                n_spec=max(1, round(DemoSizes.n_spec * scale)),
                n_cutouts=max(1, round(DemoSizes.n_cutouts * scale)),
                n_ir_only=round(DemoSizes.n_ir_only * scale),
            )
            config_path = build_demo_lake(
                path or default_demo_dir(), seed=seed, sizes=sizes, overwrite=overwrite,
            )
        except (FileExistsError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Demo lake ready. To use it:\n"
                   f"  export DATA_LAKE_CONFIG={config_path}\n"
                   f"  export LAKE_INGEST_TOKEN={DEMO_INGEST_TOKEN}\n"
                   f"  dl-describe-lake --count-total")

except ImportError:  # pragma: no cover
    cli = None  # type: ignore[assignment]
