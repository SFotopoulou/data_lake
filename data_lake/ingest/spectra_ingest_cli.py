"""``dl-ingest-spectra`` Click entry point."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np

from data_lake.cli_utils import (
    config_option,
    configure_cli_logging,
    fits_memmap_option,
    ingest_token_option,
    load_optional_config,
    logging_options,
    pick,
    require_ingest_permission,
    require_output_root,
    resolve_log_level,
    validate_quiet_verbose,
)
from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits

log = logging.getLogger(__name__)


@click.command("dl-ingest-spectra")
@click.argument("source_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_root", type=click.Path(path_type=Path), required=False)
@config_option
@ingest_token_option
@click.option("--survey", "survey_name", required=True, help="Short survey name.")
@click.option("--ra-col", default="RA", show_default=True)
@click.option("--dec-col", default="DEC", show_default=True)
@click.option(
    "--link-id-col",
    default=None,
    help=(
        "Object ID for SDSS/DESI/generic/spPlate ingest: FITS header keyword "
        "(generic/SDSS) or DESI link spec.  DESI coadds default to composite "
        "TARGETID,SURVEY,PROGRAM (fibermap TARGETID + PRIMARY SURVEY/PROGRAM, "
        "stripped); pass TARGETID alone for the legacy native-int path. "
        "Ignored by format-specific readers (2df, 6df, OzDES, …) which resolve "
        "IDs internally."
    ),
)
@click.option("--norder", default=None, type=int,
              help="HEALPix order (overrides config; default 5).")
@click.option("--wavelength-mode",
              type=click.Choice(["shared", "per_source"]),
              default=None,
              help="Wavelength storage mode (overrides config; default 'shared').")
@click.option("--mask-dtype",
              type=click.Choice(["uint8", "uint16"]),
              default=None,
              help="Mask dtype (overrides config; default 'uint8').")
@click.option("--fmt", default=None,
              type=click.Choice(
                  [
                      "sdss_boss",
                      "sdss_spplate",
                      "desi_coadd",
                      "generic",
                      "2df",
                      "6df",
                      "gama",
                      "wig",
                      "ozdes",
                      "zcosmos",
                      "vandels",
                      "vipers",
                      "vuds",
                      "vvds",
                  ],
              ),
              help="Force input format (auto-detected by default).")
@click.option(
    "--specobj-lookup",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Parquet/CSV: survey, PLATE, MJD, FIBERID, SPECOBJID (spPlate only).",
)
@click.option(
    "--specobj-lookup-from-catalog/--no-specobj-lookup-from-catalog",
    default=False,
    show_default=True,
    help="Resolve SPECOBJID from catalogs/<survey>/ instead of --specobj-lookup.",
)
@click.option(
    "--specobj-lookup-survey",
    default=None,
    help="Sidecar survey filter when the lookup file has no SURVEY column.",
)
@click.option(
    "--specobj-lookup-from-plate/--no-specobj-lookup-from-plate",
    default=False,
    show_default=True,
    help="Synthesize SPECOBJID from spPlate PLATE/MJD/FIBERID/RUN2D (no sidecar).",
)
@click.option(
    "--specobj-id-layout",
    type=click.Choice(["auto", "dr7", "dr8plus"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="specObjID bit packing for --specobj-lookup-from-plate (DR7 vs DR8+).",
)
@click.option("--on-length-mismatch",
              type=click.Choice(["error", "pad", "truncate"]),
              default="error", show_default=True)
@click.option(
    "--on-duplicate",
    type=click.Choice(["append", "error", "skip"]),
    default="skip",
    show_default=True,
    help="If a source_id already exists in a tile Zarr: skip (default), raise, or append.",
)
@click.option("--with-resolution/--no-with-resolution", default=None,
              help=(
                  "Store the DESI banded resolution matrix (n_diag × N_pix per source). "
                  "Requires DESI coadd input and wavelength-mode=shared. "
                  "Overrides config; default false."
              ))
@click.option(
    "--update-catalog/--no-update-catalog", default=True, show_default=True,
    help="Patch _spectrum_index in the Parquet catalog after ingest "
         "(skipped silently if no catalog exists for this survey).",
)
@fits_memmap_option
@logging_options
def cli(
    source_path: Path,
    output_root: Path | None,
    config_path: Path | None,
    ingest_token: str | None,
    survey_name: str,
    ra_col: str,
    dec_col: str,
    link_id_col: str | None,
    norder: int | None,
    wavelength_mode: str | None,
    mask_dtype: str | None,
    fmt: str | None,
    specobj_lookup: Path | None,
    specobj_lookup_from_catalog: bool,
    specobj_lookup_survey: str | None,
    specobj_lookup_from_plate: bool,
    specobj_id_layout: str,
    on_length_mismatch: str,
    on_duplicate: str,
    with_resolution: bool | None,
    update_catalog: bool,
    fits_memmap: str,
    log_file: Path | None,
    quiet: bool,
    verbose: bool,
) -> None:
    """Ingest 1-D FITS spectra into sharded Zarr v3 stacks.

    OUTPUT_ROOT is optional when a lake config is available (via
    --config or $DATA_LAKE_CONFIG); in that case it defaults to
    ``<lake.root>/<paths.spectra>``.
    """
    validate_quiet_verbose(quiet, verbose)
    cfg = load_optional_config(config_path)
    configure_cli_logging(
        level=resolve_log_level(quiet=quiet, verbose=verbose,
                                config_level=cfg.ingest.log_level if cfg else None),
        log_file=log_file,
        quiet=quiet,
    )
    require_ingest_permission(cfg, ingest_token)
    resolved_output = require_output_root(output_root, cfg, kind="spectra")
    resolved_norder = pick(norder, cfg.partitioning.hats_order if cfg else None, 5)

    index_map = ingest_spectra_from_fits(
        source_path=source_path,
        output_root=resolved_output,
        survey_name=survey_name,
        ra_col=ra_col,
        dec_col=dec_col,
        link_id_col=link_id_col,
        norder=resolved_norder,
        wavelength_mode=pick(wavelength_mode,
                             cfg.defaults.wavelength_mode if cfg else None,
                             "shared"),
        mask_dtype=np.dtype(pick(mask_dtype,
                                 cfg.defaults.mask_dtype if cfg else None,
                                 "uint8")),
        fmt=fmt,
        on_length_mismatch=on_length_mismatch,
        on_duplicate_source_id=on_duplicate,  # type: ignore[arg-type]
        with_resolution=pick(with_resolution,
                             cfg.defaults.with_resolution if cfg else None,
                             False),
        specobj_lookup=specobj_lookup,
        specobj_lookup_from_catalog=specobj_lookup_from_catalog,
        specobj_lookup_survey=specobj_lookup_survey,
        specobj_lookup_from_plate=specobj_lookup_from_plate,
        specobj_id_layout=specobj_id_layout.lower(),
        fits_memmap=fits_memmap.lower(),
    )

    if update_catalog and index_map:
        try:
            from data_lake.ingest.update_catalog_indices import update_index_column
            n_modified = update_index_column(
                lake_root=resolved_output,
                survey_name=survey_name,
                source_id_to_index=index_map,
                kind="spectrum",
                norder=resolved_norder,
                link_id_col=None,
            )
            click.echo(f"Patched _spectrum_index in {n_modified} catalog tile(s).")
        except FileNotFoundError:
            log.info(
                "No catalog found for survey %r — skipping _spectrum_index patch.",
                survey_name,
            )

