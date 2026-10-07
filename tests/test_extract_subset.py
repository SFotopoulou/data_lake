"""
Tests for SpectrumAccessor.extract_subset_to_zarr.

We build a tiny synthetic spectrum lake with two HEALPix tiles and known
source_ids, then verify that extract_subset_to_zarr:

* writes exactly the requested-and-present sources
* preserves flux/ivar/mask/wavelength bit-for-bit
* propagates per-source redshift into the new ``redshift`` 1-D array
* reports missing IDs without crashing (default ``missing='skip'``)
* honours ``missing='error'`` strictly
* refuses to overwrite by default and obeys ``overwrite=True``

The synthetic data is built by going through the public ingest helpers, so
this test also exercises the tile-creation path end-to-end without needing
desispec or real FITS files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data_lake.ingest.fits_to_spectra_zarr import (
    _META_DTYPE,
    SpectrumRecord,
    _meta_to_bytes,
    _open_or_create_spectrum_tile,
    _write_spectrum_info,
)
from data_lake.ingest.fits_to_parquet import (
    LAKE_JOIN_ID_COLUMN,
    assign_healpix,
    healpix_dir,
)


# ---------------------------------------------------------------------------
# Synthetic lake fixture
# ---------------------------------------------------------------------------

N_PIX = 32
NORDER = 5
SURVEY = "synthetic"

# (source_id, ra, dec, z) — chosen so the two tiles land at different npix
SOURCES = [
    (101, 30.0,  +5.0, 0.10),
    (102, 30.1,  +5.1, 0.20),
    (103, 30.2,  +5.2, 0.30),
    (201, 210.0, -10.0, 1.10),
    (202, 210.1, -10.1, 1.20),
]

# Per-source wavelength survey constants
PS_SURVEY = "ps_synthetic"
PS_SOURCES = [
    (301, 30.0,  +5.0, 0.31),
    (302, 30.1,  +5.1, 0.32),
    (401, 210.0, -10.0, 1.41),
    (402, 210.1, -10.1, 1.42),
]


def _make_record(sid: int, ra: float, dec: float, z: float) -> SpectrumRecord:
    """Build a synthetic record whose flux/ivar/mask encode its source_id.

    This makes round-trip verification trivial: row in the output Zarr is
    correct iff ``flux[row, :] == sid`` everywhere.
    """
    flux = np.full(N_PIX, sid, dtype=np.float32)
    ivar = np.full(N_PIX, 1.0 / (sid + 1), dtype=np.float32)
    mask = np.full(N_PIX, sid % 7, dtype=np.uint8)
    return SpectrumRecord(
        source_id=sid,
        ra=ra,
        dec=dec,
        flux=flux,
        ivar=ivar,
        mask=mask,
        wavelength=np.linspace(3600.0, 9800.0, N_PIX),
        meta={
            "z": z, "z_err": 0.001, "snr": 10.0,
            "exptime": 1000.0, "R": 3000.0, "instr": "TEST",
        },
    )


def _ingest_synthetic_lake(lake_root: Path) -> dict[int, int]:
    """Populate <lake_root>/spectra/<SURVEY>/... with the records above.

    Mirrors the relevant logic from ingest_spectra_from_fits but bypasses
    FITS I/O so the test runs without desispec or external files.
    """
    records = [_make_record(*row) for row in SOURCES]
    wave = records[0].wavelength
    wcs_attrs = {
        "ctype": "WAVE",
        "crval": float(wave[0]),
        "cdelt": float(wave[1] - wave[0]),
        "crpix": 1.0,
        "unit": "Angstrom",
        "air_or_vacuum": "vacuum",
        "n_pix": N_PIX,
    }

    # Group by HEALPix tile
    tile_groups: dict[int, list[SpectrumRecord]] = {}
    for rec in records:
        pix = int(assign_healpix(np.array([rec.ra]), np.array([rec.dec]), NORDER)[0])
        tile_groups.setdefault(pix, []).append(rec)

    survey_root = lake_root / "spectra" / SURVEY
    index_map: dict[int, int] = {}

    for npix, recs in tile_groups.items():
        tile_dir = survey_root / healpix_dir(NORDER, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path = tile_dir / f"Npix={npix}.zarr"

        root = _open_or_create_spectrum_tile(
            tile_path, N_PIX, "shared", np.dtype(np.uint8), wcs_attrs,
        )

        start_idx = root["flux"].shape[0]
        root["flux"].append(np.stack([r.flux for r in recs]))
        root["ivar"].append(np.stack([r.ivar for r in recs]))
        root["mask"].append(np.stack([r.mask for r in recs]))
        from data_lake.ingest.zarr_ids import zarr_join_array

        zarr_join_array(root).append(np.array([r.source_id for r in recs], dtype=np.int64))

        meta_buf = np.frombuffer(
            b"".join(_meta_to_bytes(r.meta) for r in recs),
            dtype="|V" + str(_META_DTYPE.itemsize),
        )
        root["meta"].append(meta_buf)

        if start_idx == 0:
            root["wavelength"][:] = wave.astype(np.float64)

        for i, r in enumerate(recs):
            index_map[r.source_id] = start_idx + i

    _write_spectrum_info(
        survey_root, SURVEY, NORDER, N_PIX,
        "shared", "uint8", wcs_attrs,
    )

    # Verify our setup actually used multiple tiles (otherwise the test is
    # weaker than intended).
    assert len(tile_groups) >= 2, (
        f"Synthetic SOURCES collapsed into {len(tile_groups)} tile(s); "
        "spread the RA/Dec further apart."
    )

    return index_map


@pytest.fixture
def synthetic_lake(tmp_path: Path):
    _ingest_synthetic_lake(tmp_path)
    return tmp_path


def _ingest_per_source_synthetic_lake(lake_root: Path) -> None:
    """Populate a per_source wavelength survey for round-trip tests.

    Each source receives a distinct wavelength grid (base grid shifted by
    ``source_id``), so verifying round-trip fidelity requires per-row matching.
    """
    from data_lake.ingest.zarr_ids import zarr_join_array

    base_wave = np.linspace(3600.0, 9800.0, N_PIX)
    wcs_attrs = {
        "ctype": "WAVE",
        "crval": float(base_wave[0]),
        "cdelt": float(base_wave[1] - base_wave[0]),
        "crpix": 1.0,
        "unit": "Angstrom",
        "air_or_vacuum": "vacuum",
        "n_pix": N_PIX,
        "wavelength_mode": "per_source",
    }

    tile_groups: dict[int, list] = {}
    for sid, ra, dec, z in PS_SOURCES:
        pix = int(assign_healpix(np.array([ra]), np.array([dec]), NORDER)[0])
        tile_groups.setdefault(pix, []).append((sid, ra, dec, z))

    survey_root = lake_root / "spectra" / PS_SURVEY
    for npix, recs in tile_groups.items():
        tile_dir = survey_root / healpix_dir(NORDER, npix)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path = tile_dir / f"Npix={npix}.zarr"

        root = _open_or_create_spectrum_tile(
            tile_path, N_PIX, "per_source", np.dtype(np.uint8), wcs_attrs,
        )

        flux_stack = np.stack([
            np.full(N_PIX, sid, dtype=np.float32) for sid, *_ in recs
        ])
        ivar_stack = np.stack([
            np.full(N_PIX, 1.0 / (sid + 1), dtype=np.float32) for sid, *_ in recs
        ])
        mask_stack = np.stack([
            np.full(N_PIX, sid % 7, dtype=np.uint8) for sid, *_ in recs
        ])
        # Each source gets a unique wavelength: base grid + source_id offset.
        wave_stack = np.stack([
            (base_wave + sid).astype(np.float32) for sid, *_ in recs
        ])
        sids_arr = np.array([sid for sid, *_ in recs], dtype=np.int64)
        meta_buf = np.frombuffer(
            b"".join(
                _meta_to_bytes({
                    "z": z, "z_err": 0.001, "snr": 10.0,
                    "exptime": 1000.0, "R": 3000.0, "instr": "TEST",
                })
                for _, _, _, z in recs
            ),
            dtype="|V" + str(_META_DTYPE.itemsize),
        )

        root["flux"].append(flux_stack)
        root["ivar"].append(ivar_stack)
        root["mask"].append(mask_stack)
        root["wavelength"].append(wave_stack)
        zarr_join_array(root).append(sids_arr)
        root["meta"].append(meta_buf)

    _write_spectrum_info(
        survey_root, PS_SURVEY, NORDER, N_PIX,
        "per_source", "uint8", wcs_attrs,
    )


@pytest.fixture
def per_source_lake(tmp_path: Path):
    _ingest_per_source_synthetic_lake(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractSubsetToZarr:
    def test_round_trip_subset(self, synthetic_lake: Path):
        """Extracted Zarr matches the source lake for the requested IDs."""
        import zarr

        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)

        requested = [101, 103, 201, 202]  # 4 of the 5 sources, across both tiles
        out_path = synthetic_lake / "subset.zarr"
        result = acc.extract_subset_to_zarr(
            source_ids=requested,
            output_zarr=out_path,
            show_progress=False,
        )

        assert result["n_requested"] == 4
        assert result["n_written"] == 4
        assert result["missing_ids"] == []
        assert set(result["id_to_row"].keys()) == set(requested)

        # Open output and verify content
        out_root = zarr.open_group(store=zarr.storage.LocalStore(str(out_path)),
                                    mode="r", zarr_format=3)
        flux = np.asarray(out_root["flux"][:])
        ivar = np.asarray(out_root["ivar"][:])
        mask = np.asarray(out_root["mask"][:])
        sids = np.asarray(out_root[LAKE_JOIN_ID_COLUMN][:])
        z    = np.asarray(out_root["redshift"][:])
        wave = np.asarray(out_root["wavelength"][:])

        assert flux.shape == (4, N_PIX)
        assert ivar.shape == (4, N_PIX)
        assert mask.shape == (4, N_PIX)
        assert wave.shape == (N_PIX,)

        # Round-trip via id_to_row: for each requested sid, the row in the
        # output should be all-equal to that sid (our synthetic flux encoding).
        expected_z = {sid: z_val for sid, _, _, z_val in SOURCES}
        for sid in requested:
            row = result["id_to_row"][sid]
            assert np.all(flux[row] == sid), f"flux mismatch for {sid}"
            np.testing.assert_allclose(ivar[row], 1.0 / (sid + 1))
            assert np.all(mask[row] == sid % 7)
            assert sids[row] == sid
            np.testing.assert_allclose(z[row], expected_z[sid], atol=1e-6)

        # Wavelength preserved exactly
        np.testing.assert_array_equal(
            wave, np.linspace(3600.0, 9800.0, N_PIX),
        )

        # Group attrs
        attrs = dict(out_root.attrs)
        assert attrs["source_survey"] == SURVEY
        assert attrs["wavelength_mode"] == "shared"
        assert attrs["n_sources"] == 4
        assert attrs["n_requested"] == 4
        assert attrs["n_missing"] == 0
        assert attrs["n_pix"] == N_PIX

    def test_missing_skip_reports_absent_ids(self, synthetic_lake: Path):
        """missing='skip' returns a list of IDs that weren't in the lake."""
        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        result = acc.extract_subset_to_zarr(
            source_ids=[101, 999_999, 202, 888_888],  # 2 present, 2 absent
            output_zarr=synthetic_lake / "subset_skip.zarr",
            missing="skip",
            show_progress=False,
        )
        assert result["n_written"] == 2
        assert sorted(result["missing_ids"]) == [888_888, 999_999]

    def test_missing_error_raises(self, synthetic_lake: Path):
        """missing='error' raises KeyError when any ID is absent."""
        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        with pytest.raises(KeyError, match="not found"):
            acc.extract_subset_to_zarr(
                source_ids=[101, 999_999],
                output_zarr=synthetic_lake / "subset_err.zarr",
                missing="error",
                show_progress=False,
            )

    def test_overwrite_guard(self, synthetic_lake: Path):
        """Refuse to clobber an existing path without overwrite=True."""
        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        out = synthetic_lake / "subset_guard.zarr"
        acc.extract_subset_to_zarr([101], out, show_progress=False)

        with pytest.raises(FileExistsError):
            acc.extract_subset_to_zarr([101], out, show_progress=False)

        # overwrite=True succeeds
        acc.extract_subset_to_zarr([101, 103], out, overwrite=True, show_progress=False)

    def test_all_missing_raises_valueerror(self, synthetic_lake: Path):
        """If none of the requested IDs are present, raise ValueError."""
        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        with pytest.raises(ValueError, match="None of the requested"):
            acc.extract_subset_to_zarr(
                source_ids=[10**12, 10**12 + 1],
                output_zarr=synthetic_lake / "subset_none.zarr",
                missing="skip",
                show_progress=False,
            )

    def test_extract_subset_zarr_per_source(self, per_source_lake: Path):
        """Zarr extract of a per_source survey round-trips wavelength per row."""
        import zarr

        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(per_source_lake, PS_SURVEY)
        requested = [sid for sid, *_ in PS_SOURCES]
        out = per_source_lake / "ps_subset.zarr"
        result = acc.extract_subset_to_zarr(
            source_ids=requested, output_zarr=out, show_progress=False,
        )
        assert result["n_written"] == len(requested)
        assert result["missing_ids"] == []

        out_root = zarr.open_group(
            store=zarr.storage.LocalStore(str(out)), mode="r", zarr_format=3,
        )
        assert out_root.attrs["wavelength_mode"] == "per_source"
        wave_arr = np.asarray(out_root["wavelength"][:])
        assert wave_arr.ndim == 2
        assert wave_arr.shape == (len(requested), N_PIX)
        flux_arr = np.asarray(out_root["flux"][:])

        id_to_row = result["id_to_row"]
        for sid, *_ in PS_SOURCES:
            row = id_to_row[sid]
            np.testing.assert_allclose(flux_arr[row], sid, err_msg=f"flux mismatch for {sid}")
            # Each source has a unique wavelength offset (wave + sid); verify first pixel.
            expected_wave0 = float(np.linspace(3600.0, 9800.0, N_PIX)[0]) + sid
            assert abs(float(wave_arr[row, 0]) - expected_wave0) < 1e-3, (
                f"wavelength mismatch for {sid}"
            )

    def test_extract_subset_parquet_per_source(self, per_source_lake: Path):
        """Parquet extract adds a per-row wavelength list column."""
        import pyarrow.parquet as pq

        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(per_source_lake, PS_SURVEY)
        requested = [sid for sid, *_ in PS_SOURCES]
        out = per_source_lake / "ps_subset.parquet"
        result = acc.extract_subset_to_parquet(
            source_ids=requested, output_parquet=out, show_progress=False,
        )
        assert result["n_written"] == len(requested)

        tbl = pq.read_table(str(out))
        assert "wavelength" in tbl.schema.names
        meta = tbl.schema.metadata
        assert meta[b"wavelength_mode"] == b"per_source"

        id_to_row = result["id_to_row"]
        for sid, *_ in PS_SOURCES:
            row = id_to_row[sid]
            wave_row = np.asarray(tbl.column("wavelength")[row].as_py(), dtype=np.float32)
            expected_wave0 = float(np.linspace(3600.0, 9800.0, N_PIX)[0]) + sid
            assert abs(float(wave_row[0]) - expected_wave0) < 1e-3

    def test_fits_catalog_per_source_rejected(self, per_source_lake: Path):
        """FITS catalog layout must raise ValueError for per_source surveys."""
        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(per_source_lake, PS_SURVEY)
        with pytest.raises(ValueError, match="per-file"):
            acc.extract_subset_to_fits_catalog(
                source_ids=[sid for sid, *_ in PS_SOURCES],
                output_fits=per_source_lake / "noop.fits",
                show_progress=False,
            )

    def test_extract_subset_parquet(self, synthetic_lake: Path):
        """Parquet export preserves flux encoding per source_id."""
        import pyarrow.parquet as pq

        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        out = synthetic_lake / "subset.parquet"
        result = acc.extract_subset_to_parquet(
            source_ids=[101, 103, 201],
            output_parquet=out,
            show_progress=False,
        )
        assert result["format"] == "parquet"
        assert result["n_written"] == 3
        tbl = pq.read_table(str(out))
        assert tbl.num_rows == 3
        flux0 = np.asarray(tbl.column("flux")[0].as_py())
        sid0 = int(tbl.column("source_id")[0].as_py())
        assert np.all(flux0 == sid0)

    def test_extract_subset_hdf5(self, synthetic_lake: Path):
        """HDF5 export preserves flux encoding per source_id."""
        import h5py

        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        out = synthetic_lake / "subset.h5"
        result = acc.extract_subset_to_hdf5(
            source_ids=[101, 103, 201],
            output_hdf5=out,
            show_progress=False,
        )
        assert result["format"] == "hdf5"
        assert result["n_written"] == 3
        with h5py.File(out, "r") as f:
            assert f.attrs["source_survey"] == SURVEY
            assert f.attrs["wavelength_mode"] == "shared"
            assert f["flux"].shape == (3, N_PIX)
            sids = np.asarray(f[LAKE_JOIN_ID_COLUMN][:])
            flux0 = np.asarray(f["flux"][0, :])
            assert int(sids[0]) == 101
            assert np.all(flux0 == 101)

    def test_extract_subset_fits(self, synthetic_lake: Path):
        """FITS export writes one file per spectrum."""
        from astropy.io import fits

        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        out_dir = synthetic_lake / "subset_fits"
        result = acc.extract_subset_to_fits(
            source_ids=[101, 201],
            output=out_dir,
            show_progress=False,
        )
        assert result["format"] == "fits"
        assert result["n_written"] == 2
        assert (out_dir / "spec_101.fits").is_file()
        assert (out_dir / "spec_201.fits").is_file()
        with fits.open(out_dir / "spec_101.fits") as hdul:
            flux = np.array(hdul[0].data, dtype=np.float32)
            assert np.all(flux == 101)

    def test_redshift_from_catalog_not_zarr_meta(self, synthetic_lake: Path) -> None:
        """When a catalog is present, output redshift uses catalog Z, not tile meta."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        from data_lake.io.catalog import CatalogAccessor
        from data_lake.io.spectra import SpectrumAccessor

        catalog_z = {101: 1.11, 103: 1.33, 201: 2.01}
        cat_root = synthetic_lake / "catalogs" / SURVEY
        by_tile: dict[int, list[tuple[int, float]]] = {}
        for sid, z_cat in catalog_z.items():
            _, ra, dec, _ = next(row for row in SOURCES if row[0] == sid)
            npix = int(assign_healpix(np.array([ra]), np.array([dec]), NORDER)[0])
            by_tile.setdefault(npix, []).append((sid, z_cat))
        for npix, rows in by_tile.items():
            tile_dir = cat_root / healpix_dir(NORDER, npix)
            tile_dir.mkdir(parents=True, exist_ok=True)
            sids, zs = zip(*rows)
            pq.write_table(
                pa.table({
                    "TARGETID": pa.array(sids, type=pa.int64()),
                    LAKE_JOIN_ID_COLUMN: pa.array(sids, type=pa.int64()),
                    "Z": pa.array(zs, type=pa.float64()),
                    f"_healpix_norder{NORDER}": pa.array([npix] * len(sids), type=pa.int64()),
                    "_spectrum_index": pa.array(range(len(sids)), type=pa.int64()),
                }),
                tile_dir / f"Npix={npix}.parquet",
            )
        (cat_root / "catalog_info.json").write_text(
            json.dumps({
                "hats_order": 5,
                "link_id_mode": "column:TARGETID",
                "link_id_column": LAKE_JOIN_ID_COLUMN,
                "native_id_column": "TARGETID",
            })
        )

        cat = CatalogAccessor(synthetic_lake, SURVEY)
        acc = SpectrumAccessor(synthetic_lake, SURVEY, catalog_accessor=cat)
        result = acc.extract_subset_to_zarr(
            source_ids=list(catalog_z.keys()),
            output_zarr=synthetic_lake / "subset_cat_z.zarr",
            show_progress=False,
        )
        import zarr
        out = zarr.open_group(
            store=zarr.storage.LocalStore(str(result["output_zarr"])),
            mode="r",
            zarr_format=3,
        )
        z_out = np.asarray(out["redshift"][:])
        for sid, z_exp in catalog_z.items():
            row = result["id_to_row"][sid]
            assert abs(z_out[row] - z_exp) < 1e-5, f"sid {sid}: catalog Z expected"
            z_meta = next(z for s, _, _, z in SOURCES if s == sid)
            assert abs(z_out[row] - z_meta) > 0.01, f"sid {sid}: should not use tile meta"

    def test_extract_subset_fits_catalog(self, synthetic_lake: Path):
        """FITS catalog layout: one file, all spectra in SPECTRA BINTABLE."""
        from astropy.io import fits

        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        out = synthetic_lake / "subset_catalog.fits"
        result = acc.extract_subset_to_fits_catalog(
            source_ids=[101, 103, 201],
            output_fits=out,
            show_progress=False,
        )
        assert result["fits_layout"] == "catalog"
        assert result["n_written"] == 3
        assert out.is_file()
        with fits.open(out) as hdul:
            assert hdul["SPECTRA"].data["TARGETID"][0] == 101
            flux_row = np.array(hdul["SPECTRA"].data["FLUX"][0], dtype=np.float32)
            assert np.all(flux_row == 101)
            wave = np.array(hdul["WAVELENGTH"].data).ravel()
            np.testing.assert_allclose(wave, np.linspace(3600.0, 9800.0, N_PIX))

    def test_extract_subset_cli_format_parquet(self, synthetic_lake: Path):
        from click.testing import CliRunner

        from data_lake.export.spectra_subset import cli

        runner = CliRunner()
        ids_file = synthetic_lake / "ids.txt"
        ids_file.write_text("101\n103\n")
        out = synthetic_lake / "cli_subset.parquet"
        result = runner.invoke(
            cli,
            [
                "--survey", SURVEY,
                "--target-list", str(ids_file),
                "--lake-root", str(synthetic_lake),
                "--format", "parquet",
                "--output", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.is_file()

    def test_fits_catalog_rejects_directory_output(self, synthetic_lake: Path) -> None:
        from click.testing import CliRunner

        from data_lake.export.spectra_subset import cli

        runner = CliRunner()
        ids_file = synthetic_lake / "ids2.txt"
        ids_file.write_text("101\n")
        result = runner.invoke(
            cli,
            [
                "--survey", SURVEY,
                "--target-list", str(ids_file),
                "--lake-root", str(synthetic_lake),
                "--format", "fits",
                "--fits-layout", "catalog",
                "--output", str(synthetic_lake / "qso_fits_dir"),
            ],
        )
        assert result.exit_code != 0
        assert "must be a .fits file" in (result.output or str(result.exception))

    def test_deduplicates_input(self, synthetic_lake: Path):
        """Duplicate requested IDs collapse to one output row."""
        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        result = acc.extract_subset_to_zarr(
            source_ids=[101, 101, 103, 103, 101],
            output_zarr=synthetic_lake / "subset_dedup.zarr",
            show_progress=False,
        )
        assert result["n_requested"] == 2  # unique
        assert result["n_written"] == 2


class TestExtractSubsetFluxCalibration:
    def test_explicit_flux_scale_zarr(self, synthetic_lake: Path):
        """flux_scale scales flux and ivar in exported Zarr."""
        import zarr

        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        requested = [101, 103]
        scale = 0.5
        out_path = synthetic_lake / "subset_scaled.zarr"
        result = acc.extract_subset_to_zarr(
            source_ids=requested,
            output_zarr=out_path,
            show_progress=False,
            flux_scale=scale,
        )
        assert result["flux_scale"] == scale

        out_root = zarr.open_group(
            store=zarr.storage.LocalStore(str(out_path)), mode="r", zarr_format=3,
        )
        assert out_root.attrs.get("flux_scale") == scale
        assert out_root.attrs.get("flux_calibrated") is True

        flux = np.asarray(out_root["flux"][:])
        ivar = np.asarray(out_root["ivar"][:])
        for sid in requested:
            row = result["id_to_row"][sid]
            np.testing.assert_allclose(flux[row], sid * scale)
            np.testing.assert_allclose(ivar[row], (1.0 / (sid + 1)) / (scale * scale))

    def test_cli_apply_survey_calibration(self, synthetic_lake: Path, tmp_path: Path):
        """CLI --flux-scale applies calibration and writes sidecar."""
        from click.testing import CliRunner

        from data_lake.export.spectra_subset import cli

        ids_file = synthetic_lake / "ids_scaled.txt"
        ids_file.write_text("101\n103\n")

        runner = CliRunner()
        out_zarr = synthetic_lake / "cli_scaled.zarr"
        result = runner.invoke(
            cli,
            [
                "--survey", SURVEY,
                "--target-list", str(ids_file),
                "--lake-root", str(synthetic_lake),
                "--format", "zarr",
                "--output", str(out_zarr),
                "--flux-scale", "0.25",
            ],
        )
        assert result.exit_code == 0, result.output
        sidecar = synthetic_lake / "cli_scaled.calibration.json"
        assert sidecar.is_file()
        data = json.loads(sidecar.read_text())
        assert data["flux_scale"] == 0.25

    def test_fits_catalog_chunked_merge_and_cleanup(self, synthetic_lake: Path):
        """Tiny fits_chunk_rows writes parts, merges, deletes intermediates."""
        from astropy.io import fits

        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        out = synthetic_lake / "subset_chunked.fits"
        result = acc.extract_subset_to_fits_catalog(
            source_ids=[101, 103, 201],
            output_fits=out,
            show_progress=False,
            overwrite=True,
            fits_chunk_rows=2,
            confirm=True,
        )
        assert result["n_written"] == 3
        assert result["n_parts"] == 2
        assert out.is_file()
        assert not list(synthetic_lake.glob("subset_chunked_part*.fits"))
        with fits.open(out) as hdul:
            assert hdul["SPECTRA"].header["NAXIS2"] == 3
            assert set(hdul["SPECTRA"].data["TARGETID"].tolist()) == {101, 103, 201}

    def test_fits_catalog_resume_skips_completed_part(self, synthetic_lake: Path):
        from astropy.io import fits

        from data_lake.export.to_spectrum_fits import _checkpoint_path, _part_path
        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        out = synthetic_lake / "subset_resume.fits"
        acc.extract_subset_to_fits_catalog(
            source_ids=[101, 103, 201],
            output_fits=out,
            show_progress=False,
            overwrite=True,
            fits_chunk_rows=2,
            confirm=True,
            keep_part_files=True,
        )
        part0 = _part_path(out, 0)
        part1 = _part_path(out, 1)
        assert part0.is_file() and part1.is_file()
        mtime0 = part0.stat().st_mtime_ns
        out.unlink()
        part1.unlink()
        ckpt = json.loads(_checkpoint_path(out).read_text())
        ckpt["completed_parts"] = [0]
        ckpt["part_nrows"] = [2, 0]
        ckpt["merge_done"] = False
        _checkpoint_path(out).write_text(json.dumps(ckpt, indent=2))

        acc.extract_subset_to_fits_catalog(
            source_ids=[101, 103, 201],
            output_fits=out,
            show_progress=False,
            overwrite=False,
            fits_chunk_rows=2,
            confirm=True,
            keep_part_files=True,
        )
        assert part0.stat().st_mtime_ns == mtime0
        assert out.is_file()
        with fits.open(out) as hdul:
            assert hdul["SPECTRA"].header["NAXIS2"] == 3

    def test_fits_catalog_refuses_too_many_parts(self, synthetic_lake: Path):
        from data_lake.export.to_spectrum_fits import validate_fits_part_count

        with pytest.raises(ValueError, match="100"):
            validate_fits_part_count(10_000, 1, force_many_fits_parts=False)
        with pytest.raises(ValueError, match="hard limit"):
            validate_fits_part_count(20_000, 1, force_many_fits_parts=True)

    def test_write_spectra_catalog_fits_rejects_oversized(self, tmp_path: Path, monkeypatch):
        from data_lake.export import to_spectrum_fits as mod

        monkeypatch.setattr(mod, "max_bintable_rows", lambda *a, **k: 2)
        n_pix = 4
        n_spec = 3
        with pytest.raises(ValueError, match="4 GiB"):
            mod.write_spectra_catalog_fits(
                tmp_path / "big.fits",
                source_id=np.arange(n_spec, dtype=np.int64),
                flux=np.zeros((n_spec, n_pix), dtype=np.float32),
                ivar=np.zeros((n_spec, n_pix), dtype=np.float32),
                mask=np.zeros((n_spec, n_pix), dtype=np.uint8),
                wavelength=np.linspace(1.0, 2.0, n_pix),
            )


# ---------------------------------------------------------------------------
# Tests for bounded open-tile LRU and close()
# ---------------------------------------------------------------------------


class TestOpenTileLRU:
    """SpectrumTileStore.close and SpectrumAccessor LRU / context-manager."""

    def test_tile_store_close_clears_root(self, synthetic_lake: Path):
        """close() sets _root to None and is idempotent."""
        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        tiles = list(acc.available_tiles())
        assert tiles, "synthetic lake has no tiles"
        npix = tiles[0]

        store = acc._get_tile_store(npix)
        # Force open
        _ = store._open()
        assert store._root is not None

        # First close
        store.close()
        assert store._root is None

        # Second close is idempotent
        store.close()
        assert store._root is None

    def test_accessor_lru_evicts_oldest(self, synthetic_lake: Path):
        """With max_open_tiles=1, each new tile evicts the previous one."""
        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY, max_open_tiles=1)
        tiles = list(acc.available_tiles())
        assert len(tiles) >= 2, "need at least 2 tiles for this test"

        # Open first tile
        s0 = acc._get_tile_store(tiles[0])
        _ = s0._open()
        assert len(acc._tile_stores) == 1

        # Open second tile — should evict first
        acc._get_tile_store(tiles[1])
        assert len(acc._tile_stores) == 1
        assert tiles[1] in acc._tile_stores
        assert tiles[0] not in acc._tile_stores
        # Evicted store should be closed
        assert s0._root is None

    def test_accessor_close_clears_all(self, synthetic_lake: Path):
        """close() closes every cached store and clears _tile_stores."""
        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY)
        # Touch all available tiles so they are all cached
        for npix in acc.available_tiles():
            s = acc._get_tile_store(npix)
            _ = s._open()

        assert len(acc._tile_stores) > 0
        stores = list(acc._tile_stores.values())

        acc.close()

        assert len(acc._tile_stores) == 0
        for s in stores:
            assert s._root is None

    def test_context_manager_closes_on_exit(self, synthetic_lake: Path):
        """Using SpectrumAccessor as a context manager closes stores on exit."""
        from data_lake.io.spectra import SpectrumAccessor

        with SpectrumAccessor(synthetic_lake, SURVEY) as acc:
            _ = acc.extract_subset_to_zarr(
                source_ids=[101, 103, 201],
                output_zarr=synthetic_lake / "cm_test.zarr",
                show_progress=False,
                overwrite=True,
            )
            # Inside: may have tiles open
            n_open_inside = len(acc._tile_stores)

        # Outside: all closed
        assert len(acc._tile_stores) == 0
        # The extract itself must have succeeded
        assert n_open_inside >= 0  # trivially true; closed by __exit__

    def test_extract_respects_max_open_tiles(self, synthetic_lake: Path):
        """max_open_tiles=2 keeps at most 2 stores open during a multi-tile extract."""
        import zarr

        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(synthetic_lake, SURVEY, max_open_tiles=2)

        result = acc.extract_subset_to_zarr(
            source_ids=[101, 102, 103, 201, 202],
            output_zarr=synthetic_lake / "lru_test.zarr",
            show_progress=False,
            overwrite=True,
        )
        acc.close()

        assert result["n_written"] == 5
        assert len(acc._tile_stores) == 0

    def test_shared_planning_uses_info_n_pix(self, synthetic_lake: Path):
        """Shared-mode planning reads n_pix from spectrum_info.json, not all tiles."""
        from data_lake.io.spectra import SpectrumAccessor

        # Build plan with max_open_tiles=1 so any multi-tile open would evict anyway.
        acc = SpectrumAccessor(synthetic_lake, SURVEY, max_open_tiles=1)
        plan = acc._plan_subset_extraction(
            [101, 103, 201], show_progress=False
        )
        # n_pix must match what spectrum_info.json reports
        assert plan.n_pix == N_PIX
        # At most 1 tile pinned after planning (LRU cap)
        assert len(acc._tile_stores) <= 1
        acc.close()


class TestReadTargetIdsComposite:
    """--target-id-col TARGETID,SURVEY,PROGRAM must hash like DESI ingest."""

    def test_fits_composite_matches_ingest_hash(self, tmp_path: Path):
        from astropy.table import Table

        from data_lake.export.spectra_subset import _read_target_ids
        from data_lake.ingest.fits_to_parquet import (
            composite_link_label,
            normalize_object_id,
        )

        path = tmp_path / "targets.fits"
        Table({
            "TARGETID": [123456789012345, 99],
            "SURVEY": ["main", "sv3"],
            "PROGRAM": ["dark", "bright"],
            "Z": [0.1, 0.2],
        }).write(path, overwrite=True)

        ids = _read_target_ids(path, "TARGETID,SURVEY,PROGRAM")
        expected = np.asarray([
            normalize_object_id(composite_link_label(123456789012345, "main", "dark")),
            normalize_object_id(composite_link_label(99, "sv3", "bright")),
        ], dtype=np.int64)
        np.testing.assert_array_equal(ids, expected)
        # Not the raw TARGETID
        assert ids[0] != 123456789012345

    def test_single_column_still_works(self, tmp_path: Path):
        from astropy.table import Table

        from data_lake.export.spectra_subset import _read_target_ids

        path = tmp_path / "targets.fits"
        Table({"TARGETID": [101, 102]}).write(path, overwrite=True)
        ids = _read_target_ids(path, "TARGETID")
        np.testing.assert_array_equal(ids, np.asarray([101, 102], dtype=np.int64))

    def test_missing_component_reports_clear_error(self, tmp_path: Path):
        from astropy.table import Table

        from data_lake.export.spectra_subset import _read_target_ids

        path = tmp_path / "targets.fits"
        Table({"TARGETID": [1], "SURVEY": ["main"]}).write(path, overwrite=True)
        with pytest.raises(ValueError, match="PROGRAM"):
            _read_target_ids(path, "TARGETID,SURVEY,PROGRAM")
