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

    def test_per_source_wavelength_rejected(self, tmp_path: Path):
        """The current implementation supports shared wavelength only."""
        from data_lake.io.spectra import SpectrumAccessor

        survey_root = tmp_path / "spectra" / "per_source_survey"
        survey_root.mkdir(parents=True)
        # Minimal spectrum_info.json with per_source mode; no tiles needed
        # because the guard fires before any tile is opened.
        (survey_root / "spectrum_info.json").write_text(json.dumps({
            "survey_name": "per_source_survey",
            "hats_order": NORDER,
            "n_pix": N_PIX,
            "wavelength_mode": "per_source",
        }))
        acc = SpectrumAccessor(tmp_path, "per_source_survey")
        with pytest.raises(NotImplementedError, match="wavelength_mode='shared'"):
            acc.extract_subset_to_zarr(
                source_ids=[1, 2, 3],
                output_zarr=tmp_path / "noop.zarr",
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
