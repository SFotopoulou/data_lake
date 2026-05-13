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
from data_lake.ingest.fits_to_parquet import assign_healpix, healpix_dir


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
        root["source_id"].append(np.array([r.source_id for r in recs], dtype=np.int64))

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
        sids = np.asarray(out_root["source_id"][:])
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
