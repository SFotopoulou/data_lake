"""Tests for SDSS spPlate spectrum ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits


def _write_minimal_spplate(
    path: Path,
    *,
    plate: int = 1960,
    mjd: int = 53289,
    n_fiber: int = 4,
    n_pix: int = 32,
) -> None:
    coeff0, coeff1 = 3.58, 0.0001
    flux = np.ones((n_fiber, n_pix), dtype=np.float32) * 50.0
    sigma = np.ones((n_fiber, n_pix), dtype=np.float32) * 5.0
    mask0 = np.zeros((n_fiber, n_pix), dtype=np.int32)

    phdu = fits.PrimaryHDU(flux)
    phdu.header["PLATEID"] = plate
    phdu.header["MJD"] = mjd
    phdu.header["COEFF0"] = coeff0
    phdu.header["COEFF1"] = coeff1
    ivar = np.where(sigma > 0, 1.0 / (sigma * sigma), 0.0).astype(np.float32)
    hdus = [
        phdu,
        fits.ImageHDU(ivar, name="IVAR"),
        fits.ImageHDU(mask0, name="ANDMASK"),
        fits.ImageHDU(mask0, name="ORMASK"),
    ]
    fiberid = np.arange(1, n_fiber + 1, dtype=np.int32)
    ra = np.linspace(120.0, 121.0, n_fiber)
    dec = np.full(n_fiber, 45.0)
    cols = [
        fits.Column(name="FIBERID", format="J", array=fiberid),
        fits.Column(name="RA", format="D", array=ra),
        fits.Column(name="DEC", format="D", array=dec),
        fits.Column(name="OBJID", format="5J", array=np.zeros((n_fiber, 5), dtype=np.int32)),
    ]
    hdus.append(fits.BinTableHDU.from_columns(cols, name="PLUGMAP"))
    phdu.header["RUN2D"] = "v5_13_2"
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _write_lookup(path: Path, plate: int, mjd: int, survey: str, fibers: dict[int, int]) -> None:
    rows = [
        {
            "survey": survey,
            "PLATE": plate,
            "MJD": mjd,
            "FIBERID": fid,
            "SPECOBJID": sid,
        }
        for fid, sid in fibers.items()
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


class TestSpplateDetectAndRead:
    def test_detect_format(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import _detect_format_from_path

        p = tmp_path / "spPlate-1960-53289.fits"
        _write_minimal_spplate(p)
        assert _detect_format_from_path(p) == "sdss_spplate"

    def test_read_with_lookup(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_sdss_spplate
        from data_lake.ingest.sdss_specobj_lookup import build_fiber_to_specobjid_map

        plate, mjd = 1960, 53289
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=3)
        lookup = tmp_path / "lookup.parquet"
        _write_lookup(
            lookup,
            plate,
            mjd,
            "sdss_dr17",
            {1: 9001, 2: 9002},
        )
        fiber_map = build_fiber_to_specobjid_map(
            "sdss_dr17", plate, mjd, lookup_path=lookup,
        )
        with fits.open(sp) as hdul:
            records, wcs = _read_sdss_spplate(
                hdul, path=sp, fiber_to_specobjid=fiber_map,
            )
        assert len(records) == 2
        assert {r.source_id for r in records} == {
            normalize_object_id(9001),
            normalize_object_id(9002),
        }
        assert wcs["n_pix"] == 32

    def test_skips_sentinel_plugmap_coordinates(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_parquet import assign_healpix, normalize_object_id
        from data_lake.ingest.fits_to_spectra_zarr import _read_sdss_spplate

        plate, mjd = 309, 51666
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=3)
        with fits.open(sp, mode="update", memmap=False) as hdul:
            hdul[4].data["RA"][2] = -9999.0
            hdul[4].data["DEC"][2] = -9999.0
            hdul.flush()

        fiber_map = {1: 9001, 2: 9002, 3: 9003}
        with fits.open(sp) as hdul:
            records, _ = _read_sdss_spplate(
                hdul, path=sp, fiber_to_specobjid=fiber_map, skip_unmatched=False,
            )
        assert len(records) == 2
        assert {r.source_id for r in records} == {
            normalize_object_id(9001),
            normalize_object_id(9002),
        }
        for rec in records:
            assign_healpix(np.array([rec.ra]), np.array([rec.dec]), 5)

    def test_example_spplate_with_sentinel_fibers(self) -> None:
        from data_lake.ingest.fits_to_parquet import assign_healpix
        from data_lake.ingest.fits_to_spectra_zarr import _read_sdss_spplate

        path = Path(__file__).resolve().parents[1] / "data" / "spPlate-0309-51666.fits"
        if not path.is_file():
            pytest.skip(f"Example spPlate not found: {path}")

        with fits.open(path, memmap=True) as hdul:
            fdata = hdul[5].data
            fiber_map = {int(f): int(f) for f in fdata["FIBERID"]}
            records, _ = _read_sdss_spplate(
                hdul, path=path, fiber_to_specobjid=fiber_map, skip_unmatched=False,
            )
        assert len(records) == 637
        for rec in records:
            assign_healpix(np.array([rec.ra]), np.array([rec.dec]), 5)


class TestFixLength:
    def test_pad_truncates_when_longer_than_tile(self) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import SpectrumRecord, _fix_length

        n_long = 4664
        target = 4638
        rec = SpectrumRecord(
            source_id=1,
            ra=120.0,
            dec=45.0,
            flux=np.ones(n_long, dtype=np.float32),
            ivar=np.ones(n_long, dtype=np.float32),
            mask=np.zeros(n_long, dtype=np.uint8),
            wavelength=np.linspace(4000.0, 9000.0, n_long),
            meta={},
        )
        out = _fix_length([rec], target, "pad")
        assert len(out[0].flux) == target
        assert len(out[0].wavelength) == target


class TestSpplateIngest:
    def test_ingest_to_zarr(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
        from data_lake.io.spectra import SpectrumAccessor

        plate, mjd = 1960, 53289
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=2)
        lookup = tmp_path / "lookup.parquet"
        _write_lookup(lookup, plate, mjd, "sdss_test", {1: 111, 2: 222})

        lake = tmp_path / "lake"
        index_map = ingest_spectra_from_fits(
            sp,
            lake,
            "sdss_test",
            fmt="sdss_spplate",
            specobj_lookup=lookup,
            norder=5,
            on_duplicate_source_id="skip",
        )
        assert len(index_map) == 2
        acc = SpectrumAccessor(lake, "sdss_test")
        assert acc.get_spectrum(111) is not None
        assert acc.get_spectrum(222) is not None

    def test_ingest_from_plate_specobjid(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
        from data_lake.ingest.sdss_specobj_lookup import sdss_specobjid_from_plate_fiber
        from data_lake.io.spectra import SpectrumAccessor

        plate, mjd = 1960, 53289
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=2)
        lake = tmp_path / "lake"
        index_map = ingest_spectra_from_fits(
            sp,
            lake,
            "sdss_plate_id",
            fmt="sdss_spplate",
            specobj_lookup_from_plate=True,
            norder=5,
        )
        assert len(index_map) == 2
        sid1 = sdss_specobjid_from_plate_fiber(plate, 1, mjd, "v5_13_2")
        sid2 = sdss_specobjid_from_plate_fiber(plate, 2, mjd, "v5_13_2")
        assert sid1 in index_map and sid2 in index_map
        acc = SpectrumAccessor(lake, "sdss_plate_id")
        sp1 = acc.get_spectrum(sid1)
        assert sp1.ivar.max() > 0

    @pytest.fixture
    def boss_spplate_path(self) -> Path:
        path = Path(__file__).resolve().parents[1] / "data" / "spPlate-3523-55144.fits"
        if not path.is_file():
            pytest.skip(f"Example spPlate not found: {path}")
        return path

    def test_boss_spplate_read_and_ingest(
        self, tmp_path: Path, boss_spplate_path: Path,
    ) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import (
            _read_sdss_spplate,
            ingest_spectra_from_fits,
        )
        from data_lake.ingest.sdss_specobj_lookup import build_fiber_to_specobjid_from_spplate
        from data_lake.io.spectra import SpectrumAccessor

        with fits.open(boss_spplate_path) as hdul:
            fiber_map = build_fiber_to_specobjid_from_spplate(hdul, boss_spplate_path)
            records, wcs = _read_sdss_spplate(
                hdul, path=boss_spplate_path, fiber_to_specobjid=fiber_map,
            )
        assert wcs["n_pix"] == 3854
        assert len(records) >= 400
        assert records[0].ivar.max() > 1.0

        lake = tmp_path / "lake"
        n = ingest_spectra_from_fits(
            boss_spplate_path,
            lake,
            "boss_plate_test",
            fmt="sdss_spplate",
            specobj_lookup_from_plate=True,
            norder=5,
            on_duplicate_source_id="skip",
        )
        assert len(n) == len(records)
        acc = SpectrumAccessor(lake, "boss_plate_test")
        any_sid = next(iter(n))
        assert acc.get_spectrum(any_sid).flux.shape[0] == 3854

    def test_default_triplet_hash(self, tmp_path: Path) -> None:
        """No lookup flag → composite PLATE|MJD|FIBERID hash used automatically."""
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits
        from data_lake.ingest.sdss_specobj_lookup import spplate_source_id_from_triplet
        from data_lake.io.spectra import SpectrumAccessor

        plate, mjd = 4002, 55645
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=3)
        lake = tmp_path / "lake"
        index_map = ingest_spectra_from_fits(
            sp, lake, "sdss_test", fmt="sdss_spplate", norder=5,
        )
        assert len(index_map) == 3
        for fiber_id in (1, 2, 3):
            expected_sid = spplate_source_id_from_triplet(plate, mjd, fiber_id)
            assert expected_sid in index_map, f"fiber {fiber_id} → sid {expected_sid} missing"
        acc = SpectrumAccessor(lake, "sdss_test")
        for fiber_id in (1, 2, 3):
            sid = spplate_source_id_from_triplet(plate, mjd, fiber_id)
            assert acc.get_spectrum(sid) is not None

    def test_triplet_hash_matches_composite_catalog(self, tmp_path: Path) -> None:
        """source_id from triplet hash == _source_id in catalog ingested with PLATE,MJD,FIBERID."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        from data_lake.ingest.fits_to_parquet import composite_link_label, normalize_object_id
        from data_lake.ingest.sdss_specobj_lookup import (
            build_fiber_to_source_id_from_triplet,
            spplate_source_id_from_triplet,
        )

        plate, mjd = 4002, 55645
        fibers = [1, 2, 3]

        # Simulate what catalog composite ingest does for PLATE,MJD,FIBERID rows.
        catalog_source_ids = [
            normalize_object_id(composite_link_label(plate, mjd, fid)) for fid in fibers
        ]

        # Triplet helper must produce identical values.
        triplet_map = build_fiber_to_source_id_from_triplet(plate, mjd, fibers)
        for fid, cat_sid in zip(fibers, catalog_source_ids):
            assert triplet_map[fid] == cat_sid, (
                f"fiber {fid}: triplet={triplet_map[fid]} != catalog={cat_sid}"
            )
        # Scalar helper is consistent with the map builder.
        for fid in fibers:
            assert spplate_source_id_from_triplet(plate, mjd, fid) == triplet_map[fid]

    def test_mutual_exclusion_of_lookup_modes(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits

        plate, mjd = 1, 2
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=1)
        lookup = tmp_path / "lookup.parquet"
        _write_lookup(lookup, plate, mjd, "sdss_test", {1: 99})
        with pytest.raises(ValueError, match="only one"):
            ingest_spectra_from_fits(
                sp,
                tmp_path / "lake",
                "sdss_test",
                fmt="sdss_spplate",
                specobj_lookup=lookup,
                specobj_lookup_from_plate=True,
            )


# ---------------------------------------------------------------------------
# Dynamic tile widening tests
# ---------------------------------------------------------------------------


def _make_records(source_ids, n_pix: int, *, wavelength_mode: str = "per_source"):
    """Build minimal SpectrumRecord list for widening tests."""
    from data_lake.ingest.fits_to_spectra_zarr import SpectrumRecord

    records = []
    for sid in source_ids:
        flux = np.full(n_pix, float(sid), dtype=np.float32)
        ivar = np.ones(n_pix, dtype=np.float32)
        mask = np.zeros(n_pix, dtype=np.uint8)
        wave = np.linspace(3500.0, 9000.0, n_pix).astype(np.float32)
        records.append(
            SpectrumRecord(
                source_id=sid,
                ra=120.0,
                dec=45.0,
                flux=flux,
                ivar=ivar,
                mask=mask,
                wavelength=wave if wavelength_mode == "per_source" else None,
                meta={},
            )
        )
    return records


def _make_tile(tile_path, records, n_pix, *, wavelength_mode, n_diag=None):
    """Create a Zarr tile from records at the given path."""
    import numpy as np
    import zarr

    from data_lake.ingest.fits_to_spectra_zarr import (
        _open_or_create_spectrum_tile,
        _META_DTYPE,
        _meta_to_bytes,
    )

    wcs_attrs = {"n_pix": n_pix, "coeff0": 3.5, "coeff1": 0.0001}
    mask_dtype = np.dtype(np.uint8)
    root = _open_or_create_spectrum_tile(
        tile_path,
        n_pix,
        wavelength_mode,
        mask_dtype,
        wcs_attrs,
        n_diag=n_diag,
    )

    flux = np.stack([r.flux for r in records]).astype(np.float32)
    ivar = np.stack([r.ivar for r in records]).astype(np.float32)
    mask = np.stack([r.mask for r in records]).astype(mask_dtype)
    ids = np.array([r.source_id for r in records], dtype=np.int64)
    meta = np.frombuffer(
        b"".join(_meta_to_bytes(r.meta) for r in records),
        dtype="|V" + str(_META_DTYPE.itemsize),
    )

    root["flux"].append(flux)
    root["ivar"].append(ivar)
    root["mask"].append(mask)
    from data_lake.ingest.zarr_ids import zarr_join_array

    zarr_join_array(root).append(ids)
    root["meta"].append(meta)

    if wavelength_mode == "per_source":
        waves = np.stack([r.wavelength.astype(np.float32) for r in records])
        root["wavelength"].append(waves)
    else:
        root["wavelength"][:] = np.linspace(3500.0, 9000.0, n_pix)

    if n_diag is not None:
        res = np.zeros((len(records), n_diag, n_pix), dtype=np.float32)
        root["resolution"].append(res)

    return root


class TestWidenSpectrumTile:
    """Unit tests for the ``widen_spectrum_tile`` helper."""

    def test_widen_per_source_wavelength(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import widen_spectrum_tile

        n_old, n_new = 32, 64
        records = _make_records([1, 2, 3], n_old, wavelength_mode="per_source")
        tile_path = tmp_path / "Npix=1.zarr"
        wcs = {"n_pix": n_old, "coeff0": 3.5, "coeff1": 0.0001}
        mask_dtype = np.dtype(np.uint8)
        _make_tile(tile_path, records, n_old, wavelength_mode="per_source")

        root = widen_spectrum_tile(
            tile_path,
            n_new,
            wavelength_mode="per_source",
            mask_dtype=mask_dtype,
            wcs_attrs=wcs,
        )

        assert root["flux"].shape == (3, n_new)
        assert root["ivar"].shape == (3, n_new)
        assert root["mask"].shape == (3, n_new)
        assert root["wavelength"].shape == (3, n_new)
        from data_lake.ingest.zarr_ids import zarr_join_array

        assert zarr_join_array(root).shape == (3,)

        flux_arr = np.asarray(root["flux"][:])
        # Original values preserved in the first n_old columns
        assert np.all(flux_arr[:, :n_old] == np.array([1.0, 2.0, 3.0])[:, None])
        # Padding filled with NaN
        assert np.all(np.isnan(flux_arr[:, n_old:]))

        ivar_arr = np.asarray(root["ivar"][:])
        assert np.all(ivar_arr[:, :n_old] == 1.0)
        assert np.all(ivar_arr[:, n_old:] == 0.0)

        wave_arr = np.asarray(root["wavelength"][:])
        assert np.all(wave_arr[:, n_old:] == 0.0)

    def test_widen_shared_wavelength(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import widen_spectrum_tile

        n_old, n_new = 32, 60
        records = _make_records([10, 20], n_old, wavelength_mode="per_source")
        tile_path = tmp_path / "Npix=2.zarr"
        wcs = {"n_pix": n_old, "coeff0": 3.5, "coeff1": 0.0001}
        mask_dtype = np.dtype(np.uint8)
        _make_tile(tile_path, records, n_old, wavelength_mode="shared")

        root = widen_spectrum_tile(
            tile_path,
            n_new,
            wavelength_mode="shared",
            mask_dtype=mask_dtype,
            wcs_attrs=wcs,
        )

        assert root["flux"].shape == (2, n_new)
        assert root["wavelength"].shape == (n_new,)

        wave = np.asarray(root["wavelength"][:])
        assert len(wave) == n_new
        assert np.all(wave[n_old:] == 0.0)
        assert wave[0] > 0.0  # original values preserved

    def test_widen_with_resolution_array(self, tmp_path: Path) -> None:
        from data_lake.ingest.fits_to_spectra_zarr import widen_spectrum_tile

        n_old, n_new, n_diag = 32, 48, 5
        records = _make_records([100, 200], n_old)
        tile_path = tmp_path / "Npix=3.zarr"
        wcs = {"n_pix": n_old, "coeff0": 3.5, "coeff1": 0.0001}
        mask_dtype = np.dtype(np.uint8)
        res_offsets = np.arange(-2, 3, dtype=np.int32)
        _make_tile(
            tile_path,
            records,
            n_old,
            wavelength_mode="per_source",
            n_diag=n_diag,
        )

        root = widen_spectrum_tile(
            tile_path,
            n_new,
            wavelength_mode="per_source",
            mask_dtype=mask_dtype,
            wcs_attrs=wcs,
            n_diag=n_diag,
            resolution_offsets=res_offsets,
        )

        assert root["resolution"].shape == (2, n_diag, n_new)
        res_arr = np.asarray(root["resolution"][:])
        assert np.all(res_arr[:, :, n_old:] == 0.0)

    def test_noop_when_not_wider(self, tmp_path: Path) -> None:
        """widen_spectrum_tile should return unchanged tile when new_n_pix <= old."""
        from data_lake.ingest.fits_to_spectra_zarr import widen_spectrum_tile

        n_pix = 32
        records = _make_records([1], n_pix)
        tile_path = tmp_path / "Npix=4.zarr"
        wcs = {"n_pix": n_pix, "coeff0": 3.5, "coeff1": 0.0001}
        mask_dtype = np.dtype(np.uint8)
        _make_tile(tile_path, records, n_pix, wavelength_mode="per_source")

        root = widen_spectrum_tile(
            tile_path,
            n_pix,  # same width → no-op
            wavelength_mode="per_source",
            mask_dtype=mask_dtype,
            wcs_attrs=wcs,
        )
        assert root["flux"].shape == (1, n_pix)


class TestDynamicTileWideningOnIngest:
    """Integration tests: ingest triggers dynamic tile widening."""

    def test_wider_batch_widens_per_source_tile(self, tmp_path: Path) -> None:
        """Ingest short spectra then longer spectra into same tile; old rows padded."""
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits

        plate, mjd = 1960, 53289
        short_sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        long_sp = tmp_path / f"spPlate-{plate}-{mjd+1}.fits"
        lookup_short = tmp_path / "lookup_short.parquet"
        lookup_long = tmp_path / "lookup_long.parquet"

        # First file: 2 spectra, n_pix=32
        _write_minimal_spplate(short_sp, plate=plate, mjd=mjd, n_fiber=2, n_pix=32)
        _write_lookup(lookup_short, plate, mjd, "widen_test", {1: 1001, 2: 1002})

        # Second file: 2 new spectra, n_pix=64 (wider)
        _write_minimal_spplate(long_sp, plate=plate, mjd=mjd + 1, n_fiber=2, n_pix=64)
        _write_lookup(lookup_long, plate, mjd + 1, "widen_test", {1: 2001, 2: 2002})

        lake = tmp_path / "lake"
        ingest_spectra_from_fits(
            short_sp, lake, "widen_test",
            fmt="sdss_spplate", specobj_lookup=lookup_short,
            norder=5, on_length_mismatch="pad",
        )
        ingest_spectra_from_fits(
            long_sp, lake, "widen_test",
            fmt="sdss_spplate", specobj_lookup=lookup_long,
            norder=5, on_length_mismatch="pad",
        )

        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(lake, "widen_test")
        sp_short = acc.get_spectrum(1001)
        sp_long = acc.get_spectrum(2001)

        # Both spectra are accessible
        assert sp_short is not None
        assert sp_long is not None

        # Tile n_pix expanded to accommodate longer spectra
        assert sp_long.flux.shape[0] == 64
        assert sp_short.flux.shape[0] == 64  # old row padded to new width

        # Padded tail of original short spectrum should be NaN
        assert np.all(np.isnan(sp_short.flux[32:]))

    def test_wider_batch_widens_shared_wavelength_tile(self, tmp_path: Path) -> None:
        """Same scenario using shared wavelength mode."""
        from data_lake.ingest.fits_to_spectra_zarr import ingest_spectra_from_fits

        plate, mjd = 2000, 54000
        short_sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        long_sp = tmp_path / f"spPlate-{plate}-{mjd+1}.fits"
        lookup_short = tmp_path / "lookup_s.parquet"
        lookup_long = tmp_path / "lookup_l.parquet"

        _write_minimal_spplate(short_sp, plate=plate, mjd=mjd, n_fiber=2, n_pix=32)
        _write_lookup(lookup_short, plate, mjd, "shared_widen_test", {1: 3001, 2: 3002})

        _write_minimal_spplate(long_sp, plate=plate, mjd=mjd + 1, n_fiber=2, n_pix=64)
        _write_lookup(lookup_long, plate, mjd + 1, "shared_widen_test", {1: 4001, 2: 4002})

        lake = tmp_path / "lake"
        ingest_spectra_from_fits(
            short_sp, lake, "shared_widen_test",
            fmt="sdss_spplate", specobj_lookup=lookup_short,
            norder=5, wavelength_mode="shared", on_length_mismatch="pad",
        )
        ingest_spectra_from_fits(
            long_sp, lake, "shared_widen_test",
            fmt="sdss_spplate", specobj_lookup=lookup_long,
            norder=5, wavelength_mode="shared", on_length_mismatch="pad",
        )

        from data_lake.io.spectra import SpectrumAccessor

        acc = SpectrumAccessor(lake, "shared_widen_test")
        sp_short = acc.get_spectrum(3001)
        sp_long = acc.get_spectrum(4001)
        assert sp_short is not None
        assert sp_long is not None
        assert sp_long.flux.shape[0] == 64
        assert sp_short.flux.shape[0] == 64
        assert np.all(np.isnan(sp_short.flux[32:]))


# ---------------------------------------------------------------------------
# Vectorized decoder tests (spplate_parallel_ingest)
# ---------------------------------------------------------------------------


class TestVectorizedSpplateDecoder:
    """Tests for the fast vectorized spPlate decode path."""

    def test_equivalence_with_slow_path_source_ids(self, tmp_path: Path) -> None:
        """Vectorized decoder produces source_ids identical to the triplet-hash reference."""
        from data_lake.ingest.sdss_specobj_lookup import spplate_source_id_from_triplet
        from data_lake.ingest.spplate_parallel_ingest import (
            SpplateBatchConfig,
            _decode_spplate_to_worker_result,
        )

        plate, mjd = 7000, 56789
        n_fiber = 4
        n_pix = 32
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=n_fiber, n_pix=n_pix)

        # Reference: expected IDs for fibers 1..n_fiber
        expected_sids = {
            spplate_source_id_from_triplet(plate, mjd, fid)
            for fid in range(1, n_fiber + 1)
        }

        config = SpplateBatchConfig(norder=5, triplet_hash=True)
        result = _decode_spplate_to_worker_result(sp, config)

        assert result.ok, result.error
        fast_sids = {sid for b in result.batches for sid in b.source_ids.tolist()}

        assert fast_sids == expected_sids
        assert result.n_pix == n_pix

        # Shared wavelength is set correctly
        assert result.wavelength is not None
        assert result.wavelength.shape == (n_pix,)

        # Each batch has correct pixel width
        for b in result.batches:
            assert b.flux.shape[1] == n_pix
            assert b.ivar.shape[1] == n_pix
            assert b.mask.shape[1] == n_pix

    def test_decoder_skips_all_zero_flux_fibers(self, tmp_path: Path) -> None:
        """Fibers with all-zero flux are excluded by the active-flux mask."""
        import numpy as np
        from astropy.io import fits

        from data_lake.ingest.spplate_parallel_ingest import (
            SpplateBatchConfig,
            _decode_spplate_to_worker_result,
        )

        plate, mjd = 7001, 56790
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=3, n_pix=32)

        # Zero out first fiber
        with fits.open(sp, mode="update", memmap=False) as hdul:
            hdul[0].data[0, :] = 0.0
            hdul.flush()

        config = SpplateBatchConfig(norder=5, triplet_hash=True)
        result = _decode_spplate_to_worker_result(sp, config)

        assert result.ok
        assert result.n_spectra == 2

    def test_decoder_shared_wavelength_attributes(self, tmp_path: Path) -> None:
        """wcs_attrs contains COEFF0/COEFF1-derived values and n_pix."""
        from data_lake.ingest.spplate_parallel_ingest import (
            SpplateBatchConfig,
            _decode_spplate_to_worker_result,
        )

        plate, mjd = 7002, 56791
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        n_pix = 48
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=2, n_pix=n_pix)

        config = SpplateBatchConfig(norder=5, triplet_hash=True)
        result = _decode_spplate_to_worker_result(sp, config)

        assert result.ok
        assert result.n_pix == n_pix
        assert result.wcs_attrs["n_pix"] == n_pix
        assert "crval" in result.wcs_attrs
        assert result.wavelength is not None
        assert len(result.wavelength) == n_pix

    def test_meta_bytes_length_matches_n_spectra(self, tmp_path: Path) -> None:
        """meta_bytes in each TileBatch encodes exactly n_row records."""
        from data_lake.ingest.fits_to_spectra_zarr import _META_DTYPE
        from data_lake.ingest.spplate_parallel_ingest import (
            SpplateBatchConfig,
            _decode_spplate_to_worker_result,
        )

        plate, mjd = 7003, 56792
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=4, n_pix=32)

        config = SpplateBatchConfig(norder=5, triplet_hash=True)
        result = _decode_spplate_to_worker_result(sp, config)

        assert result.ok
        total_meta_records = sum(
            len(b.meta_bytes) // _META_DTYPE.itemsize for b in result.batches
        )
        total_sids = sum(b.source_ids.size for b in result.batches)
        assert total_meta_records == total_sids


class TestSpplateBatchIngest:
    """Smoke tests for ingest_spplate_files_parallel."""

    def test_batch_ingest_two_plates(self, tmp_path: Path) -> None:
        """Batch ingest of two plate files, 1 worker; row counts match fiber count."""
        from concurrent.futures import ThreadPoolExecutor

        from data_lake.ingest.sdss_specobj_lookup import spplate_source_id_from_triplet
        from data_lake.ingest.spplate_parallel_ingest import ingest_spplate_files_parallel
        from data_lake.io.spectra import SpectrumAccessor

        plate1, mjd1 = 8000, 57000
        plate2, mjd2 = 8001, 57001
        n_fiber = 3
        lake = tmp_path / "lake"

        sp1 = tmp_path / f"spPlate-{plate1}-{mjd1}.fits"
        sp2 = tmp_path / f"spPlate-{plate2}-{mjd2}.fits"
        _write_minimal_spplate(sp1, plate=plate1, mjd=mjd1, n_fiber=n_fiber, n_pix=32)
        _write_minimal_spplate(sp2, plate=plate2, mjd=mjd2, n_fiber=n_fiber, n_pix=32)

        result = ingest_spplate_files_parallel(
            file_paths=[sp1, sp2],
            output_root=lake,
            survey_name="sdss_test_batch",
            n_workers=1,
            norder=5,
            show_progress=False,
            on_duplicate_source_id="skip",
            executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
        )

        assert result["n_files_succeeded"] == 2
        assert result["n_spectra"] == n_fiber * 2

        acc = SpectrumAccessor(lake, "sdss_test_batch")
        for plate, mjd in [(plate1, mjd1), (plate2, mjd2)]:
            for fid in range(1, n_fiber + 1):
                sid = spplate_source_id_from_triplet(plate, mjd, fid)
                sp = acc.get_spectrum(sid)
                assert sp is not None, f"Missing plate={plate} mjd={mjd} fid={fid}"
                assert sp.flux.shape[0] == 32

    def test_batch_ingest_spectrum_info_wavelength_mode_shared(
        self, tmp_path: Path
    ) -> None:
        """spectrum_info.json records wavelength_mode=shared after batch ingest."""
        import json
        from concurrent.futures import ThreadPoolExecutor

        from data_lake.ingest.spplate_parallel_ingest import ingest_spplate_files_parallel

        plate, mjd = 9000, 58000
        lake = tmp_path / "lake"
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=2, n_pix=32)

        ingest_spplate_files_parallel(
            file_paths=[sp],
            output_root=lake,
            survey_name="sdss_shared_test",
            n_workers=1,
            norder=5,
            show_progress=False,
            executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
        )

        info_path = lake / "spectra" / "sdss_shared_test" / "spectrum_info.json"
        assert info_path.exists(), "spectrum_info.json not written"
        info = json.loads(info_path.read_text())
        assert info.get("wavelength_mode") == "shared", (
            f"Expected wavelength_mode='shared', got {info.get('wavelength_mode')!r}"
        )

    def test_batch_ingest_checkpoint_resumes(self, tmp_path: Path) -> None:
        """A completed file in the checkpoint is skipped on the second run."""
        from concurrent.futures import ThreadPoolExecutor

        from data_lake.ingest.spplate_parallel_ingest import ingest_spplate_files_parallel

        plate, mjd = 9001, 58001
        lake = tmp_path / "lake"
        sp = tmp_path / f"spPlate-{plate}-{mjd}.fits"
        _write_minimal_spplate(sp, plate=plate, mjd=mjd, n_fiber=2, n_pix=32)
        ckpt = tmp_path / "checkpoint.json"

        ingest_spplate_files_parallel(
            file_paths=[sp],
            output_root=lake,
            survey_name="sdss_ckpt_test",
            n_workers=1,
            norder=5,
            show_progress=False,
            checkpoint_path=ckpt,
            executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
        )

        result2 = ingest_spplate_files_parallel(
            file_paths=[sp],
            output_root=lake,
            survey_name="sdss_ckpt_test",
            n_workers=1,
            norder=5,
            show_progress=False,
            checkpoint_path=ckpt,
            executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
        )

        assert result2["n_files_skipped"] == 1
        assert result2["n_files_processed"] == 0  # all skipped, nothing to process
        assert result2["n_spectra"] == 0


class TestSpplateBatchMixedNPix:
    """Batch ingest with different n_pix per plate (pad / widen)."""

    def test_batch_mixed_n_pix_pads_to_widest(self, tmp_path: Path) -> None:
        """Two plates with n_pix 32 and 64 ingest under default pad policy."""
        import json
        from concurrent.futures import ThreadPoolExecutor

        from data_lake.ingest.sdss_specobj_lookup import spplate_source_id_from_triplet
        from data_lake.ingest.spplate_parallel_ingest import ingest_spplate_files_parallel
        from data_lake.io.spectra import SpectrumAccessor

        plate1, mjd1 = 8100, 58100
        plate2, mjd2 = 8101, 58101
        n_fiber = 2
        lake = tmp_path / "lake"

        sp_short = tmp_path / f"spPlate-{plate1}-{mjd1}.fits"
        sp_long = tmp_path / f"spPlate-{plate2}-{mjd2}.fits"
        _write_minimal_spplate(
            sp_short, plate=plate1, mjd=mjd1, n_fiber=n_fiber, n_pix=32,
        )
        _write_minimal_spplate(
            sp_long, plate=plate2, mjd=mjd2, n_fiber=n_fiber, n_pix=64,
        )

        result = ingest_spplate_files_parallel(
            file_paths=[sp_short, sp_long],
            output_root=lake,
            survey_name="sdss_mixed_npix",
            n_workers=1,
            norder=5,
            show_progress=False,
            on_length_mismatch="pad",
            executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
        )

        assert result["n_files_failed"] == 0
        assert result["n_files_succeeded"] == 2
        assert result["n_spectra"] == n_fiber * 2

        info_path = lake / "spectra" / "sdss_mixed_npix" / "spectrum_info.json"
        info = json.loads(info_path.read_text())
        assert info["n_pix"] == 64

        acc = SpectrumAccessor(lake, "sdss_mixed_npix")
        sid_short = spplate_source_id_from_triplet(plate1, mjd1, 1)
        sid_long = spplate_source_id_from_triplet(plate2, mjd2, 1)
        sp_from_short = acc.get_spectrum(sid_short)
        sp_from_long = acc.get_spectrum(sid_long)
        assert sp_from_short is not None
        assert sp_from_long is not None
        assert sp_from_long.flux.shape[0] == 64
        assert sp_from_short.flux.shape[0] == 64
        assert np.all(np.isnan(sp_from_short.flux[32:]))

    def test_batch_mixed_n_pix_error_rejects_second_file(self, tmp_path: Path) -> None:
        """With on_length_mismatch=error, second plate with different n_pix fails."""
        from concurrent.futures import ThreadPoolExecutor

        from data_lake.ingest.spplate_parallel_ingest import ingest_spplate_files_parallel

        plate1, mjd1 = 8102, 58102
        plate2, mjd2 = 8103, 58103
        lake = tmp_path / "lake"
        sp_short = tmp_path / f"spPlate-{plate1}-{mjd1}.fits"
        sp_long = tmp_path / f"spPlate-{plate2}-{mjd2}.fits"
        _write_minimal_spplate(sp_short, plate=plate1, mjd=mjd1, n_fiber=2, n_pix=32)
        _write_minimal_spplate(sp_long, plate=plate2, mjd=mjd2, n_fiber=2, n_pix=64)

        result = ingest_spplate_files_parallel(
            file_paths=[sp_short, sp_long],
            output_root=lake,
            survey_name="sdss_strict_npix",
            n_workers=1,
            norder=5,
            show_progress=False,
            on_length_mismatch="error",
            executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
        )

        assert result["n_files_succeeded"] == 1
        assert result["n_files_failed"] == 1
        assert any("n_pix=" in (f.get("error") or "") for f in result["failures"])
