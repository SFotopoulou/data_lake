"""Tests for the Region spatial selector."""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from data_lake.discovery.region import (
    Region,
    parse_npix_arg,
    rescale_npix_nested,
)
from data_lake.ingest.fits_to_parquet import assign_healpix


class TestRescaleNpix:
    def test_identity(self) -> None:
        assert rescale_npix_nested([1, 2, 3], 5, 5) == {1, 2, 3}

    def test_finer_expands_to_children(self) -> None:
        # One order finer -> 4 children per pixel, contiguous nested block.
        assert rescale_npix_nested([0], 4, 5) == {0, 1, 2, 3}
        assert rescale_npix_nested([1], 4, 5) == {4, 5, 6, 7}
        # Two orders finer -> 16 children.
        assert rescale_npix_nested([0], 4, 6) == set(range(16))

    def test_coarser_maps_to_parent(self) -> None:
        assert rescale_npix_nested([0, 1, 2, 3], 5, 4) == {0}
        assert rescale_npix_nested([4, 5, 6, 7], 5, 4) == {1}

    def test_roundtrip_child_parent(self) -> None:
        children = rescale_npix_nested([42], 5, 8)
        assert rescale_npix_nested(children, 8, 5) == {42}


class TestConeRegion:
    def test_cone_contains_center_tile(self) -> None:
        ra, dec, norder = 120.0, 45.0, 5
        region = Region.cone(ra, dec, radius_arcsec=60.0)
        npix = region.to_npix(norder)
        center = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
        assert center in npix

    def test_cone_scales_with_order(self) -> None:
        region = Region.cone(10.0, -20.0, radius_arcsec=600.0)
        coarse = region.to_npix(4)
        fine = region.to_npix(7)
        # Finer order -> more (smaller) pixels covering the same area.
        assert len(fine) > len(coarse)


class TestBboxRegion:
    def test_bbox_contains_interior_point(self) -> None:
        norder = 6
        region = Region.bbox(ra_min=100.0, ra_max=101.0, dec_min=20.0, dec_max=21.0)
        npix = region.to_npix(norder)
        inside = int(assign_healpix(np.array([100.5]), np.array([20.5]), norder)[0])
        assert inside in npix

    def test_bbox_ra_wraparound(self) -> None:
        norder = 5
        # Box wrapping through RA=0 (359 -> 1 deg).
        region = Region.bbox(ra_min=359.0, ra_max=1.0, dec_min=-1.0, dec_max=1.0)
        npix = region.to_npix(norder)
        p_left = int(assign_healpix(np.array([359.5]), np.array([0.0]), norder)[0])
        p_right = int(assign_healpix(np.array([0.5]), np.array([0.0]), norder)[0])
        assert p_left in npix
        assert p_right in npix

    def test_bbox_clamps_dec(self) -> None:
        # dec beyond the pole should not raise.
        region = Region.bbox(ra_min=0.0, ra_max=10.0, dec_min=80.0, dec_max=95.0)
        npix = region.to_npix(4)
        assert len(npix) > 0


class TestNpixRegion:
    def test_npix_region_rescales(self) -> None:
        region = Region.from_npix([10, 11], source_norder=5)
        assert region.to_npix(5) == {10, 11}
        assert region.to_npix(6) == set(range(40, 48))

    def test_roundtrip_dict(self) -> None:
        for region in (
            Region.from_npix([1, 2, 3], 5),
            Region.cone(12.0, 34.0, 50.0),
            Region.bbox(1.0, 2.0, 3.0, 4.0),
        ):
            assert Region.from_dict(region.to_dict()).to_dict() == region.to_dict()


class TestParseNpixArg:
    def test_ints_and_ranges(self) -> None:
        assert parse_npix_arg("1,2,3") == [1, 2, 3]
        assert parse_npix_arg("10-12,20") == [10, 11, 12, 20]
