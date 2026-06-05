"""Tests for cross-match sky matcher backends."""

from __future__ import annotations

import numpy as np
import pytest

from data_lake.io.crossmatch_matchers import (
    match_sky_nn_within_radius,
    rapids_available,
    validate_match_backend,
)


class TestValidateMatchBackend:
    def test_astropy_always_valid(self) -> None:
        validate_match_backend("astropy")

    def test_rapids_requires_extra(self) -> None:
        if rapids_available():
            validate_match_backend("rapids")
        else:
            with pytest.raises(ImportError, match="rapids"):
                validate_match_backend("rapids")

    def test_invalid_backend(self) -> None:
        with pytest.raises(ValueError, match="match_backend"):
            validate_match_backend("stilts")  # type: ignore[arg-type]


class TestMatchSkyBackends:
    @pytest.fixture
    def sky_fixture(self) -> dict:
        ra_a = np.array([120.0, 10.0])
        dec_a = np.array([45.0, 0.0])
        ids_a = np.array([1001, 1002], dtype=np.int64)
        ra_b = np.array([120.0001, 50.0])
        dec_b = np.array([45.0001, 0.0])
        ids_b = np.array([2001, 2002], dtype=np.int64)
        return {
            "ra_a": ra_a,
            "dec_a": dec_a,
            "ids_a": ids_a,
            "ra_b": ra_b,
            "dec_b": dec_b,
            "ids_b": ids_b,
            "radius_deg": 2.0 / 3600.0 * 2,  # 4 arcsec
        }

    def test_astropy_nearest_within_radius(self, sky_fixture: dict) -> None:
        matched_a, matched_b, sep = match_sky_nn_within_radius(
            **sky_fixture,
            backend="astropy",
        )
        assert matched_a.tolist() == [1001]
        assert matched_b.tolist() == [2001]
        assert sep.size == 1
        assert sep[0] * 3600.0 < 4.0

    def test_empty_catalog_b(self, sky_fixture: dict) -> None:
        matched_a, matched_b, sep = match_sky_nn_within_radius(
            sky_fixture["ra_a"],
            sky_fixture["dec_a"],
            sky_fixture["ids_a"],
            np.array([]),
            np.array([]),
            np.array([], dtype=np.int64),
            sky_fixture["radius_deg"],
            backend="astropy",
        )
        assert matched_a.size == 0
        assert matched_b.size == 0
        assert sep.size == 0

    @pytest.mark.gpu
    def test_rapids_matches_astropy(self, sky_fixture: dict) -> None:
        if not rapids_available():
            pytest.skip("cuML/CuPy not installed (uv sync --extra rapids)")

        astropy_result = match_sky_nn_within_radius(**sky_fixture, backend="astropy")
        rapids_result = match_sky_nn_within_radius(
            **sky_fixture,
            backend="rapids",
            gpu_id=0,
        )
        for field_idx in range(3):
            a = astropy_result[field_idx]
            r = rapids_result[field_idx]
            assert a.shape == r.shape
            if a.dtype.kind == "f":
                np.testing.assert_allclose(a, r, rtol=0, atol=1e-5)
            else:
                np.testing.assert_array_equal(a, r)
