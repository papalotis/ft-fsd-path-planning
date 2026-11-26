"""SE2Transform: a pose transform that stands in for the old closure pairs."""

from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning.geometry.transforms import SE2Transform, rotate

RNG = np.random.default_rng(0)


def random_transform() -> SE2Transform:
    return SE2Transform(
        angle=float(RNG.uniform(-np.pi, np.pi)),
        shift=RNG.normal(size=2) * 10,
        pivot=RNG.normal(size=2) * 10,
        offset=RNG.normal(size=2) * 10,
    )


class TestSE2Transform:
    def test_identity_by_default(self) -> None:
        points = RNG.normal(size=(5, 2))
        np.testing.assert_array_equal(SE2Transform(0.0).apply_to_points(points), points)

    def test_formula(self) -> None:
        t = random_transform()
        p = RNG.normal(size=2)
        expected = rotate(p + t.shift - t.pivot, t.angle) + t.pivot + t.offset
        np.testing.assert_array_equal(t.apply_to_points(p), expected)

    def test_call_transforms_a_pose(self) -> None:
        t = random_transform()
        p = RNG.normal(size=2)
        position, yaw = t(p, 0.3)
        np.testing.assert_array_equal(position, t.apply_to_points(p))
        assert yaw == pytest.approx(0.3 + t.angle)

    @pytest.mark.parametrize("seed", range(5))
    def test_inverse_undoes_it(self, seed) -> None:
        t = random_transform()
        points = RNG.normal(size=(7, 2)) * 20
        there, yaw = t(points, 0.1)
        back, yaw_back = t.inverse()(there, yaw)
        np.testing.assert_allclose(back, points, atol=1e-9)
        assert yaw_back == pytest.approx(0.1)

    def test_inverse_of_inverse_is_the_same_transform(self) -> None:
        t = random_transform()
        twice = t.inverse().inverse()
        assert twice.angle == t.angle
        np.testing.assert_array_equal(twice.shift, t.shift)
        np.testing.assert_array_equal(twice.pivot, t.pivot)
        np.testing.assert_array_equal(twice.offset, t.offset)
