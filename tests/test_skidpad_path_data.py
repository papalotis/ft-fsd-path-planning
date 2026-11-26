"""The stored skidpad spline must reproduce the path it replaced."""

from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning.geometry.splines import SplineFitterFactory
from fsd_path_planning.relocalization.skidpad.skidpad_path_data import (
    BASE_SKIDPAD_PATH,
    load_skidpad_path_spline,
    sample_skidpad_path,
)
from fsd_path_planning.relocalization.skidpad.skidpad_relocalizer import (
    SkidpadRelocalizer,
    calculate_reference_centers_for_skidpad_path,
)

# Skidpad geometry from the Formula Student rules, in metres.
CIRCLE_RADIUS = 9.125
CENTER_DISTANCE = 18.25


class TestStoredPath:
    def test_has_the_expected_shape(self) -> None:
        assert BASE_SKIDPAD_PATH.shape == (5786, 2)
        assert BASE_SKIDPAD_PATH.dtype == np.float64

    def test_is_sampled_about_every_five_centimetres(self) -> None:
        spacing = np.linalg.norm(np.diff(BASE_SKIDPAD_PATH, axis=0), axis=1)
        assert spacing.min() > 0.049
        assert spacing.max() < 0.051

    def test_total_length_matches_the_released_path(self) -> None:
        length = np.linalg.norm(np.diff(BASE_SKIDPAD_PATH, axis=0), axis=1).sum()
        assert length == pytest.approx(289.256, abs=0.01)

    def test_sampling_twice_gives_the_same_path(self) -> None:
        np.testing.assert_array_equal(sample_skidpad_path(), BASE_SKIDPAD_PATH)


class TestGeometryIsPreserved:
    def test_the_two_circle_centres_are_where_the_rules_put_them(self) -> None:
        centers = calculate_reference_centers_for_skidpad_path(BASE_SKIDPAD_PATH)
        assert centers.shape == (2, 2)
        distance = float(np.linalg.norm(centers[0] - centers[1]))
        assert distance == pytest.approx(CENTER_DISTANCE, abs=0.05)

    def test_the_loops_have_the_expected_radius(self) -> None:
        centers = calculate_reference_centers_for_skidpad_path(BASE_SKIDPAD_PATH)
        for center, side in zip(centers, (-1.0, 1.0), strict=True):
            loop = BASE_SKIDPAD_PATH[
                np.sign(BASE_SKIDPAD_PATH[:, 1] - center[1]) == side
            ]
            if len(loop) == 0:
                continue
            radii = np.linalg.norm(loop - center, axis=1)
            assert radii.max() == pytest.approx(CIRCLE_RADIUS, abs=0.2)

    def test_the_relocalizer_uses_the_stored_path(self) -> None:
        relocalizer = SkidpadRelocalizer()
        expected = calculate_reference_centers_for_skidpad_path(BASE_SKIDPAD_PATH)
        np.testing.assert_allclose(relocalizer.reference_centers, expected)

    def test_known_global_path_is_every_second_point(self) -> None:
        relocalizer = SkidpadRelocalizer()
        np.testing.assert_array_equal(
            relocalizer.get_known_global_path(), BASE_SKIDPAD_PATH[::2]
        )


class TestSplineRoundTrip:
    def test_save_and_load_preserve_the_curve(self, tmp_path) -> None:
        spline = load_skidpad_path_spline()
        destination = tmp_path / "spline.npz"
        spline.save(destination)

        from fsd_path_planning.geometry.splines import SplineEvaluator

        reloaded = SplineEvaluator.load(destination)
        assert reloaded.max_u == spline.max_u
        assert reloaded.predict_every == spline.predict_every

        u = np.linspace(0, spline.max_u, 500)
        np.testing.assert_allclose(spline.evaluate_at(u), reloaded.evaluate_at(u))

    def test_a_fit_by_index_recovers_the_points_it_skipped(self) -> None:
        """The property the stored path relies on, on a curve of our own."""
        angles = np.linspace(0.0, 2.0, 400)
        points = np.column_stack([np.cos(angles) * 9, np.sin(angles) * 9])

        indices = np.arange(0, len(points), 20)
        factory = SplineFitterFactory(smoothing=0.0, predict_every=1.0, max_deg=3)
        spline = factory.fit(points[indices], u_fit=indices.astype(float))

        rebuilt = spline.evaluate_at(np.arange(indices[-1] + 1, dtype=float))
        error = np.linalg.norm(rebuilt - points[: len(rebuilt)], axis=1)
        assert error.max() < 1e-3
