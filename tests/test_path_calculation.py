"""Unit tests for path calculation."""
from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning.calculate_path.core_calculate_path import (
    CalculatePath,
    PathCalculationInput,
)
from fsd_path_planning.calculate_path.path_calculator_helpers import (
    PathCalculatorHelpers,
)
from fsd_path_planning.utils.cone_types import ConeTypes


# ── PathCalculatorHelpers ────────────────────────────────────────────────────

class TestPathCalculatorHelpers:
    @pytest.fixture()
    def helpers(self):
        return PathCalculatorHelpers()

    def test_chord_path_shape(self, helpers):
        path = helpers.calculate_chord_path(
            radius=10.0, maximum_angle=np.pi / 4, number_of_points=20,
        )
        assert path.shape == (20, 2)

    def test_chord_path_starts_at_origin(self, helpers):
        path = helpers.calculate_chord_path(
            radius=10.0, maximum_angle=np.pi / 4, number_of_points=20,
        )
        np.testing.assert_allclose(path[0], [0.0, 0.0], atol=1e-10)

    def test_almost_straight_path(self, helpers):
        path = helpers.calculate_almost_straight_path()
        assert path.shape[0] > 0
        assert path.shape[1] == 2


# ── CalculatePath class ──────────────────────────────────────────────────────

class TestCalculatePath:
    @pytest.fixture()
    def calculator(self):
        return CalculatePath(
            smoothing=0.2,
            predict_every=0.1,
            maximal_distance_for_valid_path=5.0,
            max_deg=3,
            mpc_path_length=20.0,
            mpc_prediction_horizon=40,
        )

    def _make_track_input(self, n=10, spacing=3.0, y_offset=2.0):
        """Create a nearly-straight track with a slight curve to avoid
        degenerate circle_fit (perfectly collinear points cause division by
        zero in the production code).
        """
        t = np.arange(n, dtype=float) * spacing
        # Slight curve: radius ~500m, enough to avoid collinear degeneracy
        curve = t ** 2 / (2 * 500.0)
        left = np.column_stack([t, curve + y_offset])
        right = np.column_stack([t, curve - y_offset])
        l2r = np.arange(n)
        r2l = np.arange(n)
        return PathCalculationInput(
            left_cones=left,
            right_cones=right,
            left_to_right_matches=l2r,
            right_to_left_matches=r2l,
            position_global=np.array([-1.0, 0.0]),
            direction_global=np.array([1.0, 0.0]),
        )

    def test_output_shape(self, calculator):
        """Path output should be (N, 4) with [spline_param, x, y, curvature]."""
        inp = self._make_track_input(n=10)
        calculator.set_new_input(inp)
        final_path, _ = calculator.run_path_calculation()

        assert final_path.ndim == 2
        assert final_path.shape[1] == 4

    def test_path_near_centerline(self, calculator):
        """Path should be approximately centered between left and right cones."""
        inp = self._make_track_input(n=10, y_offset=2.0)
        calculator.set_new_input(inp)
        final_path, _ = calculator.run_path_calculation()

        # y-coordinates of path should be near the track centerline curve
        path_y = final_path[:, 2]
        # Centerline y values are close to the curve t^2/(2*500), which is small
        assert np.all(np.abs(path_y) < 2.0)

    def test_nearly_straight_path_low_curvature(self, calculator):
        """A nearly straight track should produce low curvature."""
        inp = self._make_track_input(n=12)
        calculator.set_new_input(inp)
        final_path, _ = calculator.run_path_calculation()

        curvature = final_path[:, 3]
        np.testing.assert_allclose(curvature, 0.0, atol=0.1)

    def test_no_matches_uses_fallback(self, calculator):
        """When all matches are -1, path calculation should still produce output."""
        n = 5
        left = np.column_stack([np.arange(n, dtype=float) * 3, np.full(n, 1.5)])
        right = np.column_stack([np.arange(n, dtype=float) * 3, np.full(n, -1.5)])
        l2r = np.full(n, -1, dtype=int)
        r2l = np.full(n, -1, dtype=int)

        inp = PathCalculationInput(
            left_cones=left,
            right_cones=right,
            left_to_right_matches=l2r,
            right_to_left_matches=r2l,
            position_global=np.array([-1.0, 0.0]),
            direction_global=np.array([1.0, 0.0]),
        )
        calculator.set_new_input(inp)
        final_path, _ = calculator.run_path_calculation()

        # Should still produce a path (fallback)
        assert final_path.ndim == 2
        assert final_path.shape[1] == 4
        assert len(final_path) > 0

    def test_prediction_horizon_length(self, calculator):
        """Output path should have approximately mpc_prediction_horizon points."""
        inp = self._make_track_input(n=12)
        calculator.set_new_input(inp)
        final_path, _ = calculator.run_path_calculation()

        assert len(final_path) == calculator.scalars.mpc_prediction_horizon
