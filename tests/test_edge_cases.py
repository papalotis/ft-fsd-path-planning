"""Edge case and smoke tests for the full pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner


def _make_curved_cones(n, y_offset=2.0, spacing=3.0):
    """Create nearly-straight cone positions with slight curvature to avoid
    degenerate circle_fit (collinear points cause division by zero)."""
    t = np.arange(n, dtype=float) * spacing
    curve = t ** 2 / (2 * 500.0)
    left = np.column_stack([t, curve + y_offset])
    right = np.column_stack([t, curve - y_offset])
    return left, right


class TestEdgeCases:
    @pytest.fixture()
    def planner(self):
        return PathPlanner(MissionTypes.trackdrive)

    def test_empty_cones(self, planner):
        """No cones at all should not crash."""
        cones = [np.zeros((0, 2)) for _ in ConeTypes]
        result = planner.calculate_path_in_global_frame(
            cones,
            vehicle_position=np.array([0.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        assert result.ndim == 2
        assert result.shape[1] == 4

    def test_single_cone(self, planner):
        """Only one cone should not crash."""
        cones = [np.zeros((0, 2)) for _ in ConeTypes]
        cones[ConeTypes.LEFT] = np.array([[5.0, 2.0]])
        result = planner.calculate_path_in_global_frame(
            cones,
            vehicle_position=np.array([0.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        assert result.ndim == 2
        assert result.shape[1] == 4

    def test_all_unknown_cones(self, planner):
        """All cones as UNKNOWN (no color info) should still produce a path."""
        left, right = _make_curved_cones(12)
        unknown = np.vstack([left, right])

        cones = [np.zeros((0, 2)) for _ in ConeTypes]
        cones[ConeTypes.UNKNOWN] = unknown

        result = planner.calculate_path_in_global_frame(
            cones,
            vehicle_position=np.array([-1.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        assert result.ndim == 2
        assert result.shape[1] == 4

    def test_cones_only_left_side(self, planner):
        """Only left-side cones should not crash."""
        n = 5
        cones = [np.zeros((0, 2)) for _ in ConeTypes]
        cones[ConeTypes.LEFT] = np.column_stack(
            [np.arange(n, dtype=float) * 3, np.full(n, 2.0)]
        )
        result = planner.calculate_path_in_global_frame(
            cones,
            vehicle_position=np.array([-1.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        assert result.ndim == 2
        assert result.shape[1] == 4

    def test_cones_only_right_side(self, planner):
        """Only right-side cones should not crash."""
        n = 5
        cones = [np.zeros((0, 2)) for _ in ConeTypes]
        cones[ConeTypes.RIGHT] = np.column_stack(
            [np.arange(n, dtype=float) * 3, np.full(n, -2.0)]
        )
        result = planner.calculate_path_in_global_frame(
            cones,
            vehicle_position=np.array([-1.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        assert result.ndim == 2
        assert result.shape[1] == 4

    def test_two_cones_total(self, planner):
        """Minimum viable input: one cone per side."""
        cones = [np.zeros((0, 2)) for _ in ConeTypes]
        cones[ConeTypes.LEFT] = np.array([[3.0, 2.0]])
        cones[ConeTypes.RIGHT] = np.array([[3.0, -2.0]])
        result = planner.calculate_path_in_global_frame(
            cones,
            vehicle_position=np.array([0.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        assert result.ndim == 2
        assert result.shape[1] == 4

    def test_direction_as_angle(self, planner):
        """Direction provided as a scalar angle should work."""
        left, right = _make_curved_cones(10)
        cones = [np.zeros((0, 2)) for _ in ConeTypes]
        cones[ConeTypes.LEFT] = left
        cones[ConeTypes.RIGHT] = right
        result = planner.calculate_path_in_global_frame(
            cones,
            vehicle_position=np.array([0.0, 0.0]),
            vehicle_direction=0.0,  # angle instead of vector
        )
        assert result.ndim == 2
        assert result.shape[1] == 4

    def test_return_intermediate_results(self, planner):
        """With return_intermediate_results=True, should return a 7-tuple."""
        left, right = _make_curved_cones(10)
        cones = [np.zeros((0, 2)) for _ in ConeTypes]
        cones[ConeTypes.LEFT] = left
        cones[ConeTypes.RIGHT] = right
        result = planner.calculate_path_in_global_frame(
            cones,
            vehicle_position=np.array([-1.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
            return_intermediate_results=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 7
        final_path, sl, sr, lv, rv, l2r, r2l = result
        assert final_path.shape[1] == 4
        assert sl.shape[1] == 2 or len(sl) == 0
        assert sr.shape[1] == 2 or len(sr) == 0
