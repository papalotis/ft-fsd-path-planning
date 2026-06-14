"""Edge case and smoke tests for the full pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner


def _make_curved_cones(n, y_offset=2.0, spacing=3.0):
    """Create nearly-straight cone positions with slight curvature to avoid
    degenerate circle_fit (collinear points cause division by zero)."""
    t = np.arange(n, dtype=float) * spacing
    curve = t**2 / (2 * 500.0)
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


class TestInputValidation:
    @pytest.fixture()
    def planner(self):
        return PathPlanner(MissionTypes.trackdrive)

    def test_invalid_cone_shape_fails_early(self, planner):
        cones = [np.zeros((0, 2)) for _ in ConeTypes]
        cones[ConeTypes.LEFT] = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match=r"cones\[2\] must have shape \(N, 2\)"):
            planner.calculate_path_in_global_frame(
                cones,
                vehicle_position=np.array([0.0, 0.0]),
                vehicle_direction=np.array([1.0, 0.0]),
            )

    def test_nan_vehicle_position_fails_early(self, planner):
        cones = [np.zeros((0, 2)) for _ in ConeTypes]

        with pytest.raises(
            ValueError, match="vehicle_position must contain only finite values"
        ):
            planner.calculate_path_in_global_frame(
                cones,
                vehicle_position=np.array([np.nan, 0.0]),
                vehicle_direction=np.array([1.0, 0.0]),
            )

    def test_invalid_direction_shape_fails_early(self, planner):
        cones = [np.zeros((0, 2)) for _ in ConeTypes]

        with pytest.raises(
            ValueError, match="direction must be a float or a 2 element array"
        ):
            planner.calculate_path_in_global_frame(
                cones,
                vehicle_position=np.array([0.0, 0.0]),
                vehicle_direction=np.array([1.0, 0.0, 0.0]),
            )

    def test_non_numeric_cones_fail_early(self, planner):
        cones = [np.zeros((0, 2)) for _ in ConeTypes]
        cones[ConeTypes.LEFT] = np.array([["left", "cone"]], dtype=object)

        with pytest.raises(TypeError, match=r"cones\[2\] must contain numeric values"):
            planner.calculate_path_in_global_frame(
                cones,
                vehicle_position=np.array([0.0, 0.0]),
                vehicle_direction=np.array([1.0, 0.0]),
            )

    def test_zero_direction_vector_fails_early(self, planner):
        cones = [np.zeros((0, 2)) for _ in ConeTypes]

        with pytest.raises(ValueError, match="direction vector must not be zero"):
            planner.calculate_path_in_global_frame(
                cones,
                vehicle_position=np.array([0.0, 0.0]),
                vehicle_direction=np.array([0.0, 0.0]),
            )

    def test_wrong_number_of_cone_arrays_fails_early(self, planner):
        cones = [np.zeros((0, 2)) for _ in range(len(ConeTypes) - 1)]

        with pytest.raises(ValueError, match=r"cones must contain 5 arrays"):
            planner.calculate_path_in_global_frame(
                cones,
                vehicle_position=np.array([0.0, 0.0]),
                vehicle_direction=np.array([1.0, 0.0]),
            )

    def test_inf_in_cones_fails_early(self, planner):
        cones = [np.zeros((0, 2)) for _ in ConeTypes]
        cones[ConeTypes.LEFT] = np.array([[1.0, np.inf]])

        with pytest.raises(
            ValueError, match=r"cones\[2\] must contain only finite values"
        ):
            planner.calculate_path_in_global_frame(
                cones,
                vehicle_position=np.array([0.0, 0.0]),
                vehicle_direction=np.array([1.0, 0.0]),
            )

    def test_non_numeric_vehicle_position_fails_early(self, planner):
        cones = [np.zeros((0, 2)) for _ in ConeTypes]

        with pytest.raises(
            TypeError, match="vehicle_position must contain numeric values"
        ):
            planner.calculate_path_in_global_frame(
                cones,
                vehicle_position=np.array(["x", "y"], dtype=object),
                vehicle_direction=np.array([1.0, 0.0]),
            )

    def test_non_numeric_direction_fails_early(self, planner):
        cones = [np.zeros((0, 2)) for _ in ConeTypes]

        with pytest.raises(TypeError, match="direction must contain numeric values"):
            planner.calculate_path_in_global_frame(
                cones,
                vehicle_position=np.array([0.0, 0.0]),
                vehicle_direction=np.array(["forward", "left"], dtype=object),
            )

    def test_nan_direction_angle_fails_early(self, planner):
        cones = [np.zeros((0, 2)) for _ in ConeTypes]

        with pytest.raises(
            ValueError, match="direction must contain only finite values"
        ):
            planner.calculate_path_in_global_frame(
                cones,
                vehicle_position=np.array([0.0, 0.0]),
                vehicle_direction=np.nan,
            )
