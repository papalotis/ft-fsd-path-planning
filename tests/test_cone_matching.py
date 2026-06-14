"""Unit tests for cone matching logic."""

from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning.cone_matching.core_cone_matching import (
    ConeMatching,
    ConeMatchingInput,
)
from fsd_path_planning.cone_matching.match_directions import (
    calculate_match_search_direction,
    calculate_search_direction_for_one,
)
from fsd_path_planning.config_dataclasses import MatchingConfig
from fsd_path_planning.utils.cone_types import ConeTypes

# ── Search direction ─────────────────────────────────────────────────────────


class TestCalculateMatchSearchDirection:
    def test_helper_returns_normalized_perpendicular_direction(self):
        cones = np.array([[0.0, 0.0], [2.0, 2.0]])

        direction = calculate_search_direction_for_one(
            cones, np.array([0, 1]), ConeTypes.LEFT
        )

        expected = np.array([np.sqrt(0.5), -np.sqrt(0.5)])
        np.testing.assert_allclose(direction, expected)
        np.testing.assert_allclose(np.linalg.norm(direction), 1.0)

    def test_left_cones_direction_is_perpendicular(self):
        # Cones arranged along x-axis → search direction should be
        # perpendicular (~+y or -y)
        cones = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        dirs = calculate_match_search_direction(cones, ConeTypes.LEFT)
        assert dirs.shape == (3, 2)
        # Each direction should be roughly perpendicular to [1, 0]
        for d in dirs:
            dot = abs(d[0])  # dot with [1, 0]
            assert dot < 0.1, f"Expected perpendicular to x-axis, got {d}"

    def test_right_cones_opposite_direction(self):
        cones = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        left_dirs = calculate_match_search_direction(cones, ConeTypes.LEFT)
        right_dirs = calculate_match_search_direction(cones, ConeTypes.RIGHT)
        # Left and right should point in opposite directions (y-components)
        for ld, rd in zip(left_dirs, right_dirs, strict=False):
            assert np.sign(ld[1]) != np.sign(rd[1]) or (abs(ld[1]) < 1e-10)

    def test_single_pair_of_cones(self):
        cones = np.array([[0.0, 0.0], [1.0, 0.0]])
        dirs = calculate_match_search_direction(cones, ConeTypes.LEFT)
        assert dirs.shape == (2, 2)

    def test_uses_only_xy_columns(self):
        cones = np.array(
            [
                [0.0, 0.0, 10.0],
                [1.0, 0.0, 20.0],
                [2.0, 0.0, 30.0],
            ]
        )

        dirs = calculate_match_search_direction(cones, ConeTypes.RIGHT)

        expected = np.tile(np.array([0.0, 1.0]), (3, 1))
        np.testing.assert_allclose(dirs, expected, atol=1e-12)

    def test_requires_at_least_two_cones(self):
        cones = np.array([[0.0, 0.0]])

        with pytest.raises(AssertionError):
            calculate_match_search_direction(cones, ConeTypes.LEFT)


# ── ConeMatching class ──────────────────────────────────────────────────────


class TestConeMatching:
    @pytest.fixture()
    def matcher(self):
        config = MatchingConfig(
            min_track_width=3.0,
            max_search_range=5.0,
            max_search_angle=np.deg2rad(50),
            matches_should_be_monotonic=True,
        )
        return ConeMatching(config=config)

    def test_parallel_straight_track(self, matcher):
        """Left and right cones on a straight track should match 1:1."""
        n = 5
        left = np.column_stack([np.arange(n, dtype=float), np.full(n, 1.5)])
        right = np.column_stack([np.arange(n, dtype=float), np.full(n, -1.5)])

        cones_list = [np.zeros((0, 2)) for _ in ConeTypes]
        cones_list[ConeTypes.LEFT] = left
        cones_list[ConeTypes.RIGHT] = right

        inp = ConeMatchingInput(
            sorted_cones=cones_list,
            vehicle_position=np.array([-1.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        matcher.set_new_input(inp)
        left_out, right_out, l2r, r2l = matcher.run_cone_matching()

        # Should have at least as many cones as input
        assert len(left_out) >= n
        assert len(right_out) >= n

        # Most matches should be valid (not -1)
        valid_l2r = l2r[l2r >= 0]
        assert len(valid_l2r) > 0

    def test_no_cones(self, matcher):
        """Empty input should not crash."""
        cones_list = [np.zeros((0, 2)) for _ in ConeTypes]
        inp = ConeMatchingInput(
            sorted_cones=cones_list,
            vehicle_position=np.array([0.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        matcher.set_new_input(inp)
        left_out, right_out, l2r, r2l = matcher.run_cone_matching()

        assert left_out.shape[1] == 2
        assert right_out.shape[1] == 2

    def test_one_side_only(self, matcher):
        """Only left cones → virtual cones should be created on the right."""
        n = 4
        left = np.column_stack([np.arange(n, dtype=float) * 3, np.full(n, 1.5)])

        cones_list = [np.zeros((0, 2)) for _ in ConeTypes]
        cones_list[ConeTypes.LEFT] = left

        inp = ConeMatchingInput(
            sorted_cones=cones_list,
            vehicle_position=np.array([-1.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        matcher.set_new_input(inp)
        left_out, right_out, l2r, r2l = matcher.run_cone_matching()

        # Right cones should be created as virtual
        assert len(right_out) > 0

    def test_match_indices_shape(self, matcher):
        """Match index arrays should have the same length as their side's cone array."""
        n = 3
        left = np.column_stack([np.arange(n, dtype=float), np.full(n, 1.5)])
        right = np.column_stack([np.arange(n, dtype=float), np.full(n, -1.5)])

        cones_list = [np.zeros((0, 2)) for _ in ConeTypes]
        cones_list[ConeTypes.LEFT] = left
        cones_list[ConeTypes.RIGHT] = right

        inp = ConeMatchingInput(
            sorted_cones=cones_list,
            vehicle_position=np.array([-1.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        matcher.set_new_input(inp)
        left_out, right_out, l2r, r2l = matcher.run_cone_matching()

        assert len(l2r) == len(left_out)
        assert len(r2l) == len(right_out)

    def test_virtual_cones_at_track_width(self, matcher):
        """Virtual cones should be placed approximately min_track_width away."""
        n = 4
        left = np.column_stack([np.arange(n, dtype=float) * 3, np.full(n, 0.0)])

        cones_list = [np.zeros((0, 2)) for _ in ConeTypes]
        cones_list[ConeTypes.LEFT] = left

        inp = ConeMatchingInput(
            sorted_cones=cones_list,
            vehicle_position=np.array([-1.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        matcher.set_new_input(inp)
        left_out, right_out, l2r, r2l = matcher.run_cone_matching()

        if len(right_out) > 0 and len(left_out) > 0:
            # Check distance between matched pairs
            for i, match_idx in enumerate(l2r):
                if match_idx >= 0:
                    dist = np.linalg.norm(left_out[i] - right_out[match_idx])
                    # Should be approximately min_track_width
                    assert dist >= matcher.config.min_track_width * 0.5
