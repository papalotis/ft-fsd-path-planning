"""Unit tests for cone sorting."""

from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning.sorting_cones.core_cone_sorting import (
    ConeSorting,
    ConeSortingInput,
)
from fsd_path_planning.sorting_cones.trace_sorter import core_trace_sorter
from fsd_path_planning.sorting_cones.trace_sorter.common import breadth_first_order
from fsd_path_planning.utils.cone_types import ConeTypes

# ── breadth_first_order ──────────────────────────────────────────────────────


class TestBreadthFirstOrder:
    def test_linear_chain(self):
        # adjacency_matrix[i] has nonzero at column j if edge i→j
        # 0 → 1 → 2 → 3
        adj = np.zeros((4, 4), dtype=np.int64)
        adj[0, 1] = 1
        adj[1, 2] = 1
        adj[2, 3] = 1
        result = breadth_first_order(adj, 0)
        np.testing.assert_array_equal(result, [0, 1, 2, 3])

    def test_single_node(self):
        adj = np.zeros((1, 1), dtype=np.int64)
        result = breadth_first_order(adj, 0)
        np.testing.assert_array_equal(result, [0])

    def test_disconnected_node(self):
        # 0 → 1, node 2 disconnected
        adj = np.zeros((3, 3), dtype=np.int64)
        adj[0, 1] = 1
        result = breadth_first_order(adj, 0)
        # Should only find nodes 0 and 1
        assert 2 not in result


# ── ConeSorting class ────────────────────────────────────────────────────────


class TestConeSorting:
    @pytest.fixture()
    def sorter(self):
        return ConeSorting(
            max_n_neighbors=5,
            max_dist=6.5,
            max_dist_to_first=6.0,
            max_length=12,
            threshold_directional_angle=np.deg2rad(40),
            threshold_absolute_angle=np.deg2rad(65),
            use_unknown_cones=True,
        )

    def test_straight_track_sorting(self, sorter):
        """Left and right cones on a straight track should be sorted into two sides."""
        n = 6
        left_cones = np.column_stack([np.arange(n, dtype=float) * 3, np.full(n, 2.0)])
        right_cones = np.column_stack([np.arange(n, dtype=float) * 3, np.full(n, -2.0)])

        cones_by_type = [np.zeros((0, 2)) for _ in ConeTypes]
        cones_by_type[ConeTypes.LEFT] = left_cones
        cones_by_type[ConeTypes.RIGHT] = right_cones

        inp = ConeSortingInput(
            cones_by_type=cones_by_type,
            vehicle_position=np.array([-1.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        sorter.set_new_input(inp)
        sorted_left, sorted_right = sorter.run_cone_sorting()

        # Should return non-empty sorted arrays
        assert len(sorted_left) > 0
        assert len(sorted_right) > 0
        assert sorted_left.shape[1] == 2
        assert sorted_right.shape[1] == 2

    def test_unknown_cones_assigned(self, sorter):
        """When cones are all unknown, sorting should still produce output."""
        n = 6
        left_cones = np.column_stack([np.arange(n, dtype=float) * 3, np.full(n, 2.0)])
        right_cones = np.column_stack([np.arange(n, dtype=float) * 3, np.full(n, -2.0)])
        unknown = np.vstack([left_cones, right_cones])

        cones_by_type = [np.zeros((0, 2)) for _ in ConeTypes]
        cones_by_type[ConeTypes.UNKNOWN] = unknown

        inp = ConeSortingInput(
            cones_by_type=cones_by_type,
            vehicle_position=np.array([-1.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        sorter.set_new_input(inp)
        sorted_left, sorted_right = sorter.run_cone_sorting()

        # Should still produce sorted output
        total = len(sorted_left) + len(sorted_right)
        assert total > 0

    def test_empty_input(self, sorter):
        """Empty cone input should not crash."""
        cones_by_type = [np.zeros((0, 2)) for _ in ConeTypes]

        inp = ConeSortingInput(
            cones_by_type=cones_by_type,
            vehicle_position=np.array([0.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        sorter.set_new_input(inp)
        sorted_left, sorted_right = sorter.run_cone_sorting()

        assert sorted_left.shape == (0, 2) or len(sorted_left) == 0
        assert sorted_right.shape == (0, 2) or len(sorted_right) == 0

    def test_sorted_output_is_ordered(self, sorter):
        """Sorted cones should form a plausible trace (consecutive distances reasonable)."""
        n = 8
        left_cones = np.column_stack([np.arange(n, dtype=float) * 3, np.full(n, 2.0)])
        right_cones = np.column_stack([np.arange(n, dtype=float) * 3, np.full(n, -2.0)])

        cones_by_type = [np.zeros((0, 2)) for _ in ConeTypes]
        cones_by_type[ConeTypes.LEFT] = left_cones
        cones_by_type[ConeTypes.RIGHT] = right_cones

        inp = ConeSortingInput(
            cones_by_type=cones_by_type,
            vehicle_position=np.array([-1.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )
        sorter.set_new_input(inp)
        sorted_left, sorted_right = sorter.run_cone_sorting()

        # Consecutive distances should be reasonable (< max_dist)
        if len(sorted_left) > 1:
            dists = np.linalg.norm(np.diff(sorted_left, axis=0), axis=1)
            assert np.all(dists < 10.0)

        if len(sorted_right) > 1:
            dists = np.linalg.norm(np.diff(sorted_right, axis=0), axis=1)
            assert np.all(dists < 10.0)

    def test_experimental_caching_reuses_previous_results(self, sorter, monkeypatch):
        """Experimental caching should skip the expensive trace search on repeat input."""
        n = 8
        left_cones = np.column_stack([np.arange(n, dtype=float) * 3, np.full(n, 2.0)])
        right_cones = np.column_stack([np.arange(n, dtype=float) * 3, np.full(n, -2.0)])

        cones_by_type = [np.zeros((0, 2)) for _ in ConeTypes]
        cones_by_type[ConeTypes.LEFT] = left_cones
        cones_by_type[ConeTypes.RIGHT] = right_cones

        inp = ConeSortingInput(
            cones_by_type=cones_by_type,
            vehicle_position=np.array([-1.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )

        original = core_trace_sorter.calc_scores_and_end_configurations
        call_count = 0

        def counted_calc_scores_and_end_configurations(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            core_trace_sorter,
            "calc_scores_and_end_configurations",
            counted_calc_scores_and_end_configurations,
        )

        sorter_without_cache = ConeSorting(
            config=sorter.config,
            experimental_performance_improvements=False,
        )
        sorter_without_cache.run_cone_sorting(inp)
        sorter_without_cache.run_cone_sorting(inp)
        assert call_count == 4

        call_count = 0

        sorter_with_cache = ConeSorting(
            config=sorter.config,
            experimental_performance_improvements=True,
        )
        sorter_with_cache.run_cone_sorting(inp)
        sorter_with_cache.run_cone_sorting(inp)

        assert call_count == 2
        assert sorter_with_cache.trace_sorter.cached_results is not None
