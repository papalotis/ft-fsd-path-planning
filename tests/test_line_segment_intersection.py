"""Unit tests for line segment intersection functions."""
from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning.sorting_cones.trace_sorter.line_segment_intersection import (
    lines_segments_intersect_indicator,
    number_of_intersections,
    number_of_intersections_in_trace,
    pairwise_segment_intersection,
    trace_intersections,
)


# ── Single segment intersection ──────────────────────────────────────────────

class TestLinesSegmentsIntersectIndicator:
    def test_crossing_segments(self):
        # X-shaped crossing
        assert lines_segments_intersect_indicator(
            np.array([0.0, 0.0]), np.array([1.0, 1.0]),
            np.array([0.0, 1.0]), np.array([1.0, 0.0]),
        )

    def test_non_intersecting_parallel(self):
        # Two horizontal parallel segments
        assert not lines_segments_intersect_indicator(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]),
            np.array([0.0, 1.0]), np.array([1.0, 1.0]),
        )

    def test_non_intersecting_non_parallel(self):
        # Two segments that don't reach each other
        assert not lines_segments_intersect_indicator(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]),
            np.array([2.0, 1.0]), np.array([3.0, 2.0]),
        )

    def test_touching_at_endpoint(self):
        # T-junction: one segment ends at the other
        result = lines_segments_intersect_indicator(
            np.array([0.0, 0.0]), np.array([2.0, 0.0]),
            np.array([1.0, 0.0]), np.array([1.0, 1.0]),
        )
        assert result  # endpoint touching counts as intersection

    def test_collinear_overlapping(self):
        # Two collinear overlapping segments
        result = lines_segments_intersect_indicator(
            np.array([0.0, 0.0]), np.array([2.0, 0.0]),
            np.array([1.0, 0.0]), np.array([3.0, 0.0]),
        )
        assert result

    def test_collinear_non_overlapping(self):
        # Two collinear segments with a gap
        result = lines_segments_intersect_indicator(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]),
            np.array([2.0, 0.0]), np.array([3.0, 0.0]),
        )
        assert not result

    def test_perpendicular_crossing(self):
        assert lines_segments_intersect_indicator(
            np.array([-1.0, 0.0]), np.array([1.0, 0.0]),
            np.array([0.0, -1.0]), np.array([0.0, 1.0]),
        )


# ── Pairwise segment intersection ───────────────────────────────────────────

class TestPairwiseSegmentIntersection:
    def test_no_intersections(self):
        # Three parallel horizontal segments
        starts = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        ends = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        result = pairwise_segment_intersection(starts, ends)
        assert not result.any()

    def test_crossing_pair(self):
        starts = np.array([[0.0, 0.0], [0.0, 1.0]])
        ends = np.array([[1.0, 1.0], [1.0, 0.0]])
        result = pairwise_segment_intersection(starts, ends)
        assert result[0, 1]
        assert result[1, 0]

    def test_self_intersection_flag(self):
        starts = np.array([[0.0, 0.0]])
        ends = np.array([[1.0, 0.0]])
        result = pairwise_segment_intersection(starts, ends, intersect_with_self=True)
        assert result[0, 0]

        result_no_self = pairwise_segment_intersection(starts, ends, intersect_with_self=False)
        assert not result_no_self[0, 0]


# ── Trace intersections ─────────────────────────────────────────────────────

class TestTraceIntersections:
    def test_straight_no_intersection(self):
        # Points on a straight line → no self-intersection
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        n = number_of_intersections_in_trace(pts)
        assert n == 0

    def test_figure_eight(self):
        # A trace that crosses itself
        pts = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        n = number_of_intersections_in_trace(pts)
        assert n >= 1

    def test_simple_loop_no_self_cross(self):
        # A square loop – consecutive segments share endpoints but don't cross
        pts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        n = number_of_intersections_in_trace(pts)
        assert n == 0


# ── number_of_intersections ─────────────────────────────────────────────────

class TestNumberOfIntersections:
    def test_empty(self):
        mat = np.zeros((3, 3), dtype=bool)
        assert number_of_intersections(mat) == 0

    def test_single_crossing(self):
        mat = np.zeros((3, 3), dtype=bool)
        mat[0, 1] = mat[1, 0] = True
        assert number_of_intersections(mat) == 1
