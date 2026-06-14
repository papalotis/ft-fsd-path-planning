"""Unit tests for line segment intersection functions."""

from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning.sorting_cones.trace_sorter.line_segment_intersection import (
    batch_lines_segments_intersect_indicator,
    lines_segments_intersect_indicator,
    number_of_intersections,
    number_of_intersections_in_configurations,
    number_of_intersections_in_trace,
    pairwise_segment_intersection,
    trace_intersections,
)

# ── Single segment intersection ──────────────────────────────────────────────


class TestLinesSegmentsIntersectIndicator:
    def test_crossing_segments(self):
        # X-shaped crossing
        assert lines_segments_intersect_indicator(
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
        )

    def test_non_intersecting_parallel(self):
        # Two horizontal parallel segments
        assert not lines_segments_intersect_indicator(
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
        )

    def test_non_intersecting_non_parallel(self):
        # Two segments that don't reach each other
        assert not lines_segments_intersect_indicator(
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([2.0, 1.0]),
            np.array([3.0, 2.0]),
        )

    def test_touching_at_endpoint(self):
        # T-junction: one segment ends at the other
        result = lines_segments_intersect_indicator(
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([1.0, 1.0]),
        )
        assert result  # endpoint touching counts as intersection

    def test_collinear_overlapping(self):
        # Two collinear overlapping segments
        result = lines_segments_intersect_indicator(
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([3.0, 0.0]),
        )
        assert result

    def test_collinear_non_overlapping(self):
        # Two collinear segments with a gap
        result = lines_segments_intersect_indicator(
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([3.0, 0.0]),
        )
        assert not result

    def test_perpendicular_crossing(self):
        assert lines_segments_intersect_indicator(
            np.array([-1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, -1.0]),
            np.array([0.0, 1.0]),
        )

    def test_parallel_vertical_overlapping_segments(self):
        assert lines_segments_intersect_indicator(
            np.array([1.0, 0.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 3.0]),
        )

    def test_parallel_vertical_non_overlapping_segments(self):
        assert not lines_segments_intersect_indicator(
            np.array([1.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 3.0]),
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

        result_no_self = pairwise_segment_intersection(
            starts, ends, intersect_with_self=False
        )
        assert not result_no_self[0, 0]

    def test_mismatched_lengths_raise_value_error(self):
        starts = np.array([[0.0, 0.0], [0.0, 1.0]])
        ends = np.array([[1.0, 0.0]])

        with pytest.raises(
            ValueError,
            match="segment_starts and segment_ends must have the same length",
        ):
            pairwise_segment_intersection(starts, ends)


class TestBatchLinesSegmentsIntersectIndicator:
    def test_invalid_point_shape_raises_value_error(self):
        starts = np.array([[0.0, 0.0, 0.0]])
        ends = np.array([[1.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="segment inputs must contain 2d points"):
            batch_lines_segments_intersect_indicator(starts, ends, starts, ends)

    def test_preserves_batch_shape(self):
        segments_a_start = np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )
        segments_a_end = np.array(
            [
                [[1.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 1.0]],
            ]
        )
        segments_b_start = np.array(
            [
                [[0.0, 1.0], [2.0, 1.0]],
                [[0.0, 1.0], [2.0, 0.0]],
            ]
        )
        segments_b_end = np.array(
            [
                [[1.0, 0.0], [3.0, 1.0]],
                [[1.0, 1.0], [3.0, 0.0]],
            ]
        )

        result = batch_lines_segments_intersect_indicator(
            segments_a_start,
            segments_a_end,
            segments_b_start,
            segments_b_end,
        )

        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, np.array([[1.0, 0.0], [0.0, 0.0]]))


# ── Trace intersections ─────────────────────────────────────────────────────


class TestTraceIntersections:
    def test_trace_intersections_skip_consecutive_segments_by_default(self):
        pts = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )

        intersections = trace_intersections(pts)

        assert not intersections.any()

    def test_trace_intersections_can_include_consecutive_segments(self):
        pts = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )

        intersections = trace_intersections(
            pts,
            intersect_with_consecutive_segments=True,
        )

        assert intersections[0, 1]
        assert intersections[1, 0]
        assert intersections[1, 2]
        assert intersections[2, 1]

    def test_straight_no_intersection(self):
        # Points on a straight line → no self-intersection
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        n = number_of_intersections_in_trace(pts)
        assert n == 0

    def test_figure_eight(self):
        # A trace that crosses itself
        pts = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        n = number_of_intersections_in_trace(pts)
        assert n >= 1

    def test_simple_loop_no_self_cross(self):
        # A square loop – consecutive segments share endpoints but don't cross
        pts = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
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


class TestNumberOfIntersectionsInConfigurations:
    def test_handles_padded_configurations(self):
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 1.0],
            ]
        )
        configurations = np.array(
            [
                [0, 1, 2, 3, -1],
                [0, 2, 4, -1, -1],
            ]
        )

        result = number_of_intersections_in_configurations(points, configurations)

        np.testing.assert_array_equal(result, np.array([1, 0]))
