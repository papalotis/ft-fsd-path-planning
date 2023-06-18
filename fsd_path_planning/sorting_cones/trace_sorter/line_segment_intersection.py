#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: A module for finding line segment intersections. It can be used as part
of the cost function of the sorting algorithm.

Project: fsd_path_planning
"""
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from fsd_path_planning.types import BoolArray, FloatArray, IntArray
from fsd_path_planning.utils.math_utils import my_njit

if not TYPE_CHECKING:

    @my_njit
    def cast(  # pylint: disable=function-redefined
        type_: Any, value_: Any  # pylint: disable=unused-argument
    ) -> Any:
        "Dummy numba jit function"
        return value_


@my_njit
def _make_segments_homogeneous(
    segment_a_start: FloatArray,
    segment_a_end: FloatArray,
    segment_b_start: FloatArray,
    segment_b_end: FloatArray,
) -> FloatArray:
    homogenenous = np.ones((4, 3))
    homogenenous[0, :2] = segment_a_start
    homogenenous[1, :2] = segment_a_end
    homogenenous[2, :2] = segment_b_start
    homogenenous[3, :2] = segment_b_end

    return homogenenous


@my_njit
def _handle_line_segment_intersection_parallel_case(
    segment_a_start: FloatArray,
    segment_a_end: FloatArray,
    segment_b_start: FloatArray,
    segment_b_end: FloatArray,
    epsilon: float,
) -> bool:
    # lines are parallel only one slope calculation necessary
    difference: FloatArray = segment_a_end - segment_a_start

    if difference[0] < epsilon:
        # parallel vertical line segments
        # overlap only possible if x element is the same for both line segments
        maybe_overlap = np.abs(segment_a_start[0] - segment_b_start[0]) < epsilon
        slope = np.inf
    else:
        slope = difference[1] / difference[0]
        # parallel non vertical lines
        # overlap possible only if intercept of lines is the same
        intercept_a = segment_a_start[1] - slope * segment_a_start[0]
        intercept_b = segment_b_start[1] - slope * segment_b_start[0]
        maybe_overlap = np.abs(intercept_a - intercept_b) < epsilon

    if not maybe_overlap:
        # parallel lines, different intercept, 100% no intersection
        return False

    axis_to_use = 1 if slope > 1 else 0

    if segment_a_start[axis_to_use] < segment_b_start[axis_to_use]:
        left_segment_end_scalar = segment_a_end[axis_to_use]
        right_segment_start_scalar = min(
            segment_b_start[axis_to_use], segment_b_end[axis_to_use]
        )
    else:
        left_segment_end_scalar = segment_b_end[axis_to_use]
        right_segment_start_scalar = min(
            segment_a_start[axis_to_use], segment_a_end[axis_to_use]
        )

    return_value = cast(
        bool,
        left_segment_end_scalar >= right_segment_start_scalar,
    )
    return return_value


_DEFAULT_EPSILON = 1e-6


@my_njit
def lines_segments_intersect_indicator(
    segment_a_start: FloatArray,
    segment_a_end: FloatArray,
    segment_b_start: FloatArray,
    segment_b_end: FloatArray,
    epsilon: float = _DEFAULT_EPSILON,
) -> bool:
    """
    Given the start- and endpoint of two 2d-line segments indicate if the two line segments
    intersect.

    Args:
        segment_a_start: The start point of the first line segment.
        segment_a_end: The end point of the first line segment.
        segment_b_start: The start point of the second line segment.
        segment_b_end: The end point of the second line segment.
        epsilon: The epsilon value to use when comparing floating point values.

    Returns:
        A boolean indicating if the two line segments intersect.
    """
    # Adapted from https://stackoverflow.com/a/42727584
    homogeneous = _make_segments_homogeneous(
        segment_a_start, segment_a_end, segment_b_start, segment_b_end
    )

    line_a = np.cross(homogeneous[0], homogeneous[1])  # get first line
    line_b = np.cross(homogeneous[2], homogeneous[3])  # get second line
    inter_x, inter_y, inter_z = np.cross(line_a, line_b)  # point of intersection

    # lines are parallel <=> z is zero
    # np.allclose not allowed in nopython mode
    if np.abs(inter_z) < epsilon:
        return _handle_line_segment_intersection_parallel_case(
            segment_a_start, segment_a_end, segment_b_start, segment_b_end, epsilon
        )

    # find intersection point
    intersection_x, intersection_y = np.array([inter_x / inter_z, inter_y / inter_z])

    # bounding boxes
    segment_a_left, segment_a_right = np.sort(homogeneous[:2, 0])
    segment_b_left, segment_b_right = np.sort(homogeneous[2:, 0])
    segment_a_bottom, segment_a_top = np.sort(homogeneous[:2, 1])
    segment_b_bottom, segment_b_top = np.sort(homogeneous[2:, 1])

    # check that intersection point is in both bounding boxes
    # check with a bit of epsilon for numerical stability

    return_value = (
        (segment_a_left - epsilon <= intersection_x <= segment_a_right + epsilon)
        and (segment_b_left - epsilon <= intersection_x <= segment_b_right + epsilon)
        and (segment_a_bottom - epsilon <= intersection_y <= segment_a_top + epsilon)
        and (segment_b_bottom - epsilon <= intersection_y <= segment_b_top + epsilon)
    )

    return bool(return_value)


@my_njit
def batch_lines_segments_intersect_indicator(
    segments_a_start: FloatArray,
    segments_a_end: FloatArray,
    segments_b_start: FloatArray,
    segments_b_end: FloatArray,
) -> FloatArray:
    """
    Run the lines_segments_intersect_indicator function on a batch of line segments.
    The shape of all inputs should be the same. No broadcasting is performed.

    Args:
        segments_a_start: The start points of the first line segments.
        segments_a_end: The end points of the first line segments.
        segments_b_start: The start points of the second line segments.
        segments_b_end: The end points of the second line segments.

    Returns:
        A boolean array indicating if the two line segments intersect.
    """
    assert segments_a_start.shape[-1] == 2
    assert segments_a_start.shape == segments_a_end.shape
    assert segments_a_start.shape == segments_b_start.shape
    assert segments_a_start.shape == segments_b_end.shape

    segment_a_start_flat = segments_a_start.reshape(-1, 2)
    segment_a_end_flat = segments_a_end.reshape(-1, 2)
    segment_b_start_flat = segments_b_start.reshape(-1, 2)
    segment_b_end_flat = segments_b_end.reshape(-1, 2)

    n_values_flat = len(segment_a_start_flat)

    out_flat: FloatArray = np.zeros(n_values_flat)

    for i in range(n_values_flat):
        segment_a_start_single = segment_a_start_flat[i]
        segment_a_end_single = segment_a_end_flat[i]
        segment_b_start_single = segment_b_start_flat[i]
        segment_b_end_single = segment_b_end_flat[i]

        out_flat[i] = lines_segments_intersect_indicator(
            segment_a_start_single,
            segment_a_end_single,
            segment_b_start_single,
            segment_b_end_single,
        )

    out_shape = segments_a_start.shape[:-1]
    out = out_flat.reshape(out_shape)

    return out


@my_njit
def pairwise_segment_intersection(
    segment_starts: FloatArray,
    segment_ends: FloatArray,
    intersect_with_self: bool = False,
) -> BoolArray:
    """
    For a set of line segments, find the pairwise intersections.

    Args:
        segment_starts: The start points of the line segments.
        segment_ends: The end points of the line segments.
        intersect_with_self: Whether a line segment is considered to intersect with
            itself.

    Returns:
        A square boolean array of shape (n_segments, n_segments) where the
        intersection is True if the two line segments intersect.
    """
    assert len(segment_starts) == len(segment_ends)

    number_of_segments = len(segment_starts)

    # [0,1,2] (example)
    indices = np.arange(number_of_segments)

    # [0,0,0,1,1,1,2,2,2]
    indices_first = np.repeat(indices, number_of_segments)
    # [0,1,2,0,1,2,0,1,2] (tile is not allowed in njit)
    indices_second: IntArray = (
        indices_first.reshape(number_of_segments, -1).T.copy().reshape(-1)
    )

    # keep lower triangle
    # intersect_with_self just turns the main diagonal on (done later)
    # (we don't perform any computation on self)
    # [False, True, True, False, False, True, False, False, False]
    mask_indices_to_keep: BoolArray = indices_first < indices_second
    # [0, 0, 1]
    indices_first_keep = indices_first[mask_indices_to_keep]
    # [1, 2, 2]
    indices_second_keep = indices_second[mask_indices_to_keep]

    first_starts = segment_starts[indices_first_keep]
    first_ends = segment_ends[indices_first_keep]
    second_starts = segment_starts[indices_second_keep]
    second_ends = segment_ends[indices_second_keep]

    indicator_overlap = batch_lines_segments_intersect_indicator(
        first_starts, first_ends, second_starts, second_ends
    )

    if intersect_with_self:
        indicator_matrix = np.eye(number_of_segments, dtype=np.bool_)
    else:
        indicator_matrix = np.zeros(
            (number_of_segments, number_of_segments), dtype=np.bool_
        )

    # cannot used advanced indexing twice with nopython
    # so we have to do it manually
    for index_first_single, index_second_single, indicator_overlap_single in zip(
        indices_first_keep, indices_second_keep, indicator_overlap
    ):
        indicator_matrix[
            index_first_single, index_second_single
        ] = indicator_overlap_single
        indicator_matrix[
            index_second_single, index_first_single
        ] = indicator_overlap_single

    return indicator_matrix


@my_njit
def number_of_intersections(intersection_matrix: BoolArray) -> int:
    """
    Given sorted 2d-points find the number of intersections between the segments.

    "Sorted" means that the line segments are extracted in the order that the points
    are in the array.

    Args:
        intersection_matrix: The square boolean array of shape (n_segments, n_segments)
            where the intersection is True if the two line segments intersect.
    Returns:
        The number of intersections between the segments.
    """

    # we only count the lower triangle, otherwise we double count intersections not
    # on the diagonal because the matrix is symmetric
    lower_triangle_matrix = np.tril(intersection_matrix)  # type: ignore
    return np.count_nonzero(lower_triangle_matrix)


@my_njit
def trace_intersections(
    points: FloatArray,
    intersect_with_consecutive_segments: bool = False,
    intersect_with_self: bool = False,
) -> BoolArray:
    """
    Given sorted 2d-points find the intersections between the segments.

    "Sorted" means that the line segments are extracted in the order that the points
    are in the array.

    Args:
        points: The points to trace. Shape is (n_points, 2).
        intersect_with_consecutive_segments: Whether to include the pairwise
            intersections with consecutive segments. By definition consecutive segments
            intersect because they share a point.
        intersect_with_self: Whether a line segment is considered to intersect with
            itself.

    Returns:
        A square boolean array of shape (n_points - 1, n_points - 1) where the elements
        indicate if the two line segments intersect.
    """
    segment_starts = points[:-1]
    segment_ends = points[1:]

    intersections = pairwise_segment_intersection(
        segment_starts,
        segment_ends,
        intersect_with_self=intersect_with_self,
    )
    if not intersect_with_consecutive_segments:
        # set the diagonals next to the main diagonal (see np.eye with k=1,-1)
        # to be false
        for i in range(len(segment_starts)):
            next_diagonal_index = i + 1
            intersections[i, next_diagonal_index] = False
            intersections[next_diagonal_index, i] = False

    return intersections


@my_njit
def number_of_intersections_in_trace(
    points: FloatArray,
    intersect_with_consecutive_segments: bool = False,
    intersect_with_self: bool = False,
) -> int:
    """
    Given sorted 2d-points find the number of intersections between the segments.

    "Sorted" means that the line segments are extracted in the order that the points
    are in the array.

    Args:
        points: The points to trace. Shape is (n_points, 2).
        intersect_with_consecutive_segments: Whether to include the pairwise
            intersections with consecutive segments. By definition consecutive segments
            intersect because they share a point.
        intersect_with_self: Whether a line segment is considered to intersect with
            itself.

    Returns:
        The number of intersections between the segments.
    """
    intersections = trace_intersections(
        points,
        intersect_with_consecutive_segments=intersect_with_consecutive_segments,
        intersect_with_self=intersect_with_self,
    )
    return number_of_intersections(intersections)


@my_njit
def number_of_intersections_in_configurations(
    points: FloatArray, configurations: IntArray
) -> IntArray:
    """
    Calculate the number of intersections for a given set of configurations of 2d
    points.

    Args:
        points: The 2d point cloud. Shape is (n_points, 2).
        configurations: The configurations of the points. Shape is
            (n_configurations, n_points).

    Returns:
        The number of intersections for each configuration. Shape is
        (n_configurations,).
    """
    number_of_configurations = len(configurations)

    result_array = np.zeros(number_of_configurations, dtype=np.int64)

    for i in range(number_of_configurations):
        configuration = configurations[i]
        configuration_filtered = configuration[configuration != -1]
        points_configuration = points[configuration_filtered]
        number_of_intersections_for_configuration = number_of_intersections_in_trace(
            points_configuration
        )
        result_array[i] = number_of_intersections_for_configuration

    return result_array
