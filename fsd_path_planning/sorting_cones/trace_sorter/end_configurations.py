#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This File calculates all the possible paths
Project: fsd_path_planning
"""
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from fsd_path_planning.sorting_cones.trace_sorter.common import NoPathError
from fsd_path_planning.sorting_cones.trace_sorter.line_segment_intersection import (
    cast, lines_segments_intersect_indicator)
from fsd_path_planning.types import (BoolArray, FloatArray, GenericArray,
                                     IntArray)
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (my_in1d, my_njit,
                                                vec_angle_between)


@my_njit
def adjacency_matrix_to_borders_and_targets(
    adjacency_matrix: IntArray,
) -> Tuple[IntArray, IntArray]:
    """
    Convert an adjacency matrix to two flat arrays, one representing the neighbors of
    each node and one which indicates the starting index of each node in the neighbors
    array
    [
        [0 1 0]
        [1 1 1]
        [1 0 0]
    ]
    is converted to
    neighbors -> [1, 0, 1, 2, 0]
    borders -> [0, 1, 4, 5]
    Args:
        adjacency_matrix: The adjacency matrix to convert
    Returns:
        The neighbors and the starting position of each node in the neighbors array
    """
    source, neighbors_flat = np.where(adjacency_matrix)

    borders: IntArray = np.zeros((adjacency_matrix.shape[0], 2), dtype=np.int32) - 1
    borders[0] = [0, 0]

    for index in range(len(neighbors_flat)):
        source_idx = source[index]

        if borders[source_idx][0] == -1:
            borders[source_idx][0] = index

        borders[source_idx][1] = index + 1

    # for all the nodes that have no neighbors set their start and end to the
    # previous node's end
    for i, value in enumerate(borders):
        if value[0] == -1:
            previous = borders[i - 1][1]
            borders[i] = [previous, previous]

    final_borders = np.concatenate((borders[:, 0], borders[-1:, -1]))

    return neighbors_flat, final_borders


@my_njit
def double_stack_len(stack: GenericArray) -> GenericArray:
    """
    Double the capacity of a stack (used in the path search)
    Args:
        stack: The stack whose size will be doubled
    Returns:
        A copy of the stack with double the capacity
    """
    _len = stack.shape[0]
    new_shape = (_len * 2, *stack.shape[1:])
    new_buffer = np.full(new_shape, -1, dtype=stack.dtype)

    new_buffer[:_len] = stack
    return new_buffer


@my_njit
def resize_stack_if_needed(stack: GenericArray, stack_pointer: int) -> GenericArray:
    """
    Given a stack and its current pointer resize the stack if adding one more element
    would result in an index error
    Args:
        stack: The stack to potentially resize
        stack_pointer: The current position (filled size) of the stack
    Returns:
        np.ndarray: The stack, if resized then a copy is returned
    """
    if stack_pointer >= stack.shape[0]:
        stack = double_stack_len(stack)

    return stack


# for numba
FLOAT = float if TYPE_CHECKING else np.float32


@my_njit
def neighbor_bool_mask_can_be_added_to_attempt(
    trace: FloatArray,
    cone_type: ConeTypes,
    current_attempt: IntArray,
    position_in_stack: int,
    neighbors: IntArray,
    threshold_directional_angle: float,
    threshold_absolute_angle: float,
    car_position: FloatArray,
    car_direction: FloatArray,
    car_size: float,
) -> BoolArray:
    car_direction_normalized = car_direction / np.linalg.norm(car_direction)

    # neighbor can be added if not in current attempt
    can_be_added = ~my_in1d(neighbors, current_attempt[: position_in_stack + 1])
    for i in range(len(can_be_added)):
        if not can_be_added[i]:
            continue

        # calculate angle between second to last to last vector in attempt
        # and the vector between the last node and the candidate neighbor
        # add to current attempt only if the angle between the current last
        # vector and the potential new last vector is less than a specific
        # threshold. there are two thresholds, one is the maximum angle in a specific direction
        # for blue cones that is counter-clockwise and for yellow cones that is clockwise
        # the second threshold is an absolute angle between the two vectors.
        candidate_neighbor = trace[neighbors[i]]
        # XXX: There might be a bug where the can_be_added[i] is set to false and then
        # back to true
        if position_in_stack >= 1:
            second_to_last_in_attempt = trace[current_attempt[position_in_stack - 1]]
            last_in_attempt = trace[current_attempt[position_in_stack]]
            second_to_last_to_last = last_in_attempt - second_to_last_in_attempt
            last_to_candidate = candidate_neighbor - last_in_attempt
            angle_1 = cast(
                FLOAT,
                np.arctan2(second_to_last_to_last[1], second_to_last_to_last[0]),
            )
            angle_2 = cast(
                FLOAT,
                np.arctan2(last_to_candidate[1], last_to_candidate[0]),
            )
            # order is important here
            difference = angle_difference(angle_2, angle_1)
            len_last_to_candidate = np.linalg.norm(last_to_candidate)

            if np.abs(difference) > threshold_absolute_angle:
                can_be_added[i] = False
            elif cone_type == ConeTypes.LEFT:
                can_be_added[i] = (
                    difference < threshold_directional_angle
                    or len_last_to_candidate < 4.5
                )
            elif cone_type == ConeTypes.RIGHT:
                can_be_added[i] = (
                    difference > -threshold_directional_angle
                    or len_last_to_candidate < 4.5
                )
            else:
                raise AssertionError("Unreachable code")

        if position_in_stack == 1:
            start = trace[current_attempt[0]]
            diff = candidate_neighbor - start
            direction_offset = vec_angle_between(car_direction, diff)
            can_be_added[i] &= direction_offset < np.pi / 3

        
        if position_in_stack >= 0:
            # make sure that no intersection with car occurs
            last_in_attempt = trace[current_attempt[position_in_stack]]
            car_start = car_position - car_direction_normalized * car_size / 2
            car_end = car_position + car_direction_normalized * car_size

            can_be_added[i] &= not lines_segments_intersect_indicator(
                last_in_attempt, candidate_neighbor, car_start, car_end
            )
            

    return can_be_added


@my_njit
def angle_difference(angle1: FloatArray, angle2: FloatArray) -> FloatArray:
    """
    Calculate the difference between two angles. The range of the difference is [-pi, pi].
    The order of the angles *is* important.

    Args:
        angle1: First angle.
        angle2: Second angle.

    Returns:
        The difference between the two angles.
    """
    return cast(FloatArray, (angle1 - angle2 + 3 * np.pi) % (2 * np.pi) - np.pi)  # type: ignore


@my_njit
def _impl_find_all_end_configurations(
    trace: FloatArray,
    cone_type: ConeTypes,
    start_idx: int,
    adjacency_neighbors: IntArray,
    adjacency_borders: IntArray,
    target_length: int,
    threshold_directional_angle: float,
    threshold_absolute_angle: float,
    first_k_indices_must_be: IntArray,
    car_position: FloatArray,
    car_direction: FloatArray,
    car_size: float,
    store_all_end_configurations: bool,
) -> tuple[IntArray, Optional[tuple[IntArray, BoolArray]]]:
    """
    Finds all the possible paths up to length target length. If a path
    Args:
        start_idx: The index of the starting node
        adjacency_neighbors: The indices of the sink for each edge
        adjacency_borders: The start position of the indices of each node
        target_length: The length of the path that the search is searching for
    Returns:
        A 2d array of configurations (indices) that define all the valid paths
    """
    end_configurations: IntArray = np.full((10, target_length), -1, dtype=np.int32)
    end_configurations_pointer = 0
    stack: IntArray = np.zeros((10, 2), dtype=np.int32)
    stack_end_pointer = 0
    current_attempt: IntArray = np.zeros(target_length, dtype=np.int32) - 1
    if len(first_k_indices_must_be) > 0:
        pos = len(first_k_indices_must_be) - 1
        current_attempt[:pos] = first_k_indices_must_be[:-1]
        stack[0] = [first_k_indices_must_be[-1], pos]
    else:
        stack[0] = [start_idx, 0]

    if store_all_end_configurations:
        all_configurations_counter = 0
        all_configurations = end_configurations.copy()
        configuration_is_end = np.zeros(end_configurations.shape[0], dtype=np.bool_)

    while stack_end_pointer >= 0:
        # pop the index and the position from the stack
        next_idx, position_in_stack = stack[stack_end_pointer]
        stack_end_pointer -= 1

        # add the node to the current path
        current_attempt[position_in_stack] = next_idx
        current_attempt[position_in_stack + 1 :] = -1

        # get the neighbors of the last node in the attempt
        neighbors = adjacency_neighbors[
            adjacency_borders[next_idx] : adjacency_borders[next_idx + 1]
        ]

        can_be_added = neighbor_bool_mask_can_be_added_to_attempt(
            trace,
            cone_type,
            current_attempt,
            position_in_stack,
            neighbors,
            threshold_directional_angle,
            threshold_absolute_angle,
            car_position,
            car_direction,
            car_size,
        )

        has_valid_neighbors = position_in_stack < target_length - 1 and np.any(
            can_be_added
        )
        # check that we haven't hit target length and that we have neighbors to add
        if has_valid_neighbors:
            for i in range(len(can_be_added)):
                if not can_be_added[i]:
                    continue

                stack_end_pointer += 1

                stack = resize_stack_if_needed(stack, stack_end_pointer)
                stack[stack_end_pointer] = [
                    neighbors[i],
                    position_in_stack + 1,
                ]

        # leaf
        else:
            end_configurations = resize_stack_if_needed(
                end_configurations, end_configurations_pointer
            )

            end_configurations[end_configurations_pointer:] = current_attempt.copy()

            end_configurations_pointer += 1

        if store_all_end_configurations:
            all_configurations = resize_stack_if_needed(
                all_configurations, all_configurations_counter
            )
            configuration_is_end = resize_stack_if_needed(
                configuration_is_end, all_configurations_counter
            )
            all_configurations[all_configurations_counter] = current_attempt
            configuration_is_end[all_configurations_counter] = not has_valid_neighbors
            all_configurations_counter += 1

    return_value_end_configurations: IntArray = end_configurations[
        :end_configurations_pointer
    ]

    mask_end_configurations_with_more_that_two_nodes = (
        return_value_end_configurations != -1
    ).sum(axis=1) > 2
    return_value_end_configurations = return_value_end_configurations[
        mask_end_configurations_with_more_that_two_nodes
    ]

    if store_all_end_configurations:
        all_configurations = all_configurations[:all_configurations_counter]
        configuration_is_end = configuration_is_end[:all_configurations_counter]

    config_history = None
    if store_all_end_configurations:
        config_history = (all_configurations, configuration_is_end)

    return return_value_end_configurations, config_history


def find_all_end_configurations(
    points: FloatArray,
    cone_type: ConeTypes,
    start_idx: int,
    adjacency_matrix: IntArray,
    target_length: int,
    threshold_directional_angle: float,
    threshold_absolute_angle: float,
    first_k_indices_must_be: IntArray,
    car_position: FloatArray,
    car_direction: FloatArray,
    car_size: float,
    store_all_end_configurations: bool,
) -> tuple[IntArray, Optional[tuple[IntArray, BoolArray]]]:
    """
    Finds all the possible paths that include all the reachable nodes from the starting
    Args:
        start_idx: The index of the starting node
        adjacency_matrix: The adjacency matrix indicating which nodes are
        connected to which
        target_length: The length of the path that the search is searching for
    Raises:
        NoPathError: If no path has been found
    Returns:
        A 2d array of configurations (indices) that define a path
    """
    # print(locals())
    neighbors_flat, borders = adjacency_matrix_to_borders_and_targets(adjacency_matrix)

    (
        end_configurations,
        all_configurations_and_is_end_configuration_indicator,
    ) = _impl_find_all_end_configurations(
        points,
        cone_type,
        start_idx,
        neighbors_flat,
        borders,
        target_length,
        threshold_directional_angle,
        threshold_absolute_angle,
        first_k_indices_must_be,
        car_position,
        car_direction,
        car_size,
        store_all_end_configurations,
    )

    if len(first_k_indices_must_be) > 0 and len(end_configurations) > 0:
        mask_keep = (
            end_configurations[:, : len(first_k_indices_must_be)]
            == first_k_indices_must_be
        ).all(axis=1)
        end_configurations = end_configurations[mask_keep]

    mask_length_is_atleast_3 = (end_configurations != -1).sum(axis=1) >= 3

    end_configurations = end_configurations[mask_length_is_atleast_3]

    if len(end_configurations) == 0:
        raise NoPathError("Could not create a valid trace using the provided points")

    return end_configurations, all_configurations_and_is_end_configuration_indicator
