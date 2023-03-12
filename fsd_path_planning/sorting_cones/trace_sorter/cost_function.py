#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This File calculates the costs for the different path versions
Project: fsd_path_planning
"""

import numpy as np

from fsd_path_planning.cone_matching.functional_cone_matching import (
    calculate_match_search_direction,
)
from fsd_path_planning.sorting_cones.trace_sorter.common import get_configurations_diff
from fsd_path_planning.sorting_cones.trace_sorter.cone_distance_cost import (
    calc_distance_cost,
)
from fsd_path_planning.sorting_cones.trace_sorter.line_segment_intersection import (
    number_of_intersections_in_configurations,
)
from fsd_path_planning.types import BoolArray, FloatArray, IntArray, SortableConeTypes
from fsd_path_planning.utils.math_utils import (
    angle_difference,
    angle_from_2d_vector,
    my_njit,
    normalize,
    vec_angle_between,
)


def calc_angle_to_next(points: FloatArray, configurations: IntArray) -> FloatArray:
    """
    Calculate the angle from one cone to the previous and the next one for all
    the provided configurations
    """
    all_to_next = get_configurations_diff(points, configurations)

    mask_should_overwrite = (configurations == -1)[:, 1:]
    all_to_next[mask_should_overwrite] = 100

    from_middle_to_next = all_to_next[..., 1:, :]
    from_prev_to_middle = all_to_next[..., :-1, :]
    from_middle_to_prev = -from_prev_to_middle

    angles = vec_angle_between(from_middle_to_next, from_middle_to_prev)
    return angles


def calc_angle_cost_for_configuration(
    points: FloatArray,
    configurations: IntArray,
    cone_type: SortableConeTypes,  # pylint: disable=unused-argument
) -> FloatArray:
    """
    Calculate the angle cost of cone configurations given a set of points and many index
    lists defining the configurations
    Args:
        points: The points to be used
        configurations: An array of indices defining the configurations
        cone_type: The type of cones. It is currently unused
    Returns:
        np.array: The score of each configuration
    """

    angles = calc_angle_to_next(points, configurations)

    is_part_of_configuration = (configurations != -1)[:, 2:]

    # invert angles and normalize between [0-1]
    angles_as_cost = (np.pi - angles) / np.pi

    angles_as_cost_filtered = angles_as_cost * is_part_of_configuration

    angles_are_under_threshold = np.logical_and(
        angles < np.deg2rad(40), is_part_of_configuration
    )

    # we will multiply the score by the number of angles that are under the threshold
    cost_factors = angles_are_under_threshold.sum(axis=-1) + 1

    # get sum of costs
    costs: FloatArray = angles_as_cost_filtered.sum(
        axis=-1
    ) / is_part_of_configuration.sum(axis=-1)

    costs = costs * cost_factors
    return costs


def calc_line_segment_intersection_cost(
    points: FloatArray, configurations: IntArray
) -> FloatArray:
    """
    Calculates the number of intersections in each configuration
    Args:
        points: The underlying points
        configurations: An array of indices defining a configuration of the
        provided points
    Returns:
        np.array: The number of intersections for each configuration
    """

    exponent: IntArray = number_of_intersections_in_configurations(
        points, configurations
    )
    return_value: FloatArray = np.power(2.0, exponent) - 1.0
    return return_value


def calc_number_of_cones_cost(configurations: IntArray) -> FloatArray:
    """
    Calculates the number of cones in each configuration
    Args:
        configurations: An array of indices defining a configuration of the
        provided points
    Returns:
        A cost for each configuration
    """
    mask: BoolArray = configurations != -1
    number_of_cones: IntArray = mask.sum(axis=-1)

    # we prefer longer configurations
    cost = 1 / number_of_cones
    return cost


def calc_initial_direction_cost(
    points: FloatArray, configurations: IntArray, vehicle_direction: FloatArray
) -> FloatArray:
    points_configs_first_two = np.diff(points[configurations][:, :2], axis=1)[:, 0]

    return vec_angle_between(points_configs_first_two, vehicle_direction)


# @my_njit
def calc_change_of_direction_cost(
    points: FloatArray, configurations: IntArray
) -> FloatArray:
    """
    Calculates the change of direction cost in each configuration. This is done for each
    configuration using the following steps:
    1. Calculate the empiric first derivative of the configuration
    2. Calculate the angle of the first derivative
    3. Calculate the zero crossings of the angle along the configuration
    4. Calculate the sum of the change in the angle between the zero crossings

    Args:
        points: The underlying points
        configurations: An array of indices defining a configuration of the
        provided points
    Returns:
        A cost for each configuration
    """
    out = np.zeros(configurations.shape[0])
    for i, c in enumerate(configurations):
        c = c[c != -1]
        if len(c) == 3:
            continue

        points_of_configuration = points[c]

        diff_1 = points_of_configuration[1:] - points_of_configuration[:-1]

        diff_1 = np.diff(points_of_configuration, axis=0)
        angle = np.arctan2(diff_1[:, 1], diff_1[:, 0])
        # angle = angle_from_2d_vector(diff_1)
        difference = angle_difference(angle[:-1], angle[1:])

        mask_zero_crossing = np.sign(difference[:-1]) != np.sign(difference[1:])
        raw_cost_values = np.abs(difference[:-1] - difference[1:])

        cost_values = raw_cost_values * mask_zero_crossing
        out[i] = np.sum(cost_values)

    return out


# @my_njit
def calc_other_side_cones_cost(
    points: FloatArray, configurations: IntArray, cone_type: SortableConeTypes
) -> FloatArray:
    """
    Calculates the cost for each configuration based on the number of cones on the other
    side of the configuration
    Args:
        points: The underlying points
        configurations: An array of indices defining a configuration of the
        provided points
        cone_type: The type of cone (left/right)
    Returns:
        A cost for each configuration
    """
    found_cones_for_each_config = np.zeros(configurations.shape[0]) + 0.01
    found_wrong_side_cones_for_each_config = np.zeros(configurations.shape[0]) + 0.01

    if len(configurations) == 0:
        return found_cones_for_each_config

    for i, c in enumerate(configurations):
        c = c[c != -1]
        points_in_config = points[c]
        mask_not_in_c = np.ones(points.shape[0], dtype=bool)
        mask_not_in_c[c] = False

        points_not_in_c = points[mask_not_in_c]
        # print(len(points_in_config), cone_type)
        try:
            directions = calculate_match_search_direction(points_in_config, cone_type)
        except AssertionError:
            continue

        for point, direction in zip(points_in_config, directions):
            search_point = point + normalize(direction) * 3
            distances = np.linalg.norm(points_not_in_c - search_point, axis=1)
            if len(distances) != 0:
                found_cones_for_each_config[i] += distances.min() < 3

            wrong_side_search_point = point + normalize(direction) * -3
            wrong_side_distances = np.linalg.norm(
                points_not_in_c - wrong_side_search_point, axis=1
            )
            if len(wrong_side_distances) != 0:
                found_wrong_side_cones_for_each_config[i] += (
                    wrong_side_distances.min() < 3
                )

    # we want to have as many cones on the other side as possible
    found_cones_for_each_config = 1 / found_cones_for_each_config

    # we want to have as few cones on the same side as possible, so we just scale
    found_wrong_side_cones_for_each_config = 0 * found_wrong_side_cones_for_each_config

    return found_cones_for_each_config + found_wrong_side_cones_for_each_config


def cost_configurations(
    points: FloatArray,
    configurations: IntArray,
    cone_type: SortableConeTypes,
    vehicle_position: FloatArray,  # pylint: disable=unused-argument (future proofing, incase we want to use it)
    vehicle_direction: FloatArray,  # pylint: disable=unused-argument (future proofing)
    *,
    return_individual_costs: bool,
) -> FloatArray:
    """
    Calculates a cost for each provided configuration
    Args:
        points: The underlying points
        configurations: An array of indices defining a configuration of the
        provided points
        cone_type: The type of cone (left/right)
    Returns:
        A cost for each configuration
    """
    if configurations.shape[1] < 3:
        return np.zeros(configurations.shape[0])

    from fsd_path_planning.utils.utils import Timer

    timer_no_print = True

    with Timer("angle_cost", timer_no_print):
        angle_cost = calc_angle_cost_for_configuration(
            points, configurations, cone_type
        )

    with Timer("line_segment_intersection_cost", timer_no_print):
        line_segment_intersection_cost = calc_line_segment_intersection_cost(
            points, configurations
        )

    with Timer("residual_distance_cost", timer_no_print):
        threshold_distance = 3  # maximum allowed distance between cones is 5 meters
        residual_distance_cost = calc_distance_cost(
            points, configurations, threshold_distance
        )

    with Timer("number_of_cones_cost", timer_no_print):
        number_of_cones_cost = calc_number_of_cones_cost(configurations)

    with Timer("initial_direction_cost", timer_no_print):
        initial_direction_cost = calc_initial_direction_cost(
            points, configurations, vehicle_direction
        )
        # initial_direction_cost = np.zeros(configurations.shape[0])

    with Timer("change_of_direction_cost", timer_no_print):
        change_of_direction_cost = calc_change_of_direction_cost(points, configurations)

    with Timer("other_side_cones_cost", timer_no_print):
        other_side_cones_cost = calc_other_side_cones_cost(
            points, configurations, cone_type
        )

    factors: FloatArray = np.array([4.0, 10.0, 2.0, 200.0, 10.0, 10.0, 10.0])
    final_costs = (
        np.column_stack(
            [
                angle_cost,
                line_segment_intersection_cost,
                residual_distance_cost,
                number_of_cones_cost,
                initial_direction_cost,
                change_of_direction_cost,
                other_side_cones_cost,
            ]
        )
        * factors
    )

    if return_individual_costs:
        return final_costs

    return final_costs.sum(axis=-1)
