#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This File calculates the costs for the different path versions
Project: fsd_path_planning
"""

import numpy as np
from fsd_path_planning.sorting_cones.trace_sorter.common import get_configurations_diff
from fsd_path_planning.sorting_cones.trace_sorter.cone_distance_cost import (
    calc_distance_cost,
)
from fsd_path_planning.sorting_cones.trace_sorter.line_segment_intersection import (
    number_of_intersections_in_configurations,
)
from fsd_path_planning.types import BoolArray, FloatArray, IntArray, SortableConeTypes
from fsd_path_planning.utils.math_utils import vec_angle_between


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


def cost_configurations(
    points: FloatArray,
    configurations: IntArray,
    cone_type: SortableConeTypes,
    vehicle_direction: FloatArray,
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

    factors: FloatArray = np.array([4.0, 10.0, 0.0, 20.0, 3.0])

    angle_cost = calc_angle_cost_for_configuration(points, configurations, cone_type)

    line_segment_intersection_cost = calc_line_segment_intersection_cost(
        points, configurations
    )

    threshold_distance = 5  # maximum allowed distance between cones is 5 meters
    residual_distance_cost = calc_distance_cost(
        points, configurations, threshold_distance
    )

    number_of_cones_cost = calc_number_of_cones_cost(configurations)

    initial_direction_cost = calc_initial_direction_cost(
        points, configurations, vehicle_direction
    )
    final_costs = (
        np.column_stack(
            [
                angle_cost,
                line_segment_intersection_cost,
                residual_distance_cost,
                number_of_cones_cost,
                initial_direction_cost,
            ]
        )
        * factors
    )

    if return_individual_costs:
        return final_costs

    return final_costs.sum(axis=-1)
