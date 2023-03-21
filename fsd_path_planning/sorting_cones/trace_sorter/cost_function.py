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
from fsd_path_planning.sorting_cones.trace_sorter.nearby_cone_search import (
    number_cones_on_each_side_for_each_config,
)
from fsd_path_planning.types import BoolArray, FloatArray, IntArray, SortableConeTypes
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import angle_difference, vec_angle_between
from fsd_path_planning.utils.utils import Timer


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


def calc_wrong_direction_cost(
    points: FloatArray, configurations: IntArray, cone_type: ConeTypes
) -> FloatArray:
    """

    Args:
        points: The underlying points
        configurations: An array of indices defining a configuration of the
        provided points
    Returns:
        A cost for each configuration
    """
    out = np.zeros(configurations.shape[0])

    unwanted_direction_sign = 1 if cone_type == ConeTypes.LEFT else -1

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

        mask_wrong_direction = np.sign(difference) == unwanted_direction_sign
        mask_threshold = np.abs(difference) > np.deg2rad(40)

        mask = mask_wrong_direction & mask_threshold

        cost_values = np.abs(difference[mask].sum())

        out[i] = cost_values

    return out


def calc_cones_on_either_cost(
    points: FloatArray,
    configurations: IntArray,
    cone_type: SortableConeTypes,
) -> FloatArray:
    with Timer("calc_cones_on_either_cost", noprint=True) as _:
        n_good, n_bad = number_cones_on_each_side_for_each_config(
            points,
            configurations,
            cone_type,
            6.0,
            np.pi / 1.5,
        )

    diff = n_good - n_bad
    m_value = diff.min()

    diff += np.abs(m_value) + 1

    return 1 / diff


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
    points_xy = points[:, :2]

    # print(len(configurations))
    if len(configurations) == 0:
        return np.zeros(0)
    # if configurations.shape[1] < 3:
    #     return np.zeros(configurations.shape[0])

    timer_no_print = True

    not timer_no_print and print(cone_type)

    with Timer("angle_cost", timer_no_print):
        angle_cost = calc_angle_cost_for_configuration(
            points_xy, configurations, cone_type
        )

    with Timer("residual_distance_cost", timer_no_print):
        threshold_distance = 3  # maximum allowed distance between cones is 5 meters
        residual_distance_cost = calc_distance_cost(
            points_xy, configurations, threshold_distance
        )

    with Timer("number_of_cones_cost", timer_no_print):
        number_of_cones_cost = calc_number_of_cones_cost(configurations)

    with Timer("initial_direction_cost", timer_no_print):
        initial_direction_cost = calc_initial_direction_cost(
            points_xy, configurations, vehicle_direction
        )
        # initial_direction_cost = np.zeros(configurations.shape[0])

    with Timer("change_of_direction_cost", timer_no_print):
        change_of_direction_cost = calc_change_of_direction_cost(
            points_xy, configurations
        )

    with Timer("cones_on_either_cost", timer_no_print):
        cones_on_either_cost = calc_cones_on_either_cost(
            points_xy, configurations, cone_type
        )

    with Timer("wrong_direction_cost", timer_no_print):
        wrong_direction_cost = calc_wrong_direction_cost(
            points_xy, configurations, cone_type
        )

    # TODO: Add a cost for angle between last cone in config and two closest cones not in config

    not timer_no_print and print()

    factors: FloatArray = np.array([1000.0, 200.0, 5000.0, 1000.0, 0.0, 1000.0, 1000.0])
    factors = factors / factors.sum()
    # print(configurations)
    final_costs = (
        np.column_stack(
            [
                angle_cost,
                residual_distance_cost,
                number_of_cones_cost,
                initial_direction_cost,
                change_of_direction_cost,
                cones_on_either_cost,
                wrong_direction_cost,
            ]
        )
        * factors
    )

    if return_individual_costs:
        return final_costs

    return final_costs.sum(axis=-1)
    return final_costs.sum(axis=-1)
