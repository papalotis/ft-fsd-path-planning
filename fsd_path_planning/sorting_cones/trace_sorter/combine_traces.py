#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Combines the results of the search along the left and right traces
Project: fsd_path_planning
"""
from itertools import count, product
from typing import Iterable, Optional

import numpy as np

from fsd_path_planning.sorting_cones.trace_sorter.cost_function import (
    cost_configurations,
)
from fsd_path_planning.sorting_cones.trace_sorter.line_segment_intersection import (
    lines_segments_intersect_indicator,
)
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    angle_difference,
    angle_from_2d_vector,
    my_njit,
    vec_angle_between,
)


def calc_final_configs_for_left_and_right(
    left_scores: Optional[FloatArray],
    left_configs: Optional[IntArray],
    right_scores: Optional[FloatArray],
    right_configs: Optional[IntArray],
    cones: FloatArray,
    car_pos: FloatArray,
    car_dir: FloatArray,
) -> tuple[IntArray, IntArray, bool, bool]:
    left_score_is_none = left_scores is None
    left_config_is_none = left_configs is None
    assert left_score_is_none == left_config_is_none

    right_score_is_none = right_scores is None
    right_config_is_none = right_configs is None
    assert right_score_is_none == right_config_is_none

    n_non_none = sum(x is not None for x in (left_scores, right_scores))

    # if both sides are None, we have no valid configuration
    empty_config = np.zeros(0, dtype=np.int)
    empty_result = (empty_config, empty_config, True, True)

    if n_non_none == 0:
        return empty_result

    if n_non_none == 1:
        # only one side has a valid configuration
        return (
            *calc_final_configs_when_only_one_side_has_configs(
                left_configs,
                right_configs,
            ),
            False,
            False,
        )

    # both sides have valid configurations
    # we need to pick the best one for each side

    return calc_final_configs_when_both_available(
        left_scores,
        left_configs,
        right_scores,
        right_configs,
        cones,
        car_pos,
        car_dir,
    )


def calc_final_configs_when_only_one_side_has_configs(
    left_configs: Optional[IntArray],
    right_configs: Optional[IntArray],
) -> tuple[IntArray, IntArray]:
    empty_config = np.zeros(0, dtype=np.int)

    left_config_is_none = left_configs is None
    right_config_is_none = right_configs is None

    assert left_config_is_none != right_config_is_none

    if left_configs is None:
        left_config = empty_config
        right_config = right_configs[0]
        right_config = right_config[right_config != -1]
    elif right_configs is None:
        right_config = empty_config
        left_config = left_configs[0]
        left_config = left_config[left_config != -1]
    else:
        raise ValueError("Should not happen")

    return left_config, right_config


def calc_final_configs_when_both_available(
    left_scores: FloatArray,
    left_configs: IntArray,
    right_scores: FloatArray,
    right_configs: IntArray,
    cones: FloatArray,
    car_position: FloatArray,
    car_direction: FloatArray,
) -> tuple[IntArray, IntArray, bool, bool]:
    # we need to pick the best one for each side

    left_config = left_configs[0]
    left_config = left_config[left_config != -1]

    right_config = right_configs[0]
    right_config = right_config[right_config != -1]

    original_left_config = left_config.copy()
    original_right_config = right_config.copy()

    left_config, right_config = handle_same_cone_in_both_configs(
        cones, left_config, right_config
    )

    left_has_been_trimmed = len(left_config) < len(original_left_config)
    right_has_been_trimmed = len(right_config) < len(original_right_config)

    return (
        left_config,
        right_config,
        left_has_been_trimmed,
        right_has_been_trimmed,
    )


def handle_same_cone_in_both_configs(
    cones: FloatArray,
    left_config: IntArray,
    right_config: IntArray,
) -> tuple[Optional[IntArray], Optional[IntArray]]:
    (
        same_cone_intersection,
        left_intersection_idxs,
        right_intersection_idxs,
    ) = np.intersect1d(left_config, right_config, return_indices=True)
    if len(same_cone_intersection) == 0:
        return left_config, right_config

    left_intersection_index = min(
        left_intersection_idxs
    )  # first index of common cone in left config
    right_intersection_index = min(
        right_intersection_idxs
    )  # first index of common cone in right config

    # if both sides have the same FIRST common cone, then we try to find the
    # side to which the cone probably belongs
    (
        left_stop_idx,
        right_stop_idx,
    ) = calc_new_length_for_configs_for_same_cone_intersection(
        cones,
        left_config,
        right_config,
        left_intersection_index,
        right_intersection_index,
    )

    left_config = left_config[:left_stop_idx]
    right_config = right_config[:right_stop_idx]

    return left_config, right_config


def calc_new_length_for_configs_for_same_cone_intersection(
    cones: FloatArray,
    left_config: IntArray,
    right_config: IntArray,
    left_intersection_index: int,
    right_intersection_index: int,
) -> tuple[int, int]:
    if (
        left_config[left_intersection_index] == right_config[right_intersection_index]
        and left_intersection_index
        in range(1, len(left_config) - 1)  # not first or last
        and right_intersection_index in range(1, len(right_config) - 1)
    ):
        # XXX: currently we use the unsigned angle, in the future we should use the
        # signed angle
        angle_left = calc_angle_change_at_position(
            cones, left_config, left_intersection_index
        )
        angle_right = calc_angle_change_at_position(
            cones, right_config, right_intersection_index
        )
        angle_diff = abs(angle_left - angle_right)
        if 0 and angle_diff > 0.2:
            if angle_left > angle_right:
                # we set the
                left_stop_idx = len(left_config)
                right_stop_idx = right_intersection_index
            else:
                left_stop_idx = left_intersection_index
                right_stop_idx = len(right_config)
        else:
            left_stop_idx = left_intersection_index
            right_stop_idx = right_intersection_index
    else:
        # if the intersection is the last cone in the config, we assume that this is
        # an error because the configuration could not continue, so we only remove it
        # from that side
        left_intersection_is_at_end = left_intersection_index == len(left_config) - 1
        right_intersection_is_at_end = right_intersection_index == len(right_config) - 1

        if left_intersection_is_at_end and right_intersection_is_at_end:
            left_stop_idx = len(left_config) - 1
            right_stop_idx = len(right_config) - 1

        elif left_intersection_is_at_end:
            right_stop_idx = len(right_config)
            left_stop_idx = left_intersection_index

        elif right_intersection_is_at_end:
            left_stop_idx = len(left_config)
            right_stop_idx = right_intersection_index
        else:
            left_stop_idx = left_intersection_index
            right_stop_idx = right_intersection_index

    return left_stop_idx, right_stop_idx


def calc_angle_change_at_position(
    cones: FloatArray,
    config: IntArray,
    position_in_config: int,
) -> float:
    previous_cone, intersection_cone, next_cone = cones[
        config[position_in_config - 1 : position_in_config + 2], :2
    ]

    intersection_to_next = next_cone - intersection_cone
    prev_to_intersection = intersection_cone - previous_cone

    intersection_to_next_angle = angle_from_2d_vector(intersection_to_next)
    prev_to_intersection_angle = angle_from_2d_vector(prev_to_intersection)

    angle_diff = angle_difference(
        intersection_to_next_angle, prev_to_intersection_angle
    )

    return angle_diff


def calculate_direction_at_position(
    cones: FloatArray, config: IntArray, position: int
) -> FloatArray:
    if position == 0:
        return cones[config[1], :2] - cones[config[0], :2]
    elif position == len(config) - 1:
        return cones[config[-1], :2] - cones[config[-2], :2]
    else:
        return cones[config[position + 1], :2] - cones[config[position - 1], :2]


@my_njit
def find_first_intersection_in_trace(
    trace: np.ndarray, other_trace: np.ndarray
) -> Optional[int]:
    for i in range(len(trace) - 1):
        for j in range(len(other_trace) - 1):
            start_1 = trace[i]
            end_1 = trace[i + 1]
            start_2 = other_trace[j]
            end_2 = other_trace[j + 1]

            if lines_segments_intersect_indicator(start_1, end_1, start_2, end_2):
                return i

    return None


def handle_edge_intersection_between_both_configs(
    cones: FloatArray,
    left_config: IntArray,
    right_config: IntArray,
):
    left_trace = cones[left_config[left_config != -1], :2]
    right_trace = cones[right_config[right_config != -1], :2]

    left_intersection_index = find_first_intersection_in_trace(left_trace, right_trace)
    right_intersection_index = find_first_intersection_in_trace(right_trace, left_trace)

    if left_intersection_index is None != right_intersection_index is None:
        print(np.array_repr(left_config), np.array_repr(right_config))
        raise ValueError("Only one side has an intersection")

    if left_intersection_index is not None:
        left_config = left_config[:left_intersection_index]
        right_config = right_config[:right_intersection_index]

    return left_config, right_config
    return left_config, right_config
