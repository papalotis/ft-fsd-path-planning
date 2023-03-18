#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Combines the results of the search along the left and right traces
Project: fsd_path_planning
"""
from itertools import count, product
from typing import Iterable, Optional

import numpy as np

from fsd_path_planning.sorting_cones.trace_sorter.cost_function import \
    cost_configurations
from fsd_path_planning.sorting_cones.trace_sorter.line_segment_intersection import \
    lines_segments_intersect_indicator
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (angle_difference, my_njit,
                                                vec_angle_between)


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


def yield_config_pairs(
    left_configs: IntArray,
    left_scores: FloatArray,
    right_configs: IntArray,
    right_scores: FloatArray,
) -> Iterable[tuple[int, tuple[IntArray, float, IntArray, float, bool, bool]]]:
    max_configs = 100

    left_configs = left_configs[:max_configs]
    left_scores = left_scores[:max_configs]
    right_configs = right_configs[:max_configs]
    right_scores = right_scores[:max_configs]

    counter = count()
    left_zip = zip(left_configs, left_scores)
    right_zip = zip(right_configs, right_scores)
    for (left_config, left_score), (right_config, right_score) in product(
        left_zip, right_zip
    ):
        left_config_clean = left_config[left_config != -1]
        right_config_clean = right_config[right_config != -1]

        common_cone, left_intersection_idxs, right_intersection_idxs = np.intersect1d(
            left_config_clean, right_config_clean, return_indices=True
        )
        if len(common_cone) > 0:
            left_intersection_idx = left_intersection_idxs.min()
            right_intersection_idx = right_intersection_idxs.min()

            left_config_short = np.full(len(left_config), -1, dtype=np.int)
            left_config_short[:left_intersection_idx] = left_config[
                :left_intersection_idx
            ]

            right_config_short = np.full(len(right_config), -1, dtype=np.int)
            right_config_short[:right_intersection_idx] = right_config[
                :right_intersection_idx
            ]

            left_score_short = np.nan
            right_score_short = np.nan

            yield next(counter), (
                left_config_short,
                left_score_short,
                right_config,
                right_score,
                True,
                False,
            )

            yield next(counter), (
                left_config,
                left_score,
                right_config_short,
                right_score_short,
                False,
                True,
            )
        else:
            yield next(counter), (
                left_config,
                left_score,
                right_config,
                right_score,
                False,
                False,
            )


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

    cone_types = np.unique(cones[:, 2])
    if ConeTypes.UNKNOWN not in cone_types:
        # no unknown cones, so we can just pick the best configuration
        return (
            left_configs[0],
            right_configs[0],
            False,
            False,
        )

    final_left_configs = []
    final_right_configs = []
    final_left_scores = []
    final_right_scores = []
    final_left_has_been_shortened = []
    final_right_has_been_shortened = []

    for _, x in yield_config_pairs(
        left_configs, left_scores, right_configs, right_scores
    ):
        (
            left_config,
            left_score,
            right_config,
            right_score,
            left_shortened,
            right_shortened,
        ) = x

        final_left_configs.append(left_config)
        final_right_configs.append(right_config)
        final_left_scores.append(left_score)
        final_right_scores.append(right_score)
        final_left_has_been_shortened.append(left_shortened)
        final_right_has_been_shortened.append(right_shortened)

    final_left_configs = np.array(final_left_configs)
    final_right_configs = np.array(final_right_configs)
    final_left_scores = np.array(final_left_scores)
    final_right_scores = np.array(final_right_scores)

    final_left_scores = score_new_configs(
        cones,
        final_left_configs,
        final_left_scores,
        ConeTypes.LEFT,
        car_position,
        car_direction,
    )

    final_right_scores = score_new_configs(
        cones,
        final_right_configs,
        final_right_scores,
        ConeTypes.RIGHT,
        car_position,
        car_direction,
    )

    mask_left_is_inf = np.isinf(final_left_scores)
    final_left_scores[mask_left_is_inf] = final_right_scores[mask_left_is_inf]

    mask_right_is_inf = np.isinf(final_right_scores)
    final_right_scores[mask_right_is_inf] = final_left_scores[mask_right_is_inf]

    final_configs_together = np.column_stack((final_left_configs, final_right_configs))

    number_of_cones_in_final_configs = np.sum(final_configs_together != -1, axis=1)
    mask_under_six_cones = number_of_cones_in_final_configs < 6
    factor_under_six = mask_under_six_cones * 2.0
    factor_under_six[~mask_under_six_cones] = 1
    

    final_scores = (
        final_left_scores + final_right_scores
    ) * factor_under_six

    assert np.sum(np.isnan(final_scores)) == 0, (final_left_scores, final_right_scores)

    best_idx = np.argmin(final_scores)

    left_config = final_left_configs[best_idx]
    right_config = final_right_configs[best_idx]
    left_has_been_trimmed = final_left_has_been_shortened[best_idx]
    right_has_been_trimmed = final_right_has_been_shortened[best_idx]

    return left_config, right_config, left_has_been_trimmed, right_has_been_trimmed


def score_new_configs(
    cones: FloatArray,
    configs: IntArray,
    scores: FloatArray,
    cone_type: ConeTypes,
    car_position: FloatArray,
    car_direction: FloatArray,
) -> FloatArray:
    mask_configs_to_score = np.isnan(scores)
    mask_len_is_over_2 = np.sum(configs != -1, axis=1) > 2

    mask_config_compute_score = mask_configs_to_score & mask_len_is_over_2

    configs_to_score = configs[mask_config_compute_score]

    new_costs = cost_configurations(
        cones,
        configs_to_score,
        cone_type,
        car_position,
        car_direction,
        return_individual_costs=False,
    )

    scores = scores.copy()
    scores[mask_config_compute_score] = new_costs

    scores[np.isnan(scores)] = np.inf

    return scores


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
    if (
        len(same_cone_intersection) == 0
        or len(left_config) < 3
        or len(right_config) < 3
    ):
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
        angle_left = calc_angle_of_config_at_position(
            cones, left_config, left_intersection_index
        )
        angle_right = calc_angle_of_config_at_position(
            cones, right_config, right_intersection_index
        )

        left_direction_at_intersection = calculate_direction_at_position(
            cones, left_config, left_intersection_index
        )

        right_direction_at_intersection = calculate_direction_at_position(
            cones, right_config, right_intersection_index
        )

        cross_angle_at_intersection = vec_angle_between(
            left_direction_at_intersection, right_direction_at_intersection
        )

        if cross_angle_at_intersection > np.pi / 3:
            left_stop_idx = left_intersection_index
            right_stop_idx = right_intersection_index

        # if the angle of the left side is larger then the cone probably belongs to the
        # left side
        elif angle_left > angle_right:
            # we set the
            left_stop_idx = len(left_config)
            right_stop_idx = right_intersection_index
        else:
            left_stop_idx = left_intersection_index
            right_stop_idx = len(right_config)
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

        # left_stop_idx = (
        #     len(left_config)
        #     if right_intersection_index == len(right_config) - 1
        #     else left_intersection_index
        # )

    return left_stop_idx, right_stop_idx


def calc_angle_of_config_at_position(
    cones: FloatArray,
    config: IntArray,
    position_in_config: int,
) -> float:
    previous_cone, intersection_cone, next_cone = cones[
        config[position_in_config - 1 : position_in_config + 2], :2
    ]

    intersection_to_next = next_cone - intersection_cone
    intersection_to_prev = previous_cone - intersection_cone

    return vec_angle_between(intersection_to_prev, intersection_to_next)


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
