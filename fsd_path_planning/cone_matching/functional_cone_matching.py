#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Match cones from left and right trace to facilitate
more stable path calculation

Project: fsd_path_planning
"""

from __future__ import annotations

from typing import Literal, Tuple, cast

import numpy as np
from icecream import ic  # pylint: disable=unused-import

from fsd_path_planning.cone_matching.match_directions import (
    calculate_match_search_direction,
)
from fsd_path_planning.types import BoolArray, FloatArray, IntArray, SortableConeTypes
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    my_cdist_sq_euclidean,
    my_njit,
    rotate,
    trace_angles_between,
    vec_angle_between,
)

ic = lambda x: x  # pylint: disable=invalid-name


@my_njit
def cones_in_range_and_pov_mask(
    cones: FloatArray,
    search_directions: FloatArray,
    search_range: float,
    search_angle: float,
    other_side_cones: FloatArray,
) -> BoolArray:
    """
    Calculates the indices of the visible cones according to the car position

    Returns:
        The indices of the visible cones
    """
    search_range_squared = search_range * search_range

    # # (M, 2), (N, 2) -> (M,N)
    dist_from_cones_to_other_side_squared: FloatArray = my_cdist_sq_euclidean(
        other_side_cones, cones
    )

    # (M, N)
    dist_mask: BoolArray = dist_from_cones_to_other_side_squared < search_range_squared

    # (M, 1, 2) - (N, 2) -> (M, N, 2)
    vec_from_cones_to_other_side = np.expand_dims(other_side_cones, axis=1) - cones

    # (N, 2) -> (M, N, 2)
    search_directions_broadcasted: FloatArray = np.broadcast_to(
        search_directions, vec_from_cones_to_other_side.shape
    ).copy()  # copy is needed in numba, otherwise a reshape error occurs

    # (M, N, 2), (M, N, 2) -> (M, N)
    angles_to_car = vec_angle_between(
        search_directions_broadcasted, vec_from_cones_to_other_side
    )
    # (M, N)
    mask_angles = np.logical_and(
        -search_angle / 2 < angles_to_car, angles_to_car < search_angle / 2
    )

    visible_cones_mask = np.logical_and(dist_mask, mask_angles)

    return_value: BoolArray = visible_cones_mask
    return return_value


def find_boolean_mask_of_all_potential_matches(
    start_points: FloatArray,
    directions: FloatArray,
    other_side_cones: FloatArray,
    other_side_directions: FloatArray,
    major_radius: float,
    minor_radius: float,
    max_search_angle: float,
) -> BoolArray:
    """
    Calculate a (M,N) boolean mask that indicates for each cone in the cones array
    if a cone on the other side can be match or not.
    """

    return_value = np.zeros((len(start_points), len(other_side_cones)), dtype=bool)
    if len(start_points) == 0 or len(other_side_cones) == 0:
        return return_value

    # return mask_all

    # (2,)
    radii_square = np.array([major_radius, minor_radius]) ** 2

    # (M,)
    angles = angle_from_2d_vector(directions)
    # (M, N, 2)                       (N, 2)             (M, 1, 2)
    from_start_points_to_other_side = other_side_cones - start_points[:, None]

    start_point_direction_other_side_direction_angle_diff = vec_angle_between(
        directions[:, None], other_side_directions
    )

    for i, (
        start_point_to_other_side_cones,
        angle,
        angle_diff_start_direction_other_direction,
    ) in enumerate(
        zip(
            from_start_points_to_other_side,
            angles,
            start_point_direction_other_side_direction_angle_diff,
        )
    ):
        # (N, 2)
        rotated_start_point_to_other_side = rotate(
            start_point_to_other_side_cones, -angle
        )
        # (N,)

        s = (rotated_start_point_to_other_side**2 / radii_square).sum(axis=1)
        return_value[i] = s < 1

        angle_of_rotated_start_point_to_other_side = angle_from_2d_vector(
            rotated_start_point_to_other_side
        )

        mask_angle_is_over_threshold = (
            np.abs(angle_of_rotated_start_point_to_other_side / 2) > max_search_angle
        )

        mask_direction_diff_over_threshold = (
            angle_diff_start_direction_other_direction < np.pi / 2
        )

        return_value[i, mask_angle_is_over_threshold] = False
        return_value[i, mask_direction_diff_over_threshold] = False

    for i, (mask_cone_to_candidates, distance_to_other_side) in enumerate(
        zip(return_value, np.linalg.norm(from_start_points_to_other_side, axis=-1))
    ):
        distance_to_other_side[~mask_cone_to_candidates] = np.inf
        idxs_candidates_sorted = np.argsort(distance_to_other_side)[:2]
        mask_idx_candidate_is_valid = np.isfinite(
            distance_to_other_side[idxs_candidates_sorted]
        )
        idxs_candidates_sorted = idxs_candidates_sorted[mask_idx_candidate_is_valid]

        new_mask = np.zeros_like(mask_cone_to_candidates)
        new_mask[idxs_candidates_sorted] = True
        return_value[i] = new_mask

    return return_value


def select_best_match_candidate(
    matchable_cones: FloatArray,
    match_directions: FloatArray,
    match_boolean_mask: BoolArray,
    other_side_cones: FloatArray,
    matches_should_be_monotonic: bool,
) -> IntArray:
    """
    For each cone select a matching cone from the other side. If a cone has no potential
    match, it is marked with -1.
    """

    if len(other_side_cones) == 0:
        return np.full(len(matchable_cones), -1, dtype=int)

    matched_index_for_each_cone: IntArray = my_cdist_sq_euclidean(
        matchable_cones, other_side_cones
    ).argmin(axis=1)

    if matches_should_be_monotonic:
        # constraint matches to be monotonic
        current_max_value = matched_index_for_each_cone[0]
        for i in range(1, len(matched_index_for_each_cone)):
            current_max_value = max(current_max_value, matched_index_for_each_cone[i])
            if matched_index_for_each_cone[i] != current_max_value:
                matched_index_for_each_cone[i] = -1
            else:
                matched_index_for_each_cone[i] = current_max_value

    matched_index_for_each_cone[~match_boolean_mask.any(axis=1)] = -1
    return matched_index_for_each_cone


def calculate_positions_of_virtual_cones(
    cones: FloatArray,
    indices_of_unmatched_cones: IntArray,
    search_directions: FloatArray,
    min_track_width: float,
) -> FloatArray:
    """
    Calculate the positions of the virtual cones given the unmatched cones and the
    direction of the match search.
    """

    return_value: FloatArray = (
        cones[indices_of_unmatched_cones]
        + search_directions[indices_of_unmatched_cones] * min_track_width
    )
    return return_value


def insert_virtual_cones_to_existing(
    other_side_cones: FloatArray,
    other_side_virtual_cones: FloatArray,
    car_position: FloatArray,
) -> tuple[FloatArray, list[FloatArray]]:
    """
    Combine the virtual with the real cones into a single array.
    """
    # print(locals())
    existing_cones, cones_to_insert = (
        (other_side_cones, other_side_virtual_cones)
        if len(other_side_cones) > len(other_side_virtual_cones)
        else (other_side_virtual_cones, other_side_cones)
    )
    existing_cones = existing_cones.copy()
    cones_to_insert = cones_to_insert.copy()

    order_to_insert = (
        my_cdist_sq_euclidean(cones_to_insert, existing_cones).min(axis=1).argsort()
    )
    cones_to_insert = cones_to_insert[order_to_insert]

    history: list[FloatArray] = []

    for cone_to_insert in cones_to_insert:
        distance_to_existing_cones = np.linalg.norm(
            existing_cones - cone_to_insert, axis=1
        )
        indices_sorted_by_distances = distance_to_existing_cones.argsort()

        if len(indices_sorted_by_distances) == 1:
            index_to_insert = calculate_insert_index_for_one_cone(
                car_position, existing_cones, cone_to_insert
            )
        else:
            closest_index, second_closest_index = indices_sorted_by_distances[:2]

            if np.abs(closest_index - second_closest_index) != 1:
                continue

            virtual_to_closest = existing_cones[closest_index] - cone_to_insert
            virtual_to_second_closest = (
                existing_cones[second_closest_index] - cone_to_insert
            )
            angle_between_virtual_cones_and_closest_two = vec_angle_between(
                virtual_to_closest, virtual_to_second_closest
            )

            cone_is_between_closest_two = cast(
                bool, angle_between_virtual_cones_and_closest_two > np.pi / 2
            )

            index_to_insert = calculate_insert_index_of_new_cone(
                closest_index,
                second_closest_index,
                cone_is_between_closest_two,
            )

        existing_cones: FloatArray = np.insert(  # type: ignore
            existing_cones,
            index_to_insert,
            cone_to_insert,
            axis=0,
        )

        history.append(existing_cones.copy())

    angles = trace_angles_between(existing_cones)
    # print(np.rad2deg(angles))
    mask_low_angles = angles < np.deg2rad(85)
    mask_low_angles = np.concatenate([[False], mask_low_angles, [False]])

    if mask_low_angles.any():
        existing_cones = existing_cones[:][~mask_low_angles]
        history.append(existing_cones.copy())

    return existing_cones, history


def calculate_insert_index_for_one_cone(
    car_position: FloatArray, final_cones: FloatArray, virtual_cone: FloatArray
) -> int:
    """
    Insert a virtual cone into the real cones, when only one real cone is available.
    The position of the virtual cone in the array is based on distance to the car.
    """
    distance_to_car_other_cone: float = np.linalg.norm(
        virtual_cone - car_position,
    )  # type: ignore
    distance_to_car_existing_cone: float = np.linalg.norm(
        final_cones[0] - car_position,
    )  # type: ignore

    if distance_to_car_other_cone < distance_to_car_existing_cone:
        index_to_insert = 0
    else:
        index_to_insert = 1
    return index_to_insert


def calculate_insert_index_of_new_cone(
    closest_index: int,
    second_closest_index: int,
    cone_is_between_closest_two: bool,
) -> int:
    """
    Decide the index of the new cone to insert. It is based on the distance to the
    two closest cones and the angle that is formed between them.
    """

    if cone_is_between_closest_two:
        return min(closest_index, second_closest_index) + 1
    else:
        if closest_index < second_closest_index:
            return closest_index
        elif closest_index > second_closest_index:
            return closest_index + 1
        else:
            raise ValueError("Unreachable code")


def combine_and_sort_virtual_with_real(
    other_side_cones: FloatArray,
    other_side_virtual_cones: FloatArray,
    other_side_cone_type: SortableConeTypes,  # pylint : disable=unused-argument
    car_pos: FloatArray,
    car_dir: FloatArray,  # pylint: disable=unused-argument
) -> Tuple[FloatArray, BoolArray, list[FloatArray]]:
    """
    Combine the existing cones with the newly calculated cones into a single array.
    """

    if len(other_side_cones) == 0:
        return (
            other_side_virtual_cones,
            np.ones(len(other_side_virtual_cones), dtype=bool),
            [],
        )

    if len(other_side_virtual_cones) == 0:
        return other_side_cones, np.zeros(len(other_side_cones), dtype=bool), []

    sorted_combined_cones, history = insert_virtual_cones_to_existing(
        other_side_cones, other_side_virtual_cones, car_pos
    )

    # cones than have a distance larger than epsilon to the existing cones are virtual
    distance_of_final_cones_to_existing = my_cdist_sq_euclidean(
        sorted_combined_cones, other_side_cones
    )
    epsilon = 1e-2
    virtual_mask: BoolArray = distance_of_final_cones_to_existing > epsilon**2
    mask_is_virtual: BoolArray = np.all(virtual_mask, axis=1)

    return sorted_combined_cones, mask_is_virtual, history


def calculate_matches_for_side(
    cones: FloatArray,
    cone_type: ConeTypes,
    other_side_cones: FloatArray,
    major_radius: float,
    minor_radius: float,
    max_search_angle: float,
    matches_should_be_monotonic: bool,
) -> Tuple[FloatArray, IntArray, FloatArray]:
    """
    Find a match for each cone from one side to the other.
    """
    matchable_cones = cones[:]
    if len(matchable_cones) > 1:
        search_directions = calculate_match_search_direction(matchable_cones, cone_type)

        if len(other_side_cones) > 1:
            other_side_search_directions = calculate_match_search_direction(
                other_side_cones,
                ConeTypes.LEFT if cone_type == ConeTypes.RIGHT else ConeTypes.RIGHT,
            )
        else:
            other_side_search_directions = np.zeros((0, 2), dtype=float)
        us_to_others_match_cones_mask = find_boolean_mask_of_all_potential_matches(
            matchable_cones,
            search_directions,
            other_side_cones,
            other_side_search_directions,
            major_radius,
            minor_radius,
            max_search_angle,
        )

        matches_for_each_selectable_cone = select_best_match_candidate(
            matchable_cones,
            search_directions,
            us_to_others_match_cones_mask,
            other_side_cones,
            matches_should_be_monotonic,
        )
    else:
        matches_for_each_selectable_cone = (
            np.zeros((len(matchable_cones),), dtype=np.int32) - 1
        )
        search_directions = np.zeros((0, 2))

    return matchable_cones, matches_for_each_selectable_cone, search_directions


def calculate_cones_for_other_side(
    cones: FloatArray,
    cone_type: ConeTypes,
    major_radius: float,
    minor_radius: float,
    max_search_angle: float,
    other_side_cones: FloatArray,
    min_track_width: float,
    car_pos: FloatArray,
    car_dir: FloatArray,
    matches_should_be_monotonic: bool,
) -> Tuple[FloatArray, BoolArray]:
    """
    Calculate the virtual cones for the other side.
    """
    (
        matchable_cones,
        matches_for_each_selectable_cone,
        search_directions,
    ) = calculate_matches_for_side(
        cones,
        cone_type,
        other_side_cones,
        major_radius,
        minor_radius,
        max_search_angle,
        matches_should_be_monotonic,
    )
    mask_cone_has_match = matches_for_each_selectable_cone != -1
    # indices_to_keep = np.where(mask_cone_has_match)[0]
    indices_no_match = np.where(~mask_cone_has_match)[0]

    positions_of_virtual_cones = calculate_positions_of_virtual_cones(
        matchable_cones, indices_no_match, search_directions, min_track_width
    )

    other_side_cone_type: Literal[ConeTypes.YELLOW, ConeTypes.BLUE] = (
        ConeTypes.YELLOW if cone_type == ConeTypes.BLUE else ConeTypes.BLUE
    )

    # we do not care about the history in prod, only for debugging/visualization
    combined_and_sorted_cones, mask_is_virtual, _ = combine_and_sort_virtual_with_real(
        other_side_cones,
        positions_of_virtual_cones,
        other_side_cone_type,
        car_pos,
        car_dir,
    )

    if len(combined_and_sorted_cones) < 2:
        combined_and_sorted_cones = other_side_cones
        mask_is_virtual = np.zeros(len(other_side_cones), dtype=bool)

    return combined_and_sorted_cones, mask_is_virtual


def match_both_sides_with_virtual_cones(
    left_cones_with_virtual: FloatArray,
    right_cones_with_virtual: FloatArray,
    major_radius: float,
    minor_radius: float,
    max_search_angle: float,
    matches_should_be_monotonic: bool,
) -> Tuple[IntArray, IntArray]:
    """
    After virtual cones have been placed for each side, rerun matching algorithm
    to get final matches.
    """

    _, final_matching_from_left_to_right, _ = calculate_matches_for_side(
        left_cones_with_virtual,
        ConeTypes.LEFT,
        right_cones_with_virtual,
        major_radius,
        minor_radius,
        max_search_angle,
        matches_should_be_monotonic,
    )

    _, final_matching_from_right_to_left, _ = calculate_matches_for_side(
        right_cones_with_virtual,
        ConeTypes.RIGHT,
        left_cones_with_virtual,
        major_radius,
        minor_radius,
        max_search_angle,
        matches_should_be_monotonic,
    )

    return final_matching_from_left_to_right, final_matching_from_right_to_left


def calculate_virtual_cones_for_both_sides(
    left_cones: FloatArray,
    right_cones: FloatArray,
    car_position: FloatArray,
    car_direction: FloatArray,
    min_track_width: float,
    major_radius: float,
    minor_radius: float,
    max_search_angle: float,
    matches_should_be_monotonic: bool = True,
) -> Tuple[
    Tuple[FloatArray, BoolArray, IntArray],
    Tuple[FloatArray, BoolArray, IntArray],
]:
    """
    The main function of the module. It applies all the steps to return two results
    containing the new cones including a virtual ones, a boolean mask indicating for
    each cone if it is a virtual cone or not and an integer mask indicating for each
    cone the index of the match on the other side.
    """
    # if len(left_cones) > 20 or len(right_cones) > 20:
    #     print(locals())
    empty_bool_array: BoolArray = np.zeros(0, dtype=np.bool_)
    empty_int_array: IntArray = np.zeros(0, dtype=np.int_)
    empty_cone_array: FloatArray = np.zeros((0, 2), dtype=np.float_)

    dummy_result = empty_cone_array, empty_bool_array, empty_int_array

    ic("calculate_virtual_cones_for_both_sides: start")
    if len(left_cones) < 2 and len(right_cones) < 2:
        left_result = dummy_result
        right_result = dummy_result
        return left_result, right_result

    min_len = min(len(left_cones), len(right_cones))
    max_len = max(len(left_cones), len(right_cones))
    discard_one_side = min_len == 0 or ((max_len / min_len) > 2)
    if discard_one_side:
        if len(left_cones) < len(right_cones):
            left_cones = empty_cone_array
        else:
            right_cones = empty_cone_array

    ic("calculate_virtual_cones_for_both_sides: left match right")
    right_mask_is_virtual: BoolArray
    right_cones_with_virtual, right_mask_is_virtual = (
        calculate_cones_for_other_side(
            left_cones,
            ConeTypes.LEFT,
            major_radius,
            minor_radius,
            max_search_angle,
            right_cones,
            min_track_width,
            car_position,
            car_direction,
            matches_should_be_monotonic,
        )
        if len(left_cones) >= 2
        else (
            right_cones,
            np.zeros(len(right_cones), dtype=bool),
        )
    )

    ic("calculate_virtual_cones_for_both_sides: right match left")
    left_mask_is_virtual: BoolArray
    left_cones_with_virtual, left_mask_is_virtual = (
        calculate_cones_for_other_side(
            right_cones,
            ConeTypes.RIGHT,
            major_radius,
            minor_radius,
            max_search_angle,
            left_cones,
            min_track_width,
            car_position,
            car_direction,
            matches_should_be_monotonic,
        )
        if len(right_cones) >= 2
        else (
            left_cones,
            np.zeros(len(left_cones), dtype=bool),
        )
    )

    ic("calculate_virtual_cones_for_both_sides: match left and right w/ virtual")
    left_to_right_matches, right_to_left_matches = match_both_sides_with_virtual_cones(
        left_cones_with_virtual,
        right_cones_with_virtual,
        major_radius,
        minor_radius,
        max_search_angle,
        matches_should_be_monotonic,
    )

    left_result = (
        left_cones_with_virtual,
        left_mask_is_virtual,
        left_to_right_matches,
    )
    ic("match left and right w/ virtual")
    right_result = (
        right_cones_with_virtual,
        right_mask_is_virtual,
        right_to_left_matches,
    )

    return left_result, right_result
    return left_result, right_result

    return left_result, right_result
    return left_result, right_result
    return left_result, right_result

    return left_result, right_result
    return left_result, right_result
