#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This module provides functionality for sorting a trace of cones into a
plausible track
Project: fsd_path_planning
"""
from typing import Optional, Tuple

import numpy as np

from fsd_path_planning.cone_matching.functional_cone_matching import (
    combine_and_sort_virtual_with_real,
)
from fsd_path_planning.sorting_cones.trace_sorter.combine_traces import (
    calc_final_configs_for_left_and_right,
)
from fsd_path_planning.sorting_cones.trace_sorter.common import NoPathError
from fsd_path_planning.sorting_cones.trace_sorter.find_configs_and_scores import (
    calc_scores_and_end_configurations,
)
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes, invert_cone_type
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    my_cdist_sq_euclidean,
    points_inside_ellipse,
    rotate,
    vec_angle_between,
)


class TraceSorter:
    """
    Wraps the trace sorting functionality into a class
    """

    def __init__(
        self,
        max_n_neighbors: int,
        max_dist: float,
        max_dist_to_first: float,
        max_length: int,
        threshold_directional_angle: float,
        threshold_absolute_angle: float,
    ):
        """
        Constructor for TraceSorter class
        Args:
            max_n_neighbors: The maximum allowed number of neighbors for each node
            during sorting
            max_dist: The maximum allowed distance for two nodes to be
            considered neighbors
            max_dist_to_first: The maximum allowed distance in order for a node
            to be considered a viable first node
        """
        self.max_n_neighbors = max_n_neighbors
        self.max_dist = max_dist
        self.max_dist_to_first = max_dist_to_first
        self.max_length = max_length
        self.threshold_directional_angle = threshold_directional_angle
        self.threshold_absolute_angle = threshold_absolute_angle

    def flatten_cones_by_type_array(
        self, cones_by_type: list[FloatArray]
    ) -> FloatArray:
        """Ravel the cones_by_type_array"""
        n_all_cones = sum(map(len, cones_by_type))

        # (x, y, color)
        out = np.empty((n_all_cones, 3))
        n_start = 0
        for cone_type in ConeTypes:
            n_cones = len(cones_by_type[cone_type])
            out[n_start : n_start + n_cones, :2] = cones_by_type[cone_type].reshape(
                -1, 2
            )
            out[n_start : n_start + n_cones, 2] = cone_type
            n_start += n_cones

        return out

    def remove_last_cone_in_config_if_not_of_type(
        self, config: IntArray, cones: FloatArray, cone_type: ConeTypes
    ) -> IntArray:
        """Remove the last cone in the config if it is not of the specified type"""
        if len(config) > 0:
            last = config[-1]
            type_last = cones[last, 2]

            if type_last != cone_type and len(config) > 3:
                config = config[:-1]

        return config

    def sort_left_right(
        self,
        cones_by_type: list[FloatArray],
        car_pos: FloatArray,
        car_dir: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        cones_flat = self.flatten_cones_by_type_array(cones_by_type)

        left_scores, left_configs = self.calc_configurations_with_score_for_one_side(
            cones_flat,
            ConeTypes.LEFT,
            car_pos,
            car_dir,
        )

        right_scores, right_configs = self.calc_configurations_with_score_for_one_side(
            cones_flat,
            ConeTypes.RIGHT,
            car_pos,
            car_dir,
        )

        left_config, right_config = calc_final_configs_for_left_and_right(
            left_scores,
            left_configs,
            right_scores,
            right_configs,
            cones_flat,
            car_pos,
            car_dir,
        )

        left_config = self.remove_last_cone_in_config_if_not_of_type(
            left_config, cones_flat, ConeTypes.LEFT
        )

        right_config = self.remove_last_cone_in_config_if_not_of_type(
            right_config, cones_flat, ConeTypes.RIGHT
        )

        # remove any placeholder positions if they are present
        left_config = left_config[left_config != -1]
        right_config = right_config[right_config != -1]

        left_sorted = cones_flat[left_config]
        right_sorted = cones_flat[right_config]

        return left_sorted[:, :2], right_sorted[:, :2]

    def calc_configurations_with_score_for_one_side(
        self,
        cones: FloatArray,
        cone_type: ConeTypes,
        car_pos: FloatArray,
        car_dir: FloatArray,
    ) -> Tuple[Optional[FloatArray], Optional[IntArray]]:
        """
        Args:
            cones: The trace to be sorted.
            cone_type: The type of cone to be sorted.
            car_pos: The position of the car.
            car_dir: The direction towards which the car is facing.
        Returns:
            np.ndarray: The sorted trace, `len(return_value) <= len(trace)`
        """
        assert cone_type in (ConeTypes.LEFT, ConeTypes.RIGHT)

        no_result = None, None

        if len(cones) < 3:
            return no_result

        first_2 = self.select_first_k_starting_cones(
            car_pos,
            car_dir,
            cones,
            cone_type,
        )
        if first_2 is not None:
            start_idx = first_2[0]
            if len(first_2) > 1:
                first_k_indices_must_be = first_2.copy()
            else:
                first_k_indices_must_be = None
        else:
            start_idx = None
            first_k_indices_must_be = None

        if start_idx is None and first_k_indices_must_be is None:
            return no_result

        n_neighbors = min(self.max_n_neighbors, len(cones) - 1)
        try:
            return_value = calc_scores_and_end_configurations(
                cones,
                cone_type,
                n_neighbors,
                start_idx,
                self.threshold_directional_angle,
                self.threshold_absolute_angle,
                car_pos,
                car_dir,
                self.max_dist,
                self.max_length,
                first_k_indices_must_be,
            )

        # if no configurations can be found, then return nothing
        except NoPathError:
            return no_result

        return return_value

    def invert_cone_type(self, cone_type: ConeTypes) -> ConeTypes:
        """
        Inverts the cone type
        Args:
            cone_type: The cone type to be inverted
        Returns:
            ConeTypes: The inverted cone type
        """
        if cone_type == ConeTypes.LEFT:
            return ConeTypes.RIGHT
        if cone_type == ConeTypes.RIGHT:
            return ConeTypes.LEFT

        raise ValueError(f"Cone type {cone_type} cannot be inverted.")

    def select_starting_cone(
        self,
        car_position: FloatArray,
        car_direction: FloatArray,
        cones: FloatArray,
        cone_type: ConeTypes,
        index_to_skip: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """
        Return the index of the starting cone
            int: The index of the stating cone
        """
        trace_distances, mask_is_valid = self.mask_cone_can_be_first_in_config(
            car_position, car_direction, cones, cone_type
        )

        trace_distances_copy = trace_distances.copy()
        trace_distances_copy[~mask_is_valid] = np.inf

        if np.any(mask_is_valid) > 0:
            sorted_idxs = np.argsort(trace_distances_copy)
            start_idx = None
            for idx in sorted_idxs:
                if index_to_skip is None or idx not in index_to_skip:
                    start_idx = idx
                    break
            if trace_distances_copy[start_idx] > self.max_dist_to_first:
                start_idx = None
        else:
            start_idx = None

        return start_idx

    def mask_cone_can_be_first_in_config(
        self, car_position, car_direction, cones, cone_type
    ):
        cones_xy = cones[:, :2]  # remove cone type

        cones_relative = rotate(
            cones_xy - car_position, -angle_from_2d_vector(car_direction)
        )

        cone_relative_angles = angle_from_2d_vector(cones_relative)

        trace_distances = np.linalg.norm(cones_relative, axis=-1)

        mask_is_in_ellipse = points_inside_ellipse(
            cones_xy,
            car_position,
            car_direction,
            self.max_dist_to_first * 1.3,
            self.max_dist_to_first / 1.3,
        )
        angle_signs = np.sign(cone_relative_angles)
        valid_angle_sign = 1 if cone_type == ConeTypes.LEFT else -1
        mask_valid_side = angle_signs == valid_angle_sign
        mask_is_valid_angle = np.abs(cone_relative_angles) < np.pi / 1
        mask_is_valid_angle_min = np.abs(cone_relative_angles) > np.pi / 10
        mask_is_not_opposite_cone_type = cones[:, 2] != invert_cone_type(cone_type)
        mask_is_valid = (
            mask_is_valid_angle
            * mask_is_in_ellipse
            * mask_valid_side
            * mask_is_valid_angle_min
            * mask_is_not_opposite_cone_type
        )

        return trace_distances, mask_is_valid

    def select_first_k_starting_cones(
        self,
        car_position: FloatArray,
        car_direction: FloatArray,
        cones: FloatArray,
        cone_type: ConeTypes,
    ) -> Optional[np.ndarray]:
        """
        Return the index of the starting cones. Pick the cone that is closest in front
        of the car and the cone that is closest behind the car.
        """
        index_1 = self.select_starting_cone(
            car_position,
            car_direction,
            cones,
            cone_type,
        )

        if index_1 is None:
            return None

        # get the cone behind the car
        index_2 = self.select_starting_cone(
            car_position,
            car_direction,
            cones,
            cone_type,
            index_to_skip=np.array([index_1]),
        )

        if index_2 is None:
            return np.array([index_1], dtype=np.int_)

        cone_dir_1 = cones[index_1, :2] - cones[index_2, :2]
        cone_dir_2 = cones[index_2, :2] - cones[index_1, :2]

        angle_1 = vec_angle_between(cone_dir_1, car_direction)
        angle_2 = vec_angle_between(cone_dir_2, car_direction)

        if angle_1 > angle_2:
            index_1, index_2 = index_2, index_1

        dist = np.linalg.norm(cone_dir_1)
        if dist > self.max_dist * 1.1:
            return np.array([index_1], dtype=np.int_)

        two_cones = np.array([index_2, index_1], dtype=np.int_)

        # find the third cone
        index_3 = self.select_starting_cone(
            car_position,
            car_direction,
            cones,
            cone_type,
            index_to_skip=two_cones,
        )

        if index_3 is None:
            return two_cones

        # check if the third cone is close enough to the first cone
        min_dist_to_first_two = np.linalg.norm(
            cones[index_3, :2] - cones[two_cones, :2], axis=1
        ).min()

        if min_dist_to_first_two > self.max_dist * 1.1:
            return two_cones

        two_cones_pos = cones[two_cones, :2]
        third_cone = cones[index_3, :2][None]

        new_cones, *_ = combine_and_sort_virtual_with_real(
            two_cones_pos, third_cone, cone_type, car_position, car_direction
        )

        last, middle, first = my_cdist_sq_euclidean(new_cones, cones[:, :2]).argmin(
            axis=1
        )

        middle_to_last = cones[last, :2] - cones[middle, :2]
        middle_to_first = cones[first, :2] - cones[middle, :2]
        if vec_angle_between(middle_to_last, middle_to_first) < np.pi / 1.5:
            return two_cones

        return np.array([last, middle, first], dtype=np.int_)
