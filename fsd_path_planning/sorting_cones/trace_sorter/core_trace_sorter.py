#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This module provides functionality for sorting a trace of cones into a
plausible track
Project: fsd_path_planning
"""
from typing import Optional, Tuple

import numpy as np

from fsd_path_planning.sorting_cones.trace_sorter.common import NoPathError
from fsd_path_planning.sorting_cones.trace_sorter.sort_trace import sort_trace
from fsd_path_planning.types import FloatArray, IntArray, SortableConeTypes
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
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

    def sort(  # pylint: disable=too-many-arguments
        self,
        trace: FloatArray,
        cone_type: SortableConeTypes,
        car_pos: FloatArray,
        car_dir: FloatArray,
        start_idx: Optional[int] = None,
        first_k_indices_must_be: Optional[IntArray] = None,
    ) -> Tuple[FloatArray, IntArray]:
        """
        Sorts a provided trace. Applies basic tests to remove outlier configurations (such as
        when the nearest cone is too far)
        Args:
            trace: The trace to be sorted
            cone_type: The type of cone to be sorted
            car_pos: The position from which the sorting happens
            car_dir: The direction towards which the car goes at the start
            start_idx: The index of the starting point. If not set then
            the point closest to `car_pos` is used . Defaults to None.
        Returns:
            np.ndarray: The sorted trace, `len(return_value) <= len(trace)`
        """
        # updates the given trace to exclude passed cones
        # trace = self.remove_cones_behind(trace)
        # trace_sorted_idxs: IntArray
        no_result = None
        # nothing to sort
        if len(trace) == 0:
            return_value = no_result
        else:
            distances_to_car = np.linalg.norm(trace - car_pos, axis=-1)
            dist_to_closest_cone = distances_to_car.min()

            if len(trace) == 1:
                return_value = no_result

            elif len(trace) == 2:
                # for cases where only 2 cones are available
                # just return the cones sorted by distance since there is no better algorithm
                return_value = None

                # if the distance between the two points is too large then only get the
                # first point
                if np.abs(np.diff(distances_to_car)) > self.max_dist:
                    return_value = None

            else:
                if start_idx is None and first_k_indices_must_be is None:
                  
                    first_2 = self.select_first_two_starting_cones(
                        car_pos,
                        car_dir,
                        trace,
                        cone_type,
                    )
                    if first_2 is None:
                        pass
                    else:
                        start_idx = first_2[0]
                        if len(first_2) > 1:
                            first_k_indices_must_be = first_2.copy()
                
                
                if start_idx is None and first_k_indices_must_be is None:
                    return_value = no_result
                else:
                    n_neighbors = min(self.max_n_neighbors, len(trace) - 1)
                    try:
                        return_value = sort_trace(
                            trace,
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
                        )[1]

                    # if no configurations can be found, then just return the first trace
                    except NoPathError:
                        return_value = None




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
        
        raise ValueError(f'Cone type {cone_type} cannot be inverted.')

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
        cones_relative = rotate(cones - car_position, -angle_from_2d_vector(car_direction))

        cone_relative_angles = angle_from_2d_vector(cones_relative)

        trace_distances = np.linalg.norm(cones_relative, axis=-1)


        mask_is_in_ellipse = points_inside_ellipse(
            cones,
            car_position,
            car_direction,
            self.max_dist_to_first * 1.3,
            self.max_dist_to_first / 1.3,
        )

        angle_signs = np.sign(cone_relative_angles)
        valid_angle_sign = 1 if cone_type == ConeTypes.LEFT else -1
        mask_valid_side = angle_signs == valid_angle_sign
        mask_is_valid_angle = np.abs(cone_relative_angles) < np.pi / 1
        mask_is_valid_angle_min = np.abs(cone_relative_angles) > np.pi / 6
        mask_is_valid = (
            mask_is_valid_angle
            * mask_is_in_ellipse
            * mask_valid_side
            * mask_is_valid_angle_min
        )

        trace_distances_copy = trace_distances.copy()
        trace_distances_copy[~mask_is_valid] = np.inf

        if np.any(mask_is_valid) > 0:
            sorted_idxs = np.argsort(trace_distances_copy)
            start_idx = None
            for idx in sorted_idxs:
                if idx != index_to_skip:
                    start_idx = idx
                    break
            # start_idx = int(np.argmin(trace_distances_copy))
            if trace_distances_copy[start_idx] > self.max_dist_to_first:
                start_idx = None
        else:
            start_idx = None

        # if np.any(mask_is_valid):
        #     # return the position of the first true
        #     start_idx = int(np.argmax(mask_is_valid))
        # else:
        #     start_idx = None

        return start_idx

    def select_first_two_starting_cones(
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
            -car_direction,
            cones,
            self.invert_cone_type(cone_type),
            index_to_skip=np.array([index_1]),
        )

        if index_2 is None:
            return np.array([index_1], dtype=np.int_)

        cone_dir_1 = cones[index_1] - cones[index_2]
        cone_dir_2 = cones[index_2] - cones[index_1]

        angle_1 = vec_angle_between(cone_dir_1, car_direction)
        angle_2 = vec_angle_between(cone_dir_2, car_direction)

        if angle_1 > angle_2:
            index_1, index_2 = index_2, index_1


        dist = np.linalg.norm(cone_dir_1)
        if dist > self.max_dist * 1.1 or index_2 == index_1:
            return_value = np.array([index_1], dtype=np.int_)
        else:
            return_value = np.array([index_2, index_1], dtype=np.int_)

        return return_value
