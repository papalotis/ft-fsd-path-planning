#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This module provides functionality for sorting a trace of cones into a
plausible track
Project: fsd_path_planning
"""
from typing import Optional, Tuple, cast

import numpy as np
from fsd_path_planning.utils.math_utils import points_inside_ellipse, vec_angle_between

from fsd_path_planning.sorting_cones.trace_sorter.common import NoPathError
from fsd_path_planning.sorting_cones.trace_sorter.sort_trace import sort_trace
from fsd_path_planning.types import FloatArray, IntArray, SortableConeTypes


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
        trace_sorted_idxs: IntArray
        empty_idxs_array: IntArray = np.zeros(0, dtype=np.int_)
        # nothing to sort
        if len(trace) == 0:
            trace_sorted_idxs = empty_idxs_array
        else:
            distances_to_car = np.linalg.norm(trace - car_pos, axis=-1)
            dist_to_closest_cone = distances_to_car.min()

            if len(trace) == 1:
                trace_sorted_idxs = np.zeros(1, dtype=np.int_)

            elif len(trace) == 2:
                # for cases where only 2 cones are available
                # just return the cones sorted by distance since there is no better algorithm
                trace_sorted_idxs = distances_to_car.argsort()

                # if the distance between the two points is too large then only get the
                # first point
                if np.abs(np.diff(distances_to_car)) > self.max_dist:
                    trace_sorted_idxs = trace_sorted_idxs[:1]

            else:
                if start_idx is None:
                    angles_to_car = vec_angle_between(trace - car_pos, car_dir)
                    start_idx = self.select_starting_cone(
                        car_pos, car_dir, trace, angles_to_car, distances_to_car
                    )

                if start_idx is None:
                    trace_sorted_idxs = empty_idxs_array
                else:
                    n_neighbors = min(self.max_n_neighbors, len(trace) - 1)
                    try:
                        trace_sorted_idxs = sort_trace(
                            trace,
                            cone_type,
                            n_neighbors,
                            start_idx,
                            self.threshold_directional_angle,
                            self.threshold_absolute_angle,
                            car_dir,
                            self.max_dist,
                            self.max_length,
                            first_k_indices_must_be,
                        )

                    # if no configurations can be found, then just return the first trace
                    except NoPathError:
                        trace_sorted_idxs = np.array([start_idx], dtype=np.int_)

        sorted_trace = trace[trace_sorted_idxs]

        return sorted_trace, trace_sorted_idxs

    def select_starting_cone(
        self,
        car_position: FloatArray,
        car_direction: FloatArray,
        cones: FloatArray,
        trace_angles: FloatArray,
        trace_distances: FloatArray,
    ) -> Optional[int]:
        """
        Return the index of the starting cone
        Args:
            trace_angles: The trace from which to choose
            trace_distances: The distance to the starting
        Returns:
            int: The index of the stating cone
        """
        mask_is_in_ellipse = points_inside_ellipse(
            cones,
            car_position,
            car_direction,
            self.max_dist_to_first * 1.3,
            self.max_dist_to_first / 1.3,
        )

        mask_is_valid_angle = np.abs(trace_angles) < np.pi / 1.1
        mask_is_closest = trace_distances == trace_distances.min()
        mask_is_valid = mask_is_valid_angle * mask_is_in_ellipse * mask_is_closest

        if np.any(mask_is_valid):
            # return the position of the first true
            start_idx = int(np.argmax(mask_is_valid))
        else:
            start_idx = None

        return start_idx
