#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Class for the flow control of the cone sorting calculation.

Description: Entry point for Pathing/ConeSorting
Project: fsd_path_planning
"""
from typing import Iterable, List, Optional, cast

import numpy as np

from fsd_path_planning.sorting_cones.trace_sorter.core_trace_sorter import TraceSorter
from fsd_path_planning.sorting_cones.utils.cone_sorting_dataclasses import (
    ConeSortingState,
)
from fsd_path_planning.types import BoolArray, FloatArray, IntArray, SortableConeTypes
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import my_cdist_sq_euclidean, vec_angle_between


def cones_in_range_and_pov_mask(
    car_pos: FloatArray,
    car_dir: FloatArray,
    sight_range: float,
    sight_angle: float,
    colored_cones: FloatArray,
) -> np.ndarray:
    """
    Calculates the indices of the visible cones according to the car position

    Args:
        car_pos (np.array): The global position of the car
        car_dir (np.array): The direction of the car in global coordinates
        sight_range (float): The max distance that a cone can be seen
        sight_angle (float): The maximum angle that a cone can have to the car and still be visible in rad
        colored_cones (np.array): The cones that define the track

    Returns:
        np.array: The indices of the visible cones
    """
    dist_from_car = np.linalg.norm(car_pos - colored_cones, axis=1)
    dist_mask = dist_from_car < sight_range

    vec_from_car = colored_cones - car_pos

    angles_to_car = vec_angle_between(car_dir[None], vec_from_car)
    mask_angles = np.logical_and(
        -sight_angle / 2 < angles_to_car, angles_to_car < sight_angle / 2
    )

    visible_cones_mask = np.logical_and(dist_mask, mask_angles)

    return visible_cones_mask


class SortingFlowControl:
    """Class that controls the cone cone sorting processes."""

    def __init__(self, state: ConeSortingState) -> None:
        """Init method."""
        self.state = state

    def _iterate_through_sortable_cone_types_enum(self) -> Iterable[SortableConeTypes]:
        """Iterate over the sortable cone types."""
        return cast(Iterable[SortableConeTypes], (ConeTypes.LEFT, ConeTypes.RIGHT))

    def ravel_cones_by_type_array(self, cones_by_type: list[FloatArray]) -> FloatArray:
        """Ravel the cones_by_type_array"""
        n_all_cones = sum(map(len, cones_by_type))

        # (x, y, color)
        out = np.empty((n_all_cones, 3), dtype=np.float64)
        n_start = 0
        for cone_type in ConeTypes:
            n_cones = len(self.state.cones_by_type_array[cone_type])
            out[n_start : n_start + n_cones, :2] = self.state.cones_by_type_array[
                cone_type
            ]
            out[n_start : n_start + n_cones, 2] = cone_type
            n_start += n_cones

    def calculate_sort_indices(self, state: ConeSortingState) -> List[IntArray]:
        """Calculate the indices of sorted cones in both directions."""
        self.state = state

        sorted_indices_by_cone_type: List[IntArray] = [
            np.zeros(0, dtype=int) for _ in ConeTypes
        ]

        raveled_cones = self.ravel_cones_by_type_array(self.state.cones_by_type_array)

        ts = TraceSorter(
            self.state.max_n_neighbors,
            self.state.max_dist,
            self.state.max_dist_to_first,
            self.state.max_length,
            self.state.threshold_directional_angle,
            self.state.threshold_absolute_angle,
        )

        car_position = self.state.position_global
        car_direction = self.state.direction_global

        ts.calculate_configurations_and_scores(
            raveled_cones, ConeTypes.LEFT, car_position, car_direction
        )

        return sorted_indices_by_cone_type
