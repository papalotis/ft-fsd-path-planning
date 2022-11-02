#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Cone sorting class.

Description: Entry point for Pathing/ConeSorting
Project: fsd_path_planning
"""
from typing import Tuple

from fsd_path_planning.sorting_cones.utils.cone_sorting_dataclasses import (
    ConeSortingInput,
    ConeSortingState,
)
from fsd_path_planning.sorting_cones.utils.sorting_flow_control import (
    SortingFlowControl,
)
from fsd_path_planning.types import FloatArray
from fsd_path_planning.utils.cone_types import ConeTypes
from icecream import ic  # pylint: disable=unused-import


class ConeSorting:
    """Class that takes all Pathing/ConeSorting responsibilities."""

    def __init__(
        self,
        max_n_neighbors: int,
        max_dist: float,
        max_dist_to_first: float,
        max_range: float,
        max_angle: float,
        max_length: int,
        max_length_backwards: int,
        max_backwards_index: int,
        threshold_directional_angle: float,
        threshold_absolute_angle: float,
    ):
        """
        Init method.

        Args:
            max_n_neighbors, max_dist, max_dist_to_first: Arguments for TraceSorter.
            max_length: Argument for TraceSorter. The maximum length of a
                valid trace in the sorting algorithm.
            max_length_backwards: Argument for TraceSorter. The maximum length of a
                valid trace in the sorting algorithm for the backwards direction.
            max_range: The maximum range for which cones will be sorted
                (used in mask).
            max_angle: The maximum angle for which cones will be sorted (used in mask).
            max_backwards_index: the maximum amount of cones that will be taken in the
                backwards direction
            threshold_directional_angle: The threshold for the directional angle that is
                the minimum angle for consecutive cones to be connected in the direction
                of the trace (clockwise for left cones, counterclockwise for right cones).
            threshold_absolute_angle: The threshold for the absolute angle that is the
                minimum angle for consecutive cones to be connected regardless of the
                cone type.
        """
        self.input = ConeSortingInput()

        self.state = ConeSortingState(
            max_n_neighbors=max_n_neighbors,
            max_dist=max_dist,
            max_dist_to_first=max_dist_to_first,
            max_range=max_range,
            max_angle=max_angle,
            max_length=max_length,
            max_length_backwards=max_length_backwards,
            max_backwards_index=max_backwards_index,
            threshold_directional_angle=threshold_directional_angle,
            threshold_absolute_angle=threshold_absolute_angle,
        )

    def set_new_input(self, slam_input: ConeSortingInput) -> None:
        """Save inputs from other software nodes in variable."""
        self.input = slam_input

    def transition_input_to_state(self) -> None:
        """Parse and save the inputs in the state variable."""
        self.state.position_global, self.state.direction_global = (
            self.input.slam_position,
            self.input.slam_direction,
        )

        self.state.cones_by_type_array = self.input.slam_cones

    def run_cone_sorting(
        self,
    ) -> Tuple[FloatArray, FloatArray]:
        """
        Calculate the sorted cones.

        Returns:
            The sorted cones. The first array contains the sorted blue (left) cones and
            the second array contains the sorted yellow (right) cones.
        """
        # make transition from set inputs to usable state variables
        self.transition_input_to_state()

        sorting_calculation = SortingFlowControl(self.state)

        sorted_indices_list_by_cone_type = sorting_calculation.calculate_sort_indices(
            self.state
        )

        sorted_points_left = self.state.cones_by_type_array[ConeTypes.LEFT][
            sorted_indices_list_by_cone_type[ConeTypes.LEFT]
        ]
        sorted_points_right = self.state.cones_by_type_array[ConeTypes.RIGHT][
            sorted_indices_list_by_cone_type[ConeTypes.RIGHT]
        ]

        return sorted_points_left, sorted_points_right
