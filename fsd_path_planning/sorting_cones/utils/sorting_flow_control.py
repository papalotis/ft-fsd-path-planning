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

    def _sort(
        self,
        sorter: TraceSorter,
        cone_type: SortableConeTypes,
        cones_of_type: FloatArray,
        direction: FloatArray,
        backward_sorted_indices: Optional[IntArray] = None,
    ) -> IntArray:
        """
        Sort the visible cones.

        Sorts the given cones by applying a mask of cones to sort
        and returns an np.ndarray of the sorted indices of the visible cones.
        Args:
            sorter: TraceSorter instance for the backwards or forwards case.
            cone_type: The cone_type of the cones to sort.
            direction: The direction in which to sort.
            backward_sorted_indices: If backward was already sorted,
            those indices.
            Defaults to None.
        Returns:
            np.array[int]: The sorted indices but only of the visible cones.
        """
        if backward_sorted_indices is None:
            # get binary mask of cones that are within relevant fov
            mask_of_visible_cones = cones_in_range_and_pov_mask(
                self.state.position_global,
                direction,
                self.state.max_range,
                self.state.max_angle,
                cones_of_type,
            )
            # convert mask to indices as integers
            # indices_of_visible_cones = np.where(mask_of_visible_cones)[0]

            visible_cones = cones_of_type[mask_of_visible_cones]
        else:
            # remove backwards sorted cones from visible cones
            mask_keep = np.ones(len(cones_of_type), dtype=np.bool_)
            mask_keep[backward_sorted_indices] = False
            visible_cones = cones_of_type[mask_keep]

        start_index: Optional[int]
        if backward_sorted_indices is not None and len(backward_sorted_indices) > 0:
            backwards_cones = cones_of_type[backward_sorted_indices]
            start_index = 0
            first_k_indices_must_be = np.arange(len(backwards_cones))

            visible_cones = np.row_stack((backwards_cones, visible_cones))

        else:
            start_index = None
            first_k_indices_must_be = None

        # apply main sorting algorithm
        sorted_cones, _ = sorter.sort(
            visible_cones,
            cone_type,
            self.state.position_global,
            direction,
            start_idx=start_index,
            first_k_indices_must_be=first_k_indices_must_be,
        )
        if len(sorted_cones) == 0 or len(cones_of_type) == 0:
            return np.zeros(0, dtype=np.int64)

        return_value: IntArray = my_cdist_sq_euclidean(
            sorted_cones, cones_of_type
        ).argmin(axis=1)
        return return_value

    def _invert_sortable_cone(
        self, sortable_cone_type: SortableConeTypes
    ) -> SortableConeTypes:
        """Invert the cone_types of left and right."""
        if sortable_cone_type == ConeTypes.LEFT:
            sortable_cone_type_inverted = ConeTypes.RIGHT
        elif sortable_cone_type == ConeTypes.RIGHT:
            sortable_cone_type_inverted = ConeTypes.LEFT
        else:
            raise AssertionError("Unreachable code")
        sortable_cone_type_inverted = cast(
            SortableConeTypes, sortable_cone_type_inverted
        )

        return sortable_cone_type_inverted

    def _sort_backward(self, cone_type: SortableConeTypes) -> IntArray:
        """Sort cones in backward direction."""
        # we don't need to sort backwards as much as forwards
        backward_max_length = min(
            self.state.max_length_backwards, self.state.max_backwards_index
        )
        # backward_max_length = self.state.max_length
        backward_sorter = TraceSorter(
            self.state.max_n_neighbors,
            self.state.max_dist,
            self.state.max_dist_to_first,
            backward_max_length,
            self.state.threshold_directional_angle,
            self.state.threshold_absolute_angle,
        )

        cones_to_sort = self.state.cones_by_type_array[cone_type]

        # because we sort backwards we need to also invert the color of the cones
        # that we are sorting, so that any direction related logic works as expected
        cone_type_inverted_for_backward_sort = self._invert_sortable_cone(cone_type)
        backward_direction: FloatArray = self.state.direction_global * -1

        return self._sort(
            backward_sorter,
            cone_type_inverted_for_backward_sort,
            cones_to_sort,
            backward_direction,
        )

    def remove_known_sorted_cones_for_forward_sort(
        self,
        cones_to_sort: FloatArray,
        known_sorted_cones: FloatArray,
        car_position: FloatArray,
        car_direction: FloatArray,
    ) -> FloatArray:
        """
        When sorting forward we can remove cones which we know we have already driven through
        """

        if len(known_sorted_cones) == 0:
            return cones_to_sort

        square_distances = my_cdist_sq_euclidean(cones_to_sort, known_sorted_cones)

        # for the first 4 cones we allow them to be resorted at the end of the lap
        car_to_cones = car_position - known_sorted_cones[:4]
        distance_to_car = np.linalg.norm(car_to_cones, axis=1)
        angle_to_car_direction = vec_angle_between(car_to_cones, car_direction)
        mask_first_k_cones_keep = np.logical_and(
            distance_to_car < 10, angle_to_car_direction < np.pi / 2
        )
        index_first_k_cones_keep = np.where(mask_first_k_cones_keep)[0]
        # the first four cones can always be used for sorting
        square_distances[:, index_first_k_cones_keep] = np.inf

        distances_min_for_cones_to_sort: FloatArray = (
            np.min(square_distances, axis=1) ** 0.5
        )
        still_sortable_mask: BoolArray = distances_min_for_cones_to_sort > 1.0

        return_value: FloatArray = cones_to_sort[still_sortable_mask]

        return return_value

    def _sort_forward(
        self, cone_type: SortableConeTypes, backward_sorted_indices: IntArray
    ) -> IntArray:
        """Sort cones in forward direction."""
        forward_sorter = TraceSorter(
            self.state.max_n_neighbors,
            self.state.max_dist,
            self.state.max_dist_to_first,
            self.state.max_length,
            self.state.threshold_directional_angle,
            self.state.threshold_absolute_angle,
        )

        cones_to_sort = self.state.cones_by_type_array[cone_type]

        forward_direction = self.state.direction_global
        return self._sort(
            forward_sorter,
            cone_type,
            cones_to_sort,
            forward_direction,
            backward_sorted_indices,
        )

    def calculate_sort_indices(self, state: ConeSortingState) -> List[IntArray]:
        """Calculate the indices of sorted cones in both directions."""
        self.state = state

        assert self.state.max_angle <= np.pi

        sorted_indices_by_cone_type: List[IntArray] = [
            np.zeros(0, dtype=int) for _ in ConeTypes
        ]

        for cone_type in self._iterate_through_sortable_cone_types_enum():
            # sort cones backward
            backward_indices = self._sort_backward(cone_type)
            # we invert the order of the backward indices
            # because we want them to go in the same direction as the forward indices
            # and we only care about the close ones, so we pick the "last" three from the
            # backward sorted cones
            backward_indices_inverted = backward_indices[::-1][
                -self.state.max_backwards_index :
            ]
            # sort cones forward
            forward_indices = self._sort_forward(cone_type, backward_indices_inverted)

            if len(forward_indices) == 0:
                continue

            # # we were not able to sort forward
            # # don't propagate anything
            # if len(forward_indices) == len(backward_indices_inverted):
            #     continue

            sorted_indices_by_cone_type[cone_type] = forward_indices

        return sorted_indices_by_cone_type
