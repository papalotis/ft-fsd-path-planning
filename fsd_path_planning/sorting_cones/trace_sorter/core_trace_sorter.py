#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This module provides functionality for sorting a trace of cones into a
plausible track
Project: fsd_path_planning
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from fsd_path_planning.common_types import FloatArray, IntArray
from fsd_path_planning.sorting_cones.trace_sorter.combine_traces import (
    calc_final_configs_for_left_and_right,
)
from fsd_path_planning.sorting_cones.trace_sorter.common import NoPathError
from fsd_path_planning.sorting_cones.trace_sorter.find_configs_and_scores import (
    calc_scores_and_end_configurations,
)
from fsd_path_planning.sorting_cones.trace_sorter.start_cones_selection import (
    select_first_k_starting_cones,
)
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import my_cdist_sq_euclidean
from fsd_path_planning.utils.utils import Timer


def flatten_cones_by_type_array(cones_by_type: list[FloatArray]) -> FloatArray:
    """Ravel the cones_by_type_array"""

    if (
        isinstance(cones_by_type, np.ndarray)
        and cones_by_type.ndim == 2
        and cones_by_type.shape[1] == 3
    ):
        return cones_by_type

    n_all_cones = sum(map(len, cones_by_type))

    # (x, y, color)
    out = np.empty((n_all_cones, 3))
    n_start = 0
    for cone_type in ConeTypes:
        n_cones = len(cones_by_type[cone_type])
        out[n_start : n_start + n_cones, :2] = cones_by_type[cone_type].reshape(-1, 2)
        out[n_start : n_start + n_cones, 2] = cone_type
        n_start += n_cones

    return out


def cone_arrays_are_similar(
    cones: FloatArray | None,
    other_cones: FloatArray | None,
    threshold: float,
) -> bool:
    if cones is None or other_cones is None:
        return False

    assert cones.shape[1] == other_cones.shape[1], (cones.shape, other_cones.shape)
    if cones.shape != other_cones.shape:
        return False

    distances = my_cdist_sq_euclidean(cones[:, :2], other_cones[:, :2])

    closest_for_each = distances.min(axis=1)
    distances_all_close = np.all(closest_for_each < (threshold * threshold))

    if cones.shape[1] == 2:
        # in this case we have no color information, we only use distance as indicator
        return distances_all_close

    if not distances_all_close:
        # distances not close no need to check color
        return False

    # last dimension is color
    idx_closest = distances.argmin(axis=1)

    color_match = cones[:, 2] == other_cones[idx_closest, 2]
    return distances_all_close and color_match.all()


@dataclass
class ConeSortingCacheEntry:
    """
    Dataclass for the cache entry
    """

    input_cones: FloatArray  # x, y, color
    left_starting_cones: FloatArray
    right_starting_cones: FloatArray
    left_result: tuple[Any, ...]
    right_result: tuple[Any, ...]


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
        experimental_caching: bool = False,
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

        self.cached_results: ConeSortingCacheEntry | None = None
        self.experimental_caching = experimental_caching

    def sort_left_right(
        self,
        cones_by_type: list[FloatArray],
        car_pos: FloatArray,
        car_dir: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        timer_no_print = True
        cones_flat = flatten_cones_by_type_array(cones_by_type)

        # mask_cones_close = (
        #     my_cdist_sq_euclidean(car_pos[None], cones_flat[:, :2])[0] < 25**2
        # )

        # print(mask_cones_close.mean())

        # cones_flat = cones_flat[mask_cones_close]

        with Timer("left config search", timer_no_print):
            (
                left_scores,
                left_configs,
                left_first_cones,
            ) = left_result = self.calc_configurations_with_score_for_one_side(
                cones_flat,
                ConeTypes.LEFT,
                car_pos,
                car_dir,
            )

        with Timer("right config search", timer_no_print):
            (
                right_scores,
                right_configs,
                right_first_cones,
            ) = right_result = self.calc_configurations_with_score_for_one_side(
                cones_flat,
                ConeTypes.RIGHT,
                car_pos,
                car_dir,
            )

        self.cached_results = ConeSortingCacheEntry(
            input_cones=cones_flat,
            left_starting_cones=left_first_cones,
            right_starting_cones=right_first_cones,
            left_result=left_result,
            right_result=right_result,
        )

        (left_config, right_config) = calc_final_configs_for_left_and_right(
            left_scores,
            left_configs,
            right_scores,
            right_configs,
            cones_flat,
            car_pos,
            car_dir,
        )
        left_config = left_config[left_config != -1]
        right_config = right_config[right_config != -1]

        # remove any placeholder positions if they are present
        left_config = left_config[left_config != -1]
        right_config = right_config[right_config != -1]

        left_sorted = cones_flat[left_config]
        right_sorted = cones_flat[right_config]

        return left_sorted[:, :2], right_sorted[:, :2]

    def input_is_very_similar_to_previous_input(
        self,
        cones_by_type: FloatArray,
        starting_cones: FloatArray,
        threshold: float,
        cone_type: ConeTypes,
    ) -> bool:
        assert cone_type in (ConeTypes.LEFT, ConeTypes.RIGHT)

        # automatically fail if caching is not enabled
        if not self.experimental_caching:
            return False

        if self.cached_results is None:
            return False

        previous_starting_cones = (
            self.cached_results.left_starting_cones
            if cone_type == ConeTypes.LEFT
            else self.cached_results.right_starting_cones
        )

        # first we check if small array is similar
        previous_cones_similar = cone_arrays_are_similar(
            starting_cones, previous_starting_cones, threshold
        )

        # if it not then we need to redo the sorting
        if not previous_cones_similar:
            return False

        # if the small array is similar, we check if the large array is similar
        all_cones_are_similar = cone_arrays_are_similar(
            cones_by_type, self.cached_results.input_cones, threshold
        )
        # if the large array is not similar, we need to redo the sorting
        return all_cones_are_similar

    def calc_configurations_with_score_for_one_side(
        self,
        cones: FloatArray,
        cone_type: ConeTypes,
        car_pos: FloatArray,
        car_dir: FloatArray,
    ) -> Tuple[Optional[FloatArray], Optional[IntArray], Optional[FloatArray]]:
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

        no_result = None, None, None

        if len(cones) < 3:
            return no_result

        first_k = select_first_k_starting_cones(
            car_pos,
            car_dir,
            cones,
            cone_type,
            self.max_dist_to_first,
        )
        if first_k is not None:
            start_idx = first_k[0]
            if len(first_k) > 1:
                first_k_indices_must_be = first_k.copy()
            else:
                first_k_indices_must_be = None
        else:
            start_idx = None
            first_k_indices_must_be = None

        if start_idx is None and first_k_indices_must_be is None:
            return no_result

        assert first_k is not None

        starting_cones = cones[first_k]

        if self.input_is_very_similar_to_previous_input(
            cones, starting_cones, threshold=0.1, cone_type=cone_type
        ):
            # print("Using cached results")
            cr = self.cached_results
            return cr.left_result if cone_type == ConeTypes.LEFT else cr.right_result

        # print("Calculating new configuration")

        n_neighbors = min(self.max_n_neighbors, len(cones) - 1)
        try:
            result = calc_scores_and_end_configurations(
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

        return_value = (*result[:2], starting_cones)

        return return_value
