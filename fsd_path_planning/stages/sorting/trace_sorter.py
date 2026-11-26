#!/usr/bin/env python3
"""
Description: This module provides functionality for sorting a trace of cones into a
plausible track
Project: fsd_path_planning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fsd_path_planning.common_types import FloatArray, IntArray, TraceOutcome
from fsd_path_planning.config_dataclasses import SortingConfig
from fsd_path_planning.geometry.angles import angle_from_2d_vector, vec_angle_between
from fsd_path_planning.geometry.distances import cdist_sq_euclidean
from fsd_path_planning.geometry.shapes import points_inside_ellipse
from fsd_path_planning.geometry.transforms import rotate
from fsd_path_planning.stages.sorting.adjacency import (
    AdjacencyMatrixCache,
)
from fsd_path_planning.stages.sorting.combine import (
    calc_final_configs_for_left_and_right,
)
from fsd_path_planning.stages.sorting.nearby_cones import (
    NearbyConeSearcher,
)
from fsd_path_planning.stages.sorting.scoring import (
    calc_scores_and_end_configurations,
)
from fsd_path_planning.utils.cone_types import ConeTypes, Side, invert_cone_type


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

    distances = cdist_sq_euclidean(cones[:, :2], other_cones[:, :2])

    closest_for_each = distances.min(axis=1)
    distances_all_close = np.all(closest_for_each < (threshold * threshold))

    if cones.shape[1] == 2:
        # in this case we have no color information, we only use distance as indicator
        return bool(distances_all_close)

    if not distances_all_close:
        # distances not close no need to check color
        return False

    # last dimension is color
    idx_closest = distances.argmin(axis=1)

    color_match = cones[:, 2] == other_cones[idx_closest, 2]
    return bool(distances_all_close and color_match.all())


def _final_outcome(search_outcome: TraceOutcome, n_sorted: int) -> TraceOutcome:
    """A side's outcome once both sides have been combined."""
    if n_sorted > 0:
        return TraceOutcome.FOUND
    if search_outcome is TraceOutcome.FOUND:
        return TraceOutcome.DROPPED_WHEN_COMBINED
    return search_outcome


@dataclass
class ConeSortingCacheEntry:
    """
    Dataclass for the cache entry
    """

    input_cones: FloatArray  # x, y, color
    left_starting_cones: FloatArray | None
    right_starting_cones: FloatArray | None
    left_result: tuple[Any, ...]
    right_result: tuple[Any, ...]
    left_outcome: TraceOutcome = TraceOutcome.FOUND
    right_outcome: TraceOutcome = TraceOutcome.FOUND


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
        config: SortingConfig | None = None,
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

        # Everything the search needs beyond the scalars above - the cost
        # weights, the car size, the search geometry - comes from the config.
        # Built from the scalars when the caller passes none, so that the two
        # can never disagree.
        self.config = (
            config
            if config is not None
            else SortingConfig(
                max_n_neighbors=max_n_neighbors,
                max_dist=max_dist,
                max_dist_to_first=max_dist_to_first,
                max_length=max_length,
                threshold_directional_angle=threshold_directional_angle,
                threshold_absolute_angle=threshold_absolute_angle,
            )
        )

        self.cached_results: ConeSortingCacheEntry | None = None
        self.experimental_caching = experimental_caching
        self.adjacency_cache = AdjacencyMatrixCache()
        self.nearby_searcher = NearbyConeSearcher()

    def sort_left_right(
        self,
        cones_by_type: list[FloatArray],
        car_pos: FloatArray,
        car_dir: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        left, right, _, _ = self.sort_left_right_with_outcomes(
            cones_by_type, car_pos, car_dir
        )
        return left, right

    def sort_left_right_with_outcomes(
        self,
        cones_by_type: list[FloatArray],
        car_pos: FloatArray,
        car_dir: FloatArray,
    ) -> tuple[FloatArray, FloatArray, TraceOutcome, TraceOutcome]:
        """The sorted left and right cones, and how each side's search ended."""
        cones_flat = flatten_cones_by_type_array(cones_by_type)

        left_result, left_search_outcome = self._search_one_side(
            cones_flat, Side.LEFT.cone_type, car_pos, car_dir
        )
        left_scores, left_configs, left_first_cones = left_result

        right_result, right_search_outcome = self._search_one_side(
            cones_flat, Side.RIGHT.cone_type, car_pos, car_dir
        )
        right_scores, right_configs, right_first_cones = right_result

        self.cached_results = ConeSortingCacheEntry(
            input_cones=cones_flat,
            left_starting_cones=left_first_cones,
            right_starting_cones=right_first_cones,
            left_result=left_result,
            right_result=right_result,
            left_outcome=left_search_outcome,
            right_outcome=right_search_outcome,
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

        return (
            left_sorted[:, :2],
            right_sorted[:, :2],
            _final_outcome(left_search_outcome, len(left_sorted)),
            _final_outcome(right_search_outcome, len(right_sorted)),
        )

    def input_is_very_similar_to_previous_input(
        self,
        cones_by_type: FloatArray,
        starting_cones: FloatArray,
        threshold: float,
        cone_type: ConeTypes,
    ) -> bool:
        Side.of(cone_type)  # raises for anything but LEFT and RIGHT

        # automatically fail if caching is not enabled
        if not self.experimental_caching:
            return False

        if self.cached_results is None:
            return False

        previous_starting_cones = (
            self.cached_results.left_starting_cones
            if Side.of(cone_type) is Side.LEFT
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
    ) -> tuple[FloatArray | None, IntArray | None, FloatArray | None]:
        """
        Args:
            cones: The trace to be sorted.
            cone_type: The type of cone to be sorted.
            car_pos: The position of the car.
            car_dir: The direction towards which the car is facing.
        Returns:
            The scores and the candidate traces, and the starting cones; all
            three None when no trace was found.
        """
        result, _ = self._search_one_side(cones, cone_type, car_pos, car_dir)
        return result

    def _search_one_side(
        self,
        cones: FloatArray,
        cone_type: ConeTypes,
        car_pos: FloatArray,
        car_dir: FloatArray,
    ) -> tuple[
        tuple[FloatArray | None, IntArray | None, FloatArray | None], TraceOutcome
    ]:
        """The candidate traces for one side, and how the search ended."""
        Side.of(cone_type)  # raises for anything but LEFT and RIGHT

        no_result = None, None, None

        if len(cones) < 3:
            return no_result, TraceOutcome.TOO_FEW_CONES

        first_k = self.select_first_k_starting_cones(
            car_pos,
            car_dir,
            cones,
            cone_type,
        )
        if first_k is not None:
            start_idx = first_k[0]
            first_k_indices_must_be = first_k.copy() if len(first_k) > 1 else None
        else:
            start_idx = None
            first_k_indices_must_be = None

        if start_idx is None and first_k_indices_must_be is None:
            return no_result, TraceOutcome.NO_START_CONE

        assert first_k is not None
        assert start_idx is not None

        starting_cones = cones[first_k]

        if self.input_is_very_similar_to_previous_input(
            cones, starting_cones, threshold=0.1, cone_type=cone_type
        ):
            cr = self.cached_results
            assert cr is not None
            if Side.of(cone_type) is Side.LEFT:
                return cr.left_result, cr.left_outcome
            return cr.right_result, cr.right_outcome

        n_neighbors = min(self.max_n_neighbors, len(cones) - 1)
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
            adjacency_cache=self.adjacency_cache,
            nearby_searcher=self.nearby_searcher,
            config=self.config,
        )

        _, end_configurations, _ = result
        if len(end_configurations) == 0:
            return no_result, TraceOutcome.NO_VALID_TRACE

        return (*result[:2], starting_cones), TraceOutcome.FOUND

    def select_starting_cone(
        self,
        car_position: FloatArray,
        car_direction: FloatArray,
        cones: FloatArray,
        cone_type: ConeTypes,
        index_to_skip: np.ndarray | None = None,
    ) -> int | None:
        """
        Return the index of the starting cone
            int: The index of the stating cone
        """
        trace_distances, mask_is_valid = self.mask_cone_can_be_first_in_config(
            car_position, car_direction, cones, cone_type
        )
        if index_to_skip is not None:
            mask_is_valid[index_to_skip] = False

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
            self.max_dist_to_first * 1.5,
            self.max_dist_to_first / 1.5,
        )
        angle_signs = np.sign(cone_relative_angles)
        valid_angle_sign = 1 if cone_type == ConeTypes.LEFT else -1
        mask_valid_side = angle_signs == valid_angle_sign
        mask_is_valid_angle = np.abs(cone_relative_angles) < np.pi - np.pi / 5
        mask_is_valid_angle_min = np.abs(cone_relative_angles) > np.pi / 10
        mask_is_right_color = cones[:, 2] == cone_type

        mask_side = (
            mask_valid_side * mask_is_valid_angle * mask_is_valid_angle_min
        ) + mask_is_right_color

        mask_is_not_opposite_cone_type = cones[:, 2] != invert_cone_type(cone_type)
        mask_is_valid = mask_is_in_ellipse * mask_side * mask_is_not_opposite_cone_type

        return trace_distances, mask_is_valid

    def select_first_k_starting_cones(
        self,
        car_position: FloatArray,
        car_direction: FloatArray,
        cones: FloatArray,
        cone_type: ConeTypes,
    ) -> np.ndarray | None:
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

        cones_to_car = cones[:, :2] - car_position
        angle_to_car = vec_angle_between(cones_to_car, car_direction)

        mask_should_not_be_selected = np.abs(angle_to_car) < np.pi / 2
        idxs_to_skip = np.where(mask_should_not_be_selected)[0]
        if index_1 not in idxs_to_skip:
            idxs_to_skip = np.concatenate([idxs_to_skip, np.array([index_1])])

        # get the cone behind the car
        index_2 = self.select_starting_cone(
            car_position,
            car_direction,
            cones,
            cone_type,
            index_to_skip=idxs_to_skip,
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
        if dist > self.max_dist * 1.1 or dist < 1.4:
            return np.array([index_1], dtype=np.int_)

        two_cones = np.array([index_2, index_1], dtype=np.int_)

        return two_cones
