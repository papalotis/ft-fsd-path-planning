#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Algorithm that for sorted cone configurations, finds the number of cones
that are on the wrong side of the track. For example, if we are considering the left
edge of the track, we do not want to see any cones to the right of them.

Project: fsd_path_planning
"""
from __future__ import annotations

from collections import deque
from sys import maxsize
from typing import Dict, Optional, Tuple

import numpy as np

from fsd_path_planning.cone_matching.match_directions import (
    calculate_search_direction_for_one,
)
from fsd_path_planning.types import BoolArray, FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    my_cdist_sq_euclidean,
    my_njit,
    vec_angle_between,
)

SEARCH_DIRECTIONS_CACHE_KEY_TYPE = Tuple[int, int, int]
SEARCH_DIRECTIONS_CACHE_TYPE = Dict[SEARCH_DIRECTIONS_CACHE_KEY_TYPE, FloatArray]


# my_njit = lambda x: x  # for debugging only


SENTINEL_VALUE = maxsize - 10


@my_njit
def create_search_directions_cache() -> SEARCH_DIRECTIONS_CACHE_TYPE:
    return {(SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE): np.array([-1.0, -1.0])}


@my_njit
def calculate_match_search_direction_for_one_if_not_in_cache(
    cones_xy: FloatArray,
    key: SEARCH_DIRECTIONS_CACHE_KEY_TYPE,
    cone_type: ConeTypes,
    cache_dict: SEARCH_DIRECTIONS_CACHE_TYPE,
) -> FloatArray:
    if key not in cache_dict:
        cache_dict[key] = calculate_search_direction_for_one(
            cones_xy, key[0::2], cone_type
        )

    return cache_dict[key]


@my_njit
def pre_caluclate_search_directions(
    cones: FloatArray,
    configs: IntArray,
    cone_type: ConeTypes,
    existing_cache: Optional[SEARCH_DIRECTIONS_CACHE_TYPE] = None,
) -> SEARCH_DIRECTIONS_CACHE_TYPE:
    cones_xy = cones[:, :2]
    if existing_cache is not None:
        cache = existing_cache
    else:
        cache = create_search_directions_cache()

    for c in configs:
        c = c[c != -1]
        assert len(c) >= 2

        key_first = (c[0], SENTINEL_VALUE, c[1])
        key_last = (c[-2], SENTINEL_VALUE, c[-1])

        calculate_match_search_direction_for_one_if_not_in_cache(
            cones_xy, key_first, cone_type, cache
        )
        calculate_match_search_direction_for_one_if_not_in_cache(
            cones_xy, key_last, cone_type, cache
        )

        for j in range(1, len(c) - 1):
            key = (c[j - 1], c[j], c[j + 1])
            calculate_match_search_direction_for_one_if_not_in_cache(
                cones_xy, key, cone_type, cache
            )

    return cache


@my_njit
def sorted_set_diff(a: IntArray, b: IntArray) -> IntArray:
    """Returns the set difference between a and b, assume a,b are sorted."""
    # we cannot use np.setdiff1d because it is not supported by numba
    mask = np.ones(len(a), dtype=np.bool_)
    mask[np.searchsorted(a, b)] = False
    return a[mask]


@my_njit
def find_nearby_cones_for_idxs(
    idxs: IntArray, distance_matrix_sqaured: FloatArray, search_range: float
) -> IntArray:
    mask = distance_matrix_sqaured[idxs] < search_range * search_range
    all_idxs = np.unique(mask.nonzero()[1])

    # calculate the set difference between all_idxs and idxs
    return sorted_set_diff(all_idxs, idxs)


ANGLE_MASK_CACHE_KEY_TYPE = Tuple[Tuple[int, int, int], int, int]
ANGLE_MASK_CACHE_TYPE = Dict[ANGLE_MASK_CACHE_KEY_TYPE, Tuple[bool, bool]]


@my_njit
def create_angle_cache() -> ANGLE_MASK_CACHE_TYPE:
    # the key is a tuple which consists of the following:
    # - the key of the search direction cache (previous cone, current cone, next cone)
    # - the index of the cone we are considering
    # - the index of the cone we are comparing to
    d = {
        (
            (SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE),
            SENTINEL_VALUE,
            SENTINEL_VALUE,
        ): (False, False),
    }

    return d


@my_njit
def angle_between_search_direction_of_cone_and_other_cone_is_too_large(
    all_cone_directions: FloatArray,
    directions_key: SEARCH_DIRECTIONS_CACHE_KEY_TYPE,
    cone_idx: int,
    other_cone_idx: int,
    search_directions_cache: SEARCH_DIRECTIONS_CACHE_TYPE,
    search_angle: float,
) -> tuple[bool, bool]:
    from_cone_to_other_cone = all_cone_directions[cone_idx, other_cone_idx]

    search_direction = search_directions_cache[directions_key]

    good_angle = (
        vec_angle_between(from_cone_to_other_cone, search_direction) < search_angle / 2
    )
    bad_angle = (
        vec_angle_between(from_cone_to_other_cone, -search_direction) < search_angle / 2
    )

    return good_angle, bad_angle


@my_njit
def angle_between_search_direction_of_cone_and_other_cone_is_too_large_if_not_in_cache(
    all_cone_directions: FloatArray,
    directions_key: SEARCH_DIRECTIONS_CACHE_KEY_TYPE,
    cone_idx: int,
    other_cone_idx: int,
    search_directions_cache: SEARCH_DIRECTIONS_CACHE_TYPE,
    angle_cache: ANGLE_MASK_CACHE_TYPE,
    search_angle: float,
) -> bool:
    key = (directions_key, cone_idx, other_cone_idx)
    if key not in angle_cache:
        angle_cache[
            key
        ] = angle_between_search_direction_of_cone_and_other_cone_is_too_large(
            all_cone_directions,
            directions_key,
            cone_idx,
            other_cone_idx,
            search_directions_cache,
            search_angle,
        )

    return angle_cache[key]


@my_njit
def calculate_visible_cones_for_one_cone(
    cone_idx: int,
    cone_within_distance_matrix_mask: BoolArray,
    search_direction_key: SEARCH_DIRECTIONS_CACHE_KEY_TYPE,
    cone_to_cone_vecs: FloatArray,
    search_angle: float,
    search_direction_cache: SEARCH_DIRECTIONS_CACHE_TYPE,
    angles_between_search_direction_and_other_cone_cache: ANGLE_MASK_CACHE_TYPE,
    idxs_to_check,
) -> tuple[BoolArray, BoolArray]:
    angle_good_mask = np.zeros(len(idxs_to_check), dtype=np.bool_)
    angle_bad_mask = np.zeros(len(idxs_to_check), dtype=np.bool_)
    for i in range(len(idxs_to_check)):
        idx = idxs_to_check[i]

        if not cone_within_distance_matrix_mask[cone_idx, idx]:
            continue

        (
            value_good,
            value_bad,
        ) = angle_between_search_direction_of_cone_and_other_cone_is_too_large_if_not_in_cache(
            cone_to_cone_vecs,
            search_direction_key,
            cone_idx,
            idx,
            search_direction_cache,
            angles_between_search_direction_and_other_cone_cache,
            search_angle,
        )
        angle_good_mask[i] = value_good
        angle_bad_mask[i] = value_bad

    mask_distance = cone_within_distance_matrix_mask[cone_idx]

    good_mask = angle_good_mask & mask_distance[idxs_to_check]
    bad_mask = angle_bad_mask & mask_distance[idxs_to_check]

    return good_mask, bad_mask


@my_njit
def _impl_number_cones_on_each_side_for_each_config(
    cones: FloatArray,
    configs: IntArray,
    cone_type: ConeTypes,
    search_distance: float,
    search_angle: float,
    existing_search_directions_cache: Optional[SEARCH_DIRECTIONS_CACHE_TYPE] = None,
    existing_angles_mask_cache: Optional[ANGLE_MASK_CACHE_TYPE] = None,
    distance_matrix_square: Optional[FloatArray] = None,
    cones_to_cones_vecs: Optional[FloatArray] = None,
) -> tuple[IntArray, IntArray]:
    """
    For each configuration, find the number of cones that are on the expected side of
    the track, and the number of cones that are on the wrong side of the track.

    Args:
        cones: array of cone positions and types
        configs: array of sorted cone configurations
        cone_type: the type of cone we are considering
        search_distance: the distance to search for cones
        search_angle: the angle to search for cones

    Returns:
        A tuple of two arrays, the first is the number of cones on the correct side of
        the track, the second is the number of cones on the wrong side of the track.
    """
    cones_xy = cones[:, :2]

    idxs_in_all_configs = np.unique(configs)
    idxs_in_all_configs = idxs_in_all_configs[idxs_in_all_configs != -1]

    if cones_to_cones_vecs is None:
        cones_to_cones_vecs = cones_xy - np.expand_dims(cones_xy, 1)

    if distance_matrix_square is None:
        distance_matrix_square = my_cdist_sq_euclidean(cones_xy, cones_xy)
        np.fill_diagonal(
            distance_matrix_square, 1e6
        )  # for some reason np.inf doesn't work here

    distance_matrix_mask = distance_matrix_square < search_distance * search_distance

    search_directions_cache = pre_caluclate_search_directions(
        cones, configs, cone_type, existing_search_directions_cache
    )

    close_idxs = find_nearby_cones_for_idxs(
        idxs_in_all_configs, distance_matrix_square, search_distance
    )

    n_bad_cones_for_all = np.zeros(len(configs), dtype=np.int_)
    n_good_cones_for_all = np.zeros(len(configs), dtype=np.int_)

    if existing_angles_mask_cache is None:
        angle_cache = create_angle_cache()
    else:
        angle_cache = existing_angles_mask_cache

    # print(len(angle_cache))

    for i, c in enumerate(configs):
        c = c[c != -1]

        extra_idxs = sorted_set_diff(idxs_in_all_configs, c)
        other_idxs = np.concatenate((close_idxs, extra_idxs))

        for j in range(len(c)):
            if j == 0:
                key = (c[j], SENTINEL_VALUE, c[j + 1])
            elif j == len(c) - 1:
                key = (c[j - 1], SENTINEL_VALUE, c[j])
            else:
                key = (c[j - 1], c[j], c[j + 1])

            mask_good, mask_bad = calculate_visible_cones_for_one_cone(
                c[j],
                distance_matrix_mask,
                key,
                cones_to_cones_vecs,
                search_angle,
                search_directions_cache,
                angle_cache,
                idxs_to_check=other_idxs,
            )

            n_good_cones_for_all[i] += mask_good.sum()
            n_bad_cones_for_all[i] += mask_bad.sum()

    return n_good_cones_for_all, n_bad_cones_for_all


class NearbyConeSearcher:
    def __init__(self) -> None:
        self.caches_cache: deque[
            tuple[tuple[int, ConeTypes], tuple[dict, dict, FloatArray, FloatArray]]
        ] = deque(maxlen=20)

    def get_caches(
        self, cones: np.ndarray, cone_type: ConeTypes
    ) -> tuple[dict, dict, FloatArray, FloatArray]:
        array_buffer = cones.tobytes()
        array_hash = hash(array_buffer)
        cache_key = (array_hash, cone_type)

        try:
            index_of_hashed_values = next(
                i for i, (k, _) in enumerate(self.caches_cache) if k == cache_key
            )
        except StopIteration:
            index_of_hashed_values = None
        if index_of_hashed_values is None:
            cones_xy = cones[:, :2]
            distance_matrix_square = my_cdist_sq_euclidean(cones_xy, cones_xy)
            np.fill_diagonal(distance_matrix_square, 1e7)
            cones_to_cones = cones_xy - cones_xy[:, None]

            self.caches_cache.append(
                (
                    cache_key,
                    (
                        create_search_directions_cache(),
                        create_angle_cache(),
                        distance_matrix_square,
                        cones_to_cones,
                    ),
                )
            )
            index_of_hashed_values = -1

        return self.caches_cache[index_of_hashed_values][1]

    def number_of_cones_on_each_side_for_each_config(
        self,
        cones: np.ndarray,
        configs: np.ndarray,
        cone_type: ConeTypes,
        max_distance: float,
        max_angle: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        cached_values = self.get_caches(cones, cone_type)
        return _impl_number_cones_on_each_side_for_each_config(
            cones, configs, cone_type, max_distance, max_angle, *cached_values[:]
        )


NEARBY_CONE_SEARCH_CACHE = NearbyConeSearcher()


def clear_nearby_cone_search_cache() -> None:
    global NEARBY_CONE_SEARCH_CACHE
    NEARBY_CONE_SEARCH_CACHE = NearbyConeSearcher()


def number_cones_on_each_side_for_each_config(
    cones: np.ndarray,
    configs: np.ndarray,
    cone_type: ConeTypes,
    max_distance: float,
    max_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    return NEARBY_CONE_SEARCH_CACHE.number_of_cones_on_each_side_for_each_config(
        cones, configs, cone_type, max_distance, max_angle
    )
