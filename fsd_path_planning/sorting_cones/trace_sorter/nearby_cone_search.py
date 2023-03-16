#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Algorithm that for sorted cone configurations, finds the number of cones
that are on the wrong side of the track. For example, if we are considering the left
edge of the track, we do not want to see any cones to the right of them.

Project: fsd_path_planning
"""
import numpy as np

from fsd_path_planning.cone_matching.functional_cone_matching import (
    calculate_match_search_direction, cones_in_range_and_pov_mask)
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import my_cdist_sq_euclidean, my_njit


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


@my_njit
def number_cones_on_each_side_for_each_config(
    cones: FloatArray,
    configs: IntArray,
    cone_type: ConeTypes,
    search_distance: float,
    search_angle: float,
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

    idxs_in_all_configs = np.unique(configs.flatten())
    idxs_in_all_configs = idxs_in_all_configs[idxs_in_all_configs != -1]

    distance_matrix = my_cdist_sq_euclidean(cones_xy, cones_xy)

    close_idxs = find_nearby_cones_for_idxs(
        idxs_in_all_configs, distance_matrix, search_distance * 1.1
    )
    n_bad_cones_for_all = np.zeros(len(configs), dtype=np.int_)
    n_good_cones_for_all = np.zeros(len(configs), dtype=np.int_)

    for i, c in enumerate(configs):
        c = c[c != -1]
        p = cones_xy[c]
        match_directions = calculate_match_search_direction(p, cone_type)
        anti_match_directions = -match_directions

        extra_idxs = sorted_set_diff(idxs_in_all_configs, c)
        other_idxs = np.concatenate((close_idxs, extra_idxs))
        other_cones = cones_xy[other_idxs]

        n_bad_cones = cones_in_range_and_pov_mask(
            p, anti_match_directions, search_distance, search_angle, other_cones
        ).sum()

        n_good_cones = cones_in_range_and_pov_mask(
            p, match_directions, search_distance, search_angle, other_cones
        ).sum()

        n_bad_cones_for_all[i] = n_bad_cones
        n_good_cones_for_all[i] = n_good_cones

    return n_good_cones_for_all, n_bad_cones_for_all
