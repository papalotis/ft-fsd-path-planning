#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This file provides the core algorithm for sorting a trace of cones into a
plausible track
Project: fsd_path_planning
"""
import sys
from typing import Optional, cast

import numpy as np

from fsd_path_planning.sorting_cones.trace_sorter.adjacency_matrix import \
    create_adjacency_matrix
from fsd_path_planning.sorting_cones.trace_sorter.cost_function import \
    cost_configurations
from fsd_path_planning.sorting_cones.trace_sorter.end_configurations import \
    find_all_end_configurations
from fsd_path_planning.types import FloatArray, IntArray, SortableConeTypes
from fsd_path_planning.utils.utils import Timer


def calc_scores_and_end_configurations(
    trace: FloatArray,
    cone_type: SortableConeTypes,
    n_neighbors: int,
    start_idx: int,
    threshold_directional_angle: float,
    threshold_absolute_angle: float,
    vehicle_position: FloatArray,
    vehicle_direction: FloatArray,
    max_dist: float = np.inf,
    max_length: int = sys.maxsize,
    first_k_indices_must_be: Optional[IntArray] = None,
) -> tuple[FloatArray, IntArray]:
    """
    Sorts a set of points such that the sum of the angles between the points is minimal.
    If a point is too far away, from any neighboring points, it is considered an outlier
    and is removed from the ordering
    Args:
        trace: The points to be ordered
        cone_type: The type of cone to be sorted (left/right)
        n_neighbors: The number of neighbors to be considered. For exhaustive
        search set to `len(trace) - 1`
        start_idx: The index of the point to be set first in the ordering.
        max_dist: The maximum valid distance between neighbors
        Defaults to np.inf
        max_length: The maximum valid length of the tree
        Defaults to np.inf
        cone_type:: The type of cone that is being sorted (left or right
        trace)
    Raises:
        ValueError: If `n_neighbors` is greater than len(trace) - 1
        RuntimeError: If no valid path can be computed
    Returns:
        A list of indexes of the points in the optimal ordering, as well as the
        the costs of all end configurations and their corresponding indices
    """
    no_print = True
    with Timer("create_adjacency_matrix", no_print):
        adjacency_matrix, reachable_nodes = create_adjacency_matrix(
            cones=trace,
            n_neighbors=n_neighbors,
            start_idx=start_idx,
            max_dist=max_dist,
            cone_type=cone_type,
        )

    target_length = min(reachable_nodes.shape[0], max_length)

    if first_k_indices_must_be is None:
        first_k_indices_must_be = np.arange(0)

    with Timer("find_all_end_configurations", no_print):
        all_end_configurations, _ = find_all_end_configurations(
            trace,
            cone_type,
            start_idx,
            adjacency_matrix,
            target_length,
            threshold_directional_angle,
            threshold_absolute_angle,
            first_k_indices_must_be,
            vehicle_position,
            vehicle_direction,
            car_size=2.1,
            # this is only used for testing/debugging/visualization purposes and should be
            # set to False in production
            store_all_end_configurations=False,
        )

    # remove last cone from config, only if the last cone not of the type we are sorting

    last_cone_in_each_config_idx = (
        np.argmax(all_end_configurations == -1, axis=1) - 1
    ) % all_end_configurations.shape[1]

    last_cone_in_each_config = all_end_configurations[
        np.arange(all_end_configurations.shape[0]), last_cone_in_each_config_idx
    ]

    mask_last_cone_is_not_of_type = trace[last_cone_in_each_config, 2] != cone_type
    mask_config_has_over_3_cones = last_cone_in_each_config_idx > 2

    mask_should_trim = mask_last_cone_is_not_of_type & mask_config_has_over_3_cones

    last_cone_in_each_config_idx_masked = last_cone_in_each_config_idx[mask_should_trim]

    all_end_configurations[mask_should_trim, last_cone_in_each_config_idx_masked] = -1

    with Timer("cost_configurations", no_print):
        costs = cost_configurations(
            points=trace,
            configurations=all_end_configurations,
            cone_type=cone_type,
            vehicle_position=vehicle_position,
            vehicle_direction=vehicle_direction,
            return_individual_costs=False,
        )
    costs_sort_idx = np.argsort(costs)
    costs = cast(FloatArray, costs[costs_sort_idx])
    all_end_configurations = cast(IntArray, all_end_configurations[costs_sort_idx])

    return (costs, all_end_configurations)
