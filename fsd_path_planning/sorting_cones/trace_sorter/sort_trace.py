#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This File provides the core algorithm for sorting a trace of cones into a
plausible track
Project: fsd_path_planning
"""
import sys
from typing import Optional

import numpy as np

from fsd_path_planning.sorting_cones.trace_sorter.adjacency_matrix import (
    create_adjacency_matrix,
)
from fsd_path_planning.sorting_cones.trace_sorter.cost_function import (
    cost_configurations,
)
from fsd_path_planning.sorting_cones.trace_sorter.end_configurations import (
    find_all_end_configurations,
)
from fsd_path_planning.types import FloatArray, IntArray, SortableConeTypes


def sort_trace(
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
) -> tuple[IntArray, tuple[FloatArray, IntArray]]:
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

    if n_neighbors >= len(trace):
        raise ValueError(
            f"sort_trace was called with n_neighbors ({n_neighbors}) >= len(trace)"
            f" ({len(trace)})."
        )

    adjacency_matrix, reachable_nodes = create_adjacency_matrix(
        trace=trace, n_neighbors=n_neighbors, start_idx=start_idx, max_dist=max_dist
    )

    target_length = min(reachable_nodes.shape[0], max_length)

    if first_k_indices_must_be is None:
        first_k_indices_must_be = np.arange(0)

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
        store_all_end_configurations=False,  # this is only used for testing/debugging/visualization purposes and should be set to False in production
    )

    best_configuration: IntArray

    costs = cost_configurations(
        points=trace,
        configurations=all_end_configurations,
        cone_type=cone_type,
        vehicle_position=vehicle_position,
        vehicle_direction=vehicle_direction,
        return_individual_costs=False,
    )
    costs_sort_idx = np.argsort(costs)
    costs = costs[costs_sort_idx]
    all_end_configurations = all_end_configurations[costs_sort_idx]
    best_configuration = all_end_configurations[0]

    best_configuration = best_configuration[best_configuration != -1]
    return best_configuration, (costs, all_end_configurations)
