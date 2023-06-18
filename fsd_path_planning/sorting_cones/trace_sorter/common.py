#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This File provides several functions used in several other files in the
sorting algorithm
Project: fsd_path_planning
"""
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.math_utils import my_njit

if not TYPE_CHECKING:

    @my_njit
    def cast(  # pylint: disable=function-redefined
        type_: Any, value_: Any  # pylint: disable=unused-argument
    ) -> Any:
        "Dummy numba jit function"
        return value_


class NoPathError(RuntimeError):
    """
    A special exception thrown when no path can be found (i.e. no configuration)
    """


def get_configurations_diff(points: FloatArray, configurations: IntArray) -> FloatArray:
    """
    Gets the difference from each point to its next for each order defined by configurations
    Args:
        points: The points for which the differences should be calculated
        configurations: (n,m), all the configurations that define the orders
    Returns:
        np.array: The difference from one point to the next for each configuration
    """
    result: FloatArray
    result = points[configurations[..., :-1]]
    result -= points[configurations[..., 1:]]
    return result


@my_njit
def breadth_first_order(adjacency_matrix: IntArray, start_idx: int) -> IntArray:
    """
    Returns the nodes reachable from `start_idx` in BFS order
    Args:
        adjacency_matrix: The adjacency matrix describing the graph
        start_idx: The index of the starting node
    Returns:
        np.array: An array containing the nodes reachable from the starting node in BFS order
    """
    visited = np.zeros(adjacency_matrix.shape[0], dtype=np.uint8)
    queue = np.full(adjacency_matrix.shape[0], fill_value=-1)

    queue[0] = start_idx
    visited[start_idx] = 1

    queue_pointer = 0
    queue_end_pointer = 0

    while queue_pointer <= queue_end_pointer:
        node = queue[queue_pointer]

        next_nodes = np.argwhere(adjacency_matrix[node])[:, 0]
        for i in next_nodes:
            if not visited[i]:
                queue_end_pointer += 1
                queue[queue_end_pointer] = i
                visited[i] = 1
        queue_pointer += 1
    return cast(IntArray, queue[:queue_pointer])
