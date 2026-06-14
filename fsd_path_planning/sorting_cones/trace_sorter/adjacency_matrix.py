#!/usr/bin/env python3
"""
Description: This File calculates the Adjacency Matrix
Project: fsd_path_planning
"""

import numpy as np

from fsd_path_planning.sorting_cones.trace_sorter.common import breadth_first_order
from fsd_path_planning.types import BoolArray, FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes, invert_cone_type
from fsd_path_planning.utils.math_utils import calc_pairwise_distances


class AdjacencyMatrixCache:
    """Instance-scoped cache for distance matrix and k-nearest-neighbor calculations."""

    def __init__(self) -> None:
        self._matrix_hash: int | None = None
        self._distance_matrix: FloatArray | None = None
        self._idxs_hash: int | None = None
        self._idxs_calculated: IntArray | None = None

    def calculate_distance_matrix(self, cones_xy: FloatArray) -> FloatArray:
        input_hash = hash(cones_xy.tobytes())
        if input_hash != self._matrix_hash:
            self._matrix_hash = input_hash
            self._distance_matrix = calc_pairwise_distances(
                cones_xy, dist_to_self=np.inf
            )

        assert self._distance_matrix is not None
        return self._distance_matrix.copy()

    def find_k_closest_in_point_cloud(
        self, pairwise_distances: FloatArray, k: int
    ) -> IntArray:
        """
        Finds the indices of the k closest points for each point in a point
        cloud from its pairwise distances.

        Args:
            pairwise_distances: A square matrix containing the distance from each
            point to every other point
            k: The number closest points (indices) to return of each point
        Returns:
            np.array: An (n,k) array containing the indices of the `k` closest points.
        """
        input_hash = hash((pairwise_distances.tobytes(), k))
        if input_hash != self._idxs_hash:
            self._idxs_hash = input_hash
            self._idxs_calculated = np.argsort(pairwise_distances, axis=1)[:, :k]

        assert self._idxs_calculated is not None
        return self._idxs_calculated.copy()


# Module-level instance for backward compatibility
_DEFAULT_CACHE = AdjacencyMatrixCache()

LAST_MATRIX_CALC_HASH = None
LAST_MATRIX_CALC_DISTANCE_MATRIX = None


def calculate_distance_matrix(cones_xy: FloatArray) -> FloatArray:
    return _DEFAULT_CACHE.calculate_distance_matrix(cones_xy)


LAST_IDXS_CALCULATED = None
LAST_IDXS_HASH = None


def find_k_closest_in_point_cloud(pairwise_distances: FloatArray, k: int) -> IntArray:
    return _DEFAULT_CACHE.find_k_closest_in_point_cloud(pairwise_distances, k)


def create_adjacency_matrix(
    cones: FloatArray,
    n_neighbors: int,
    start_idx: int,
    max_dist: float,
    cone_type: ConeTypes,
    cache: AdjacencyMatrixCache | None = None,
) -> tuple[BoolArray, IntArray]:
    """
    Creates the adjacency matrix that defines the possible points each point
    can be connected with
    Args:
        cones: The trace containing all the points
        n_neighbors: The maximum number of neighbors each node can have
        start_idx: The index from which the trace starts
        max_dist: The maximum distance two points can have in order for them to
        be considered possible neighbors
    Returns:
        Tuple[np.array, np.array]: Three values are returned. First a square boolean
        matrix indicating at each position if two nodes are connected. The second 1d
        matrix contains the reachable nodes from `start_idx`.
    """

    n_points = cones.shape[0]

    cones_xy = cones[:, :2]
    cones_color = cones[:, 2]

    if cache is None:
        cache = _DEFAULT_CACHE

    pairwise_distances: FloatArray = cache.calculate_distance_matrix(cones_xy)

    mask_is_other_cone_type = cones_color == invert_cone_type(cone_type)
    pairwise_distances[mask_is_other_cone_type, :] = np.inf
    pairwise_distances[:, mask_is_other_cone_type] = np.inf

    # do not connect points that are very close to each other
    # pairwise_distances[pairwise_distances < 1.5] = np.inf

    k_closest_each = cache.find_k_closest_in_point_cloud(
        pairwise_distances, n_neighbors
    )

    sources = np.repeat(np.arange(n_points), n_neighbors)
    targets = k_closest_each.flatten()

    adjacency_matrix = np.zeros((n_points, n_points), dtype=int)

    adjacency_matrix[sources, targets] = (
        1  # for each node set its closest n_neighbor to 1
    )
    adjacency_matrix[pairwise_distances > (max_dist * max_dist)] = (
        0  # but if distance is too high set to 0 again
    )

    # remove all edges that don't have a revere i.e. convert to undirected graph
    adjacency_matrix = np.logical_and(adjacency_matrix, adjacency_matrix.T)

    reachable_nodes = breadth_first_order(adjacency_matrix, start_idx)

    # completely disconnect nodes that are not reachable from start node
    # assume that all nodes will be disconnected
    nodes_to_disconnect = np.ones(n_points, dtype=bool)
    # but for the reachable nodes don't do anything
    nodes_to_disconnect[reachable_nodes] = False

    # if we are sorting left (blue) cones, then we want to disconnect
    # all right (yellow) cones
    # nodes_to_disconnect[cones_color == invert_cone_type(cone_type)] = True

    # disconnect the remaining nodes in both directions
    # adjacency_matrix[:, nodes_to_disconnect] = 0
    # adjacency_matrix[nodes_to_disconnect, :] = 0

    return adjacency_matrix, reachable_nodes
