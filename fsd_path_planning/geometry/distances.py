"""
Euclidean distances between point sets.

Project: fsd_path_planning
"""

from __future__ import annotations

import numpy as np

from fsd_path_planning._numba import my_njit


@my_njit
def my_cdist_sq_euclidean(arr_a: np.ndarray, arr_b: np.ndarray) -> np.ndarray:
    """
    Calculates the pairwise square euclidean distances from each point in `X` to each
    point in `Y`

    Credit:
        Uses https://stackoverflow.com/a/56084419 which in turn uses
        https://github.com/droyed/eucl_dist

    Args:
        arr_a (np.array): A 2d array of shape (m,k)
        arr_b (np.array): A 2d array of shape (n,k)

    Returns:
        np.array: A matrix of shape (m,n) containing the square euclidean distance
        between all the points in `X` and `Y`
    """
    n_x, dim = arr_a.shape
    x_ext = np.empty((n_x, 3 * dim))
    x_ext[:, :dim] = 1
    x_ext[:, dim : 2 * dim] = arr_a
    x_ext[:, 2 * dim :] = np.square(arr_a)

    n_y = arr_b.shape[0]
    y_ext = np.empty((3 * dim, n_y))
    y_ext[:dim] = np.square(arr_b).T
    y_ext[dim : 2 * dim] = -2 * arr_b.T
    y_ext[2 * dim :] = 1

    return np.dot(x_ext, y_ext)


@my_njit
def calc_pairwise_distances(
    points: np.ndarray, dist_to_self: float = 0.0
) -> np.ndarray:
    """
    Given a set of points, creates a distance matrix from each point to every point

    Args:
        points (np.ndarray): The points for which the distance matrix should be
        calculated dist_to_self (np.ndarray, optional): The distance to set the
        diagonal. Defaults to 0.0.

    Returns:
        np.ndarray: The 2d distance matrix
    """
    pairwise_distances = my_cdist_sq_euclidean(points, points)

    if dist_to_self != 0:
        for i in range(len(points)):
            pairwise_distances[i, i] = dist_to_self
    return pairwise_distances
