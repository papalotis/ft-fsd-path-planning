"""
Rigid transforms of 2D points.

Project: fsd_path_planning
"""

from __future__ import annotations

import numpy as np

from fsd_path_planning._numba import my_njit


@my_njit
def rotate(points: np.ndarray, theta: float | np.ndarray) -> np.ndarray:
    """
    Rotates the points in `points` by angle `theta` around the origin

    Args:
        points (np.array): The points to rotate. Shape (n,2)
        theta (float): The angle by which to rotate in radians

    Returns:
        np.array: The points rotated
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((cos_theta, -sin_theta), (sin_theta, cos_theta))).T
    return np.dot(points, rotation_matrix)
