"""
Rigid transforms of 2D points and poses.

Project: fsd_path_planning
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fsd_path_planning._numba import njit
from fsd_path_planning.common_types import FloatArray


@njit
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


@dataclass(frozen=True)
class SE2Transform:
    """A rigid 2D transform of a pose.

        position -> R(angle) @ (position + shift - pivot) + pivot + offset
        yaw      -> yaw + angle

    That is: shift, rotate about the pivot, then offset. Each part defaults to
    nothing, so "rotate about the origin, then translate" is
    ``SE2Transform(angle, offset=t)``. Calling it transforms a pose, so it can
    stand wherever a ``(position, yaw) -> (position, yaw)`` function was
    expected.
    """

    angle: float
    shift: FloatArray = field(default_factory=lambda: np.zeros(2))
    pivot: FloatArray = field(default_factory=lambda: np.zeros(2))
    offset: FloatArray = field(default_factory=lambda: np.zeros(2))

    def __call__(self, position_2d: FloatArray, yaw: float) -> tuple[FloatArray, float]:
        return self.apply_to_points(position_2d), yaw + self.angle

    def apply_to_points(self, points: FloatArray) -> FloatArray:
        """Transform a position, or an (N, 2) array of them."""
        rotated = rotate(points + self.shift - self.pivot, self.angle)
        return rotated + self.pivot + self.offset

    def inverse(self) -> SE2Transform:
        """The transform that undoes this one."""
        return SE2Transform(
            angle=-self.angle, shift=-self.offset, pivot=self.pivot, offset=-self.shift
        )
