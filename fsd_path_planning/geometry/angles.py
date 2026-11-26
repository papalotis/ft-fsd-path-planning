"""
Angles of and between 2D vectors.

Project: fsd_path_planning
"""

from __future__ import annotations

import numpy as np

from fsd_path_planning._numba import njit
from fsd_path_planning.common_types import FloatArray
from fsd_path_planning.geometry.vectors import norm_of_last_axis, vec_dot


@njit
def vec_angle_between(
    vecs1: np.ndarray, vecs2: np.ndarray, clip_cos_theta: bool = True
) -> np.ndarray:
    """
    Calculates the angle between the vectors of the last dimension

    Args:
        vecs1 (np.ndarray): An array of shape (...,2)
        vecs2 (np.ndarray): An array of shape (...,2)
        clip_cos_theta (bool): Clip the values of the dot products so that they are
        between -1 and 1. Defaults to True.

    Returns:
        np.ndarray: A vector, such that each element i contains the angle between
        vectors vecs1[i] and vecs2[i]
    """
    if vecs1.shape[-1] != 2:
        raise ValueError("vecs1 must contain 2d vectors")
    if vecs2.shape[-1] != 2:
        raise ValueError("vecs2 must contain 2d vectors")

    cos_theta = vec_dot(vecs1, vecs2)

    cos_theta /= norm_of_last_axis(vecs1) * norm_of_last_axis(vecs2)

    cos_theta = np.asarray(cos_theta)

    cos_theta_flat = cos_theta.ravel()

    if clip_cos_theta:
        cos_theta_flat[cos_theta_flat < -1] = -1
        cos_theta_flat[cos_theta_flat > 1] = 1

    return np.arccos(cos_theta)


@njit
def unit_2d_vector_from_angle(rad: float | FloatArray) -> FloatArray:
    """
    Creates unit vectors for each value in the rad array

    Args:
        rad (float | FloatArray): The angles (in radians) for which the
            vectors should be created

    Returns:
        FloatArray: The created unit vectors
    """
    rad = np.asarray(rad)
    new_shape = rad.shape + (2,)
    res = np.empty(new_shape, dtype=rad.dtype)
    res[..., 0] = np.cos(rad)
    res[..., 1] = np.sin(rad)
    return res


@njit
def _angles_of_2d_vectors(vecs: FloatArray) -> FloatArray:
    """The angle of each vector in `vecs`, shaped like `vecs` without its last axis."""
    vecs_flat = vecs.reshape(-1, 2)
    angles = np.arctan2(vecs_flat[:, 1], vecs_flat[:, 0])
    return angles.reshape(vecs.shape[:-1])


def angle_from_2d_vector(vecs: FloatArray) -> FloatArray:
    """
    Calculates the angle of each vector in `vecs`. If `vecs` is just a single 2d vector
    then one angle is calculated and a scalar is returned

    >>> import numpy as np
    >>> x = np.array([[1, 0], [1, 1], [0, 1]])
    >>> angle_from_2d_vector(x)
    array([0.        , 0.78539816, 1.57079633])
    >>> angle_from_2d_vector(np.array([0.0, 1.0]))
    np.float64(1.5707963267948966)

    Args:
        vecs (np.array): The vectors for which the angle is calculated

    Raises:
        ValueError: If `vecs` has the wrong shape a ValueError is raised

    Returns:
        np.array: The angle of each vector in `vecs`, or a NumPy scalar for a
        single vector
    """
    vecs = np.asarray(vecs)
    if vecs.shape[-1] != 2:
        raise ValueError("vecs must be a 2d vector")

    angles = _angles_of_2d_vectors(np.ascontiguousarray(vecs, dtype=float))
    if vecs.ndim == 1:
        # A NumPy scalar rather than a 0-d array. The value is the same, but a
        # 0-d array passed on to a Numba function makes Numba compile another
        # version of it, which the warm-up cannot know about in advance.
        return angles[()]
    return angles


@njit
def angle_difference(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """
    Calculate the difference between two angles. The range of the difference
    is [-pi, pi].
    The order of the angles *is* important.

    Args:
        angle1: First angle.
        angle2: Second angle.

    Returns:
        The difference between the two angles.
    """
    return (angle1 - angle2 + 3 * np.pi) % (2 * np.pi) - np.pi  # type: ignore
