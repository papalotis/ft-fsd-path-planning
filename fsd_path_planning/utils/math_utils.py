#!/usin_roll/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: A module with common mathematical functions

Taken from ft-as-utils

Project: fsd_path_planning
"""
from typing import Tuple, TypeVar, cast

import numpy as np
from numba import jit

T = TypeVar("T")


def my_njit(func: T) -> T:
    """
    numba.njit is an untyped decorator. This wrapper helps type checkers keep the
    type information after applying the decorator. Furthermore, it sets some performance
    flags

    Args:
        func (T): The function to jit

    Returns:
        T: The jitted function
    """
    jit_func: T = jit(nopython=True, cache=True, nogil=True, fastmath=True)(func)

    return jit_func


@my_njit
def vec_dot(vecs1: np.ndarray, vecs2: np.ndarray) -> np.ndarray:
    """
    Mutliplies vectors in an array elementwise

    Args:
        vecs1 (np.array): The first "list" of vectors
        vecs2 (np.array): The second "list" of vectors

    Returns:
        np.array: The results
    """
    return np.sum(vecs1 * vecs2, axis=-1)


@my_njit
def norm_of_last_axis(arr: np.ndarray) -> np.ndarray:
    original_shape = arr.shape
    arr_row_col = np.ascontiguousarray(arr).reshape(-1, arr.shape[-1])
    result = np.empty(arr_row_col.shape[0])
    for i in range(arr_row_col.shape[0]):
        vec = arr_row_col[i]
        result[i] = np.sqrt(vec_dot(vec, vec))

    result = result.reshape(original_shape[:-1])

    return result


@my_njit
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
    assert vecs1.shape[-1] == 2
    assert vecs2.shape[-1] == 2

    cos_theta = vec_dot(vecs1, vecs2)

    cos_theta /= norm_of_last_axis(vecs1) * norm_of_last_axis(vecs2)

    cos_theta = np.asarray(cos_theta)

    cos_theta_flat = cos_theta.ravel()

    if clip_cos_theta:
        cos_theta_flat[cos_theta_flat < -1] = -1
        cos_theta_flat[cos_theta_flat > 1] = 1

    return np.arccos(cos_theta)


@my_njit
def rotate(points: np.ndarray, theta: float) -> np.ndarray:
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


@my_njit
def my_in1d(test_values: np.ndarray, source_container: np.ndarray) -> np.ndarray:
    """
    Calculate a boolean mask for a 1d array indicating if an element in `test_values` is
    present in `source container` which is also 1d

    Args:
        test_values (np.ndarray): The values to test if they are inside the container
        source_container (np.ndarray): The container

    Returns:
        np.ndarray: A boolean array with the same length as `test_values`. If
        `return_value[i]` is `True` then `test_value[i]` is in `source_container`
    """
    source_sorted = np.sort(source_container)
    is_in = np.zeros(test_values.shape[0], dtype=np.bool_)
    for i, test_val in enumerate(test_values):
        for source_val in source_sorted:
            if test_val == source_val:
                is_in[i] = True
                break

            if source_val > test_val:
                break

    return is_in


def trace_calculate_consecutive_radii(trace: np.ndarray) -> np.ndarray:
    """
    Expects a (n,2) array and returns the radius of the circle that passes
    between all consecutive point triples. The radius between index 0,1,2, then 1,2,3
    and so on

    Args:
        trace (np.ndarray): The points for which the radii will be calculated

    Returns:
        np.ndarray: The radii for each consecutive point triple
    """

    # TODO: Vectorize this function. Limit is the indexer
    indexer = np.arange(3)[None, :] + 1 * np.arange(trace.shape[-2] - 2)[:, None]

    points = trace[indexer]
    radii = calculate_radius_from_points(points)
    return radii


def trace_distance_to_next(trace: np.ndarray) -> np.ndarray:
    """
    Calculates the distance of one point in the trace to the next. Obviously the last
    point doesn't have any distance associated

    Args:
        trace (np.array): The points of the trace

    Returns:
        np.array: A vector containing the distances from one point to the next
    """
    return np.linalg.norm(np.diff(trace, axis=-2), axis=-1)


def trace_angles_between(trace: np.ndarray) -> np.ndarray:
    """
    Calculates the angles in a trace from each point to its next

    Args:
        trace (np.array): The trace containing a series of 2d vectors

    Returns:
        np.array: The angle from each vector to its next, with `len(return_value) ==
        len(trace) - 1`
    """
    all_to_next = np.diff(trace, axis=-2)
    from_middle_to_next = all_to_next[..., 1:, :]
    from_middle_to_prev = -all_to_next[..., :-1, :]
    angles = vec_angle_between(from_middle_to_next, from_middle_to_prev)
    return angles


@my_njit
def unit_2d_vector_from_angle(rad: np.ndarray) -> np.ndarray:
    """
    Creates unit vectors for each value in the rad array

    Args:
        rad (np.array): The angles (in radians) for which the vectors should be created

    Returns:
        np.array: The created unit vectors
    """
    rad = np.asarray(rad)
    new_shape = rad.shape + (2,)
    res = np.empty(new_shape, dtype=rad.dtype)
    res[..., 0] = np.cos(rad)
    res[..., 1] = np.sin(rad)
    return res


# Calculates the angle of each vector in `vecs`
# TODO: Look into fixing return type when a single vector is provided (return float)
@my_njit
def angle_from_2d_vector(vecs: np.ndarray) -> np.ndarray:
    """
    Calculates the angle of each vector in `vecs`. If `vecs` is just a single 2d vector
    then one angle is calculated and a scalar is returned

    >>> import numpy as np
    >>> x = np.array([[1, 0], [1, 1], [0, 1]])
    >>> angle_from_2d_vector(x)
    >>> array([0.        , 0.78539816, 1.57079633])

    Args:
        vecs (np.array): The vectors for which the angle is calculated

    Raises:
        ValueError: If `vecs` has the wrong shape a ValueError is raised

    Returns:
        np.array: The angle of each vector in `vecs`
    """
    assert vecs.shape[-1] == 2, "vecs must be a 2d vector"

    vecs_flat = vecs.reshape(-1, 2)

    angles = np.arctan2(vecs_flat[:, 1], vecs_flat[:, 0])
    return_value = angles.reshape(vecs.shape[:-1])

    # if vecs.ndim == 1:
    #     return return_value[0]

    return return_value


@my_njit
def normalize_last_axis(vecs: np.ndarray) -> np.ndarray:
    """
    Returns a normalized version of vecs

    Args:
        vecs (np.ndarray): The vectors to normalize
    Returns:
        np.ndarray: The normalized vectors
    """
    vecs_flat = vecs.reshape(-1, vecs.shape[-1])
    out = np.zeros(vecs.shape, dtype=vecs.dtype)
    for i, vec in enumerate(vecs_flat):
        out[i] = vec / np.linalg.norm(vec)

    return out.reshape(vecs.shape)


@my_njit
def lerp(
    values_to_lerp: np.ndarray,
    start1: np.ndarray,
    stop1: np.ndarray,
    start2: np.ndarray,
    stop2: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolates (lerps) from one sin_pitchace `[start1, stop1]` to another
    `[start2, stop2]`. `start1 >= stop1` and `start2 >= stop2` are allowed. If ns is a
    2d array, then start1, stop1, start2, stop2 must be 1d vectors. This allows for
    lerping in any n-dim sin_pitchace

    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> lerp(x, 0, 10, 30, 100)
    >>> array([37., 44., 51.])

    Args:
        values_to_lerp (np.array): The points to interpolate
        start1 (np.array): The beginning of the original sin_pitchace
        stop1 (np.array): The end of the original sin_pitchace
        start2 (np.array): The beginning of the target sin_pitchace
        stop2 (np.array): The end of the target sin_pitchace

    Returns:
        np.array: The interpolated points
    """
    return (values_to_lerp - start1) / (stop1 - start1) * (stop2 - start2) + start2


def calculate_radius_from_points(points: np.ndarray) -> np.ndarray:
    """
    Given a three points this function calculates the radius of the circle that passes
    through these points

    Based on: https://math.stackexchange.com/questions/133638/
    how-does-this-equation-to-find-the-radius-from-3-points-actually-work

    Args:
        points (np.ndarray): The points for which should be used to calculate the radius

    Returns:
        np.ndarray: The calculated radius
    """
    # implements the equation discussed here:
    #
    # assert points.shape[-2:] == (3, 2)
    # get side lengths
    points_circular = points[..., [0, 1, 2, 0], :]
    len_sides = trace_distance_to_next(points_circular)

    # calc prod of sides
    prod_of_sides = np.prod(len_sides, axis=-1, keepdims=True)

    # calc area of triangle
    # https://www.mathopenref.com/heronsformula.html

    # calc half of perimeter
    perimeter = np.sum(len_sides, axis=-1, keepdims=True)
    half_perimeter = perimeter / 2
    half_perimeter_minus_sides = half_perimeter - len_sides
    area_sqr = (
        np.prod(half_perimeter_minus_sides, axis=-1, keepdims=True) * half_perimeter
    )
    area = np.sqrt(area_sqr)

    radius = prod_of_sides / (area * 4)

    radius = radius[..., 0]
    return radius


Numeric = TypeVar("Numeric", float, np.ndarray)


def linearly_combine_values_over_time(
    tee: float, delta_time: float, previous_value: Numeric, new_value: Numeric
) -> Numeric:
    """
    Linear combination of two values over time
    (see https://de.wikipedia.org/wiki/PT1-Glied)
    Args:
        tee (float): The parameter selecting how much we keep from the previous value
        and how much we update from the new
        delta_time (float): The time difference between the previous and new value
        previous_value (Numeric): The previous value
        new_value (Numeric): The next value

    Returns:
        Numeric: The combined value
    """
    tee_star = 1 / (tee / delta_time + 1)
    combined_value: Numeric = tee_star * (new_value - previous_value) + previous_value
    return combined_value


def odd_square(values: Numeric) -> Numeric:
    return cast(Numeric, np.sign(values) * np.square(values))


def euler_angles_to_quaternion(euler_angles: np.ndarray) -> np.ndarray:
    """
    Converts Euler angles to a quaternion representation.

    Args:
        euler_angles (np.ndarray): Euler angles as an [...,3] array. Order is
        [roll, pitch, yaw]

    Returns:
        np.ndarray: The quaternion representation in [..., 4] [x, y, z, w] order
    """
    roll_index, pitch_index, yaw_index = 0, 1, 2
    sin_values = np.sin(euler_angles * 0.5)
    cos_values = np.cos(euler_angles * 0.5)

    cos_yaw = cos_values[..., yaw_index]
    sin_yaw = sin_values[..., yaw_index]
    cos_pitch = cos_values[..., pitch_index]
    sin_pitch = sin_values[..., pitch_index]
    cos_roll = cos_values[..., roll_index]
    sin_roll = sin_values[..., roll_index]

    quaternion_x = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
    quaternion_y = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
    quaternion_z = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw
    quaternion_w = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw

    return_value = np.stack(
        [quaternion_x, quaternion_y, quaternion_z, quaternion_w], axis=-1
    )
    return return_value


def quaternion_to_euler_angles(quaternion: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion to Euler angles. Based on
    https://stackoverflow.com/a/37560411.

    Args:
        quaternion (np.ndarray): The quaternion as an [..., 4] array. Order is
        [x, y, z, w]

    Returns:
        np.ndarray: The Euler angles as an [..., 3] array. Order is [roll, pitch, yaw]
    """
    x_index, y_index, z_index, w_index = 0, 1, 2, 3
    x_value = quaternion[..., x_index]
    y_value = quaternion[..., y_index]
    z_value = quaternion[..., z_index]
    w_value = quaternion[..., w_index]

    y_square = y_value * y_value
    temporary_0 = -2.0 * (y_square + z_value * z_value) + 1.0
    temporary_1 = +2.0 * (x_value * y_value + w_value * z_value)
    temporary_2 = -2.0 * (x_value * z_value - w_value * y_value)
    temporary_3 = +2.0 * (y_value * z_value + w_value * x_value)
    temporary_4 = -2.0 * (x_value * x_value + y_square) + 1.0

    temporary_2 = np.clip(temporary_2, -1.0, 1.0)

    roll = np.arctan2(temporary_3, temporary_4)
    pitch = np.arcsin(temporary_2)
    yaw = np.arctan2(temporary_1, temporary_0)

    return_value = np.stack([roll, pitch, yaw], axis=-1)
    return return_value


@my_njit
def points_inside_ellipse(
    points: np.ndarray,
    center: np.ndarray,
    major_direction: np.ndarray,
    major_radius: float,
    minor_radius: float,
) -> np.ndarray:
    """
    Checks if a set of points are inside an ellipse.

    Args:
        points: The points as an [..., 2] array.
        center: The center of the ellipse as an [2] array.
        major_direction: The major direction of the ellipse as an [2] array.
        major_radius: The major radius of the ellipse.
        minor_radius: The minor radius of the ellipse.

    Returns:
        An [...] array of booleans.
    """

    # Center the points around the center
    # [..., 2]
    centered_points = points - center
    # Calculate angle of the major direction with the x-axis
    # [1]
    major_direction_angle = np.arctan2(major_direction[1], major_direction[0])
    # Rotate the points around the center of the ellipse
    # [..., 2]
    rotated_points = rotate(centered_points, -major_direction_angle)
    # [2]
    radii_square = np.array([major_radius, minor_radius]) ** 2
    # [...]    [..., 2]              [2]
    criterion_value = (rotated_points**2 / radii_square).sum(axis=-1)

    mask_is_inside = criterion_value < 1
    return mask_is_inside


def center_of_circle_from_3_points(
    point_1: np.ndarray,
    point_2: np.ndarray,
    point_3: np.ndarray,
    atol: float = 1e-6,
) -> np.ndarray:
    """
    Calculates the center of a circle from three points.

    Adapted from http://paulbourke.net/geometry/circlesphere/Circle.cpp (CalcCircle)

    Args:
        point_1: The first point as an [2] array.
        point_2: The second point as an [2] array.
        point_3: The third point as an [2] array.

    Returns:
        The center of the circle as an [2] array.
    """
    y_delta_1 = point_2[1] - point_1[1]
    x_delta_1 = point_2[0] - point_1[0]
    y_delta_2 = point_3[1] - point_2[1]
    x_delta_2 = point_3[0] - point_2[0]

    if np.isclose(x_delta_1, 0.0, atol=atol) and np.isclose(x_delta_2, 0.0, atol=atol):
        center_x = (point_2[0] + point_3[0]) / 2
        center_y = (point_1[1] + point_2[1]) / 2
        return np.array([center_x, center_y])  # early return

    slope_1 = y_delta_1 / x_delta_1
    slope_2 = y_delta_2 / x_delta_2
    if np.isclose(slope_1, slope_2, atol=atol):
        raise ValueError("Points are colinear")

    center_x = (
        slope_1 * slope_2 * (point_1[1] - point_3[1])
        + slope_2 * (point_1[0] + point_2[0])
        - slope_1 * (point_2[0] + point_3[0])
    ) / (2 * (slope_2 - slope_1))

    center_y = (
        -(center_x - (point_1[0] + point_2[0]) / 2) / slope_1
        + (point_1[1] + point_2[1]) / 2
    )

    center = np.array([center_x, center_y])
    return center


@my_njit
def circle_fit(coords: np.ndarray, max_iter: int = 99) -> np.ndarray:
    """
    Fit a circle to a set of points. This function is adapted from the hyper_fit function
    in the circle-fit package (https://pypi.org/project/circle-fit/). The function is
    a njit version of the original function with some input validation removed. Furthermore,
    the residuals are not calculated or returned.

    Args:
        coords: The coordinates of the points as an [N, 2] array.
        max_iter: The maximum number of iterations.

    Returns:
        An array with 3 elements:
        - center x
        - center y
        - radius
    """

    X = coords[:, 0]
    Y = coords[:, 1]

    n = X.shape[0]

    Xi = X - X.mean()
    Yi = Y - Y.mean()
    Zi = Xi * Xi + Yi * Yi

    # compute moments
    Mxy = (Xi * Yi).sum() / n
    Mxx = (Xi * Xi).sum() / n
    Myy = (Yi * Yi).sum() / n
    Mxz = (Xi * Zi).sum() / n
    Myz = (Yi * Zi).sum() / n
    Mzz = (Zi * Zi).sum() / n

    # computing the coefficients of characteristic polynomial
    Mz = Mxx + Myy
    Cov_xy = Mxx * Myy - Mxy * Mxy
    Var_z = Mzz - Mz * Mz

    A2 = 4 * Cov_xy - 3 * Mz * Mz - Mzz
    A1 = Var_z * Mz + 4.0 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz
    A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy
    A22 = A2 + A2

    # finding the root of the characteristic polynomial
    y = A0
    x = 0.0
    for _ in range(max_iter):
        Dy = A1 + x * (A22 + 16.0 * x * x)
        x_new = x - y / Dy
        if x_new == x or not np.isfinite(x_new):
            break
        y_new = A0 + x_new * (A1 + x_new * (A2 + 4.0 * x_new * x_new))
        if abs(y_new) >= abs(y):
            break
        x, y = x_new, y_new

    det = x * x - x * Mz + Cov_xy
    X_center = (Mxz * (Myy - x) - Myz * Mxy) / det / 2.0
    Y_center = (Myz * (Mxx - x) - Mxz * Mxy) / det / 2.0

    x = X_center + X.mean()
    y = Y_center + Y.mean()
    r = np.sqrt(abs(X_center**2 + Y_center**2 + Mz))

    return np.array([x, y, r])


if __name__ == "__main__":
    p1, p2, p3 = unit_2d_vector_from_angle(np.array([0, 0.3, 0.31]))
    print(center_of_circle_from_3_points(p1, p2, p3))

    p1, p2, p3 = unit_2d_vector_from_angle(np.array([0, 0.8, 4])) + 10
    print(center_of_circle_from_3_points(p1, p2, p3))

    p1, p2, p3 = np.array([-1, 0.0]), np.array([0, 1.0]), np.array([1.0, 0.0])
    print(center_of_circle_from_3_points(p1, p2, p3))

    p1, p2, p3 = np.array([0, 0.0]), np.array([0, 1.0]), np.array([0.0, 2.0])
    print(center_of_circle_from_3_points(p1, p2, p3))


# @my_njit
def angle_difference(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """
    Calculate the difference between two angles. The range of the difference is [-pi, pi].
    The order of the angles *is* important.

    Args:
        angle1: First angle.
        angle2: Second angle.

    Returns:
        The difference between the two angles.
    """
    return (angle1 - angle2 + 3 * np.pi) % (2 * np.pi) - np.pi  # type: ignore
