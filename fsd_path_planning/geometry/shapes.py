"""
Circles through points, circle fitting and ellipse membership.

Project: fsd_path_planning
"""

from __future__ import annotations

import numpy as np

from fsd_path_planning._numba import my_njit
from fsd_path_planning.geometry.traces import trace_distance_to_next
from fsd_path_planning.geometry.transforms import rotate


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
    Fit a circle to a set of points. This function is adapted from the
    hyper_fit function in the circle-fit package
    (https://pypi.org/project/circle-fit/). The function is
    a njit version of the original function with some input validation
    removed. Furthermore, the residuals are not calculated or returned.

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
    if abs(det) < 1e-15:
        det = 1e-15  # prevent division by zero for collinear points
    X_center = (Mxz * (Myy - x) - Myz * Mxy) / det / 2.0
    Y_center = (Myz * (Mxx - x) - Mxz * Mxy) / det / 2.0

    x = X_center + X.mean()
    y = Y_center + Y.mean()
    r = np.sqrt(abs(X_center**2 + Y_center**2 + Mz))

    return np.array([x, y, r])
