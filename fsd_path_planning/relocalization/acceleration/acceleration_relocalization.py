#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Place the car in the known accelearation map and relocalize it.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from fsd_path_planning.relocalization.relocalization_base_class import (
    RelocalizationCallable,
    Relocalizer,
)
from fsd_path_planning.types import FloatArray
from fsd_path_planning.utils.math_utils import rotate


def select_random_subset(points, subset_size):
    """
    Randomly selects a subset of points from the given set of points.

    Parameters:
    - points: A numpy array of shape (n, 2) where n is the number of points.
    - subset_size: The number of points to select.

    Returns:
    - A subset of the given points.
    """
    indices = np.random.choice(points.shape[0], subset_size, replace=False)
    return points[indices]


def linear_fit(points):
    """
    Performs a linear fit on the given points.

    Parameters:
    - points: A numpy array of shape (n, 2) where n is the number of points.

    Returns:
    - Coefficients of the linear fit (slope and intercept).
    """
    x = points[:, 0]
    y = points[:, 1]
    coefficients = np.polyfit(x, y, 1)
    return coefficients


def calculate_error(points, coefficients):
    """
    Calculates the sum of squared errors for the given points and linear fit.

    Parameters:
    - points: A numpy array of shape (n, 2) where n is the number of points.
    - coefficients: Coefficients of the linear fit (slope and intercept).

    Returns:
    - The sum of squared errors.
    """
    x = points[:, 0]
    y = points[:, 1]
    y_fit = np.polyval(coefficients, x)
    error = np.sum((y - y_fit) ** 2)
    return error


def random_subset_fit_error(points, subset_size):
    """
    Selects a random subset of points, performs a linear fit, and calculates the error.

    Parameters:
    - points: A numpy array of shape (n, 2) where n is the number of points.
    - subset_size: The number of points to select for the subset.

    Returns:
    - The sum of squared errors for the linear fit on the selected subset of points.
    """
    subset = select_random_subset(points, subset_size)
    coefficients = linear_fit(subset)
    error = calculate_error(subset, coefficients)
    return coefficients, error


def best_fit(points, subset_size, iterations):
    """
    Calls the random_subset_fit_error function many times and returns the coefficients with the smallest error.

    Parameters:
    - points: A numpy array of shape (n, 2) where n is the number of points.
    - subset_size: The number of points to select for each subset.
    - iterations: The number of iterations to perform.

    Returns:
    - Coefficients of the linear fit with the smallest error.
    """
    best_coefficients = None
    smallest_error = np.inf

    for _ in range(iterations):
        coefficients, error = random_subset_fit_error(points, subset_size)
        if error < smallest_error:
            smallest_error = error
            best_coefficients = coefficients

    return best_coefficients


# # Example usage:
# points = np.array([[1, 2], [2, 3], [3, 5], [4, 4], [5, 5], [6, 7], [7, 8], [8, 9]])

# subset_size = 4
# iterations = 1000
# best_coefficients = best_fit(points, subset_size, iterations)
# print(f"Best coefficients: {best_coefficients}")


class AccelerationRelocalizer(Relocalizer):
    def do_relocalization_once(
        self,
        cones: List[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> Tuple[RelocalizationCallable, RelocalizationCallable] | None:
        if self._original_vehicle_position is None:
            return

        all_cones = np.row_stack(cones)

        if len(all_cones) < 3:
            return

        vehicle_yaw = np.arctan2(*vehicle_direction[::-1])

        # convert to local frame
        all_cones_local = rotate(all_cones - vehicle_position, -vehicle_yaw)

        # get cones expected to be on the left
        mask_over_0 = all_cones_local[:, 1] > 0
        mask_under_2_left = all_cones_local[:, 1] < 2
        mask = mask_over_0 & mask_under_2_left

        cones_expected_left = all_cones_local[mask]

        cones_expected_left = cones_expected_left[cones_expected_left[:, 0].argsort()]

        if len(cones_expected_left) < 4:
            return

        coeffs = best_fit(cones_expected_left, subset_size=3, iterations=100)

        slope = coeffs[0]

        angle_to_fix = np.arctan(slope) + vehicle_yaw

        def transform_to_known_frame(position_2d, yaw):
            return (
                rotate(position_2d - self._original_vehicle_position, -angle_to_fix),
                yaw - angle_to_fix,
            )

        def transform_to_base_frame(position_2d, yaw):
            base_position = rotate(position_2d, angle_to_fix) + self._original_vehicle_position
            base_yaw = yaw + angle_to_fix
            return base_position, base_yaw

        return transform_to_known_frame, transform_to_base_frame

    def get_known_global_path(self) -> FloatArray:
        return BASE_ACCELERATION_PATH


def create_acceleartion_path() -> FloatArray:
    path_x = np.arange(-10, 150, 0.2)
    rng = np.random.default_rng(42)
    # add a bit of noise so that path calculations do not crash
    path_y = rng.normal(0, 0.01, len(path_x))

    path_2_y = np.arange(0, 5, 0.2)
    path_2_x = rng.normal(0, 0.01, len(path_2_y)) + path_x[-1]

    path_3_x = path_x[::-1]
    path_3_y = path_y[::-1] + path_2_y[-1]

    path_4_y = path_2_y[::-1]
    path_4_x = rng.normal(0, 0.01, len(path_4_y)) + path_x[0]

    path_x_final = np.concatenate(
        [
            path_x,
            path_2_x,
            path_3_x,
            path_4_x,
        ]
    )
    path_y_final = np.concatenate(
        [
            path_y,
            path_2_y,
            path_3_y,
            path_4_y,
        ]
    )

    return np.array([path_x_final, path_y_final]).T


BASE_ACCELERATION_PATH = create_acceleartion_path()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.scatter(*create_acceleartion_path().T, c=np.arange(len(create_acceleartion_path())))

    plt.axis("equal")
    plt.show()
