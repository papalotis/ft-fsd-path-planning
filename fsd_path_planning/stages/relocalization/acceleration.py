#!/usr/bin/env python3
"""
Description: Place the car in the known accelearation map and relocalize it.
"""

from __future__ import annotations

import numpy as np

from fsd_path_planning.common_types import FloatArray
from fsd_path_planning.geometry.transforms import SE2Transform, rotate
from fsd_path_planning.stages.relocalization.base import (
    RelocalizationCallable,
    Relocalizer,
)

DEFAULT_RANDOM_SEED = 42


def select_random_subset(points, subset_size, rng: np.random.Generator | None = None):
    """
    Randomly selects a subset of points from the given set of points.

    Parameters:
    - points: A numpy array of shape (n, 2) where n is the number of points.
    - subset_size: The number of points to select.
    - rng: The generator to draw from. A fresh seeded one is used when omitted,
      so that a call without a generator is still reproducible.

    Returns:
    - A subset of the given points.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    indices = rng.choice(points.shape[0], subset_size, replace=False)
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


def random_subset_fit_error(
    points, subset_size, rng: np.random.Generator | None = None
):
    """
    Selects a random subset of points, performs a linear fit, and calculates the error.

    Parameters:
    - points: A numpy array of shape (n, 2) where n is the number of points.
    - subset_size: The number of points to select for the subset.
    - rng: The generator to draw the subset from.

    Returns:
    - The sum of squared errors for the linear fit on the selected subset of points.
    """
    subset = select_random_subset(points, subset_size, rng)
    coefficients = linear_fit(subset)
    error = calculate_error(subset, coefficients)
    return coefficients, error


def best_fit(
    points: FloatArray,
    subset_size: int,
    iterations: int,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """
    Calls the random_subset_fit_error function many times and returns the coefficients
    with the smallest error.

    Parameters:
    - points: A numpy array of shape (n, 2) where n is the number of points.
    - subset_size: The number of points to select for each subset.
    - iterations: The number of iterations to perform.
    - rng: The generator to draw the subsets from. A fresh seeded one is used
      when omitted.

    Returns:
    - Coefficients of the linear fit with the smallest error.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RANDOM_SEED)

    best_coefficients = np.zeros(2)
    smallest_error = np.inf

    for _ in range(iterations):
        coefficients, error = random_subset_fit_error(points, subset_size, rng)
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
    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED) -> None:
        """
        Args:
            random_seed: Seed for the generator that draws the subsets of the
                line fit. The relocalizer owns its generator so that a run can
                be replayed; drawing from the global NumPy generator made the
                relocalization result depend on whatever else had used it.
        """
        super().__init__()
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    def do_relocalization_once(
        self,
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> tuple[RelocalizationCallable, RelocalizationCallable] | None:
        if self._original_vehicle_position is None:
            return

        all_cones = np.vstack([c.reshape(-1, 2) for c in cones])

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

        coeffs = best_fit(
            cones_expected_left, subset_size=3, iterations=100, rng=self._rng
        )

        slope = coeffs[0]

        angle_to_fix = np.arctan(slope) + vehicle_yaw

        to_known_frame = SE2Transform(
            angle=-angle_to_fix, shift=-self._original_vehicle_position
        )
        return to_known_frame, to_known_frame.inverse()

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

    return np.array([path_x_final, path_y_final], dtype=np.float64).T


BASE_ACCELERATION_PATH = create_acceleartion_path()
