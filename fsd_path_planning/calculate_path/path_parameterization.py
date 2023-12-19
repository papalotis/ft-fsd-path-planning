#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Path parameterization for MPC

Description: Path parameterization for MPC. MPC requires specific features of a path
not just the 2D points of the path. This module provides the necessary features.
Project: fsd_path_planning
"""
from dataclasses import dataclass
from typing import Tuple, cast

import numpy as np
from icecream import ic  # pylint: disable=unused-import
from scipy.ndimage import uniform_filter1d

from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.math_utils import (
    Numeric,
    angle_from_2d_vector,
    circle_fit,
    my_njit,
    trace_distance_to_next,
)
from fsd_path_planning.utils.spline_fit import SplineEvaluator, SplineFitterFactory


@dataclass
class PathParameterizerState:
    "State of the PathParameterizer"

    prediction_horizon: int


def angle_difference(angle1: Numeric, angle2: Numeric) -> Numeric:
    """
    Calculate the difference between two angles. The range of the difference is [-pi, pi].
    The order of the angles *is* important.

    Args:
        angle1: First angle.
        angle2: Second angle.

    Returns:
        The difference between the two angles.
    """
    return cast(Numeric, (angle1 - angle2 + 3 * np.pi) % (2 * np.pi) - np.pi)  # type: ignore


@my_njit
def calculate_path_curvature(
    path: FloatArray, window_size: int, path_is_closed: bool
) -> FloatArray:
    """
    Calculate the curvature of the path.

    Args:
        path: The path as a 2D array of points.
        window_size: The size of the window to use for the curvature calculation.
        path_is_closed: Whether the path is closed or not.

    Returns:
        The curvature of the path.
    """
    windows = create_cyclic_sliding_window_indices(
        window_size=window_size, step_size=1, signal_length=len(path)
    )

    path_curvature = np.zeros(len(path))
    for i, window in enumerate(windows):
        if not path_is_closed:
            diff = window[1:] - window[:-1]
            if np.any(diff != 1):
                idx_cutoff = int(np.argmax(diff != 1) + 1)
                if i < window_size:
                    window = window[idx_cutoff:]
                else:
                    window = window[:idx_cutoff]

        points_in_window = path[window]

        _, _, radius = circle_fit(points_in_window)
        radius = min(
            max(radius, 1.0), 3000.0
        )  # np.clip didn't work for some reason (numba bug?)
        curvature = 1 / radius
        three_idxs = np.array([0, int(len(points_in_window) / 2), -1])
        three_points = points_in_window[three_idxs]
        hom_points = np.column_stack((np.ones(3), three_points))
        sign = np.linalg.det(hom_points)

        path_curvature[i] = curvature * np.sign(sign)

    return path_curvature


@my_njit
def create_cyclic_sliding_window_indices(
    window_size: int, step_size: int, signal_length: int
) -> IntArray:
    if window_size % 2 == 0:
        raise ValueError(f"Window size must be odd.")
    half_window_size = window_size // 2

    indexer = (
        np.arange(-half_window_size, half_window_size + 1)
        + np.arange(0, signal_length, step_size).reshape(-1, 1)
    ) % signal_length
    return indexer


class PathParameterizer:
    """
    Path parameterization / curvature calculation
    """

    def __init__(self, prediction_horizon: int) -> None:
        """
        Initialize the path parameterization

        Args:
            prediction_horizon: The prediction horizon of the MPC
        """
        self._state = PathParameterizerState(prediction_horizon=prediction_horizon)

    def _refit_spline(self, path: FloatArray) -> SplineEvaluator:
        """
        Refit the spline of the path. This way we can get accurate values for the
        derivatives of the spline. In order to increase computational efficiency,
        points will be skipped if they are considered to be too close to each other (
        i.e. the resolution of the path is too high).

        Args:
            path: The path.

        Returns:
            The spline of the path.
        """
        # Example: if prediction horizon is 10, and the desired path length is 30m, then
        # we need a point every 3m. In order to avoid losing information, we
        # need to predict with our spline every 1.5m (half the distance between expected
        # points). For better quality, we will to predict with our spline every 1m
        # (three times) so in the example every 1m we will have a point.
        distances_between_points = trace_distance_to_next(path)
        path_length = distances_between_points.sum()
        mean_point_distance = distances_between_points[:10].mean()
        predict_every = path_length / self._state.prediction_horizon / 3
        try:
            skip_factor = max(int(predict_every / mean_point_distance), 1)
        except ValueError:
            skip_factor = 1

        path_skipped = path[::skip_factor]

        # very little smoothing, because we assume that the incoming points are already
        # smooth, and max_deg is set to 3 because we cannot have a constant second derivative
        # for good curvature calculations
        factory = SplineFitterFactory(
            smoothing=0.01, predict_every=predict_every, max_deg=3
        )
        spline_fitter = factory.fit(path_skipped)
        return spline_fitter

    def _calculate_path_curvature(
        self, path_spline: SplineEvaluator, path_is_closed: bool
    ) -> FloatArray:
        """
        Calculate the curvature of the path.

        Args:
            path_spline: The spline of the path.

        Returns:
            The curvature of the path.
        """

        points = path_spline.predict(der=0)
        window_size = 501 if path_is_closed else min(len(points) // 5, 30)
        window_size += window_size % 2 == 0

        path_curvature = calculate_path_curvature(
            path=points,
            window_size=window_size,
            path_is_closed=path_is_closed,
        )
        mode = "wrap" if path_is_closed else "nearest"

        filter_size = max(2, window_size // 2)

        filtered_curvature: FloatArray = uniform_filter1d(
            path_curvature, size=filter_size, mode=mode
        )

        return filtered_curvature

    def _calculate_smallest_distance_to_path(
        self, path_spline: SplineEvaluator, point: FloatArray
    ) -> Tuple[float, int]:
        """
        Calculate the closest point on the path to a point.

        Args:
            path_spline: The spline of the path.
            point: The point.

        Returns:
            The closest point and its index on the path to the point.
        """

        path_resampled = path_spline.predict(der=0)
        distances_of_path_to_vehicle: FloatArray = np.linalg.norm(
            path_resampled - point, axis=-1
        )

        index_position_on_path_closest_to_pose = np.argmin(distances_of_path_to_vehicle)
        smallest_distance_to_point: float = distances_of_path_to_vehicle[
            index_position_on_path_closest_to_pose
        ]
        return smallest_distance_to_point, cast(
            int, index_position_on_path_closest_to_pose
        )

    def _calculate_relative_direction_difference_between_path_and_vehicle(
        self,
        path_spline: SplineEvaluator,
        vehicle_direction: FloatArray,
        index_of_closest_point: int,
    ) -> float:
        """
        Calculate the relative direction difference between the path and the vehicle.
        The relative direction difference is the difference between the direction of the vehicle
        and the direction of the path at the closest point.

        Args:
            path_spline: The spline of the path.
            vehicle_direction: The direction of the vehicle. The direction is a 2D vector.
            index_of_closest_point: The index of the closest point on the path.

        Returns:
            The relative direction difference between the path and the vehicle in rad.
        """
        path_direction = path_spline.predict(der=1)[index_of_closest_point]
        path_direction_as_angle = cast(float, angle_from_2d_vector(path_direction))
        vehicle_direction_as_angle = cast(
            float, angle_from_2d_vector(vehicle_direction)
        )

        relative_angle_difference = angle_difference(
            path_direction_as_angle, vehicle_direction_as_angle
        )
        return relative_angle_difference

    def _sample_path_parameters_for_prediction_horizon(
        self,
        path_spline: SplineEvaluator,
        path_curvature: FloatArray,
    ) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """
        Sample the path parameters for the prediction horizon. The splines are predicted
        in a high resolution and the parameters are sampled in a low resolution.

        Args:
            path_spline: The spline of the path.
            path_curvature: The curvature of the path.
            path_evaluation_parameters: The spline parameter values for which the path have
            been evaluated.

        Returns:
            The sampled path parameters for the prediction horizon. The parameters are:
                The spline parameter (theta) for the rest of the values
                The x position of the path.
                The y position of the path.
                The curvature of the path.
        """
        path_evaluation_parameters = path_spline.calculate_u_eval()
        path = path_spline.predict(der=0)

        indices: IntArray = np.linspace(
            0,
            len(path) - 1,
            self._state.prediction_horizon,
            dtype=np.int_,
        )
        # check that no index appears twice
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Indices of resampled path appear twice")

        path_x_sampled, path_y_resampled = path[indices].T
        path_curvature_resampled = path_curvature[indices]
        path_evaluation_parameters_resampled = path_evaluation_parameters[indices]
        return (
            path_evaluation_parameters_resampled,
            path_x_sampled,
            path_y_resampled,
            path_curvature_resampled,
        )

    def parameterize_path(
        self,
        path: FloatArray,
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
        path_is_closed: bool,
    ) -> FloatArray:
        """
        Parameterize the path. Calculate the curvature of the path and sample the path

        Args:
            path: The path as a 2D array of shape (N, 2).
            vehicle_position: The position of the vehicle. The position is a 2D vector.
            vehicle_direction: The direction of the vehicle. The direction is a 2D vector.
            path_is_closed: Should be `True` if the path is closed (full track).
        Returns:
            The path parameterized as a 2D array of shape (N, 4). The parameters are:
                The spline parameter (theta) for the rest of the values
                The x position of the path.
                The y position of the path.
                The curvature of the path.
        """

        spline_of_path = self._refit_spline(path)

        path_curvature = self._calculate_path_curvature(spline_of_path, path_is_closed)

        path_info = self._sample_path_parameters_for_prediction_horizon(
            spline_of_path, path_curvature
        )

        return np.column_stack(path_info)
