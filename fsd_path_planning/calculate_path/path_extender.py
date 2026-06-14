"""Path extension and trimming utilities."""

from __future__ import annotations

from typing import cast

import numpy as np

from fsd_path_planning.types import BoolArray, FloatArray
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    circle_fit,
    normalize_last_axis,
    trace_distance_to_next,
    unit_2d_vector_from_angle,
    vec_angle_between,
)


def connect_path_to_car(
    path_update: FloatArray,
    vehicle_position: FloatArray,
    vehicle_direction: FloatArray,
) -> FloatArray:
    """Connect the path to the car's current position if needed."""
    distance_to_first_point = np.linalg.norm(vehicle_position - path_update[0])
    car_to_first_point = path_update[0] - vehicle_position
    angle_to_first_point = vec_angle_between(car_to_first_point, vehicle_direction)

    if distance_to_first_point < 0.5 or angle_to_first_point > np.pi / 2:
        return path_update

    new_point = (
        vehicle_position + normalize_last_axis(car_to_first_point[None])[0] * 0.2
    )
    return np.vstack((new_point, path_update))


def extend_path(
    path_update: FloatArray,
    vehicle_position: FloatArray,
    vehicle_direction: FloatArray,
    mpc_path_length: float,
    circle_fit_tail_points: int,
    min_extension_radius: float,
    max_extension_radius: float,
    circular_arc_threshold: float,
    straight_extension_points: int,
) -> FloatArray:
    """Extend the path with a circular arc or straight line if too short."""
    car_to_path = path_update - vehicle_position
    mask_in_front = np.dot(car_to_path, vehicle_direction) > 0

    for i, value in enumerate(mask_in_front.copy()):
        if value:
            mask_in_front[i:] = True
            break

    mask_in_front[-circle_fit_tail_points:] = True

    if not mask_in_front.any():
        return path_update

    path_infront = path_update[mask_in_front]
    cum_length = trace_distance_to_next(path_infront).cumsum()
    path_length = cum_length[-1]

    if path_length > mpc_path_length:
        return path_update

    relevant_path = path_infront[-circle_fit_tail_points:]
    center_x, center_y, radius = circle_fit(relevant_path)
    center = np.array([center_x, center_y])

    radius_to_use = min(max(radius, min_extension_radius), max_extension_radius)

    if radius_to_use < circular_arc_threshold:
        relevant_centered = relevant_path - center
        three_points = relevant_centered[[0, len(relevant_centered) // 2, -1]]

        homogeneous = np.column_stack((np.ones(3), three_points))
        orientation_sign = np.sign(np.linalg.det(homogeneous))

        start_angle = float(angle_from_2d_vector(three_points[0]))
        end_angle = start_angle + orientation_sign * np.pi
        new_angles = np.linspace(start_angle, end_angle)
        new_points_raw = unit_2d_vector_from_angle(new_angles) * radius_to_use
        new_points = new_points_raw - new_points_raw[0] + path_update[-1]
    else:
        second_last = path_update[-2]
        last = path_update[-1]
        direction = last - second_last
        direction = direction / np.linalg.norm(direction)
        new_points = last + direction * np.arange(straight_extension_points)[:, None]

    new_points = new_points[1:]
    return np.vstack((path_update, new_points))


def remove_path_behind_car(
    path: FloatArray,
    vehicle_position: FloatArray,
) -> FloatArray:
    """Remove the part of the path that is behind the car."""
    distance_cost: FloatArray = np.linalg.norm(vehicle_position - path, axis=1)
    idx_start = int(distance_cost.argmin())
    return path[idx_start:]


def remove_path_not_in_prediction_horizon(
    path: FloatArray,
    mpc_path_length: float,
    fallback_path: FloatArray,
) -> FloatArray:
    """Truncate the path to the MPC prediction horizon length."""
    distances = trace_distance_to_next(path)
    cum_dist = np.cumsum(distances)

    if len(cum_dist) <= 1:
        return fallback_path

    mask_over: BoolArray = cum_dist > mpc_path_length
    first_over = cast(int, mask_over.argmax())

    if first_over == 0 and not mask_over[0]:
        first_over = len(cum_dist)

    return path[:first_over]
