"""
Helpers that the planner no longer uses.

Reachable only through the deprecated fsd_path_planning.utils.math_utils,
with a warning, until they are removed in 0.7.0.

Project: fsd_path_planning
"""

from __future__ import annotations

from typing import TypeVar, cast

import numpy as np

from fsd_path_planning._numba import njit


@njit
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
