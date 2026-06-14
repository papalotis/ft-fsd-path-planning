from __future__ import annotations

from typing import Any

import numpy as np

from fsd_path_planning.types import FloatArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import unit_2d_vector_from_angle


def convert_direction_to_array(direction: Any) -> FloatArray:
    direction_array = np.squeeze(np.asarray(direction))

    if not np.issubdtype(direction_array.dtype, np.number):
        raise TypeError("direction must contain numeric values")

    direction_array = direction_array.astype(float, copy=False)

    if direction_array.shape == (2,):
        if not np.isfinite(direction_array).all():
            raise ValueError("direction must contain only finite values")
        if np.allclose(direction_array, 0.0):
            raise ValueError("direction vector must not be zero")
        return direction_array

    if direction_array.shape in [(1,), ()]:
        scalar_direction = float(direction_array)
        if not np.isfinite(scalar_direction):
            raise ValueError("direction must contain only finite values")
        return unit_2d_vector_from_angle(scalar_direction)

    raise ValueError("direction must be a float or a 2 element array")


def validate_vehicle_position(vehicle_position: FloatArray) -> FloatArray:
    position = np.asarray(vehicle_position)

    if not np.issubdtype(position.dtype, np.number):
        raise TypeError("vehicle_position must contain numeric values")

    position = position.astype(float, copy=False)
    if position.shape != (2,):
        raise ValueError(f"vehicle_position must have shape (2,), got {position.shape}")

    if not np.isfinite(position).all():
        raise ValueError("vehicle_position must contain only finite values")

    return position


def validate_and_normalize_cones(cones: list[FloatArray]) -> list[FloatArray]:
    if len(cones) != len(ConeTypes):
        raise ValueError(
            f"cones must contain {len(ConeTypes)} arrays ordered by ConeTypes, "
            f"got {len(cones)}"
        )

    normalized_cones: list[FloatArray] = []
    for cone_type, cone_array in enumerate(cones):
        array = np.asarray(cone_array)

        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(
                f"cones[{cone_type}] must contain numeric values, got {array.dtype}"
            )

        array = array.astype(float, copy=False)

        if array.size == 0:
            normalized_cones.append(np.zeros((0, 2), dtype=float))
            continue

        if array.ndim == 1:
            if array.shape != (2,):
                raise ValueError(
                    f"cones[{cone_type}] must have shape (N, 2), got {array.shape}"
                )
            array = array.reshape(1, 2)

        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(
                f"cones[{cone_type}] must have shape (N, 2), got {array.shape}"
            )

        if not np.isfinite(array).all():
            raise ValueError(f"cones[{cone_type}] must contain only finite values")

        normalized_cones.append(array)

    return normalized_cones
