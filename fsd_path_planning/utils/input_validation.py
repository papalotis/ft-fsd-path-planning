from __future__ import annotations

from typing import Any

import numpy as np

from fsd_path_planning.common_types import FloatArray
from fsd_path_planning.geometry.angles import unit_2d_vector_from_angle
from fsd_path_planning.utils.cone_types import ConeTypes


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


def _split_flat_cone_array(cones: FloatArray) -> list[FloatArray]:
    """Split an (N, 3) array of [x, y, cone_type] rows into one array per type."""
    if not np.isfinite(cones).all():
        raise ValueError("cones must contain only finite values")

    type_column = cones[:, 2]
    if not np.all(type_column == np.round(type_column)):
        raise ValueError(
            "the third column of a flat cone array must hold whole ConeTypes"
            " values, got non-integer entries"
        )

    type_indices = type_column.astype(int)
    known_types = {int(cone_type) for cone_type in ConeTypes}
    unknown = sorted(set(type_indices.tolist()) - known_types)
    if unknown:
        raise ValueError(
            "the third column of a flat cone array must hold ConeTypes values"
            f" ({min(known_types)}-{max(known_types)}), got {unknown}"
        )

    return [
        np.ascontiguousarray(cones[type_indices == int(cone_type), :2], dtype=float)
        for cone_type in ConeTypes
    ]


def normalize_cones(cones: Any) -> list[FloatArray]:
    """Bring either accepted cone format into one list of (N, 2) arrays.

    Two inputs are accepted:

    - a sequence of one (N, 2) array per cone type, ordered by
      :class:`~fsd_path_planning.utils.cone_types.ConeTypes`, which is returned
      validated and converted to float;
    - a single (N, 3) array whose rows are ``[x, y, cone_type]``, which is split
      into that per-type list. Within a type the input order is kept, but the
      cones are regrouped by type, so a caller that relied on the row order of a
      flat array may see different tie-breaking in sorting.

    Raises:
        TypeError: if the values are not numeric.
        ValueError: on the wrong shape, non-finite values, or a type column that
            does not hold whole ConeTypes values.
    """
    if isinstance(cones, np.ndarray) and cones.ndim == 2 and cones.shape[1] == 3:
        if not np.issubdtype(cones.dtype, np.number):
            raise TypeError("cones must contain numeric values")
        return _split_flat_cone_array(cones.astype(float, copy=False))

    return validate_and_normalize_cones(cones)


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
