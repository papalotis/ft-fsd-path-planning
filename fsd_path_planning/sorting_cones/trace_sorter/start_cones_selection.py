import numpy as np

from fsd_path_planning.common_types import ConeTypes, FloatArray
from fsd_path_planning.utils.cone_types import invert_cone_type
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    points_inside_ellipse,
    rotate,
    vec_angle_between,
)


def mask_cone_can_be_first_in_config(
    car_position: FloatArray,
    car_direction: FloatArray,
    cones: FloatArray,
    cone_type: ConeTypes,
    ellipse_max_dist_to_first_large_axis: float,
    ellipse_max_dist_to_first_small_axis: float,
) -> tuple[FloatArray, np.ndarray]:
    cones_xy = cones[:, :2]  # remove cone type

    cones_relative = rotate(
        cones_xy - car_position, -angle_from_2d_vector(car_direction)
    )

    cone_relative_angles = angle_from_2d_vector(cones_relative)

    trace_distances = np.linalg.norm(cones_relative, axis=-1)

    mask_is_in_ellipse = points_inside_ellipse(
        cones_xy,
        car_position,
        car_direction,
        ellipse_max_dist_to_first_large_axis,
        ellipse_max_dist_to_first_small_axis,
    )
    angle_signs = np.sign(cone_relative_angles)
    valid_angle_sign = 1 if cone_type == ConeTypes.LEFT else -1
    mask_valid_side = angle_signs == valid_angle_sign
    mask_is_valid_angle = np.abs(cone_relative_angles) < np.pi - np.pi / 5
    mask_is_valid_angle_min = np.abs(cone_relative_angles) > np.pi / 10
    mask_is_right_color = cones[:, 2] == cone_type

    mask_side = (
        mask_valid_side * mask_is_valid_angle * mask_is_valid_angle_min
    ) + mask_is_right_color

    mask_is_not_opposite_cone_type = cones[:, 2] != invert_cone_type(cone_type)
    mask_is_valid = mask_is_in_ellipse * mask_side * mask_is_not_opposite_cone_type

    return trace_distances, mask_is_valid


def select_starting_cone(
    car_position: FloatArray,
    car_direction: FloatArray,
    cones: FloatArray,
    cone_type: ConeTypes,
    max_distance_to_first: float,
    index_to_skip: np.ndarray | None = None,
) -> int | None:
    """
    Return the index of the starting cone
        int: The index of the stating cone
    """
    trace_distances, mask_is_valid = mask_cone_can_be_first_in_config(
        car_position,
        car_direction,
        cones,
        cone_type,
        max_distance_to_first * 1.5,
        max_distance_to_first / 1.5,
    )
    if index_to_skip is not None:
        mask_is_valid[index_to_skip] = False

    trace_distances_copy = trace_distances.copy()
    trace_distances_copy[~mask_is_valid] = np.inf

    if np.any(mask_is_valid) > 0:
        sorted_idxs = np.argsort(trace_distances_copy)
        start_idx = None
        for idx in sorted_idxs:
            if index_to_skip is None or idx not in index_to_skip:
                start_idx = idx
                break
        if trace_distances_copy[start_idx] > max_distance_to_first:
            start_idx = None
    else:
        start_idx = None

    return start_idx


def select_first_k_starting_cones(
    car_position: FloatArray,
    car_direction: FloatArray,
    cones: FloatArray,
    cone_type: ConeTypes,
    max_distance_to_first: float,
) -> np.ndarray | None:
    """
    Return the index of the starting cones. Pick the cone that is closest in front
    of the car and the cone that is closest behind the car.
    """
    index_1 = select_starting_cone(
        car_position, car_direction, cones, cone_type, max_distance_to_first
    )

    if index_1 is None:
        return None

    cones_to_car = cones[:, :2] - car_position
    angle_to_car = vec_angle_between(cones_to_car, car_direction)

    mask_should_not_be_selected = np.abs(angle_to_car) < np.pi / 2
    idxs_to_skip = np.where(mask_should_not_be_selected)[0]
    if index_1 not in idxs_to_skip:
        idxs_to_skip = np.concatenate([idxs_to_skip, np.array([index_1])])

    # get the cone behind the car
    index_2 = select_starting_cone(
        car_position,
        car_direction,
        cones,
        cone_type,
        index_to_skip=idxs_to_skip,
        max_distance_to_first=max_distance_to_first,
    )

    if index_2 is None:
        return np.array([index_1], dtype=np.int_)

    cone_dir_1 = cones[index_1, :2] - cones[index_2, :2]
    cone_dir_2 = cones[index_2, :2] - cones[index_1, :2]

    angle_1 = vec_angle_between(cone_dir_1, car_direction)
    angle_2 = vec_angle_between(cone_dir_2, car_direction)

    if angle_1 > angle_2:
        index_1, index_2 = index_2, index_1

    dist = np.linalg.norm(cone_dir_1)
    if dist > max_distance_to_first * 1.1 or dist < 1.4:
        return np.array([index_1], dtype=np.int_)

    two_cones = np.array([index_2, index_1], dtype=np.int_)

    return two_cones
