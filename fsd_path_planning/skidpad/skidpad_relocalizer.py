#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Place the car in the known skidpad map and relocalize it.
"""


from itertools import combinations

import numpy as np
from icecream import ic  # pylint: disable=unused-import
from sklearn.cluster import DBSCAN

from fsd_path_planning.skidpad.skidpad_path_data import BASE_SKIDPAD_PATH
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    calc_pairwise_distances,
    circle_fit,
    rotate,
)

PowersetCirceFitResult = list[tuple[FloatArray, IntArray]]


def circle_fit_powerset(points: np.ndarray) -> PowersetCirceFitResult:
    out = []
    idxs = range(len(points))

    max_powerset_size = 5
    max_actual_powerset_size = min(max_powerset_size, len(points))

    rng = np.random.RandomState(42)

    for i in range(3, max_actual_powerset_size + 1):
        for idxs in combinations(idxs, i):
            set_ = np.array(idxs)
            # get the points in the set
            points_of_set = points[set_]

            # get the mean distance to the closest point
            distance = (
                calc_pairwise_distances(points_of_set, dist_to_self=np.inf) ** 0.5
            )
            min_distances = distance.min(axis=0)
            mean_distance = min_distances.mean()

            # estimate the circle parameters
            # add tiny noise to avoid colinear points
            points_of_set = points_of_set + rng.randn(*points_of_set.shape) * 1e-3

            res = circle_fit(points_of_set)

            residual = np.abs(
                np.linalg.norm(res[:2] - points_of_set, axis=1) - res[2]
            ).mean()

            if (
                abs(res[2] - 7.625) < 1.0
                and abs(mean_distance - 2.4) < 1.5
                and residual < 0.4
            ):
                # print(range_smallest_distance, min_distances, mean_distance, distance)
                # print(second_smallest_distance)
                out.append((res, set_))

    return out


def calculate_circle_centers(potential_circles: PowersetCirceFitResult) -> FloatArray:
    potential_centers = np.array([circle[:2] for circle, _ in potential_circles])

    labels = DBSCAN(eps=3, min_samples=1).fit(potential_centers).labels_

    unique_labels = np.unique(labels)
    assert len(unique_labels) > 1

    best_centers = None
    best_distance = 1000

    for label_1, label_2 in combinations(unique_labels, 2):
        cluster_centers = np.array(
            [
                np.median(potential_centers[labels == label], axis=0)
                for label in [label_1, label_2]
            ]
        )

        # distance between the two centers
        distance = abs(18.25 - np.linalg.norm(cluster_centers[0] - cluster_centers[1]))
        if distance < best_distance:
            best_distance = distance
            best_centers = cluster_centers

    if best_distance > 0.5:
        raise ValueError(
            "Could not find two clusters that have the same distance to the center of the skidpad"
        )

    cluster_centers = best_centers

    # sign_y_values_of_centers = np.sign(cluster_centers[:, 1])
    # if set(sign_y_values_of_centers) != {-1, 1}:
    #     raise ValueError("Found center only on one side")

    return cluster_centers


def calculate_transformation(
    reference_centers: FloatArray,
    cluster_centers: FloatArray,
    original_vehicle_position: FloatArray,
    original_vehicle_direction: FloatArray,
) -> tuple[callable, callable]:
    # Your code here
    """
    Given two reference points and two new points calculate
    """
    # convert centers to vehicle frame
    original_vehicle_yaw = angle_from_2d_vector(original_vehicle_direction)

    centers_in_vehicle_frame = rotate(
        cluster_centers - original_vehicle_position, -original_vehicle_yaw
    )

    mask_is_right = centers_in_vehicle_frame[:, 1] < 0.0

    right_calculated_center = cluster_centers[mask_is_right][0]
    left_calculated_center = cluster_centers[~mask_is_right][0]

    right_reference_center = reference_centers[0]
    left_reference_center = reference_centers[1]

    # translate the calculated centers so that the right center aligns with the right reference center
    translation = np.squeeze(right_reference_center - right_calculated_center)

    # calculate the angle between the two reference points
    from_right_reference_to_left_reference = (
        left_reference_center - right_reference_center
    )
    reference_angle = angle_from_2d_vector(from_right_reference_to_left_reference)

    # calculate the angle between the two calculated points
    from_right_calculated_to_left_calculated = (
        left_calculated_center - right_calculated_center
    )

    calculated_angle = angle_from_2d_vector(from_right_calculated_to_left_calculated)

    # calculate the rotation
    rotation = np.squeeze(reference_angle - calculated_angle)

    def transform_pose(position_2d, yaw):
        # handle position
        position_translated = position_2d + translation
        # translate to reference right center
        position_translated_centered = position_translated - right_reference_center
        # rotate by rotation
        position_rotated = rotate(position_translated_centered, rotation)
        # translate back
        position_rotated = position_rotated + right_reference_center

        # handle yaw
        yaw_rotated = yaw + rotation

        return position_rotated, yaw_rotated

    def transform_back_to_original(position_2d, yaw):
        # handle position
        position_translated = position_2d - translation
        # translate to skidpad right center
        position_translated_centered = position_translated - right_calculated_center
        # rotate by negative rotation
        position_rotated = rotate(position_translated_centered, -rotation)
        # translate back
        position_rotated = position_rotated + right_calculated_center

        # handle yaw
        yaw_rotated = yaw - rotation

        return position_rotated, yaw_rotated

    return transform_pose, transform_back_to_original


def calculate_reference_centers_for_skidpad_path(
    skidpad_path: FloatArray,
) -> FloatArray:
    points_with_definitely_neg_y = skidpad_path[skidpad_path[:, 1] < -2]

    center_neg_y = circle_fit(points_with_definitely_neg_y)[:2]

    points_with_definitely_pos_y = skidpad_path[skidpad_path[:, 1] > 2]

    center_pos_y = circle_fit(points_with_definitely_pos_y)[:2]

    return np.array([center_neg_y, center_pos_y])


class SkidpadRelocalizer:
    # this class attempts to relocalize the car in the skidpad map
    # since we will not detect all cones at once and we can sit in
    # the start line for a bit, this class will attempt to relocalize
    # a few times, each time new cones are detected until a plausible
    # transformation is found

    def __init__(self):
        self.reference_centers = calculate_reference_centers_for_skidpad_path(
            BASE_SKIDPAD_PATH
        )

        self._transform_to_skidpad_frame: callable[
            [FloatArray, float], tuple[FloatArray, float]
        ] | None = None
        self._transform_to_original_frame: callable[
            [FloatArray, float], tuple[FloatArray, float]
        ] | None = None

        self._original_vehicle_position: FloatArray | None = None
        self._original_vehicle_direction: FloatArray | None = None

    def attempt_relocalization_calculation(
        self,
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> bool:
        if self.is_relocalized:
            return True

        if self._original_vehicle_position is None:
            self._original_vehicle_position = vehicle_position

        if self._original_vehicle_direction is None:
            self._original_vehicle_direction = vehicle_direction

        cones_array = np.row_stack(cones)
        cones_array_xy = cones_array[:, :2]

        # only keep 20 closest cones
        # this is an optimization to speed up the calculation
        # because otherwise we enter combinatorics hell :)
        distances = np.linalg.norm(cones_array_xy - vehicle_position, axis=1)
        idxs = np.argsort(distances)[:20]
        cones_array_xy = cones_array_xy[idxs]

        # calculate the centers of the circles
        # print(cones_array_xy)
        potential_circles = circle_fit_powerset(cones_array_xy)

        if len(potential_circles) < 3:
            return False

        try:
            circle_centers = calculate_circle_centers(potential_circles)
        except (ValueError, AssertionError):
            return False

        # calculate the transformation
        try:
            (
                transform_to_skidpad_frame,
                transform_to_original_frame,
            ) = calculate_transformation(
                self.reference_centers,
                circle_centers,
                self._original_vehicle_position,
                self._original_vehicle_direction,
            )
        except IndexError:
            return False

        # save the transformation
        self._transform_to_skidpad_frame = transform_to_skidpad_frame
        self._transform_to_original_frame = transform_to_original_frame

        return True

    def transform_to_skidpad_frame(
        self, position_2d: FloatArray, yaw: float
    ) -> tuple[FloatArray, float]:
        if self._transform_to_skidpad_frame is None:
            raise ValueError("No transformation calculated yet")

        return self._transform_to_skidpad_frame(position_2d, yaw)

    def transform_to_original_frame(
        self, position_2d: FloatArray, yaw: float
    ) -> tuple[FloatArray, float]:
        if self._transform_to_original_frame is None:
            raise ValueError("No transformation calculated yet")

        return self._transform_to_original_frame(position_2d, yaw)

    @property
    def is_relocalized(self):
        return self._transform_to_skidpad_frame is not None
