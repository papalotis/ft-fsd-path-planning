#!/usr/bin/env python3
"""
Description: Place the car in the known skidpad map and relocalize it.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from fsd_path_planning.common_types import FloatArray, IntArray
from fsd_path_planning.config_dataclasses import SkidpadConfig
from fsd_path_planning.geometry.angles import angle_from_2d_vector
from fsd_path_planning.geometry.distances import calc_pairwise_distances
from fsd_path_planning.geometry.shapes import circle_fit
from fsd_path_planning.geometry.transforms import SE2Transform, rotate
from fsd_path_planning.stages.relocalization.base import (
    RelocalizationCallable,
    Relocalizer,
)
from fsd_path_planning.stages.relocalization.skidpad_path_data import BASE_SKIDPAD_PATH

PowersetCirceFitResult = list[tuple[FloatArray, IntArray]]


def circle_fit_powerset(
    points: np.ndarray, config: SkidpadConfig | None = None
) -> PowersetCirceFitResult:
    """Circles through subsets of the points that look like a skidpad circle."""
    config = config or SkidpadConfig()
    out = []
    all_indices = range(len(points))

    # Only 3-cone subsets are evaluated, which reproduces the behaviour of the
    # released algorithm: there, a loop variable shadowed `all_indices`, so the
    # 4- and 5-cone passes silently iterated over an empty sequence. Raising this
    # limit changes relocalization results and costs an order of magnitude more
    # time per frame, so it is scheduled as a deliberate change after the
    # refactor (random subset sampling), not as a side effect of this cleanup.
    max_powerset_size = config.max_powerset_size
    max_actual_powerset_size = min(max_powerset_size, len(points))

    rng = np.random.RandomState(42)

    for i in range(3, max_actual_powerset_size + 1):
        for idxs in combinations(all_indices, i):
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
                abs(res[2] - config.circle_radius) < config.circle_radius_tolerance
                and abs(mean_distance - config.mean_cone_distance)
                < config.mean_cone_distance_tolerance
                and residual < config.max_residual
            ):
                out.append((res, set_))

    return out


# Circle centres closer than this, directly or through a chain of others, are
# taken to be estimates of the same centre, in metres. The default of
# SkidpadConfig.center_cluster_distance.
CENTER_CLUSTER_DISTANCE = SkidpadConfig().center_cluster_distance


def cluster_within_distance(points: FloatArray, distance: float) -> IntArray:
    """Label points that are linked by a chain of neighbours within `distance`.

    This is what DBSCAN with ``min_samples=1`` computes: every point is a core
    point, so a cluster is a connected component of the graph that joins points
    at most `distance` apart. Clusters are numbered in the order of their first
    point, as DBSCAN numbers them, so the result is the same as the
    scikit-learn call this replaces.
    """
    n_points = len(points)
    if n_points == 0:
        return np.zeros(0, dtype=np.intp)

    pairs = cKDTree(points).query_pairs(r=distance, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_points, n_points)
    )
    _, labels = connected_components(graph, directed=False)

    _, first_member = np.unique(labels, return_index=True)
    rank = np.empty(len(first_member), dtype=np.intp)
    rank[np.argsort(first_member)] = np.arange(len(first_member))
    return rank[labels]


def calculate_circle_centers(
    potential_circles: PowersetCirceFitResult, config: SkidpadConfig | None = None
) -> FloatArray | None:
    """The two circle centres, or None when the candidates do not show two.

    None is an expected outcome while the car waits at the start and sees only
    part of the skidpad: either all candidate centres fall into one cluster, or
    no two clusters lie the skidpad's centre distance apart.
    """
    config = config or SkidpadConfig()
    potential_centers = np.array([circle[:2] for circle, _ in potential_circles])

    labels = cluster_within_distance(potential_centers, config.center_cluster_distance)

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None

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
        distance = abs(
            config.center_distance
            - np.linalg.norm(cluster_centers[0] - cluster_centers[1])
        )
        if distance < best_distance:
            best_distance = distance
            best_centers = cluster_centers

    if best_distance > config.center_distance_tolerance:
        return None

    return best_centers


def calculate_transformation(
    reference_centers: FloatArray,
    cluster_centers: FloatArray,
    original_vehicle_position: FloatArray,
    original_vehicle_direction: FloatArray,
) -> tuple[RelocalizationCallable, RelocalizationCallable] | None:
    """The transforms between the original frame and the skidpad frame.

    Matches the two detected circle centres to the two reference centres by
    which side of the car's starting pose they lie on. Returns None when both
    detected centres lie on the same side, so they cannot be told apart.
    """
    # convert centers to vehicle frame
    original_vehicle_yaw = angle_from_2d_vector(original_vehicle_direction)

    centers_in_vehicle_frame = rotate(
        cluster_centers - original_vehicle_position, -original_vehicle_yaw
    )

    mask_is_right = centers_in_vehicle_frame[:, 1] < 0.0
    if mask_is_right.all() or not mask_is_right.any():
        return None

    right_calculated_center = cluster_centers[mask_is_right][0]
    left_calculated_center = cluster_centers[~mask_is_right][0]

    right_reference_center = reference_centers[0]
    left_reference_center = reference_centers[1]

    # translate the calculated centers so that the right center aligns with
    # the right reference center
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

    # Two explicit transforms rather than one and its inverse(): the way back
    # pivots about the detected centre, as it always has, which keeps the
    # results bit for bit the same.
    to_skidpad_frame = SE2Transform(
        angle=rotation, shift=translation, pivot=right_reference_center
    )
    to_original_frame = SE2Transform(
        angle=-rotation, shift=-translation, pivot=right_calculated_center
    )
    return to_skidpad_frame, to_original_frame


def calculate_reference_centers_for_skidpad_path(
    skidpad_path: FloatArray,
) -> FloatArray:
    points_with_definitely_neg_y = skidpad_path[skidpad_path[:, 1] < -2]

    center_neg_y = circle_fit(points_with_definitely_neg_y)[:2]

    points_with_definitely_pos_y = skidpad_path[skidpad_path[:, 1] > 2]

    center_pos_y = circle_fit(points_with_definitely_pos_y)[:2]

    return np.array([center_neg_y, center_pos_y])


class SkidpadRelocalizer(Relocalizer):
    # this class attempts to relocalize the car in the skidpad map
    # since we will not detect all cones at once and we can sit in
    # the start line for a bit, this class will attempt to relocalize
    # a few times, each time new cones are detected until a plausible
    # transformation is found

    def __init__(self, config: SkidpadConfig | None = None):
        super().__init__()
        self.config = config or SkidpadConfig()

        self.reference_centers = calculate_reference_centers_for_skidpad_path(
            BASE_SKIDPAD_PATH
        )

    def do_relocalization_once(
        self,
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> tuple[RelocalizationCallable, RelocalizationCallable] | None:
        cones_array = np.vstack(cones)
        cones_array_xy = cones_array[:, :2]

        # only keep the closest cones: the number of subsets grows
        # combinatorially with the number of cones
        distances = np.linalg.norm(cones_array_xy - vehicle_position, axis=1)
        idxs = np.argsort(distances)[: self.config.max_cones_for_relocalization]
        cones_array_xy = cones_array_xy[idxs]

        potential_circles = circle_fit_powerset(cones_array_xy, self.config)
        if len(potential_circles) < self.config.min_candidate_circles:
            return None

        circle_centers = calculate_circle_centers(potential_circles, self.config)
        if circle_centers is None:
            return None

        assert self._original_vehicle_position is not None
        assert self._original_vehicle_direction is not None
        return calculate_transformation(
            self.reference_centers,
            circle_centers,
            self._original_vehicle_position,
            self._original_vehicle_direction,
        )

    def get_known_global_path(self) -> FloatArray:
        return BASE_SKIDPAD_PATH[::2]
