from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.distance import cdist

from fsd_path_planning.common_types import ConeTypes, FloatArray
from fsd_path_planning.sorting_cones.trace_sorter.sort_weights_estimation.load_lyt import (
    load_lyt_file,
)
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    my_cdist_sq_euclidean,
    rotate,
    vec_angle_between,
)
from fsd_path_planning.utils.spline_fit import SplineFitterFactory


def calculate_centerline_for_whole_track(
    left_cones: FloatArray,
    right_cones: FloatArray,
    orange_cones: FloatArray,
    car_position: FloatArray,
    car_direction: FloatArray,
    point_every: float,
) -> FloatArray:
    left_cones = np.array(left_cones)
    right_cones = np.array(right_cones)
    orange_cones = np.array(orange_cones)

    (
        left_cones_with_orange,
        right_cones_with_orange,
    ) = add_orange_cones_to_left_or_right_cones(left_cones, right_cones, orange_cones)

    (
        left_cones_centered,
        right_cones_centered,
    ) = center_cones_on_car_position_and_direction(
        left_cones_with_orange, right_cones_with_orange, car_position, car_direction
    )

    centers = calculate_centers(left_cones_centered, right_cones_centered)

    centers_sorted = sort_centers(centers)

    spline_points = spline_fit_centers(centers_sorted, 0.1, point_every)
    spline_points_in_original_frame = reframe_points_to_car_position_and_direction(
        spline_points, car_position, car_direction
    )
    return spline_points_in_original_frame


def roll_spline_points_to_start_at_car_position(
    spline_points: FloatArray, car_position: FloatArray
) -> FloatArray:
    distances = np.linalg.norm(spline_points - car_position, axis=1)

    idx = np.argmin(distances)

    spline_points_rolled = np.roll(spline_points, -idx, axis=0)

    return spline_points_rolled


def angles_formed_by_point_and_target_points(
    source: FloatArray, targets: FloatArray
) -> Iterable[float]:
    for target_1, target_2 in combinations(targets, 2):
        vec_1 = target_1 - source
        vec_2 = target_2 - source

        angle = np.abs(np.arctan2(vec_1[1], vec_1[0]) - np.arctan2(vec_2[1], vec_2[0]))
        angle = abs(3.1415 - angle)

        yield angle


def classify_orange_cones_as_left_or_right(
    left_cones: FloatArray, right_cones: FloatArray, orange_cones: FloatArray
) -> Iterable[ConeTypes]:
    for orange_cone in orange_cones:
        n = 4
        idx_closest_n_right = np.argsort(
            np.linalg.norm(right_cones - orange_cone, axis=1)
        )[:n]
        idx_closest_n_left = np.argsort(
            np.linalg.norm(left_cones - orange_cone, axis=1)
        )[:n]

        closest_n_right = right_cones[idx_closest_n_right]
        closest_n_left = left_cones[idx_closest_n_left]

        largest_angle_right = max(
            angles_formed_by_point_and_target_points(orange_cone, closest_n_right)
        )
        largest_angle_left = max(
            angles_formed_by_point_and_target_points(orange_cone, closest_n_left)
        )

        # print(np.rad2deg(largest_angle_right), np.rad2deg(largest_angle_left))

        largest_angle = max(largest_angle_right, largest_angle_left)
        if largest_angle < np.deg2rad(175):
            yield ConeTypes.UNKNOWN

        elif largest_angle_right > largest_angle_left:
            yield ConeTypes.RIGHT
        else:
            yield ConeTypes.LEFT


def add_orange_cones_to_left_or_right_cones(
    left_cones: FloatArray,
    right_cones: FloatArray,
    orange_cones: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    if len(orange_cones) == 0:
        return left_cones, right_cones

    orange_cone_types = np.array(
        list(
            classify_orange_cones_as_left_or_right(
                left_cones, right_cones, orange_cones
            )
        )
    )

    left_cones_with_orange = np.row_stack(
        [left_cones, orange_cones[orange_cone_types == ConeTypes.LEFT]]
    )
    right_cones_with_orange = np.row_stack(
        [right_cones, orange_cones[orange_cone_types == ConeTypes.RIGHT]]
    )

    return left_cones_with_orange, right_cones_with_orange


def calculate_centers_from_one_source(
    source_cones: FloatArray, target_cones: FloatArray
) -> FloatArray:
    distances = cdist(source_cones, target_cones)

    # min_distance = 3.1
    # mask = distances < min_distance
    # print(mask.sum())
    # distances[mask] = np.inf

    # find closest cone to each cone
    min_indices = np.argmin(distances, axis=1)
    # centers
    centers = (source_cones + target_cones[min_indices]) / 2

    return centers


def center_cones_on_car_position_and_direction(
    left_cones: FloatArray,
    right_cones: FloatArray,
    car_position: FloatArray,
    car_direction: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    angle = angle_from_2d_vector(car_direction)
    left_centered = left_cones - car_position
    right_centered = right_cones - car_position

    left_centered_rotated = rotate(left_centered, -angle)
    right_centered_rotated = rotate(right_centered, -angle)

    return left_centered_rotated, right_centered_rotated


def reframe_points_to_car_position_and_direction(
    centers: FloatArray,
    car_position: FloatArray,
    car_direction: FloatArray,
) -> FloatArray:
    angle = angle_from_2d_vector(car_direction)

    centers_rotated = rotate(centers, angle)

    centers_uncentered = centers_rotated + car_position

    return centers_uncentered


def calculate_centers(left_cones: FloatArray, right_cones: FloatArray) -> FloatArray:
    left_source_centers = calculate_centers_from_one_source(left_cones, right_cones)
    right_source_centers = calculate_centers_from_one_source(right_cones, left_cones)

    if len(left_source_centers) > len(right_source_centers):
        source_centers = left_source_centers
        target_centers = right_source_centers
    else:
        source_centers = right_source_centers
        target_centers = left_source_centers

    # centers = calculate_centers_from_one_source(source_centers, target_centers)
    centers = np.concatenate([source_centers, target_centers])

    centers = np.unique(centers, axis=0)
    return centers


def sort_centers(centers: FloatArray) -> FloatArray:
    idx_first = np.argmin(np.linalg.norm(centers, axis=1))

    idxs = [idx_first]

    while len(idxs) < len(centers):
        mask_centers_not_used = np.ones(len(centers), dtype=bool)
        mask_centers_not_used[idxs] = False

        centers_not_used = centers[mask_centers_not_used]

        distances = cdist([centers[idxs[-1]]], centers_not_used)[0]
        # make distance to self be inf
        distances[distances < 0.01] = np.inf

        if len(idxs) == 1:
            # in the first instance we want to go forward
            # we asssume that at the beginning has a 0 degree direction
            mask_centers_negative_x = centers_not_used[:, 0] < 0
            distances[mask_centers_negative_x] = np.inf

        if len(idxs) == 1:
            direction = np.array([[1.0, 0]])
        else:
            direction = (centers[idxs[-1]] - centers[idxs[-2]])[None]

        directions_to_points = centers_not_used - centers[idxs[-1]]
        angles = vec_angle_between(direction, directions_to_points)

        mask_angles = angles < np.pi / 2

        # max_distance = 10 if len(idxs) > 4 else 15
        # mask_distance = (distances < max_distance) & (distances > 0.1)

        distances[~mask_angles] = np.inf

        if np.all(distances == np.inf):
            break

        internal_idx_next_point = np.argmin(distances)
        next_point = centers_not_used[internal_idx_next_point]

        external_idx_next_point = np.all(centers == next_point, axis=1).nonzero()[0][0]

        if len(idxs) > 1:
            new_to_first = centers[external_idx_next_point] - centers[idxs[0]]
            new_to_second = centers[external_idx_next_point] - centers[idxs[1]]

            angle = vec_angle_between(new_to_first[None], new_to_second[None])[0]

            if angle > np.pi / 1.5:
                break

        idxs.append(external_idx_next_point)

    centers_sorted = centers[idxs]
    remaining_idxs = [(i, 0) for i in range(len(centers)) if i not in idxs]

    while len(remaining_idxs) > 0:
        i, attempt = remaining_idxs.pop(0)
        point = centers[i]

        two_closest_points_idx = cdist([point], centers_sorted)[0].argsort()[:2]

        if np.abs(np.diff(two_closest_points_idx))[0] != 1:
            if attempt < 3:
                remaining_idxs.append((i, attempt + 1))

            continue

        vec_1 = centers_sorted[two_closest_points_idx[0]] - point
        vec_2 = centers_sorted[two_closest_points_idx[1]] - point

        angle = vec_angle_between(vec_1[None], vec_2[None])[0]

        if angle < np.pi / 2:
            continue

        smaller_idx = min(two_closest_points_idx)

        centers_sorted = np.insert(centers_sorted, smaller_idx + 1, point, axis=0)

    return centers_sorted


def spline_fit_centers(
    centers: FloatArray, smoothing: float, predict_every: float
) -> FloatArray:
    assert predict_every > 0.0, f"predict_every must be positive, got {predict_every}"

    centers_stacked = np.row_stack([centers] * 3)

    spline_fitter = SplineFitterFactory(smoothing, predict_every, max_deg=3)

    spline_evaluator = spline_fitter.fit(centers_stacked)

    all_points = spline_evaluator.predict(der=0)

    start_idx = len(all_points) // 3 + int(1 / predict_every)
    start_point = all_points[start_idx]

    potential_end_idx = int(start_idx * 1.5)

    while True:
        current_point_distance_to_start = np.linalg.norm(
            all_points[potential_end_idx] - start_point
        )

        if current_point_distance_to_start > (predict_every * 2):
            potential_end_idx += 1
        else:
            break

    end_idx = potential_end_idx

    points_centerline_spline = all_points[start_idx:end_idx]

    # find closest point to origin and make it be the first point
    distances = np.linalg.norm(points_centerline_spline, axis=1)
    idx_closest_point = np.argmin(distances)

    points_centerline_spline_rolled = np.roll(
        points_centerline_spline, -idx_closest_point, axis=0
    )

    return points_centerline_spline_rolled


def sort_cones_by_centerline(cones: FloatArray, centerline: FloatArray) -> FloatArray:
    idxs_closest = my_cdist_sq_euclidean(centerline, cones).argmin(axis=1)

    _, indices = np.unique(idxs_closest, return_index=True)

    idxs = np.argsort(indices)

    return cones[idxs]


def one_track_edge_from_centerline(
    cones: FloatArray, centerline: FloatArray, point_every: float
) -> tuple[FloatArray, FloatArray]:
    sorted_cones = sort_cones_by_centerline(cones, centerline)

    smooth_track_edge = spline_fit_centers(
        sorted_cones,
        smoothing=0.1,
        predict_every=point_every,
    )

    idx_closest_to_centerline_start = np.linalg.norm(
        smooth_track_edge - centerline[0], axis=1
    ).argmin()

    # roll the array so that the first point is the closest to the centerline start
    smooth_track_edge = np.roll(
        smooth_track_edge, -idx_closest_to_centerline_start, axis=0
    )

    n_iter = 3

    for _ in range(n_iter):
        # print(f"filling gaps, iteration {_}")
        distance_to_next = np.linalg.norm(np.diff(smooth_track_edge, axis=0), axis=1)
        distance_first_last = np.linalg.norm(
            smooth_track_edge[0] - smooth_track_edge[-1]
        )
        distance_to_next = np.concatenate([distance_to_next, [distance_first_last]])

        # mask_set_distance_to_inf = np.ones_like(distance_to_next, dtype=bool)
        # mask_set_distance_to_inf[:20] = False
        # mask_set_distance_to_inf[-20:] = False

        # distance_to_next[mask_set_distance_to_inf] = np.inf

        idx_largest_distance = np.argmax(distance_to_next)

        if distance_to_next[idx_largest_distance] < (point_every * 1.2):
            break

        start_point = smooth_track_edge[idx_largest_distance]
        end_point = smooth_track_edge[
            (idx_largest_distance + 1) % len(smooth_track_edge)
        ]

        distance_to_fill = np.linalg.norm(start_point - end_point)
        direction = (end_point - start_point) / distance_to_fill

        distances = np.arange(point_every, distance_to_fill, point_every)
        # print(distances)

        new_points = start_point + distances[:, None] * direction

        smooth_track_edge = np.insert(
            smooth_track_edge, idx_largest_distance + 1, new_points, axis=0
        )
    else:
        pass
        # print(f"Could not fill the gaps after {n_iter} iterations.")

    return smooth_track_edge


def track_edges_from_centerline(
    left_cones: FloatArray,
    right_cones: FloatArray,
    centerline: FloatArray,
    point_every: float,
) -> tuple[FloatArray, FloatArray]:
    left_edge = one_track_edge_from_centerline(left_cones, centerline, point_every)
    right_edge = one_track_edge_from_centerline(right_cones, centerline, point_every)

    return left_edge, right_edge


def centerline_and_track_edges(
    left_cones: FloatArray,
    right_cones: FloatArray,
    orange_cones: FloatArray,
    point_every: float,
    origin_position: FloatArray,
    origin_direction: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    first_centerline = calculate_centerline_for_whole_track(
        left_cones,
        right_cones,
        orange_cones,
        origin_position,
        origin_direction,
        point_every,
    )

    left_track_edge, right_track_edge = track_edges_from_centerline(
        left_cones, right_cones, first_centerline, point_every
    )

    # final_centerline = calculate_centerline_for_whole_track(
    #     left_track_edge,
    #     right_track_edge,
    #     np.zeros((0, 2)),
    #     origin_position,
    #     origin_direction,
    #     point_every,
    # )
    final_centerline = first_centerline

    # point_connect_first_to_last = final_centerline[0] * 0.8 + final_centerline[-1] * 0.2

    # final_centerline = np.concatenate([final_centerline, [point_connect_first_to_last]])

    final_centerline_smooth = (
        SplineFitterFactory(0.1, predict_every=point_every / 3, max_deg=3)
        .fit(final_centerline[::])
        .predict(der=0)[::3]
    )

    return final_centerline_smooth, left_track_edge, right_track_edge


def get_sorted_left_right_coness_from_lyt(
    lyt_path: Path,
) -> tuple[list[FloatArray], dict[str, float], FloatArray, FloatArray, FloatArray]:
    lyt_path = Path(lyt_path)

    point_every = 0.1

    assert lyt_path.is_file(), lyt_path
    assert lyt_path.suffix == ".lyt", lyt_path

    cones, start_info = load_lyt_file(lyt_path)

    start_position = np.array([start_info["position_x"], start_info["position_y"]])
    start_rad = np.deg2rad(start_info["heading_degrees"])
    start_direction = np.array([np.cos(start_rad), np.sin(start_rad)])

    left_cones = cones[ConeTypes.LEFT]
    right_cones = cones[ConeTypes.RIGHT]
    orange_cones = cones[ConeTypes.ORANGE_BIG]

    centerline, left_track_edge, right_track_edge = centerline_and_track_edges(
        left_cones,
        right_cones,
        orange_cones,
        point_every,
        start_position,
        start_direction,
    )

    cones_left_final_sorted = do_final_sorting_for_original_cones(
        left_cones, orange_cones, left_track_edge
    )

    cones_right_final_sorted = do_final_sorting_for_original_cones(
        right_cones, orange_cones, right_track_edge
    )

    return cones_left_final_sorted, cones_right_final_sorted


def do_final_sorting_for_original_cones(
    cones_of_side, orange_cones, track_edge_of_side
):
    cones_with_orange = np.row_stack([cones_of_side, orange_cones])

    distances_cones_with_orange_to_track_edge = cdist(
        cones_with_orange, track_edge_of_side
    ).min(axis=1)

    mask_distance_small = distances_cones_with_orange_to_track_edge < 0.2

    only_cones_of_side = cones_with_orange[mask_distance_small]

    cones_of_side_sorted = sort_cones_by_centerline(
        only_cones_of_side, track_edge_of_side
    )

    return cones_of_side_sorted


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lyt_path = Path("/mnt/c/LFS/data/layout/LA2_tims_miese_strecke.lyt")

    left_sorted, right_sorted = get_sorted_left_right_coness_from_lyt(lyt_path)

    plt.plot(*left_sorted.T, ".-")
    plt.plot(*right_sorted.T, ".-")
    plt.axis("equal")
    plt.show()
