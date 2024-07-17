from __future__ import annotations

import random
import sys
from itertools import count
from pathlib import Path
from typing import Iterator

import numpy as np
from joblib import Memory
from tqdm import tqdm

from fsd_path_planning.common_types import BoolArray, FloatArray, IntArray
from fsd_path_planning.config import get_cone_sorting_config
from fsd_path_planning.sorting_cones.trace_sorter.common import NoPathError
from fsd_path_planning.sorting_cones.trace_sorter.cost_function import (
    cost_configurations,
)
from fsd_path_planning.sorting_cones.trace_sorter.find_configs_and_scores import (
    calc_scores_and_end_configurations,
)
from fsd_path_planning.sorting_cones.trace_sorter.sort_weights_estimation.sort_whole_track_with_perfect_cones import (
    get_sorted_left_right_coness_from_lyt,
)
from fsd_path_planning.sorting_cones.trace_sorter.start_cones_selection import (
    select_first_k_starting_cones,
)
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    my_cdist_sq_euclidean,
    rotate,
    unit_2d_vector_from_angle,
)
from fsd_path_planning.utils.mission_types import MissionTypes


def find_project_root_path():
    file_path = Path(__file__)
    while file_path.name != "fsd_path_planning":
        file_path = file_path.parent
    return file_path.parent


def cache_location():
    return find_project_root_path() / ".cache"


memory = Memory(location=cache_location(), verbose=0)


def calculate_configurations_for_track_part_for_cone_type(
    cone_type: ConeTypes,
    cones: FloatArray,  # (N, 3) last dim is type
    car_position: FloatArray,
    car_direction: FloatArray,
    max_distance_to_first: float,
    n_neighbors: int,
    threshold_directional_angle: float,
    threshold_absolute_angle: float,
    max_distance_between_neighboring_cones: float,
    max_length: int,
) -> IntArray | None:
    first_k_cones = select_first_k_starting_cones(
        car_position,
        car_direction,
        cones,
        cone_type,
        max_distance_to_first,
    )

    assert first_k_cones is not None
    try:
        _, end_configurations, _ = calc_scores_and_end_configurations(
            trace=cones,
            cone_type=cone_type,
            n_neighbors=n_neighbors,
            start_idx=first_k_cones[0],
            threshold_directional_angle=threshold_directional_angle,
            threshold_absolute_angle=threshold_absolute_angle,
            vehicle_position=car_position,
            vehicle_direction=car_direction,
            max_dist=max_distance_between_neighboring_cones,
            max_length=max_length,
            first_k_indices_must_be=first_k_cones,
        )
    except NoPathError:
        return None

    return end_configurations


def calculate_configurations_for_track_part(
    cones: FloatArray,  # (N, 3) last dim is type
    car_position: FloatArray,
    car_direction: FloatArray,
    max_distance_to_first: float,
    n_neighbors: int,
    threshold_directional_angle: float,
    threshold_absolute_angle: float,
    max_distance_between_neighboring_cones: float,
    max_length: int,
) -> tuple[IntArray, IntArray]:
    result = tuple(
        calculate_configurations_for_track_part_for_cone_type(
            cone_type,
            cones,
            car_position,
            car_direction,
            max_distance_to_first,
            n_neighbors,
            threshold_directional_angle,
            threshold_absolute_angle,
            max_distance_between_neighboring_cones,
            max_length,
        )
        for cone_type in [ConeTypes.LEFT, ConeTypes.RIGHT]
    )
    assert len(result) == 2

    return result


def generate_car_position_and_direction(
    rng: np.random.Generator,
    left_cones: FloatArray,
    right_cones: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    # pick random left cone
    idx_left_cone = rng.integers(len(left_cones))
    left_cone = left_cones[idx_left_cone]
    # pick closest right cone
    idx_right_cone = np.linalg.norm(right_cones - left_cone, axis=1).argmin()
    right_cone = right_cones[idx_right_cone]

    # car position equal to linear interpolation between the two cones
    # min mix factor between 0 and 1 (but not too close to 0 or 1)
    # max mix factor between 1 - min_mix_factor

    min_mix_factor = 0.3
    max_mix_factor = 1 - min_mix_factor

    mix_factor = rng.uniform(min_mix_factor, max_mix_factor)

    car_position = mix_factor * left_cone[:2] + (1 - mix_factor) * right_cone[:2]

    idx_left_cone_next = (idx_left_cone + 1) % len(left_cones)
    idx_right_cone_next = (idx_right_cone + 1) % len(right_cones)

    direction_at_left_cone = left_cones[idx_left_cone_next] - left_cone
    direction_at_right_cone = right_cones[idx_right_cone_next] - right_cone

    direction_at_left_cone = direction_at_left_cone / np.linalg.norm(
        direction_at_left_cone
    )
    direction_at_right_cone = direction_at_right_cone / np.linalg.norm(
        direction_at_right_cone
    )

    mix_factor_direction_min = 0.2
    mix_factor_direction_max = 1 - mix_factor_direction_min

    mix_factor_direction = rng.uniform(
        mix_factor_direction_min, mix_factor_direction_max
    )

    car_direction = (
        mix_factor_direction * direction_at_left_cone
        + (1 - mix_factor_direction) * direction_at_right_cone
    )

    return car_position, car_direction


def create_false_detection(
    all_cones: FloatArray,
    cones_detected: FloatArray,
    rng: np.random.Generator,
    cone_type: ConeTypes,
) -> FloatArray:
    # select random cone
    idx_cone = rng.integers(len(cones_detected))
    base_cone = cones_detected[idx_cone]

    idx_cone_detected_in_all = np.linalg.norm(all_cones - base_cone, axis=1).argmin()
    idx_next_cone = (idx_cone_detected_in_all + 1) % len(all_cones)
    next_cone = all_cones[idx_next_cone]

    base_direction = next_cone - base_cone
    base_direction = base_direction / np.linalg.norm(base_direction)
    base_angle = angle_from_2d_vector(base_direction)

    offset_core_direction = np.pi / 4 if cone_type == ConeTypes.LEFT else -np.pi / 4
    offset_angle = rng.normal(offset_core_direction, 1.0)  # np.pi / 8)

    # rarely add a large offset
    max_magnitude = 3.0 if rng.random() > 0.9 else 10.0
    offset_magnitude = rng.uniform(0.8, max_magnitude)

    offset_direction = unit_2d_vector_from_angle(base_angle + offset_angle)

    false_detection = base_cone + offset_magnitude * offset_direction

    return false_detection


def probability_cone_detected(
    cone_relative_to_car: FloatArray, indexed_distance_from_car: int
) -> float:
    distance_to_cone = np.linalg.norm(cone_relative_to_car)
    x = distance_to_cone

    prob_distance = 1.5 - np.tanh(0.07 * x) ** 10

    prob_index = 0.999 - np.tanh((0.1 * indexed_distance_from_car) ** 5)
    prob = prob_distance * prob_index

    return 1 - np.clip(prob, 0, 1)


def create_cone_inputs(
    rng: np.random.Generator,
    cones_of_type: float,
    car_position: FloatArray,
    car_direction: FloatArray,
    cone_type: ConeTypes,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """
    Creates cones for sorting. Randomly deletes some cones and adds some noise and
    false detections close to real cones.
    """

    # randomly delete some cones
    car_angle = angle_from_2d_vector(car_direction)
    cones_relative_to_car = rotate(cones_of_type - car_position, -car_angle)
    idx_closest_to_car = np.linalg.norm(cones_relative_to_car, axis=1).argmin()

    cones_relative_to_car_rolled = np.roll(
        cones_relative_to_car, -idx_closest_to_car, axis=0
    )

    mask_keep = np.array(
        [
            rng.uniform()
            > probability_cone_detected(cone, min(i, len(cones_relative_to_car) - i))
            for i, cone in enumerate(cones_relative_to_car_rolled)
        ]
    )

    mask_keep_unrolled = np.roll(mask_keep, idx_closest_to_car)
    cones_missing = cones_of_type[mask_keep_unrolled]

    # add false detections
    n_false_detections = max(round(rng.normal(len(cones_missing) / 3, 4.0)), 0)
    false_detections = np.array(
        [
            create_false_detection(cones_of_type, cones_missing, rng, cone_type)
            for _ in range(n_false_detections)
        ]
    ).reshape(-1, 2)
    cones_with_false_detections = np.row_stack([cones_missing, false_detections])

    # add noise
    noise_factor = 0.1
    noise = rng.uniform(-noise_factor, noise_factor, cones_with_false_detections.shape)
    cones_with_noise = cones_with_false_detections + noise

    mask_is_real = np.ones(cones_with_noise.shape[0], dtype=bool)
    if n_false_detections > 0:
        mask_is_real[-n_false_detections:] = False

    return cones_missing, cones_with_noise, mask_is_real


def create_sorting_inputs(
    seed: int,
    left_cones: FloatArray,
    right_cones: FloatArray,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    BoolArray,
    FloatArray,
    FloatArray,
    BoolArray,
    FloatArray,
]:
    """
    Creates a cone sorting scenario. Selects track position and direction.
    Gathers cones there and returns them with the car position and direction.
    It also returns the expected configurations for the cones.
    """

    rng = np.random.default_rng(seed)

    car_position, car_direction = generate_car_position_and_direction(
        rng, left_cones, right_cones
    )

    original_left, left_cones_with_noise, mask_is_real_left = create_cone_inputs(
        rng,
        left_cones,
        car_position,
        car_direction,
        ConeTypes.LEFT,
    )

    original_right, right_cones_with_noise, mask_is_real_right = create_cone_inputs(
        rng,
        right_cones,
        car_position,
        car_direction,
        ConeTypes.RIGHT,
    )

    return (
        car_position,
        car_direction,
        left_cones_with_noise,
        mask_is_real_left,
        original_left,
        right_cones_with_noise,
        mask_is_real_right,
        original_right,
    )


def create_sorting_inputs_from_lyt_file(
    seed: int, lyt_file: Path
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    BoolArray,
    FloatArray,
    FloatArray,
    BoolArray,
    FloatArray,
]:
    left_cones, right_cones = get_sorted_left_right_coness_from_lyt(lyt_file)

    return create_sorting_inputs(
        seed,
        left_cones,
        right_cones,
    )


@memory.cache
def calculate_sample_end_configurations_from_lyt_file(
    seed: int, lyt_path: Path, remove_color: bool = False
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    BoolArray,
    BoolArray,
    FloatArray,
    FloatArray,
    IntArray,
    IntArray,
]:
    (
        car_position,
        car_direction,
        left_final,
        left_is_real,
        left_original,
        right_final,
        right_is_real,
        right_original,
    ) = create_sorting_inputs_from_lyt_file(
        seed,
        lyt_path,
    )

    len_left = len(left_final)
    cones_together = np.zeros((len_left + len(right_final), 3))
    cones_together[:len_left, :2] = left_final
    cones_together[:len_left, 2] = ConeTypes.LEFT
    cones_together[len_left:, :2] = right_final
    cones_together[len_left:, 2] = ConeTypes.RIGHT

    if remove_color:
        cones_together[:, 2] = ConeTypes.UNKNOWN

    args = get_cone_sorting_config(MissionTypes.trackdrive)

    (
        left_end_configurations,
        right_end_configurations,
    ) = calculate_configurations_for_track_part(
        cones_together,
        car_position,
        car_direction,
        max_distance_to_first=args["max_dist_to_first"],
        n_neighbors=args["max_n_neighbors"],
        threshold_directional_angle=args["threshold_directional_angle"],
        threshold_absolute_angle=args["threshold_absolute_angle"],
        max_distance_between_neighboring_cones=args["max_dist"],
        max_length=args["max_length"],
    )

    return (
        cones_together,
        left_original,
        right_original,
        left_is_real,
        right_is_real,
        car_position,
        car_direction,
        left_end_configurations,
        right_end_configurations,
    )


def apply_cost_function(
    seed: int, lyt_path: Path, weights: FloatArray
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
]:
    (
        cones,
        left_original,
        right_original,
        mask_left_is_real,
        mask_right_is_real,
        car_position,
        car_direction,
        left_configs,
        right_configs,
    ) = calculate_sample_end_configurations_from_lyt_file(
        seed, lyt_path, remove_color=True
    )

    cost_configurations_left = (
        cost_configurations(
            cones,
            left_configs,
            ConeTypes.LEFT,
            car_position,
            car_direction,
            weights,
            return_individual_costs=False,
        )
        if left_configs is not None
        else np.inf
    )

    cost_configurations_right = (
        cost_configurations(
            cones,
            right_configs,
            ConeTypes.RIGHT,
            car_position,
            car_direction,
            weights,
            return_individual_costs=False,
        )
        if right_configs is not None
        else np.inf
    )

    if left_configs is not None:
        idx_best_left = np.argmin(cost_configurations_left)
        best_left = left_configs[idx_best_left]
        best_left = best_left[best_left != -1]

        cones_left = cones[best_left, :2]

    else:
        cones_left = np.zeros((0, 2))

    if right_configs is not None:
        idx_best_right = np.argmin(cost_configurations_right)
        best_right = right_configs[idx_best_right]
        best_right = best_right[best_right != -1]

        cones_right = cones[best_right, :2]

    else:
        cones_right = np.zeros((0, 2))

    mask_is_real = np.concatenate([mask_left_is_real, mask_right_is_real])
    false_cones = cones[~mask_is_real]

    false_cones_left = false_cones[false_cones[:, 2] == ConeTypes.LEFT]
    false_cones_right = false_cones[false_cones[:, 2] == ConeTypes.RIGHT]

    real_cones_left = cones[: len(mask_left_is_real)][mask_left_is_real]
    real_cones_right = cones[len(mask_left_is_real) :][mask_right_is_real]

    return (
        left_original,
        right_original,
        cones_left,
        cones_right,
        false_cones_left,
        false_cones_right,
        false_cones,
        real_cones_left,
        real_cones_right,
    )


def grade_config_for_weights(seed: int, lyt_path: Path, weights: FloatArray) -> float:
    (
        left_original,
        right_original,
        cones_left,
        cones_right,
        false_cones_left,
        false_cones_right,
        false_cones,
        real_cones_left,
        real_cones_right,
    ) = apply_cost_function(seed, lyt_path, weights)

    # find number of incorrectly sorted left cones
    # idx_left_cones_in_original_left_cones = np.all(
    #     (left_original[:, None, :2] - cones_left[:, :2]) < 0.2, axis=2
    # )
    # print(idx_left_cones_in_original_left_cones)

    # find number of real right cones in left cones
    # find number of real left cones in right cones
    n_wrong_real_left = np.all(
        real_cones_right[:, None, :2] == cones_left[:, :2], axis=2
    ).sum()

    n_wrong_real_right = np.all(
        real_cones_left[:, None, :2] == cones_right[:, :2], axis=2
    ).sum()

    # find number of false cones in left and right cones
    n_false_in_left = np.all(
        false_cones[:, None, :2] == cones_left[:, :2], axis=2
    ).sum()

    n_false_in_right = np.all(
        false_cones[:, None, :2] == cones_right[:, :2], axis=2
    ).sum()

    return sum(
        [
            n_wrong_real_left,
            n_wrong_real_right,
            n_false_in_left,
            n_false_in_right,
        ]
    )

    plt.plot(left_original[:, 0], left_original[:, 1], ".", label="left original")
    plt.plot(right_original[:, 0], right_original[:, 1], ".", label="right original")
    plt.plot(cones_left[:, 0], cones_left[:, 1], ".-", label="left")
    plt.plot(cones_right[:, 0], cones_right[:, 1], ".-", label="right")
    plt.plot(false_cones_left[:, 0], false_cones_left[:, 1], "o", label="false left")
    plt.plot(false_cones_right[:, 0], false_cones_right[:, 1], "o", label="false right")
    plt.plot(false_cones[:, 0], false_cones[:, 1], "o", label="false")
    plt.axis("equal")
    plt.legend()
    plt.show()


def create_inputs_iterator(
    initial_seed: int, n_iters: int | None = None
) -> Iterator[tuple[int, Path]]:
    base_path = Path("/mnt/c/LFS/data/layout")
    tracks = [base_path / f"LA2_eval_track_{i:02}.lyt" for i in range(1, 10)] + [
        (base_path / f"LA2_{track_name}").with_suffix(".lyt")
        for track_name in ["fsg23", "fs_east_22", "fsg_real", "fs_spain_real"]
    ]

    rng = random.Random(initial_seed)

    iterator = range(n_iters) if n_iters is not None else count()
    for i in iterator:
        seed = rng.randint(0, sys.maxsize - 1)
        idx_track = rng.randint(0, len(tracks) - 1)
        lyt_path = tracks[idx_track]

        yield seed, lyt_path


@memory.cache
def evaluate_weights(initial_seed: int, n_iters: int, weights: FloatArray) -> float:
    return sum(
        grade_config_for_weights(seed, lyt_path, weights)
        for seed, lyt_path in create_inputs_iterator(initial_seed, n_iters=n_iters)
    )


def generate_random_weights(rng_seed: int) -> FloatArray:
    rng = np.random.default_rng(rng_seed)
    w = np.array([rng.uniform(0, 1) for _ in range(7)])
    return w / w.sum()


def evolve_weights(weights: FloatArray, scores: FloatArray) -> FloatArray:
    weights = weights.copy()
    # sort weights by scores
    idx_sort = np.argsort(scores)
    weights = weights[idx_sort]

    # select best 10% of weights
    n_best = len(weights) // 10
    best_weights = weights[:n_best]

    # select random 90% of weights
    n_random = len(weights) // 10 * 9
    random_weights = np.array([generate_random_weights(i) for i in range(n_random)])

    # combine best and random weights
    new_weights = np.row_stack([best_weights, random_weights])

    return new_weights


def optimize_weights(
    initial_seed: int,
    n_iters_evolution: int,
    n_iters_evaluation: int,
    n_initial_pop: int,
) -> FloatArray:
    base_weights = np.array([1000.0, 200.0, 5000.0, 1000.0, 0.0, 1000.0, 1000.0])
    base_weights = base_weights / base_weights.sum()

    randomized_weights = [
        generate_random_weights(initial_seed + i) for i in range(n_initial_pop)
    ]
    all_weights = np.array([base_weights] + randomized_weights)

    for i in range(n_iters_evolution):
        eval_results = [
            evaluate_weights(initial_seed, n_iters_evaluation, weights)
            for weights in tqdm(all_weights)
        ]

        print(hash(all_weights.tobytes()))

        best_score = np.min(eval_results)

        print(f"Generation {i} best score: {best_score}")

        all_weights = evolve_weights(all_weights, eval_results)

        if i == n_iters_evolution - 1:
            idx_best = np.argmin(eval_results)
            best_weights = all_weights[idx_best]

            return best_weights


if __name__ == "__main__":
    print(
        optimize_weights(
            initial_seed=0,
            n_iters_evolution=10,
            n_iters_evaluation=1000,
            n_initial_pop=500,
        ).tolist()
    )

    raise SystemExit(0)

    import matplotlib.pyplot as plt
    from tqdm import trange

    rng = random.Random(42)

    tracks = [
        Path(f"/mnt/c/LFS/data/layout/LA2_eval_track_{i:02}.lyt") for i in range(1, 10)
    ]

    for i in trange(10):
        seed = rng.randint(0, 1000000)
        idx_track = rng.randint(0, len(tracks) - 1)
        lyt_path = tracks[idx_track]

        # if i < 9:
        #     continue

        # r = calculate_sample_end_configurations_from_lyt_file(seed, lyt_path)
        # r = apply_cost_function(seed, lyt_path)
        r = grade_config_for_weights(
            seed,
            lyt_path,
            np.array([1000.0, 200.0, 5000.0, 1000.0, 0.0, 1000.0, 1000.0]),
        )

        print(seed, lyt_path)
        print(r)

    # (
    #     car_position,
    #     car_direction,
    #     left_final,
    #     left_is_real,
    #     right_final,
    #     right_is_real,
    # ) = create_sorting_inputs_from_lyt_file(
    #     seed,
    #     lyt_path,
    # )

    # plt.scatter(*left_final[left_is_real].T, label="left real")
    # plt.scatter(*left_final[~left_is_real].T, label="left false")
    # plt.scatter(*right_final[right_is_real].T, label="right real")
    # plt.scatter(*right_final[~right_is_real].T, label="right false")

    # plt.scatter(*car_position, label="car position", marker="x")
    # plt.quiver(*car_position, *car_direction, label="car direction")

    # plt.legend()
    # plt.axis("equal")
    # plt.show()
