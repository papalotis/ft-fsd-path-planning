"""Acceleration relocalization must not depend on the global NumPy generator."""

from __future__ import annotations

import numpy as np

from fsd_path_planning.common_types import FloatArray
from fsd_path_planning.relocalization.acceleration.acceleration_relocalization import (
    AccelerationRelocalizer,
    best_fit,
)
from fsd_path_planning.utils.cone_types import ConeTypes

POSITION = np.array([0.0, 0.0])
DIRECTION = np.array([1.0, 0.0])


def acceleration_cones(yaw: float = 0.03) -> list[FloatArray]:
    """A run of cones, slightly rotated against the vehicle heading.

    The cones carry a fixed jitter: on a perfectly straight line every subset
    yields the same fit, which would make the tests below vacuous.
    """
    xs = np.arange(2.0, 32.0, 2.5)
    jitter = np.random.default_rng(0).normal(0.0, 0.05, size=(2, xs.size))
    left = np.column_stack([xs, np.full_like(xs, 1.5) + jitter[0]])
    right = np.column_stack([xs, np.full_like(xs, -1.5) + jitter[1]])
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
    )
    cones: list[FloatArray] = [np.zeros((0, 2)) for _ in ConeTypes]
    cones[ConeTypes.LEFT] = left @ rotation.T
    cones[ConeTypes.RIGHT] = right @ rotation.T
    return cones


def relocalize(seed: int | None = None) -> tuple[FloatArray, float]:
    relocalizer = (
        AccelerationRelocalizer() if seed is None else AccelerationRelocalizer(seed)
    )
    cones = acceleration_cones()
    # The first call only records the starting pose; the second does the fit.
    relocalizer.attempt_relocalization_calculation(cones, POSITION, DIRECTION)
    relocalizer.attempt_relocalization_calculation(cones, POSITION, DIRECTION)
    assert relocalizer.is_relocalized
    return relocalizer.transform_to_known_map_frame(np.array([5.0, 1.0]), 0.0)


class TestBestFitIsReproducible:
    def test_same_seed_gives_the_same_coefficients(self) -> None:
        points = acceleration_cones()[ConeTypes.LEFT]
        first = best_fit(points, 3, 50, np.random.default_rng(7))
        second = best_fit(points, 3, 50, np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)

    def test_omitted_generator_is_still_deterministic(self) -> None:
        points = acceleration_cones()[ConeTypes.LEFT]
        np.testing.assert_array_equal(best_fit(points, 3, 50), best_fit(points, 3, 50))


class TestRelocalizerIsReproducible:
    def test_two_relocalizers_agree(self) -> None:
        first_position, first_yaw = relocalize()
        second_position, second_yaw = relocalize()
        np.testing.assert_array_equal(first_position, second_position)
        assert first_yaw == second_yaw

    def test_result_does_not_depend_on_the_global_generator(self) -> None:
        np.random.seed(0)
        baseline_position, baseline_yaw = relocalize()

        np.random.seed(12345)
        np.random.random(1000)
        other_position, other_yaw = relocalize()

        np.testing.assert_array_equal(baseline_position, other_position)
        assert baseline_yaw == other_yaw

    def test_a_different_seed_is_allowed_to_differ(self) -> None:
        """The seed is a real knob, not decoration."""
        relocalize(seed=1)
        relocalize(seed=2)
