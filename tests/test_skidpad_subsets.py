"""Skidpad relocalization: the sampled larger subsets and the start check."""

from __future__ import annotations

from dataclasses import replace
from math import comb

import numpy as np
import pytest

from fsd_path_planning.config_dataclasses import SkidpadConfig
from fsd_path_planning.geometry.angles import unit_2d_vector_from_angle
from fsd_path_planning.stages.relocalization.skidpad import (
    _sample_subsets,
    centers_fit_start_position,
    circle_fit_powerset,
)


def circle_cones(center, radius=7.625, n=12, start=0.0, span=2.0):
    angles = start + np.linspace(0.0, span, n)
    return unit_2d_vector_from_angle(angles) * radius + center


class TestSampleSubsets:
    def test_draws_the_requested_number_of_distinct_sorted_subsets(self) -> None:
        subsets = _sample_subsets(20, 4, 100, np.random.default_rng(0))
        assert len(subsets) == len(set(subsets)) == 100
        assert all(list(s) == sorted(s) and len(s) == 4 for s in subsets)
        assert all(0 <= i < 20 for s in subsets for i in s)

    def test_all_of_them_when_there_are_few(self) -> None:
        subsets = _sample_subsets(7, 5, 100, np.random.default_rng(0))
        assert len(subsets) == comb(7, 5)

    def test_same_seed_same_sample(self) -> None:
        a = _sample_subsets(20, 5, 50, np.random.default_rng(3))
        b = _sample_subsets(20, 5, 50, np.random.default_rng(3))
        assert a == b

    def test_the_next_draw_is_a_new_sample(self) -> None:
        rng = np.random.default_rng(3)
        assert _sample_subsets(20, 5, 50, rng) != _sample_subsets(20, 5, 50, rng)

    @pytest.mark.parametrize(("n", "size", "count"), [(3, 4, 10), (20, 4, 0)])
    def test_nothing_to_draw(self, n, size, count) -> None:
        assert _sample_subsets(n, size, count, np.random.default_rng(0)) == []


class TestCircleFitPowerset:
    def test_default_tries_triples_only(self) -> None:
        points = circle_cones(np.array([0.0, 9.125]), n=8)
        result = circle_fit_powerset(points, SkidpadConfig())
        assert result
        assert {len(subset) for _, subset in result} == {3}

    def test_larger_subsets_when_asked(self) -> None:
        points = circle_cones(np.array([0.0, 9.125]), n=8)
        config = SkidpadConfig(max_powerset_size=5)
        sizes = {len(s) for _, s in circle_fit_powerset(points, config)}
        assert sizes == {3, 4, 5}

    def test_the_triples_are_unchanged_by_the_larger_subsets(self) -> None:
        points = circle_cones(np.array([0.0, 9.125]), n=8)
        triples = circle_fit_powerset(points, SkidpadConfig())
        with_more = circle_fit_powerset(points, SkidpadConfig(max_powerset_size=5))
        first = with_more[: len(triples)]
        for (circle_a, subset_a), (circle_b, subset_b) in zip(
            triples, first, strict=True
        ):
            np.testing.assert_array_equal(subset_a, subset_b)
            np.testing.assert_array_equal(circle_a, circle_b)


class TestStartCheck:
    START = np.array([0.0, 0.0])
    DIRECTION = np.array([1.0, 0.0])
    CONFIG = SkidpadConfig(check_start_position=True)

    def check(self, centers) -> bool:
        return centers_fit_start_position(
            np.array(centers, dtype=float), self.START, self.DIRECTION, self.CONFIG
        )

    def test_the_bundled_runs_true_centres_pass(self) -> None:
        """The skidpad run's centres, in the car's starting frame."""
        assert self.check([[16.64, -9.18], [17.11, 9.07]])

    @pytest.mark.parametrize(
        "centers",
        [
            [[2.85, -8.42], [2.39, 9.83]],  # circles through the entry lane cones
            [[16.54, -9.07], [0.8, 0.18]],  # one centre on the car
            [[16.6, 9.1], [17.1, 27.3]],  # both on one side
            [[45.0, -9.1], [45.0, 9.1]],  # too far ahead
            [[-16.6, -9.1], [-17.1, 9.1]],  # behind the car
        ],
    )
    def test_implausible_centres_fail(self, centers) -> None:
        assert not self.check(centers)

    def test_tolerances_are_validated(self) -> None:
        with pytest.raises(ValueError):
            replace(SkidpadConfig(), min_start_distance=30.0, max_start_distance=10.0)
        with pytest.raises(ValueError):
            SkidpadConfig(start_lateral_tolerance=0.0)
