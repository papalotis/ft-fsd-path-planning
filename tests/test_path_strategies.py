"""Path calculation strategies: which basis is chosen, and how it is fitted."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fsd_path_planning.calculate_path.core_calculate_path import (
    CalculatePath,
    PathCalculationInput,
)
from fsd_path_planning.calculate_path.skidpad_calculate_path import (
    SkidpadCalculatePath,
)
from fsd_path_planning.calculate_path.strategies import (
    DEFAULT_BASIS_STRATEGIES,
    FollowKnownMap,
    KeepPrevious,
    KnownMapWindow,
    MatchCenterline,
    SplineThroughBasis,
)
from fsd_path_planning.config_dataclasses import PathConfig

POSITION = np.array([0.0, 0.0])
DIRECTION = np.array([1.0, 0.0])


def straight_input(n_cones: int = 8, global_path=None) -> PathCalculationInput:
    xs = np.arange(n_cones, dtype=float) * 3.0
    left = np.column_stack([xs, np.full(n_cones, 1.5)])
    right = np.column_stack([xs, np.full(n_cones, -1.5)])
    matches = np.arange(n_cones)
    return PathCalculationInput(
        left, right, matches, matches.copy(), POSITION, DIRECTION, global_path
    )


def circle_path(radius: float = 30.0, n: int = 600) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.sin(t), radius - radius * np.cos(t)])


class TestDefaults:
    def test_default_strategies(self) -> None:
        calculator = CalculatePath()
        assert calculator.basis_strategies == DEFAULT_BASIS_STRATEGIES
        assert [type(s) for s in calculator.basis_strategies] == [
            KnownMapWindow,
            KeepPrevious,
            MatchCenterline,
        ]
        assert isinstance(calculator.fit_strategy, SplineThroughBasis)

    def test_skidpad_follows_the_known_map(self) -> None:
        calculator = SkidpadCalculatePath()
        assert isinstance(calculator.fit_strategy, FollowKnownMap)
        calculator.index_along_path = 7
        assert calculator.fit_strategy.index_along_path == 7


class TestBasisChoice:
    def test_matched_cones_give_the_centreline(self) -> None:
        calculator = CalculatePath()
        calculator.input = straight_input()
        basis = calculator.select_path_basis()
        np.testing.assert_allclose(basis[:, 1], 0.0, atol=1e-9)

    def test_too_few_cones_keep_the_previous_path(self) -> None:
        calculator = CalculatePath()
        calculator.input = straight_input(n_cones=2)
        basis = calculator.select_path_basis()
        np.testing.assert_array_equal(basis, calculator.previous_paths[-1][:, 1:3])

    def test_a_known_map_wins_over_cones(self) -> None:
        global_path = circle_path()
        calculator = CalculatePath()
        calculator.input = straight_input(global_path=global_path)
        basis = calculator.select_path_basis()
        assert len(basis) > 0
        assert np.linalg.norm(basis - POSITION, axis=1).max() < 30.0

    def test_window_radius_comes_from_the_config(self) -> None:
        global_path = circle_path()
        calculator = CalculatePath(
            config=replace(PathConfig(), known_map_window_radius=10.0)
        )
        calculator.input = straight_input(global_path=global_path)
        basis = calculator.select_path_basis()
        assert np.linalg.norm(basis - POSITION, axis=1).max() < 10.0

    def test_custom_strategies_are_used_in_order(self) -> None:
        marker = np.full((5, 2), 42.0)

        class Always:
            def basis(self, calculator, stage_input):
                return marker

        calculator = CalculatePath(basis_strategies=[Always(), MatchCenterline()])
        calculator.input = straight_input()
        assert calculator.select_path_basis() is marker


class TestFollowKnownMap:
    def test_follows_from_the_closest_point(self) -> None:
        global_path = circle_path()
        calculator = SkidpadCalculatePath()
        calculator.input = straight_input(global_path=global_path)
        update = calculator.fit_matches_as_spline(np.zeros((0, 2)))
        np.testing.assert_array_equal(update[0], global_path[0])
        length = np.linalg.norm(np.diff(update, axis=0), axis=1).sum()
        assert abs(length - PathConfig().known_map_length_ahead) < 1.0

    def test_without_a_known_map_it_uses_the_trivial_path(self) -> None:
        calculator = SkidpadCalculatePath()
        calculator.input = straight_input()
        update = calculator.fit_matches_as_spline(np.zeros((0, 2)))
        np.testing.assert_array_equal(update, calculator.calculate_trivial_path())
