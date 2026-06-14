from __future__ import annotations

import warnings

import numpy as np
import pytest

from fsd_path_planning.calculate_path.core_calculate_path import (
    CalculatePath,
    PathCalculationInput,
)
from fsd_path_planning.cone_matching.core_cone_matching import (
    ConeMatching,
    ConeMatchingInput,
)
from fsd_path_planning.config_dataclasses import (
    MatchingConfig,
    PathConfig,
    SortingConfig,
)
from fsd_path_planning.sorting_cones.core_cone_sorting import (
    ConeSorting,
    ConeSortingInput,
)
from fsd_path_planning.types import FloatArray
from fsd_path_planning.utils.cone_types import ConeTypes

API_DEPRECATION_SNIPPETS = (
    "Passing individual parameters",
    "set_new_input() is deprecated",
)


def _make_sorting_input() -> ConeSortingInput:
    left: FloatArray = np.array([[0.0, 2.0], [3.0, 2.0], [6.0, 2.0]], dtype=float)
    right: FloatArray = np.array([[0.0, -2.0], [3.0, -2.0], [6.0, -2.0]], dtype=float)
    cones_by_type: list[FloatArray] = [np.zeros((0, 2), dtype=float) for _ in ConeTypes]
    cones_by_type[ConeTypes.LEFT] = left
    cones_by_type[ConeTypes.RIGHT] = right
    return ConeSortingInput(
        cones_by_type=cones_by_type,
        vehicle_position=np.array([-1.0, 0.0]),
        vehicle_direction=np.array([1.0, 0.0]),
    )


def _make_matching_input() -> ConeMatchingInput:
    left: FloatArray = np.array([[0.0, 1.5], [3.0, 1.5], [6.0, 1.5]], dtype=float)
    right: FloatArray = np.array([[0.0, -1.5], [3.0, -1.5], [6.0, -1.5]], dtype=float)
    sorted_cones: list[FloatArray] = [np.zeros((0, 2), dtype=float) for _ in ConeTypes]
    sorted_cones[ConeTypes.LEFT] = left
    sorted_cones[ConeTypes.RIGHT] = right
    return ConeMatchingInput(
        sorted_cones=sorted_cones,
        vehicle_position=np.array([-1.0, 0.0]),
        vehicle_direction=np.array([1.0, 0.0]),
    )


def _make_path_input() -> PathCalculationInput:
    left = np.array([[0.0, 2.0], [3.0, 2.01], [6.0, 2.04], [9.0, 2.09]])
    right = np.array([[0.0, -2.0], [3.0, -1.99], [6.0, -1.96], [9.0, -1.91]])
    matches = np.arange(len(left), dtype=int)
    return PathCalculationInput(
        left_cones=left,
        right_cones=right,
        left_to_right_matches=matches,
        right_to_left_matches=matches,
        vehicle_position=np.array([-1.0, 0.0]),
        vehicle_direction=np.array([1.0, 0.0]),
    )


class TestDeprecatedConstructors:
    def test_cone_sorting_legacy_constructor_warns_and_works(self):
        with pytest.deprecated_call(
            match="Passing individual parameters to ConeSorting"
        ):
            sorter = ConeSorting(max_n_neighbors=5, max_dist=6.5)

        result = sorter.run_cone_sorting(_make_sorting_input())
        assert result.left_cones.shape[1] == 2

    def test_cone_matching_legacy_constructor_warns_and_works(self):
        with pytest.deprecated_call(
            match="Passing individual parameters to ConeMatching"
        ):
            matcher = ConeMatching(min_track_width=3.0, max_search_range=5.0)

        result = matcher.run_cone_matching(_make_matching_input())
        assert result.left_cones_with_virtual.shape[1] == 2

    def test_calculate_path_legacy_constructor_warns_and_works(self):
        with pytest.deprecated_call(
            match="Passing individual parameters to CalculatePath"
        ):
            calculator = CalculatePath(smoothing=0.2, mpc_path_length=20.0)

        result = calculator.run_path_calculation(_make_path_input())
        assert result.final_path.shape[1] == 4


class TestDeprecatedSetNewInput:
    def test_cone_sorting_set_new_input_warns_and_works(self):
        sorter = ConeSorting(config=SortingConfig())

        with pytest.deprecated_call(match=r"set_new_input\(\) is deprecated"):
            sorter.set_new_input(_make_sorting_input())

        result = sorter.run_cone_sorting()
        assert result.right_cones.shape[1] == 2

    def test_cone_matching_set_new_input_warns_and_works(self):
        matcher = ConeMatching(config=MatchingConfig())

        with pytest.deprecated_call(match=r"set_new_input\(\) is deprecated"):
            matcher.set_new_input(_make_matching_input())

        result = matcher.run_cone_matching()
        assert result.right_to_left_matches.ndim == 1

    def test_calculate_path_set_new_input_warns_and_works(self):
        calculator = CalculatePath(config=PathConfig())

        with pytest.deprecated_call(match=r"set_new_input\(\) is deprecated"):
            calculator.set_new_input(_make_path_input())

        result = calculator.run_path_calculation()
        assert result.centerline_basis.shape[1] == 2


class TestNewApisDoNotWarn:
    def test_cone_sorting_new_api_does_not_warn(self):
        sorter = ConeSorting(config=SortingConfig())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sorter.run_cone_sorting(_make_sorting_input())

        assert not any(
            isinstance(w.message, DeprecationWarning)
            and any(snippet in str(w.message) for snippet in API_DEPRECATION_SNIPPETS)
            for w in caught
        )

        assert result.left_cones.shape[1] == 2

    def test_cone_matching_new_api_does_not_warn(self):
        matcher = ConeMatching(config=MatchingConfig())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = matcher.run_cone_matching(_make_matching_input())

        assert not any(
            isinstance(w.message, DeprecationWarning)
            and any(snippet in str(w.message) for snippet in API_DEPRECATION_SNIPPETS)
            for w in caught
        )

        assert result.left_to_right_matches.ndim == 1

    def test_calculate_path_new_api_does_not_warn(self):
        calculator = CalculatePath(config=PathConfig())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calculator.run_path_calculation(_make_path_input())

        api_deprecations = [
            w
            for w in caught
            if isinstance(w.message, DeprecationWarning)
            and any(snippet in str(w.message) for snippet in API_DEPRECATION_SNIPPETS)
        ]

        assert not api_deprecations

        assert result.final_path.shape[1] == 4
