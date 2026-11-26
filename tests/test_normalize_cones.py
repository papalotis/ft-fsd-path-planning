"""Both accepted cone formats must arrive at the pipeline as one per-type list."""

from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner, normalize_cones
from fsd_path_planning.types import FloatArray


def per_type_cones() -> list[FloatArray]:
    xs = np.arange(0.0, 21.0, 3.0)
    cones: list[FloatArray] = [np.zeros((0, 2)) for _ in ConeTypes]
    cones[ConeTypes.LEFT] = np.column_stack([xs, np.full_like(xs, 2.0)])
    cones[ConeTypes.RIGHT] = np.column_stack([xs, np.full_like(xs, -2.0)])
    cones[ConeTypes.UNKNOWN] = np.array([[9.0, 0.5]])
    return cones


def flat_cones(cones: list[FloatArray]) -> FloatArray:
    rows = [
        np.column_stack([array, np.full(len(array), float(cone_type))])
        for cone_type, array in enumerate(cones)
        if len(array)
    ]
    return np.vstack(rows)


class TestPerTypeInput:
    def test_passes_through_unchanged(self) -> None:
        cones = per_type_cones()
        for expected, actual in zip(cones, normalize_cones(cones), strict=True):
            np.testing.assert_array_equal(expected, actual)

    def test_wrong_number_of_arrays_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="5 arrays"):
            normalize_cones([np.zeros((0, 2))] * 4)

    def test_a_five_by_two_array_is_still_one_cone_per_type(self) -> None:
        """A (5, 2) array means one cone of each type, not a flat array."""
        result = normalize_cones(np.zeros((5, 2)))
        assert len(result) == len(ConeTypes)
        assert all(array.shape == (1, 2) for array in result)


class TestFlatInput:
    def test_splits_into_one_array_per_type(self) -> None:
        cones = per_type_cones()
        result = normalize_cones(flat_cones(cones))

        assert len(result) == len(ConeTypes)
        for expected, actual in zip(cones, result, strict=True):
            assert actual.shape == expected.shape
            np.testing.assert_array_equal(expected, actual)

    def test_keeps_input_order_within_a_type(self) -> None:
        flat = np.array(
            [
                [3.0, 2.0, float(ConeTypes.LEFT)],
                [0.0, -2.0, float(ConeTypes.RIGHT)],
                [0.0, 2.0, float(ConeTypes.LEFT)],
            ]
        )
        result = normalize_cones(flat)
        np.testing.assert_array_equal(
            result[ConeTypes.LEFT], np.array([[3.0, 2.0], [0.0, 2.0]])
        )

    def test_integer_dtype_is_accepted(self) -> None:
        flat = np.array([[0, 2, int(ConeTypes.LEFT)], [0, -2, int(ConeTypes.RIGHT)]])
        result = normalize_cones(flat)
        assert result[ConeTypes.LEFT].dtype == np.float64

    def test_absent_types_become_empty_arrays(self) -> None:
        flat = np.array([[0.0, 2.0, float(ConeTypes.LEFT)]])
        result = normalize_cones(flat)
        assert result[ConeTypes.RIGHT].shape == (0, 2)

    @pytest.mark.parametrize("bad_type", [-1.0, 5.0, 99.0])
    def test_unknown_cone_type_is_rejected(self, bad_type: float) -> None:
        with pytest.raises(ValueError, match="ConeTypes values"):
            normalize_cones(np.array([[0.0, 2.0, bad_type]]))

    def test_fractional_cone_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="whole ConeTypes"):
            normalize_cones(np.array([[0.0, 2.0, 1.5]]))

    def test_non_finite_values_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            normalize_cones(np.array([[0.0, np.nan, float(ConeTypes.LEFT)]]))


class TestPipelineAcceptsBothFormats:
    @pytest.mark.parametrize(
        "mission",
        [MissionTypes.trackdrive, MissionTypes.acceleration, MissionTypes.skidpad],
    )
    def test_flat_input_does_not_raise(self, mission: MissionTypes) -> None:
        """Acceleration used to fail on flat input with a broadcasting error."""
        planner = PathPlanner(mission)
        path = planner.calculate_path_in_global_frame(
            flat_cones(per_type_cones()), np.array([-1.0, 0.0]), np.array([1.0, 0.0])
        )
        assert path.ndim == 2
        assert path.shape[1] == 4

    def test_trackdrive_agrees_between_the_two_formats(self) -> None:
        cones = per_type_cones()
        position, direction = np.array([-1.0, 0.0]), np.array([1.0, 0.0])

        from_list = PathPlanner(MissionTypes.trackdrive).calculate_path_in_global_frame(
            cones, position, direction
        )
        from_flat = PathPlanner(MissionTypes.trackdrive).calculate_path_in_global_frame(
            flat_cones(cones), position, direction
        )
        np.testing.assert_allclose(from_list, from_flat)
