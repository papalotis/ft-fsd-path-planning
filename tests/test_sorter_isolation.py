"""Two sorters must not influence each other through shared caches."""

from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning.common_types import FloatArray
from fsd_path_planning.stages.sorting import adjacency as adjacency_matrix
from fsd_path_planning.stages.sorting import nearby_cones as nearby_cone_search
from fsd_path_planning.stages.sorting.trace_sorter import TraceSorter
from fsd_path_planning.utils.cone_types import ConeTypes


def make_sorter() -> TraceSorter:
    return TraceSorter(
        max_n_neighbors=5,
        max_dist=6.5,
        max_dist_to_first=6.0,
        max_length=12,
        threshold_directional_angle=np.deg2rad(40),
        threshold_absolute_angle=np.deg2rad(65),
    )


def straight_track(offset: float = 0.0, spacing: float = 3.0) -> list[FloatArray]:
    xs = np.arange(0.0, 7 * spacing, spacing) + offset
    left = np.column_stack([xs, np.full_like(xs, 2.0)])
    right = np.column_stack([xs, np.full_like(xs, -2.0)])
    cones: list[FloatArray] = [np.zeros((0, 2)) for _ in ConeTypes]
    cones[ConeTypes.LEFT] = left
    cones[ConeTypes.RIGHT] = right
    return cones


def curved_track() -> list[FloatArray]:
    angles = np.linspace(0.0, np.pi / 3, 7)
    inner = np.column_stack([np.cos(angles) * 9, np.sin(angles) * 9])
    outer = np.column_stack([np.cos(angles) * 12, np.sin(angles) * 12])
    cones: list[FloatArray] = [np.zeros((0, 2)) for _ in ConeTypes]
    cones[ConeTypes.LEFT] = outer - [11.0, 0.0]
    cones[ConeTypes.RIGHT] = inner - [11.0, 0.0]
    return cones


POSITION = np.array([-1.0, 0.0])
DIRECTION = np.array([1.0, 0.0])


class TestNoSharedModuleState:
    @pytest.mark.parametrize(
        "module, name",
        [
            (adjacency_matrix, "_DEFAULT_CACHE"),
            (adjacency_matrix, "LAST_MATRIX_CALC_HASH"),
            (adjacency_matrix, "LAST_IDXS_CALCULATED"),
            (nearby_cone_search, "NEARBY_CONE_SEARCH_CACHE"),
        ],
    )
    def test_process_wide_cache_is_gone(self, module, name: str) -> None:
        assert not hasattr(module, name)

    def test_each_sorter_owns_its_caches(self) -> None:
        first, second = make_sorter(), make_sorter()
        assert first.adjacency_cache is not second.adjacency_cache
        assert first.nearby_searcher is not second.nearby_searcher


class TestInterleavedSortersAgree:
    def test_second_sorter_does_not_change_the_first(self) -> None:
        tracks = [straight_track(), curved_track(), straight_track(offset=0.4)]

        alone = make_sorter()
        expected = [alone.sort_left_right(t, POSITION, DIRECTION) for t in tracks]

        interleaved = make_sorter()
        other = make_sorter()
        actual = []
        for track in tracks:
            other.sort_left_right(curved_track(), POSITION, DIRECTION)
            actual.append(interleaved.sort_left_right(track, POSITION, DIRECTION))

        for (exp_left, exp_right), (act_left, act_right) in zip(
            expected, actual, strict=True
        ):
            np.testing.assert_array_equal(exp_left, act_left)
            np.testing.assert_array_equal(exp_right, act_right)

    def test_a_fresh_sorter_matches_a_reused_one(self) -> None:
        track = straight_track()
        reused = make_sorter()
        reused.sort_left_right(curved_track(), POSITION, DIRECTION)

        from_reused = reused.sort_left_right(track, POSITION, DIRECTION)
        from_fresh = make_sorter().sort_left_right(track, POSITION, DIRECTION)

        np.testing.assert_array_equal(from_reused[0], from_fresh[0])
        np.testing.assert_array_equal(from_reused[1], from_fresh[1])
