"""The nearby-cone-search cache must key on the search geometry it was built for."""

from __future__ import annotations

import numpy as np

from fsd_path_planning.sorting_cones.trace_sorter.nearby_cone_search import (
    NearbyConeSearcher,
)
from fsd_path_planning.utils.cone_types import ConeTypes


def _cones() -> np.ndarray:
    left = [[float(x), 2.0, float(ConeTypes.LEFT)] for x in range(0, 9, 3)]
    right = [[float(x), -2.0, float(ConeTypes.RIGHT)] for x in range(0, 9, 3)]
    return np.array(left + right, dtype=float)


def _configs() -> np.ndarray:
    return np.array([[0, 1, 2]], dtype=int)


class TestCacheIdentity:
    def test_same_arguments_reuse_one_entry(self) -> None:
        searcher = NearbyConeSearcher()
        cones = _cones()
        first = searcher.get_caches(cones, ConeTypes.LEFT, 6.0, np.pi / 1.5)
        second = searcher.get_caches(cones, ConeTypes.LEFT, 6.0, np.pi / 1.5)
        assert first is second
        assert len(searcher.caches_cache) == 1

    def test_different_search_angle_gets_its_own_entry(self) -> None:
        searcher = NearbyConeSearcher()
        cones = _cones()
        wide = searcher.get_caches(cones, ConeTypes.LEFT, 6.0, np.pi / 1.5)
        narrow = searcher.get_caches(cones, ConeTypes.LEFT, 6.0, np.pi / 6)
        assert wide is not narrow
        assert len(searcher.caches_cache) == 2

    def test_different_search_distance_gets_its_own_entry(self) -> None:
        searcher = NearbyConeSearcher()
        cones = _cones()
        near = searcher.get_caches(cones, ConeTypes.LEFT, 6.0, np.pi / 1.5)
        far = searcher.get_caches(cones, ConeTypes.LEFT, 12.0, np.pi / 1.5)
        assert near is not far

    def test_different_cone_type_gets_its_own_entry(self) -> None:
        searcher = NearbyConeSearcher()
        cones = _cones()
        left = searcher.get_caches(cones, ConeTypes.LEFT, 6.0, np.pi / 1.5)
        right = searcher.get_caches(cones, ConeTypes.RIGHT, 6.0, np.pi / 1.5)
        assert left is not right


class TestSearchAngleIsHonoured:
    def test_narrow_angle_sees_no_more_cones_than_a_wide_one(self) -> None:
        """A reused cache from a wider search would report too many cones."""
        searcher = NearbyConeSearcher()
        cones, configs = _cones(), _configs()

        wide_good, wide_bad = searcher.number_of_cones_on_each_side_for_each_config(
            cones, configs, ConeTypes.LEFT, 6.0, np.pi / 1.5
        )
        narrow_good, narrow_bad = searcher.number_of_cones_on_each_side_for_each_config(
            cones, configs, ConeTypes.LEFT, 6.0, np.deg2rad(1)
        )

        assert narrow_good.sum() <= wide_good.sum()
        assert narrow_bad.sum() <= wide_bad.sum()

    def test_result_does_not_depend_on_call_order(self) -> None:
        cones, configs = _cones(), _configs()

        def run(angles: list[float]) -> list[tuple[int, int]]:
            searcher = NearbyConeSearcher()
            out = []
            for angle in angles:
                good, bad = searcher.number_of_cones_on_each_side_for_each_config(
                    cones, configs, ConeTypes.LEFT, 6.0, angle
                )
                out.append((int(good.sum()), int(bad.sum())))
            return out

        wide, narrow = np.pi / 1.5, np.deg2rad(1)
        forwards = run([wide, narrow])
        backwards = list(reversed(run([narrow, wide])))
        assert forwards == backwards
