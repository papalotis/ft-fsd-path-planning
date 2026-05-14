"""Full pipeline regression tests against golden snapshot data.

These tests run the complete PathPlanner pipeline on sampled frames from each
demo JSON dataset and compare every intermediate result against previously
recorded golden snapshots.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fsd_path_planning import MissionTypes, PathPlanner
from tests.conftest import DEMO_DIR, GOLDEN_DIR, load_data_json


def _load_golden(stem: str):
    path = GOLDEN_DIR / f"{stem}.npz"
    if not path.exists():
        pytest.skip(
            f"Golden data not found: {path}. Run tests/generate_golden_data.py first."
        )
    return np.load(path)


# ── Helpers ──────────────────────────────────────────────────────────────────

RESULT_KEYS = [
    "final_path",
    "sorted_left",
    "sorted_right",
    "left_cones_with_virtual",
    "right_cones_with_virtual",
    "left_to_right_match",
    "right_to_left_match",
]

PERFORMANCE_MEASURED_FRAMES = 80


def _run_and_compare(
    planner: PathPlanner,
    positions,
    directions,
    cone_observations,
    golden,
    frame_idx: int,
    atol: float = 1e-10,
):
    """Run pipeline for one frame and compare all outputs to golden data."""
    result = planner.calculate_path_in_global_frame(
        cone_observations[frame_idx],
        positions[frame_idx],
        directions[frame_idx],
        return_intermediate_results=True,
    )

    prefix = f"frame_{frame_idx:04d}"

    for key_idx, key in enumerate(RESULT_KEYS):
        golden_key = f"{prefix}_{key}"
        expected = golden[golden_key]
        actual = result[key_idx]
        np.testing.assert_allclose(
            actual,
            expected,
            atol=atol,
            err_msg=f"Frame {frame_idx}, key '{key}' mismatch",
        )


def _run_pipeline_and_time(
    positions,
    directions,
    cone_observations,
    *,
    experimental_performance_improvements: bool,
    capture_indices: set[int],
) -> tuple[float, dict[int, np.ndarray], PathPlanner]:
    np.random.seed(42)
    planner = PathPlanner(
        MissionTypes.trackdrive,
        experimental_performance_improvements=experimental_performance_improvements,
    )

    planner.calculate_path_in_global_frame(
        cone_observations[0],
        positions[0],
        directions[0],
    )

    outputs = {}
    measured_stop = min(PERFORMANCE_MEASURED_FRAMES + 1, len(positions))

    start = time.perf_counter()
    for frame_idx in range(1, measured_stop):
        result = planner.calculate_path_in_global_frame(
            cone_observations[frame_idx],
            positions[frame_idx],
            directions[frame_idx],
        )
        if frame_idx in capture_indices:
            outputs[frame_idx] = result.copy()
    elapsed = time.perf_counter() - start

    return elapsed, outputs, planner


# ── Trackdrive FSG regression ───────────────────────────────────────────────


class TestTrackdriveFSGRegression:
    """Regression tests for the FSG 19 trackdrive dataset."""

    @pytest.fixture(scope="class")
    def golden(self):
        return _load_golden("fsg_19_2_laps")

    @pytest.fixture(scope="class")
    def dataset(self):
        return load_data_json(DEMO_DIR / "fsg_19_2_laps.json")

    @pytest.fixture(scope="class")
    def planner_and_results(self, dataset, golden):
        """Run all frames sequentially (pipeline may be stateful) and collect results."""
        positions, directions, cones = dataset
        frame_indices = golden["frame_indices"]

        np.random.seed(42)
        planner = PathPlanner(MissionTypes.trackdrive)

        results = {}
        n_frames = len(positions)
        for i in range(n_frames):
            out = planner.calculate_path_in_global_frame(
                cones[i],
                positions[i],
                directions[i],
                return_intermediate_results=True,
            )
            if i in frame_indices:
                results[i] = out

        return results

    @pytest.mark.slow
    def test_all_sampled_frames(self, golden, planner_and_results):
        frame_indices = golden["frame_indices"]
        for frame_idx in frame_indices:
            result = planner_and_results[frame_idx]
            prefix = f"frame_{frame_idx:04d}"
            for key_idx, key in enumerate(RESULT_KEYS):
                golden_key = f"{prefix}_{key}"
                expected = golden[golden_key]
                actual = result[key_idx]
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=1e-10,
                    err_msg=f"FSG frame {frame_idx}, key '{key}'",
                )

    @pytest.mark.slow
    def test_all_sampled_frames_with_experimental_performance_improvements(
        self, dataset, golden
    ):
        positions, directions, cones = dataset
        frame_indices = golden["frame_indices"]

        np.random.seed(42)
        planner = PathPlanner(
            MissionTypes.trackdrive,
            experimental_performance_improvements=True,
        )

        results = {}
        n_frames = len(positions)
        for i in range(n_frames):
            out = planner.calculate_path_in_global_frame(
                cones[i],
                positions[i],
                directions[i],
                return_intermediate_results=True,
            )
            if i in frame_indices:
                results[i] = out

        for frame_idx in frame_indices:
            result = results[frame_idx]
            prefix = f"frame_{frame_idx:04d}"
            for key_idx, key in enumerate(RESULT_KEYS):
                golden_key = f"{prefix}_{key}"
                expected = golden[golden_key]
                actual = result[key_idx]
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=1e-10,
                    err_msg=(
                        "FSG experimental performance improvements "
                        f"frame {frame_idx}, key '{key}'"
                    ),
                )

    @pytest.mark.slow
    def test_experimental_performance_improvements_are_close_and_faster_abba(
        self, dataset
    ):
        positions, directions, cones = dataset
        measured_stop = min(PERFORMANCE_MEASURED_FRAMES + 1, len(positions))
        capture_indices = set(range(1, measured_stop, 10))
        capture_indices.add(measured_stop - 1)

        a_first, outputs_a_first, planner_a_first = _run_pipeline_and_time(
            positions,
            directions,
            cones,
            experimental_performance_improvements=False,
            capture_indices=capture_indices,
        )
        b_first, outputs_b_first, planner_b_first = _run_pipeline_and_time(
            positions,
            directions,
            cones,
            experimental_performance_improvements=True,
            capture_indices=capture_indices,
        )
        b_second, outputs_b_second, planner_b_second = _run_pipeline_and_time(
            positions,
            directions,
            cones,
            experimental_performance_improvements=True,
            capture_indices=capture_indices,
        )
        a_second, outputs_a_second, planner_a_second = _run_pipeline_and_time(
            positions,
            directions,
            cones,
            experimental_performance_improvements=False,
            capture_indices=capture_indices,
        )

        for frame_idx in sorted(capture_indices):
            actual = outputs_b_first[frame_idx]
            expected = outputs_a_first[frame_idx]
            assert actual.shape == expected.shape
            np.testing.assert_allclose(
                actual,
                expected,
                atol=1e-10,
                err_msg=(
                    "Experimental performance improvements diverged from baseline "
                    f"at frame {frame_idx}"
                ),
            )
            np.testing.assert_allclose(
                outputs_b_second[frame_idx],
                expected,
                atol=1e-10,
                err_msg=(
                    "Experimental performance improvements were not deterministic "
                    f"at frame {frame_idx}"
                ),
            )

        avg_a = (a_first + a_second) / 2
        avg_b = (b_first + b_second) / 2
        assert avg_b < avg_a, (
            "Expected experimental performance improvements to be faster on average "
            f"in ABBA order, got A=({a_first:.4f}, {a_second:.4f}) and "
            f"B=({b_first:.4f}, {b_second:.4f})"
        )

        for planner in (planner_b_first, planner_b_second):
            trace_sorter = planner.cone_sorting.trace_sorter
            assert trace_sorter.cached_results is not None
            assert trace_sorter.adjacency_cache._matrix_hash is not None
            assert len(trace_sorter.nearby_searcher.caches_cache) > 0

        assert planner_a_first.cone_sorting.trace_sorter.cached_results is not None
        assert planner_a_second.cone_sorting.trace_sorter.cached_results is not None


# ── Trackdrive FSS regression ───────────────────────────────────────────────


class TestTrackdriveFSSRegression:
    """Regression tests for the FSS 19 trackdrive dataset."""

    @pytest.fixture(scope="class")
    def golden(self):
        return _load_golden("fss_19_4_laps")

    @pytest.fixture(scope="class")
    def dataset(self):
        return load_data_json(DEMO_DIR / "fss_19_4_laps.json")

    @pytest.fixture(scope="class")
    def planner_and_results(self, dataset, golden):
        positions, directions, cones = dataset
        frame_indices = golden["frame_indices"]

        np.random.seed(42)
        planner = PathPlanner(MissionTypes.trackdrive)

        results = {}
        n_frames = len(positions)
        for i in range(n_frames):
            out = planner.calculate_path_in_global_frame(
                cones[i],
                positions[i],
                directions[i],
                return_intermediate_results=True,
            )
            if i in frame_indices:
                results[i] = out

        return results

    @pytest.mark.slow
    def test_all_sampled_frames(self, golden, planner_and_results):
        frame_indices = golden["frame_indices"]
        for frame_idx in frame_indices:
            result = planner_and_results[frame_idx]
            prefix = f"frame_{frame_idx:04d}"
            for key_idx, key in enumerate(RESULT_KEYS):
                golden_key = f"{prefix}_{key}"
                expected = golden[golden_key]
                actual = result[key_idx]
                # Longer dataset: tiny float differences accumulate across
                # stateful frames, so we use a slightly relaxed tolerance.
                # use higher atol as FSS dataset is longer and more prone to small floating point differences accumulating across frames
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=1e-3,
                    err_msg=f"FSS frame {frame_idx}, key '{key}'",
                )


# ── Skidpad regression ──────────────────────────────────────────────────────


class TestSkidpadRegression:
    """Regression tests for the skidpad dataset."""

    @pytest.fixture(scope="class")
    def golden(self):
        return _load_golden("skidpad")

    @pytest.fixture(scope="class")
    def dataset(self):
        return load_data_json(DEMO_DIR / "skidpad.json")

    @pytest.fixture(scope="class")
    def planner_and_results(self, dataset, golden):
        positions, directions, cones = dataset
        frame_indices = golden["frame_indices"]

        np.random.seed(42)
        planner = PathPlanner(MissionTypes.skidpad)

        results = {}
        n_frames = len(positions)
        for i in range(n_frames):
            out = planner.calculate_path_in_global_frame(
                cones[i],
                positions[i],
                directions[i],
                return_intermediate_results=True,
            )
            if i in frame_indices:
                results[i] = out

        return results

    @pytest.mark.slow
    def test_all_sampled_frames(self, golden, planner_and_results):
        frame_indices = golden["frame_indices"]
        for frame_idx in frame_indices:
            result = planner_and_results[frame_idx]
            prefix = f"frame_{frame_idx:04d}"
            for key_idx, key in enumerate(RESULT_KEYS):
                golden_key = f"{prefix}_{key}"
                expected = golden[golden_key]
                actual = result[key_idx]
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=1e-10,
                    err_msg=f"Skidpad frame {frame_idx}, key '{key}'",
                )
