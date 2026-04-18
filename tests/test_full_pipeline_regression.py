"""Full pipeline regression tests against golden snapshot data.

These tests run the complete PathPlanner pipeline on sampled frames from each
demo JSON dataset and compare every intermediate result against previously
recorded golden snapshots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fsd_path_planning import MissionTypes, PathPlanner
from tests.conftest import DEMO_DIR, GOLDEN_DIR, load_data_json, sample_frame_indices


def _load_golden(stem: str):
    path = GOLDEN_DIR / f"{stem}.npz"
    if not path.exists():
        pytest.skip(f"Golden data not found: {path}. Run tests/generate_golden_data.py first.")
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
                cones[i], positions[i], directions[i],
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
                    actual, expected, atol=1e-10,
                    err_msg=f"FSG frame {frame_idx}, key '{key}'",
                )


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
                cones[i], positions[i], directions[i],
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
                np.testing.assert_allclose(
                    actual, expected, atol=1e-10,
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
                cones[i], positions[i], directions[i],
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
                    actual, expected, atol=1e-10,
                    err_msg=f"Skidpad frame {frame_idx}, key '{key}'",
                )
