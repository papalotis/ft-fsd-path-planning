"""Shared fixtures for the fsd_path_planning test suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fsd_path_planning import MissionTypes, PathPlanner

DEMO_DIR = Path(__file__).resolve().parent.parent / "fsd_path_planning" / "demo"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden_data"


def load_data_json(
    data_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    """Load a demo JSON file. Replicates logic from fsd_path_planning.demo.json_demo."""
    data = json.loads(data_path.read_text())
    positions = np.array([d["car_position"] for d in data])
    directions = np.array([d["car_direction"] for d in data])
    cone_observations = [
        [np.array(c).reshape(-1, 2) for c in d["slam_cones"]] for d in data
    ]
    return positions, directions, cone_observations


# ---------------------------------------------------------------------------
# Demo data fixtures (session-scoped – loaded once for the whole test run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fsg_data():
    """FSG 19 trackdrive – 2 laps."""
    return load_data_json(DEMO_DIR / "fsg_19_2_laps.json")


@pytest.fixture(scope="session")
def fss_data():
    """FSS 19 trackdrive – 4 laps."""
    return load_data_json(DEMO_DIR / "fss_19_4_laps.json")


@pytest.fixture(scope="session")
def skidpad_data():
    """Skidpad dataset."""
    return load_data_json(DEMO_DIR / "skidpad.json")


# ---------------------------------------------------------------------------
# PathPlanner fixtures (session-scoped – amortises JIT warmup)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def planner_trackdrive(fsg_data):
    """Session-scoped trackdrive PathPlanner, warmed up on the first frame."""
    planner = PathPlanner(MissionTypes.trackdrive)
    positions, directions, cones = fsg_data
    # warmup call to trigger JIT compilation
    planner.calculate_path_in_global_frame(cones[0], positions[0], directions[0])
    return planner


@pytest.fixture(scope="session")
def planner_skidpad(skidpad_data):
    """Session-scoped skidpad PathPlanner, warmed up on the first frame."""
    planner = PathPlanner(MissionTypes.skidpad)
    positions, directions, cones = skidpad_data
    planner.calculate_path_in_global_frame(cones[0], positions[0], directions[0])
    return planner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sample_frame_indices(total: int, step: int = 10) -> list[int]:
    """Return indices: first, last, and every `step`-th frame."""
    indices = list(range(0, total, step))
    if (total - 1) not in indices:
        indices.append(total - 1)
    return sorted(set(indices))
