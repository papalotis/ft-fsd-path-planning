#!/usr/bin/env python3
"""Run the planner over the demo datasets and dump every stage output to an .npz.

This script is executed as a subprocess, once per git ref under test, with the
ref's checkout prepended to ``sys.path``. It must therefore stay dependency-free
beyond the library itself and must not import anything from ``tools``.

Usage:
    python _collect.py <package_root> <datasets_dir> <output.npz> <spec_json>

``spec_json`` is a JSON list of run specifications, each a mapping with the keys
``dataset``, ``mission``, ``no_color`` and ``cache``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Stage outputs returned by calculate_path_in_global_frame(..., True), in order.
STAGE_NAMES = (
    "path",
    "sorted_left",
    "sorted_right",
    "left_with_virtual",
    "right_with_virtual",
    "left_to_right",
    "right_to_left",
)


def load_dataset(path: Path, no_color: bool):
    data = json.loads(path.read_text())
    positions = np.array([frame["car_position"] for frame in data], dtype=float)
    directions = np.array([frame["car_direction"] for frame in data], dtype=float)

    observations = []
    for frame in data:
        cones = [np.array(c, dtype=float).reshape(-1, 2) for c in frame["slam_cones"]]
        if no_color:
            merged = np.vstack(cones) if cones else np.zeros((0, 2))
            cones = [merged] + [np.zeros((0, 2)) for _ in range(len(cones) - 1)]
        observations.append(cones)

    return positions, directions, observations


def run_one(spec, package_root: Path, datasets_dir: Path, store: dict) -> dict:
    from fsd_path_planning import MissionTypes, PathPlanner

    mission = getattr(MissionTypes, spec["mission"])
    dataset = Path(spec["dataset"])
    positions, directions, observations = load_dataset(
        datasets_dir / dataset, spec["no_color"]
    )

    def make_planner():
        # The experimental cache flag is the second positional parameter on every
        # revision of PathPlanner we compare, so pass it positionally.
        return PathPlanner(mission, spec["cache"])

    # Warm up the JIT on a throwaway planner so timings exclude compilation.
    warmup = make_planner()
    warmup.calculate_path_in_global_frame(observations[0], positions[0], directions[0])

    np.random.seed(42)
    planner = make_planner()
    key = spec["key"]
    durations = []
    relocalized_at = -1

    for i, (position, direction, cones) in enumerate(
        zip(positions, directions, observations, strict=True)
    ):
        start = time.perf_counter()
        outputs = planner.calculate_path_in_global_frame(
            cones, position, direction, return_intermediate_results=True
        )
        durations.append(time.perf_counter() - start)

        for name, value in zip(STAGE_NAMES, outputs, strict=True):
            store[f"{key}|{i}|{name}"] = np.asarray(value)

        if relocalized_at < 0 and getattr(planner, "relocalization_info", None):
            relocalized_at = i

    store[f"{key}|meta|durations_ms"] = np.array(durations) * 1000
    store[f"{key}|meta|relocalized_at"] = np.array(relocalized_at)
    store[f"{key}|meta|n_frames"] = np.array(len(positions))
    return store


def main() -> int:
    package_root, datasets_dir, output, spec_json = sys.argv[1:5]
    sys.path.insert(0, package_root)

    specs = json.loads(spec_json)
    store: dict = {}
    for spec in specs:
        run_one(spec, Path(package_root), Path(datasets_dir), store)

    np.savez_compressed(output, **store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
