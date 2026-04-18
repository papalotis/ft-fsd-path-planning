#!/usr/bin/env python3
"""Generate golden snapshot data for regression tests.

Runs the full pipeline on each demo JSON file, captures all intermediate results
at sampled frames, and saves them as .npz files in tests/golden_data/.

Usage:
    python tests/generate_golden_data.py
    python tests/generate_golden_data.py --regenerate   # overwrite existing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fsd_path_planning import MissionTypes, PathPlanner
from tests.conftest import DEMO_DIR, GOLDEN_DIR, load_data_json, sample_frame_indices

# Dataset definitions: (json filename, mission type)
DATASETS = [
    ("fsg_19_2_laps.json", MissionTypes.trackdrive),
    ("fss_19_4_laps.json", MissionTypes.trackdrive),
    ("skidpad.json", MissionTypes.skidpad),
]


def generate_golden_for_dataset(
    json_name: str,
    mission: MissionTypes,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    data_path = DEMO_DIR / json_name
    stem = data_path.stem  # e.g. "fsg_19_2_laps"
    out_file = output_dir / f"{stem}.npz"

    if out_file.exists() and not overwrite:
        print(f"  SKIP {out_file} (exists, use --regenerate to overwrite)")
        return

    print(f"  Loading {data_path.name} ...")
    positions, directions, cone_observations = load_data_json(data_path)
    n_frames = len(positions)
    frame_indices = sample_frame_indices(n_frames, step=10)

    print(f"  {n_frames} total frames, sampling {len(frame_indices)} frames")
    print(f"  Mission: {mission.name}")

    # Seed for determinism (acceleration relocalizer uses np.random)
    np.random.seed(42)

    # Create and warm up the planner
    planner = PathPlanner(mission)
    planner.calculate_path_in_global_frame(
        cone_observations[0], positions[0], directions[0]
    )

    # We need to process frames sequentially because some planners (skidpad)
    # maintain internal state (relocalization). Process ALL frames but only
    # store sampled ones.
    stored: dict[str, np.ndarray] = {}
    stored["frame_indices"] = np.array(frame_indices)

    # Reset planner for clean run
    np.random.seed(42)
    planner = PathPlanner(mission)

    for i in range(n_frames):
        result = planner.calculate_path_in_global_frame(
            cone_observations[i],
            positions[i],
            directions[i],
            return_intermediate_results=True,
        )

        if i in frame_indices:
            (
                final_path,
                sorted_left,
                sorted_right,
                left_cones_with_virtual,
                right_cones_with_virtual,
                left_to_right_match,
                right_to_left_match,
            ) = result

            prefix = f"frame_{i:04d}"
            stored[f"{prefix}_final_path"] = final_path
            stored[f"{prefix}_sorted_left"] = sorted_left
            stored[f"{prefix}_sorted_right"] = sorted_right
            stored[f"{prefix}_left_cones_with_virtual"] = left_cones_with_virtual
            stored[f"{prefix}_right_cones_with_virtual"] = right_cones_with_virtual
            stored[f"{prefix}_left_to_right_match"] = left_to_right_match
            stored[f"{prefix}_right_to_left_match"] = right_to_left_match

    np.savez_compressed(out_file, **stored)
    print(f"  Saved {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden regression data")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Overwrite existing golden data files",
    )
    args = parser.parse_args()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    for json_name, mission in DATASETS:
        print(f"\n=== {json_name} ===")
        generate_golden_for_dataset(json_name, mission, GOLDEN_DIR, args.regenerate)

    print("\nDone.")


if __name__ == "__main__":
    main()
