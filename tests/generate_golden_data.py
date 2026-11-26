#!/usr/bin/env python3
"""Record the golden baseline the slow regression tests compare against.

Runs the full pipeline over every demo dataset, with and without colour
information, and stores every stage output of every tenth frame (plus the
last) in tests/golden_data/. Each file also records the commit and the library
versions it was produced with.

The tests compare with tolerances (1 cm of sideways movement of the planned
line), so a baseline recorded on one machine holds on another. Re-record only
when the output is meant to change, in a commit of its own that says why.

Usage:
    python tests/generate_golden_data.py              # write missing files
    python tests/generate_golden_data.py --regenerate # overwrite all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numba  # noqa: E402
import scipy  # noqa: E402

from fsd_path_planning import MissionTypes, PathPlanner  # noqa: E402
from tests.conftest import (  # noqa: E402
    DEMO_DIR,
    GOLDEN_DIR,
    GOLDEN_RUNS,
    golden_file,
    load_data_json,
    sample_frame_indices,
    strip_colour,
)

RESULT_KEYS = (
    "final_path",
    "sorted_left",
    "sorted_right",
    "left_cones_with_virtual",
    "right_cones_with_virtual",
    "left_to_right_match",
    "right_to_left_match",
)


def current_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def record(stem: str, mission: MissionTypes, colour: bool, overwrite: bool) -> None:
    out_file = golden_file(stem, colour)
    if out_file.exists() and not overwrite:
        print(f"  SKIP {out_file.name} (exists, use --regenerate to overwrite)")
        return

    positions, directions, observations = load_data_json(DEMO_DIR / f"{stem}.json")
    if not colour:
        observations = [strip_colour(cones) for cones in observations]
    frame_indices = sample_frame_indices(len(positions), step=10)
    wanted = set(frame_indices)

    stored: dict[str, np.ndarray] = {
        "frame_indices": np.array(frame_indices),
        "meta_commit": np.array(current_commit()),
        "meta_date": np.array(date.today().isoformat()),
        "meta_versions": np.array(
            f"python {sys.version.split()[0]}, numpy {np.__version__},"
            f" scipy {scipy.__version__}, numba {numba.__version__}"
        ),
        "meta_mission": np.array(mission.name),
    }

    # Every frame is run, in order, because the planner carries state from one
    # frame to the next; only the sampled ones are stored.
    planner = PathPlanner(mission)
    for i, (position, direction, cones) in enumerate(
        zip(positions, directions, observations, strict=True)
    ):
        result = planner.calculate_path_in_global_frame(
            cones, position, direction, return_intermediate_results=True
        )
        if i in wanted:
            for key, value in zip(RESULT_KEYS, result, strict=True):
                stored[f"frame_{i:04d}_{key}"] = np.asarray(value)

    np.savez_compressed(out_file, **stored)
    size = out_file.stat().st_size / 1024
    print(f"  wrote {out_file.name}: {len(frame_indices)} frames, {size:.0f} kB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regenerate", action="store_true", help="overwrite existing files"
    )
    args = parser.parse_args()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for stem, mission, colour in GOLDEN_RUNS:
        print(f"{stem} ({'colour' if colour else 'no colour'})")
        record(stem, mission, colour, args.regenerate)


if __name__ == "__main__":
    main()
