#!/usr/bin/env python3
"""
How well skidpad relocalization copes with imperfect cone maps.

The bundled skidpad run is clean simulation data: every variant of the subset
search relocalizes on the same frame, onto the same transform. To tell the
variants apart this replays the run with the cone map degraded on every
frame: position noise, missed cones, and false detections. For each trial it
records

- the frame on which the relocalizer first succeeds, or that it never did;
- the error of the transform it found: the largest distance, over the
  skidpad reference path, between where the found transform and the clean
  run's transform put it.

A relocalization with an error over WRONG_THRESHOLD metres is counted as
wrong: the car would follow a path that is off the track.

    python tools/skidpad_quality.py               # the default comparison
    python tools/skidpad_quality.py --trials 50

Project: fsd_path_planning
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fsd_path_planning import ConeTypes  # noqa: E402
from fsd_path_planning.config_dataclasses import SkidpadConfig  # noqa: E402
from fsd_path_planning.stages.relocalization.base import (  # noqa: E402
    RelocalizationInput,
)
from fsd_path_planning.stages.relocalization.skidpad import (  # noqa: E402
    SkidpadRelocalizer,
)
from fsd_path_planning.stages.relocalization.skidpad_path_data import (  # noqa: E402
    BASE_SKIDPAD_PATH,
)

DATA = REPO_ROOT / "fsd_path_planning" / "demo" / "skidpad.json"
MAX_FRAMES = 150
WRONG_THRESHOLD = 0.5  # metres

VARIANTS = {
    "triples (default)": SkidpadConfig(),
    "sampled 4-5": SkidpadConfig(max_powerset_size=5),
    "exhaustive 4-5": SkidpadConfig(max_powerset_size=5, max_sampled_subsets=10**7),
    "triples + start": SkidpadConfig(check_start_position=True),
    "sampled + start": SkidpadConfig(max_powerset_size=5, check_start_position=True),
}

# (position noise sigma in m, probability a cone is missed, false cones/frame)
LEVELS = [(0.0, 0.0, 0), (0.05, 0.0, 0), (0.1, 0.1, 0), (0.2, 0.2, 1), (0.3, 0.3, 2)]


def load_data_json(path: Path):
    data = json.loads(path.read_text())
    positions = np.array([d["car_position"] for d in data])
    directions = np.array([d["car_direction"] for d in data])
    cones = [[np.array(c).reshape(-1, 2) for c in d["slam_cones"]] for d in data]
    return positions, directions, cones


@dataclass
class Trial:
    frame: int | None
    error: float | None
    frame_time_ms: float


class Degrader:
    """Degrades each frame's cone map the same way a SLAM map would be wrong.

    Errors stick to a cone: each cone gets one position offset and one
    missed-or-not draw, kept for the whole run, as a SLAM estimate that is off
    stays off. Per-frame independent noise would instead give the relocalizer
    a fresh chance at a lucky false match on every frame. The false cones are
    redrawn each frame, as spurious detections come and go.
    """

    def __init__(self, rng, sigma, p_miss, n_false, per_frame_noise=False):
        self.rng, self.sigma, self.p_miss, self.n_false = rng, sigma, p_miss, n_false
        self.per_frame_noise = per_frame_noise
        self._cones: dict[tuple[int, int], tuple[np.ndarray, bool]] = {}

    def _cone(self, xy):
        key = (round(xy[0] * 2), round(xy[1] * 2))  # the same cone, frame to frame
        if key not in self._cones:
            self._cones[key] = (
                self.rng.normal(0.0, self.sigma, 2),
                bool(self.rng.random() < self.p_miss),
            )
        return self._cones[key]

    def __call__(self, cones, position):
        out = []
        for array in cones:
            kept = []
            for xy in array:
                offset, missed = self._cone(xy)
                if self.per_frame_noise:
                    offset = self.rng.normal(0.0, self.sigma, 2)
                if not missed:
                    kept.append(xy + offset)
            out.append(np.array(kept).reshape(-1, 2))
        if self.n_false:
            offsets = self.rng.uniform(-15.0, 15.0, (self.n_false, 2))
            out[ConeTypes.UNKNOWN] = np.vstack(
                [out[ConeTypes.UNKNOWN], position + offsets]
            )
        return out


def relocalize(config, positions, directions, cones, degrade_frame) -> Trial:
    relocalizer = SkidpadRelocalizer(config)
    times = []
    for i in range(min(MAX_FRAMES, len(positions))):
        frame_cones = degrade_frame(cones[i], positions[i])
        start = time.perf_counter()
        result = relocalizer.run(
            RelocalizationInput(frame_cones, positions[i], directions[i])
        )
        times.append(time.perf_counter() - start)
        if result.is_relocalized:
            return Trial(i, None, float(np.median(times) * 1000)), relocalizer
    return Trial(None, None, float(np.median(times) * 1000)), None


def path_error(relocalizer, reference) -> float:
    path = BASE_SKIDPAD_PATH[::20]
    found, _ = relocalizer.transform_to_original_frame(path, 0.0)
    expected, _ = reference.transform_to_original_frame(path, 0.0)
    return float(np.linalg.norm(found - expected, axis=1).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--per-frame-noise",
        action="store_true",
        help="redraw every cone's position error on every frame (harsher)",
    )
    args = parser.parse_args()

    positions, directions, cones = load_data_json(DATA)
    _, reference = relocalize(
        SkidpadConfig(), positions, directions, cones, lambda c, p: c
    )
    assert reference is not None

    print(
        f"{'noise m':>7} {'miss':>5} {'false':>5}  {'variant':<19}"
        f" {'relocalized':>11} {'wrong':>5} {'frame p50':>9}"
        f" {'error p50 m':>11} {'error max m':>11} {'ms/frame':>8}"
    )
    for sigma, p_miss, n_false in LEVELS:
        for name, config in VARIANTS.items():
            trials = []
            for trial in range(args.trials):
                degrader = Degrader(
                    np.random.default_rng(1000 + trial),
                    sigma,
                    p_miss,
                    n_false,
                    per_frame_noise=args.per_frame_noise,
                )
                variant_config = replace(config, random_seed=trial)
                result, relocalizer = relocalize(
                    variant_config, positions, directions, cones, degrader
                )
                if relocalizer is not None:
                    result.error = path_error(relocalizer, reference)
                trials.append(result)

            done = [t for t in trials if t.frame is not None]
            wrong = [
                t for t in done if t.error is not None and t.error > WRONG_THRESHOLD
            ]
            errors = [t.error for t in done if t.error is not None]
            frames = [t.frame for t in done]
            print(
                f"{sigma:7.2f} {p_miss:5.1f} {n_false:5d}  {name:<19}"
                f" {len(done):>5}/{len(trials):<5} {len(wrong):>5}"
                f" {np.median(frames) if frames else float('nan'):9.0f}"
                f" {np.median(errors) if errors else float('nan'):11.3f}"
                f" {max(errors) if errors else float('nan'):11.3f}"
                f" {np.median([t.frame_time_ms for t in trials]):8.1f}"
            )
        print()


if __name__ == "__main__":
    main()
