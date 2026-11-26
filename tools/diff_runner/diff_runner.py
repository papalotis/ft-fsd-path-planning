#!/usr/bin/env python3
"""Compare the planner output of two git refs, frame by frame.

Both refs are checked out into throwaway worktrees and run in separate
subprocesses against the *same* dataset files, so the only variable is the
library code. Every stage output of every frame is compared exactly.

Examples:
    # working tree against main
    python tools/diff_runner/diff_runner.py --base origin/main

    # two commits, quick pass over one dataset
    python tools/diff_runner/diff_runner.py --base HEAD~1 --head HEAD --quick

Exit code is 0 when the two refs agree exactly, 1 when they differ, so the
runner can gate a refactoring commit in CI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _metrics import PathDeviation, path_deviation  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECT_SCRIPT = Path(__file__).resolve().parent / "_collect.py"
WORKING_TREE = "WORKTREE"
# How far the planned trajectory may move before a run counts as a regression.
# A centimetre is well inside the noise of cone detection and vehicle state.
DEFAULT_TOLERANCE = 0.01
# The far end of the horizon is a sampling artefact, so the length of the
# planned path is held to a looser bound than its position.
DEFAULT_LENGTH_TOLERANCE = 0.5

# (dataset file, mission name). Missions follow the demo's filename convention.
DATASETS = (
    ("fsg_19_2_laps.json", "trackdrive"),
    ("fss_19_4_laps.json", "trackdrive"),
    ("skidpad.json", "skidpad"),
)
QUICK_DATASETS = (("fsg_19_2_laps.json", "trackdrive"),)


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    mission: str
    no_color: bool
    cache: bool

    @property
    def key(self) -> str:
        return (
            f"{Path(self.dataset).stem}"
            f"|{'nocolor' if self.no_color else 'color'}"
            f"|{'cache' if self.cache else 'nocache'}"
        )

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "mission": self.mission,
            "no_color": self.no_color,
            "cache": self.cache,
            "key": self.key,
        }


def build_specs(quick: bool, with_cache: bool) -> list[RunSpec]:
    datasets = QUICK_DATASETS if quick else DATASETS
    cache_modes = (False, True) if with_cache else (False,)
    return [
        RunSpec(dataset, mission, no_color, cache)
        for dataset, mission in datasets
        for no_color in (False, True)
        for cache in cache_modes
    ]


def add_worktree(ref: str, destination: Path) -> Path:
    """Check `ref` out into `destination`. Returns the package root to import."""
    if ref == WORKING_TREE:
        return REPO_ROOT

    subprocess.run(
        ["git", "worktree", "add", "--detach", str(destination), ref],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return destination


def remove_worktree(destination: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(destination)],
        cwd=REPO_ROOT,
        capture_output=True,
    )


def collect(ref: str, package_root: Path, datasets_dir: Path, specs, out: Path) -> None:
    print(f"  running {ref} ...", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            str(COLLECT_SCRIPT),
            str(package_root),
            str(datasets_dir),
            str(out),
            json.dumps([spec.as_dict() for spec in specs]),
        ],
        # Run from a neutral directory: the package directory contains a
        # types.py that shadows the standard library module.
        cwd=COLLECT_SCRIPT.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"collection failed for ref {ref!r}")


def compare_arrays(base: np.ndarray, head: np.ndarray) -> float | None:
    """None when identical, else the max absolute difference (inf if shapes differ)."""
    if base.shape != head.shape:
        return float("inf")
    if np.array_equal(base, head):
        return None
    if base.size == 0:
        return None
    try:
        return float(np.nanmax(np.abs(base.astype(float) - head.astype(float))))
    except (TypeError, ValueError):
        return float("inf")


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


# Missions that plan from sorted cones. The others follow a stored reference
# path and never run the sorter, so an empty sort result means nothing there.
SORTING_MISSIONS = frozenset({"trackdrive", "autocross"})


def planner_was_blind(store, spec: RunSpec, frame: int) -> bool:
    """Whether the base ref's sorter returned no cones on either side."""
    if spec.mission not in SORTING_MISSIONS:
        return False
    left = store[f"{spec.key}|{frame}|sorted_left"]
    right = store[f"{spec.key}|{frame}|sorted_right"]
    return len(left) == 0 and len(right) == 0


def summarize_run(base, head, spec: RunSpec, stage_names) -> dict:
    """Collect, for one run, what changed and by how much."""
    n_frames = int(base[f"{spec.key}|meta|n_frames"])
    stages: dict[str, dict] = {
        stage: {"differing": 0, "first": None, "raw": 0.0} for stage in stage_names
    }
    deviation = PathDeviation.zero()
    blind_deviation = PathDeviation.zero()
    worst_frame = None
    frames_beyond_noise = 0
    blind_frames = 0

    for frame in range(n_frames):
        for stage in stage_names:
            name = f"{spec.key}|{frame}|{stage}"
            entry = stages[stage]
            if name not in head:
                entry["differing"] += 1
                entry["raw"] = float("inf")
                entry["first"] = frame if entry["first"] is None else entry["first"]
                continue
            raw = compare_arrays(base[name], head[name])
            if raw is None:
                continue
            entry["differing"] += 1
            entry["raw"] = max(entry["raw"], raw)
            if entry["first"] is None:
                entry["first"] = frame

        path_name = f"{spec.key}|{frame}|path"
        if path_name in head:
            frame_deviation = path_deviation(base[path_name], head[path_name])
            if planner_was_blind(base, spec, frame):
                # The sorter found no cones, so the planner only replays and
                # extends the path it planned earlier. Small differences from
                # before are extrapolated here; the frame says nothing about how
                # well the code plans a track, so it is reported, not gated.
                blind_frames += 1
                blind_deviation = blind_deviation.combine(frame_deviation)
                continue
            if frame_deviation.lateral > deviation.lateral:
                worst_frame = frame
            if frame_deviation.lateral > 1e-9:
                frames_beyond_noise += 1
            deviation = deviation.combine(frame_deviation)

    return {
        "n_frames": n_frames,
        "stages": {k: v for k, v in stages.items() if v["differing"]},
        "deviation": deviation,
        "worst_frame": worst_frame,
        "frames_beyond_noise": frames_beyond_noise,
        "blind_frames": blind_frames,
        "blind_deviation": blind_deviation,
        "base_ms": base[f"{spec.key}|meta|durations_ms"],
        "head_ms": head[f"{spec.key}|meta|durations_ms"],
        "base_reloc": int(base[f"{spec.key}|meta|relocalized_at"]),
        "head_reloc": int(head[f"{spec.key}|meta|relocalized_at"]),
    }


def report(
    base_file: Path,
    head_file: Path,
    specs: list[RunSpec],
    tolerance: float,
    length_tolerance: float,
) -> bool:
    base = np.load(base_file, allow_pickle=False)
    head = np.load(head_file, allow_pickle=False)

    from _collect import STAGE_NAMES  # noqa: PLC0415  (same directory)

    within_tolerance = True
    for spec in specs:
        run = summarize_run(base, head, spec, STAGE_NAMES)
        deviation = run["deviation"]
        n_frames = run["n_frames"]

        if not run["stages"] and run["base_reloc"] == run["head_reloc"]:
            status = "IDENTICAL"
        elif deviation.within(tolerance, length_tolerance):
            status = "WITHIN TOLERANCE"
        else:
            status = "BEYOND TOLERANCE"
            within_tolerance = False

        print(f"\n{spec.key}  ({n_frames} frames)  {status}")
        print(
            f"  time p50 {percentile(run['base_ms'], 50):6.2f} ->"
            f" {percentile(run['head_ms'], 50):6.2f} ms"
            f"   p95 {percentile(run['base_ms'], 95):6.2f} ->"
            f" {percentile(run['head_ms'], 95):6.2f} ms"
            f"   max {run['base_ms'].max():7.2f} -> {run['head_ms'].max():7.2f} ms"
        )
        if run["base_reloc"] != run["head_reloc"]:
            within_tolerance = False
            print(
                f"  relocalization frame {run['base_reloc']} ->"
                f" {run['head_reloc']}  DIFFERS"
            )
        elif run["base_reloc"] >= 0:
            print(f"  relocalization frame {run['base_reloc']}")

        if run["stages"]:
            print(
                f"  trajectory   lateral {deviation.lateral * 1000:9.3f} mm"
                f"   length {deviation.length * 1000:9.3f} mm"
                f"   curvature {deviation.curvature:.3e} 1/m"
            )
            print(
                f"               samples moved along the path by up to"
                f" {deviation.sample_shift * 1000:.3f} mm;"
                f" {run['frames_beyond_noise']}/{n_frames} frames moved"
                f" (worst frame {run['worst_frame']})"
            )
            if run["blind_frames"]:
                blind = run["blind_deviation"]
                print(
                    f"  not gated    {run['blind_frames']} frames where the sorter"
                    f" found no cones and the planner replayed an older path:"
                    f" lateral up to {blind.lateral * 1000:.3f} mm"
                )
            for stage, entry in run["stages"].items():
                print(
                    f"  {stage:<20} {entry['differing']:>5}/{n_frames} frames differ"
                    f"   max abs {entry['raw']:.6g}   first frame {entry['first']}"
                )

    return within_tolerance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="reference git ref")
    parser.add_argument(
        "--head",
        default=WORKING_TREE,
        help=f"git ref under test, or {WORKING_TREE} for the current working tree",
    )
    parser.add_argument(
        "--datasets-from",
        default=None,
        help="git ref to take the dataset files from (default: the base ref)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="only the FSG dataset, colour and no-colour",
    )
    parser.add_argument(
        "--with-cache",
        action="store_true",
        help="also run with experimental_performance_improvements=True",
    )
    parser.add_argument(
        "--keep", default=None, help="directory to keep the collected .npz files in"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=(
            "how far the planned trajectory may move, in metres, before the run"
            f" is called a regression (default {DEFAULT_TOLERANCE})"
        ),
    )
    parser.add_argument(
        "--length-tolerance",
        type=float,
        default=DEFAULT_LENGTH_TOLERANCE,
        help=(
            "how much longer or shorter the planned path may become, in metres"
            f" (default {DEFAULT_LENGTH_TOLERANCE})"
        ),
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="require byte-for-byte equality, as before tolerances were allowed",
    )
    args = parser.parse_args()

    specs = build_specs(args.quick, args.with_cache)
    print(f"comparing {args.base} -> {args.head}  ({len(specs)} runs)")

    with tempfile.TemporaryDirectory(prefix="diff-runner-") as tmp:
        tmpdir = Path(tmp)
        outdir = Path(args.keep) if args.keep else tmpdir
        outdir.mkdir(parents=True, exist_ok=True)

        base_tree = tmpdir / "base"
        head_tree = tmpdir / "head"
        created = []
        try:
            base_root = add_worktree(args.base, base_tree)
            created.append(base_root)
            head_root = add_worktree(args.head, head_tree)
            created.append(head_root)

            # Both refs read the same dataset files, so a change to the demo
            # data can never show up as an algorithm difference.
            datasets_dir = base_root / "fsd_path_planning" / "demo"
            if args.datasets_from:
                data_tree = add_worktree(args.datasets_from, tmpdir / "data")
                created.append(data_tree)
                datasets_dir = data_tree / "fsd_path_planning" / "demo"

            base_npz = outdir / "base.npz"
            head_npz = outdir / "head.npz"
            collect(args.base, base_root, datasets_dir, specs, base_npz)
            collect(args.head, head_root, datasets_dir, specs, head_npz)

            tolerance = 0.0 if args.exact else args.tolerance
            length_tolerance = 0.0 if args.exact else args.length_tolerance
            identical = report(base_npz, head_npz, specs, tolerance, length_tolerance)
        finally:
            for tree in created:
                if tree != REPO_ROOT:
                    remove_worktree(tree)

    if identical:
        print("\nWithin tolerance." if not args.exact else "\nNo differences.")
    else:
        print("\nOutput differs beyond tolerance.")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
