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

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECT_SCRIPT = Path(__file__).resolve().parent / "_collect.py"
WORKING_TREE = "WORKTREE"

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


def report(base_file: Path, head_file: Path, specs: list[RunSpec]) -> bool:
    base = np.load(base_file, allow_pickle=False)
    head = np.load(head_file, allow_pickle=False)

    from _collect import STAGE_NAMES  # noqa: PLC0415  (same directory)

    all_match = True
    for spec in specs:
        n_frames = int(base[f"{spec.key}|meta|n_frames"])
        rows = []
        for stage in STAGE_NAMES:
            n_differing = 0
            first_diff = None
            worst = 0.0
            for frame in range(n_frames):
                name = f"{spec.key}|{frame}|{stage}"
                if name not in head:
                    n_differing += 1
                    first_diff = frame if first_diff is None else first_diff
                    worst = float("inf")
                    continue
                diff = compare_arrays(base[name], head[name])
                if diff is not None:
                    n_differing += 1
                    worst = max(worst, diff)
                    if first_diff is None:
                        first_diff = frame
            if n_differing:
                all_match = False
                rows.append((stage, n_differing, worst, first_diff))

        base_ms = base[f"{spec.key}|meta|durations_ms"]
        head_ms = head[f"{spec.key}|meta|durations_ms"]
        base_reloc = int(base[f"{spec.key}|meta|relocalized_at"])
        head_reloc = int(head[f"{spec.key}|meta|relocalized_at"])

        status = "IDENTICAL" if not rows else "DIFFERS"
        print(f"\n{spec.key}  ({n_frames} frames)  {status}")
        print(
            f"  time p50 {percentile(base_ms, 50):6.2f} ->"
            f" {percentile(head_ms, 50):6.2f} ms"
            f"   p95 {percentile(base_ms, 95):6.2f} ->"
            f" {percentile(head_ms, 95):6.2f} ms"
            f"   max {base_ms.max():7.2f} -> {head_ms.max():7.2f} ms"
        )
        if base_reloc != head_reloc:
            all_match = False
            print(f"  relocalization frame {base_reloc} -> {head_reloc}  DIFFERS")
        elif base_reloc >= 0:
            print(f"  relocalization frame {base_reloc}")
        for stage, n_differing, worst, first_diff in rows:
            print(
                f"  {stage:<20} {n_differing:>5}/{n_frames} frames differ"
                f"   max abs {worst:.6g}   first frame {first_diff}"
            )

    return all_match


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

            sys.path.insert(0, str(COLLECT_SCRIPT.parent))
            identical = report(base_npz, head_npz, specs)
        finally:
            for tree in created:
                if tree != REPO_ROOT:
                    remove_worktree(tree)

    print("\n" + ("No differences." if identical else "Output differs."))
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
