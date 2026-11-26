#!/usr/bin/env python3
"""Regenerate the stored skidpad reference path.

The reference path used to live in the repository as a 5,800-line Python
literal. It is a smooth curve sampled roughly every 5 cm, so a cubic spline
through every 20th point reproduces it to well under a millimetre at a fraction
of the size.

The spline is parameterized by point index rather than arc length, so
evaluating it at ``arange(n_points)`` returns the original points in their
original places. ``n_points`` is the spline's ``max_u`` plus one.

Run it against the released literal to reproduce the shipped file:

    python tools/generate_skidpad_path.py --from-literal old_skidpad_path.py
    python tools/generate_skidpad_path.py --check
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fsd_path_planning.utils.spline_fit import (  # noqa: E402
    SplineFitterFactory,
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "fsd_path_planning"
    / "relocalization"
    / "skidpad"
    / "data"
    / "skidpad_path_spline.npz"
)
# Every 20th point of a path sampled every ~5 cm, so one knot per metre.
DEFAULT_STRIDE = 20


def fit_path(points: np.ndarray, stride: int):
    """Fit a cubic spline through every `stride`-th point, parameterized by index."""
    indices = np.unique(np.append(np.arange(0, len(points), stride), len(points) - 1))
    factory = SplineFitterFactory(smoothing=0.0, predict_every=1.0, max_deg=3)
    return factory.fit(points[indices], u_fit=indices.astype(float)), indices


def load_reference_points(from_literal: Path | None) -> np.ndarray:
    if from_literal is not None:
        namespace = runpy.run_path(str(from_literal))
        return np.asarray(namespace["BASE_SKIDPAD_PATH"], dtype=float)

    from fsd_path_planning.relocalization.skidpad.skidpad_path_data import (
        BASE_SKIDPAD_PATH,
    )

    return np.asarray(BASE_SKIDPAD_PATH, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-literal",
        type=Path,
        default=None,
        help="python file defining BASE_SKIDPAD_PATH; defaults to the installed path",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report the reconstruction error, write nothing",
    )
    args = parser.parse_args()

    points = load_reference_points(args.from_literal)
    spline, indices = fit_path(points, args.stride)
    rebuilt = spline.evaluate_at(np.arange(len(points), dtype=float))
    error = np.linalg.norm(rebuilt - points, axis=1)

    print(f"points              {len(points)}")
    print(f"fitted through      {len(indices)} of them (stride {args.stride})")
    print(f"max error           {error.max() * 1000:.3f} mm")
    print(f"mean error          {error.mean() * 1000:.4f} mm")

    if args.check:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    spline.save(args.output)
    print(f"written             {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
