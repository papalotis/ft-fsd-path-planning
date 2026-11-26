#!/usr/bin/env python3
"""
Skidpad reference path.

The path is stored as the cubic spline it was fitted to, not as the thousands
of points it is sampled into. The spline is parameterized by point index, so
evaluating it at every whole index returns the sampled path. Regenerate the
stored file with ``tools/generate_skidpad_path.py``.

Project: FaSTTUBe Chabo Pathing
"""

from __future__ import annotations

from importlib import resources

import numpy as np

from fsd_path_planning.types import FloatArray
from fsd_path_planning.utils.spline_fit import SplineEvaluator

SPLINE_RESOURCE = "skidpad_path_spline.npz"


def load_skidpad_path_spline() -> SplineEvaluator:
    """Load the stored skidpad spline."""
    data_package = resources.files(__package__) / "data"
    with (data_package / SPLINE_RESOURCE).open("rb") as handle:
        return SplineEvaluator.load(handle)


def sample_skidpad_path(spline: SplineEvaluator | None = None) -> FloatArray:
    """Sample the skidpad path at the points the spline was fitted through.

    The spline's parameter is the index of a point in the original path, so its
    `max_u` is the index of the last one.
    """
    if spline is None:
        spline = load_skidpad_path_spline()
    n_points = int(round(spline.max_u)) + 1
    return spline.evaluate_at(np.arange(n_points, dtype=float))


BASE_SKIDPAD_PATH: FloatArray = sample_skidpad_path()
