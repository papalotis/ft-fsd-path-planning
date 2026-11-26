#!/usr/bin/env python3
"""Measure how far two planned paths really differ.

Comparing the two point arrays element by element answers the wrong question.
The path is a curve sampled at a fixed number of points, so a change that moves
the samples a little along the same curve looks as large as one that steers the
car somewhere else. These metrics separate the two:

- ``lateral``: how far the two curves are apart, in metres, measured over the
  arc length they share. This is the one that says whether the car would drive
  somewhere else.
- ``length``: how much longer or shorter the planned path became.
- ``curvature``: the largest difference in curvature at matched positions along
  the path, in 1/m.
- ``sample_shift``: how far the sample points moved along the curve. Large
  values here with a small ``lateral`` mean the same trajectory, resampled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Positions along the shared arc length at which the two curves are compared.
RESAMPLE_POINTS = 200


@dataclass(frozen=True)
class PathDeviation:
    lateral: float
    length: float
    curvature: float
    sample_shift: float

    @classmethod
    def zero(cls) -> PathDeviation:
        return cls(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def incomparable(cls) -> PathDeviation:
        infinity = float("inf")
        return cls(infinity, infinity, infinity, infinity)

    def combine(self, other: PathDeviation) -> PathDeviation:
        return PathDeviation(
            max(self.lateral, other.lateral),
            max(self.length, other.length),
            max(self.curvature, other.curvature),
            max(self.sample_shift, other.sample_shift),
        )

    def within(self, lateral_tolerance: float, length_tolerance: float) -> bool:
        """Whether the car would drive the same line, to the given tolerances.

        Lateral deviation is the safety-relevant number: it says how far the
        planned line moved sideways. The length of the path matters much less,
        because the far end of the horizon is a sampling artefact, so it gets a
        tolerance of its own.
        """
        return self.lateral <= lateral_tolerance and self.length <= length_tolerance


def arc_length(xy: np.ndarray) -> np.ndarray:
    """Cumulative distance along a polyline, starting at zero."""
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def resample(xy: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Linearly resample `values` at the given arc-length positions."""
    s = arc_length(xy)
    if s[-1] <= 0:
        return np.repeat(values[:1], len(at), axis=0)
    if values.ndim == 1:
        return np.interp(at, s, values)
    return np.column_stack([np.interp(at, s, column) for column in values.T])


def path_deviation(base: np.ndarray, head: np.ndarray) -> PathDeviation:
    """Compare two parameterized paths, each (N, 4) of theta, x, y, curvature."""
    if base.ndim != 2 or head.ndim != 2 or base.shape[1] < 3 or head.shape[1] < 3:
        return PathDeviation.incomparable()
    if len(base) < 2 or len(head) < 2:
        return (
            PathDeviation.zero()
            if np.array_equal(base, head)
            else (PathDeviation.incomparable())
        )

    base_xy, head_xy = base[:, 1:3], head[:, 1:3]
    base_s, head_s = arc_length(base_xy), arc_length(head_xy)

    shared = min(base_s[-1], head_s[-1])
    at = np.linspace(0.0, shared, RESAMPLE_POINTS)

    lateral = float(
        np.linalg.norm(
            resample(base_xy, base_xy, at) - resample(head_xy, head_xy, at), axis=1
        ).max()
    )
    length = abs(float(base_s[-1] - head_s[-1]))

    curvature = 0.0
    if base.shape[1] > 3 and head.shape[1] > 3:
        curvature = float(
            np.abs(
                resample(base_xy, base[:, 3], at) - resample(head_xy, head[:, 3], at)
            ).max()
        )

    # How far each sample moved along the curve, for paths sampled alike.
    sample_shift = 0.0
    if len(base) == len(head):
        sample_shift = float(np.abs(base_s - head_s).max())

    return PathDeviation(lateral, length, curvature, sample_shift)


def cone_deviation(base: np.ndarray, head: np.ndarray) -> float:
    """Largest distance between matched cones of an intermediate stage."""
    if base.shape != head.shape:
        return float("inf")
    if base.size == 0:
        return 0.0
    if base.ndim == 1:
        return float(np.abs(base.astype(float) - head.astype(float)).max())
    return float(np.linalg.norm(base.astype(float) - head.astype(float), axis=-1).max())
