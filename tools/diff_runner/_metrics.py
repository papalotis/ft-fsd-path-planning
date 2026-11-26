#!/usr/bin/env python3
"""Measure how far two planned paths really differ.

Comparing the two point arrays element by element answers the wrong question.
The path is a curve sampled at a fixed number of points, so a change that moves
the samples a little along the same curve looks as large as one that steers the
car somewhere else. These metrics separate the two:

- ``lateral``: the cross-track distance between the two curves, in metres: how
  far a point on one lies from the nearest point on the other, over the stretch
  both cover. This is the one that says whether the car would drive somewhere
  else, and it is blind to samples sliding along the same curve.
- ``length``: how much longer or shorter the planned path became.
- ``curvature``: the largest difference in curvature at matched positions along
  the path, in 1/m.
- ``sample_shift``: how far the sample points moved along the curve. Large
  values here with a small ``lateral`` mean the same trajectory, resampled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep

# Acceptance rule agreed for the refactor: the planned line may move sideways by
# up to about a centimetre. Path length gets a looser bound of its own, because
# the far end of the horizon moves by a sampling step without the line moving.
LATERAL_TOLERANCE = 0.01  # metres
LENGTH_TOLERANCE = 0.5  # metres
# Cones produced by a stage (sorted, virtual) may move by the same amount.
CONE_TOLERANCE = 0.01  # metres

# Missions that plan from sorted cones. The others follow a stored reference
# path and never run the sorter, so an empty sort result means nothing there.
SORTING_MISSIONS = frozenset({"trackdrive", "autocross"})

# Positions along the shared arc length at which curvature is compared.
RESAMPLE_POINTS = 200
# Points on the smooth curve through each path's samples, for the cross-track
# distance: about 5 cm apart over a 20 m path.
DENSE_POINTS = 400


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


def densify(xy: np.ndarray, n_points: int = DENSE_POINTS) -> np.ndarray:
    """A dense sampling of the smooth curve through the given points.

    The planner's output is itself sampled from a cubic spline, so a cubic
    through its samples recovers the curve it came from far better than
    straight lines between samples half a metre apart, which on a tight corner
    are off by several millimetres on their own.
    """
    keep = np.concatenate([[True], np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-9])
    xy = xy[keep]
    if len(xy) < 2:
        return xy
    s = arc_length(xy)
    degree = min(3, len(xy) - 1)
    try:
        tck, _ = splprep(xy.T, u=s, s=0, k=degree)
    except (ValueError, TypeError):
        return xy
    return np.array(splev(np.linspace(0.0, s[-1], n_points), tck)).T


def project(points: np.ndarray, curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to a polyline, and whether it projects inside it.

    A point whose nearest location on the curve is one of its two ends, beyond
    which the curve does not continue, is outside the stretch the two curves
    share; its distance says how much longer one path is, not how far apart
    they are, so it is flagged rather than counted as sideways movement.
    """
    start, end = curve[:-1], curve[1:]
    segment = end - start
    squared = np.maximum((segment**2).sum(axis=1), 1e-18)
    relative = points[:, None, :] - start[None, :, :]
    t = (relative * segment[None]).sum(axis=-1) / squared[None]
    clamped = np.clip(t, 0.0, 1.0)
    nearest = start[None] + clamped[..., None] * segment[None]
    distance = np.linalg.norm(points[:, None, :] - nearest, axis=-1)

    best = distance.argmin(axis=1)
    rows = np.arange(len(points))
    best_t = t[rows, best]
    overhang = ((best == 0) & (best_t < 0.0)) | (
        (best == len(segment) - 1) & (best_t > 1.0)
    )
    return distance[rows, best], ~overhang


def cross_track(base_xy: np.ndarray, head_xy: np.ndarray) -> float:
    """Largest sideways distance between two curves, over the stretch they share."""
    base_dense, head_dense = densify(base_xy), densify(head_xy)
    if len(base_dense) < 2 or len(head_dense) < 2:
        return float("inf")
    worst = 0.0
    for points, curve in ((head_dense, base_dense), (base_dense, head_dense)):
        distance, inside = project(points, curve)
        if inside.any():
            worst = max(worst, float(distance[inside].max()))
    return worst


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
    if base.shape == head.shape and np.array_equal(base, head):
        return PathDeviation.zero()

    base_xy, head_xy = base[:, 1:3], head[:, 1:3]
    base_s, head_s = arc_length(base_xy), arc_length(head_xy)

    lateral = cross_track(base_xy, head_xy)
    length = abs(float(base_s[-1] - head_s[-1]))

    # Curvature is compared at the same distance along each path, over the
    # stretch both cover.
    shared = min(base_s[-1], head_s[-1])
    at = np.linspace(0.0, shared, RESAMPLE_POINTS)
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


def planner_was_blind(mission: str, sorted_left, sorted_right) -> bool:
    """Whether a sorting mission's sorter found no cones on either side.

    On such a frame the planner only replays and extends the path it planned
    earlier, so any earlier difference is extrapolated as the car keeps moving.
    The frame says nothing about how well the code plans a track, so it is
    reported but not held to the tolerance.
    """
    return (
        mission in SORTING_MISSIONS and len(sorted_left) == 0 and len(sorted_right) == 0
    )
