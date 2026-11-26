"""
Compile the planner's Numba functions before the car moves.

Numba compiles a function the first time it is called with a new combination of
argument types, and each branch of the planner reaches different functions with
different array layouts. Planning one frame therefore compiles only part of the
code; the rest compiles the first time a later frame takes another branch. With
a warm on-disk cache that costs tens of milliseconds, but on a fresh install, or
after the code changed, a single frame took 6.5 seconds in the middle of a run.

:func:`warmup` drives throwaway planners through a set of synthetic situations
that between them reach every compiled function the planner uses on the bundled
datasets: a corner with coloured cones, the same without colour, a track with
one side missing, a short trace, an acceleration run and a skidpad run through
relocalization. Call it once while the vehicle is still standing, for example
when the node starts or enters AS-READY.

Project: fsd_path_planning
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable, Iterator

import numpy as np

from fsd_path_planning.common_types import FloatArray
from fsd_path_planning.geometry.angles import (
    angle_difference,
    unit_2d_vector_from_angle,
)
from fsd_path_planning.geometry.transforms import rotate
from fsd_path_planning.pipeline import PathPlanner
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.mission_types import MissionTypes

logger = logging.getLogger(__name__)

Frame = tuple[list[FloatArray], FloatArray, FloatArray]

# Skidpad geometry from the Formula Student rules, in metres, in the skidpad
# frame: origin between the two circles, x towards the exit.
SKIDPAD_CENTER_OFFSET = 9.125
SKIDPAD_INNER_RADIUS = 7.625
SKIDPAD_OUTER_RADIUS = 10.625


def _empty_cones() -> list[FloatArray]:
    return [np.zeros((0, 2)) for _ in ConeTypes]


def _corner(n_left: int = 12, n_right: int = 12, radius: float = 12.0) -> Frame:
    """A left-hand corner ahead of a car at the origin heading along +x."""
    width = 3.0
    left_angles = np.linspace(0.0, 2.2, n_left)
    right_angles = np.linspace(0.0, 2.2, n_right)
    center = np.array([0.0, radius])
    left = center + (radius - width / 2) * np.column_stack(
        [np.sin(left_angles), -np.cos(left_angles)]
    )
    right = center + (radius + width / 2) * np.column_stack(
        [np.sin(right_angles), -np.cos(right_angles)]
    )
    cones = _empty_cones()
    cones[ConeTypes.LEFT] = left
    cones[ConeTypes.RIGHT] = right
    return cones, np.array([-1.5, 0.0]), np.array([1.0, 0.0])


def _without_colour(frame: Frame) -> Frame:
    cones, position, direction = frame
    unknown = _empty_cones()
    unknown[ConeTypes.UNKNOWN] = np.vstack(cones)
    return unknown, position, direction


def _drive(frames: list[Frame], steps: int, step: float) -> Iterator[Frame]:
    """Move the car forward along +x through the given frames, frame by frame."""
    for i in range(steps):
        cones, position, direction = frames[i % len(frames)]
        yield cones, position + np.array([i * step, 0.0]), direction


def _trackdrive_frames() -> Iterator[Frame]:
    corner = _corner()
    yield from _drive([corner], steps=6, step=0.6)
    yield from _drive([_without_colour(corner)], steps=6, step=0.6)

    # One side of the track only: matching has to place virtual cones.
    left_only = _corner()
    left_only[0][ConeTypes.RIGHT] = np.zeros((0, 2))
    yield from _drive([left_only], steps=4, step=0.6)
    right_only = _corner()
    right_only[0][ConeTypes.LEFT] = np.zeros((0, 2))
    yield from _drive([right_only], steps=4, step=0.6)

    # One side seen less far ahead than the other: the cones beyond its end find
    # no match, so matching inserts virtual cones among the real ones and then
    # prunes them by angle.
    for side in (ConeTypes.RIGHT, ConeTypes.LEFT):
        for visible in (9, 6):
            shorter = _corner()
            shorter[0][side] = shorter[0][side][:visible]
            yield from _drive([shorter], steps=2, step=0.6)

    # Few cones, and a frame with none at all.
    yield from _drive([_corner(n_left=3, n_right=2)], steps=3, step=0.6)
    yield _empty_cones(), np.array([0.0, 0.0]), np.array([1.0, 0.0])


def _acceleration_frames() -> Iterator[Frame]:
    xs = np.arange(2.0, 60.0, 4.0)
    cones = _empty_cones()
    cones[ConeTypes.LEFT] = np.column_stack([xs, np.full_like(xs, 1.5)])
    cones[ConeTypes.RIGHT] = np.column_stack([xs, np.full_like(xs, -1.5)])
    yield from _drive([(cones, np.array([0.0, 0.0]), np.array([1.0, 0.0]))], 5, 0.5)


def _skidpad_frames() -> Iterator[Frame]:
    """The skidpad layout, approached from the entry, as SLAM would map it."""
    cones = []
    for side in (-1.0, 1.0):
        center = np.array([0.0, side * SKIDPAD_CENTER_OFFSET])
        for radius, count in ((SKIDPAD_INNER_RADIUS, 16), (SKIDPAD_OUTER_RADIUS, 13)):
            angles = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
            ring = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
            # The outer rings are open where the loops meet.
            if radius == SKIDPAD_OUTER_RADIUS:
                ring = ring[np.abs(ring[:, 1]) > 3.0]
            cones.append(ring)
    all_cones = _empty_cones()
    all_cones[ConeTypes.UNKNOWN] = np.vstack(cones)
    start = np.array([-16.0, 0.0])
    yield from _drive([(all_cones, start, np.array([1.0, 0.0]))], steps=12, step=0.8)


SCENARIOS: dict[MissionTypes, Callable[[], Iterator[Frame]]] = {
    MissionTypes.trackdrive: _trackdrive_frames,
    MissionTypes.acceleration: _acceleration_frames,
    MissionTypes.skidpad: _skidpad_frames,
}


def _compile_scalar_helpers() -> None:
    """Compile the scalar versions of the shared geometry helpers.

    Some branches call these with plain floats only in situations that are hard
    to stage with a synthetic track, such as the two traces crossing when the
    cones have no colour. Calling them once here is cheaper than a scenario.
    """
    angle_difference(0.25, 0.5)
    rotate(np.zeros((3, 2)), np.float64(0.25))
    rotate(np.zeros(2), np.float64(0.25))
    unit_2d_vector_from_angle(np.float64(0.25))


def warmup() -> float:
    """Compile every Numba function the planner uses, before it is needed.

    Runs throwaway planners through synthetic trackdrive, acceleration and
    skidpad situations and discards the results. Planners created afterwards
    start with nothing left to compile. Safe to call more than once; later calls
    are fast.

    Returns:
        The time the warm-up took, in seconds.
    """
    start = time.perf_counter()
    _compile_scalar_helpers()
    for mission, frames in SCENARIOS.items():
        for experimental_performance_improvements in (False, True):
            planner = PathPlanner(mission, experimental_performance_improvements)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for cones, position, direction in frames():
                    planner.calculate_path_in_global_frame(cones, position, direction)
    elapsed = time.perf_counter() - start
    logger.info("Planner warm-up took %.2f s", elapsed)
    return elapsed
