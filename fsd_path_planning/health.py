"""
The error policy, and the health record a running planner keeps.

A few steps of path calculation have a fallback for when they fail: a spline
that will not fit, a path that cannot be parameterized. What happens then is
the error policy, set by `PipelineConfig.on_error`:

- ``"recover"`` (the default, and the behaviour of every earlier release):
  the fallback runs, and the error is logged with its stack trace and counted
  on `PathPlanner.health`.
- ``"raise"``: the error propagates, so that a bug is seen where it happens.
  For CI, tests, offline replays and profiling. Setting the environment
  variable ``FSD_PATH_PLANNING_STRICT=1`` selects it whatever the config says.

Expected outcomes are not errors and never raise: too few cones to build a
new path, or a path too far from the car to trust, simply reuse the previous
path. Those frames are marked degraded.

Recovery can be bounded (`PipelineConfig.max_consecutive_recoveries`,
`max_distance_without_good_path`): past the bound, the planner returns an
empty path and the status NO_VALID_PATH instead of extending a stale path
indefinitely. Both bounds are off by default.

Project: fsd_path_planning
"""

from __future__ import annotations

import logging
import os
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal

import numpy as np

from fsd_path_planning.common_types import FloatArray

logger = logging.getLogger("fsd_path_planning")

ErrorPolicy = Literal["recover", "raise"]

STRICT_ENV_VAR = "FSD_PATH_PLANNING_STRICT"


def strict_mode_from_environment() -> bool:
    """Whether ``FSD_PATH_PLANNING_STRICT`` asks for the ``"raise"`` policy."""
    value = os.environ.get(STRICT_ENV_VAR, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def resolve_error_policy(configured: ErrorPolicy) -> ErrorPolicy:
    """The policy in effect: ``"raise"`` if the environment asks for it."""
    if strict_mode_from_environment():
        return "raise"
    return configured


@dataclass(frozen=True)
class Recovery:
    """One error that a fallback recovered from."""

    frame: int
    """The planner's frame counter when it happened (0 for the first frame)."""
    stage: str
    kind: str
    """What failed, for example "spline_fit_failed"."""
    message: str
    traceback: str


class RecoveryHandler:
    """Applies the error policy at each fallback, and records what it did.

    A stage calls `recover` from inside its ``except`` block, before running
    its fallback. With the ``"raise"`` policy that re-raises the exception;
    with ``"recover"`` it logs it and records it, and the stage carries on.
    """

    def __init__(self, policy: ErrorPolicy = "recover") -> None:
        self.policy: ErrorPolicy = resolve_error_policy(policy)
        self.frame = 0
        self._this_frame: list[Recovery] = []

    def recover(self, stage: str, kind: str, error: BaseException) -> None:
        if self.policy == "raise":
            raise error
        record = Recovery(
            frame=self.frame,
            stage=stage,
            kind=kind,
            message=f"{type(error).__name__}: {error}",
            traceback="".join(traceback.format_exception(error)),
        )
        logger.warning(
            "fsd_path_planning recovered from %s in %s at frame %d: %s",
            kind,
            stage,
            self.frame,
            record.message,
            exc_info=error,
        )
        self._this_frame.append(record)

    def start_frame(self, frame: int) -> None:
        self.frame = frame
        self._this_frame = []

    @property
    def recoveries_this_frame(self) -> tuple[Recovery, ...]:
        return tuple(self._this_frame)


class PathStatus(Enum):
    """How trustworthy the path of a frame is."""

    OK = "ok"
    """Built fresh from this frame's cones, or from the known map."""
    DEGRADED = "degraded"
    """The previous path, reused for an expected reason: too few cones, or the
    new path was too far from the car."""
    RECOVERED = "recovered"
    """An error occurred and a fallback produced the path."""
    NO_VALID_PATH = "no_valid_path"
    """A recovery bound was exceeded; the planner returned an empty path."""


@dataclass(frozen=True)
class PlannerHealth:
    """A running planner's health, after its most recent frame.

    Meant to be published as diagnostics, for example by a ROS node.
    """

    frames: int
    """Frames planned so far."""
    status: PathStatus
    """The status of the most recent frame."""
    recoveries_total: int
    recoveries_by_kind: Mapping[str, int]
    consecutive_recoveries: int
    """Frames in a row, up to the most recent, on which an error was recovered."""
    frames_since_good_path: int
    """Frames since the last OK frame; 0 if the most recent frame was OK."""
    distance_since_good_path: float
    """How far the car has moved since the last OK frame, in meters."""
    last_recovery: Recovery | None
    recoveries_this_frame: tuple[Recovery, ...] = ()


@dataclass
class HealthTracker:
    """Keeps the counters behind `PlannerHealth`, frame by frame."""

    max_consecutive_recoveries: int | None = None
    max_distance_without_good_path: float | None = None

    frames: int = 0
    recoveries_total: int = 0
    recoveries_by_kind: dict[str, int] = field(default_factory=dict)
    consecutive_recoveries: int = 0
    frames_since_good_path: int = 0
    distance_since_good_path: float = 0.0
    last_recovery: Recovery | None = None
    _last_position: FloatArray | None = None
    health: PlannerHealth | None = None

    def update(
        self,
        vehicle_position: FloatArray,
        recoveries: tuple[Recovery, ...],
        degraded: bool,
    ) -> PlannerHealth:
        """Account for one frame and return the health after it."""
        if self._last_position is not None:
            step = float(np.linalg.norm(vehicle_position - self._last_position))
        else:
            step = 0.0
        self._last_position = np.array(vehicle_position, dtype=float)
        self.frames += 1

        for recovery in recoveries:
            self.recoveries_total += 1
            self.recoveries_by_kind[recovery.kind] = (
                self.recoveries_by_kind.get(recovery.kind, 0) + 1
            )
            self.last_recovery = recovery
        self.consecutive_recoveries = (
            self.consecutive_recoveries + 1 if recoveries else 0
        )

        if recoveries:
            status = PathStatus.RECOVERED
        elif degraded:
            status = PathStatus.DEGRADED
        else:
            status = PathStatus.OK

        if status is PathStatus.OK:
            self.frames_since_good_path = 0
            self.distance_since_good_path = 0.0
        else:
            self.frames_since_good_path += 1
            self.distance_since_good_path += step

        if self._bound_exceeded():
            status = PathStatus.NO_VALID_PATH

        self.health = PlannerHealth(
            frames=self.frames,
            status=status,
            recoveries_total=self.recoveries_total,
            recoveries_by_kind=MappingProxyType(dict(self.recoveries_by_kind)),
            consecutive_recoveries=self.consecutive_recoveries,
            frames_since_good_path=self.frames_since_good_path,
            distance_since_good_path=self.distance_since_good_path,
            last_recovery=self.last_recovery,
            recoveries_this_frame=recoveries,
        )
        return self.health

    def _bound_exceeded(self) -> bool:
        if (
            self.max_consecutive_recoveries is not None
            and self.consecutive_recoveries > self.max_consecutive_recoveries
        ):
            return True
        return (
            self.max_distance_without_good_path is not None
            and self.distance_since_good_path > self.max_distance_without_good_path
        )
