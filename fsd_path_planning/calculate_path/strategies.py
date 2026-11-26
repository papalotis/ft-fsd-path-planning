"""
The interchangeable steps of path calculation.

A path is calculated in two steps that differ between missions:

1. The *basis*: the points the new path is built through. The first basis
   strategy that applies to the frame provides them:

   - `KnownMapWindow`: a window of the known global path around the car, for
     missions that have relocalized onto a known map.
   - `KeepPrevious`: too few cones on both sides; reuse the previous path, or
     a straight path from the car before there is one.
   - `MatchCenterline`: the midpoints between matched left and right cones.

2. The *fit*: turning the basis into the path update.

   - `SplineThroughBasis`: a smoothing spline through the basis, falling back
     to the previous path if the fit fails. The default.
   - `FollowKnownMap`: ignore the basis and follow the known global path from
     the point closest to the car, searching only near where the car was on
     the previous frame. Skidpad, whose figure-of-eight crosses itself.

`CalculatePath` holds one tuple of basis strategies and one fit strategy, and
runs the rest of the calculation (connecting to the car, extending,
parameterizing) the same way for every mission.

Project: fsd_path_planning
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from fsd_path_planning.common_types import FloatArray
from fsd_path_planning.geometry.traces import trace_distance_to_next

if TYPE_CHECKING:
    from fsd_path_planning.calculate_path.core_calculate_path import (
        CalculatePath,
        PathCalculationInput,
    )


class PathBasisStrategy(Protocol):
    """Chooses the points a new path is built through."""

    def basis(
        self, calculator: CalculatePath, stage_input: PathCalculationInput
    ) -> FloatArray | None:
        """The basis for this frame, or None if the strategy does not apply."""
        ...


class PathFitStrategy(Protocol):
    """Turns the basis into the path update."""

    def fit(
        self,
        calculator: CalculatePath,
        stage_input: PathCalculationInput,
        basis: FloatArray,
    ) -> FloatArray:
        """The path update, an (N, 2) array of points."""
        ...


class KnownMapWindow:
    """The part of the known global path within reach of the car."""

    def basis(
        self, calculator: CalculatePath, stage_input: PathCalculationInput
    ) -> FloatArray | None:
        global_path = stage_input.global_path
        if global_path is None:
            return None

        distance = np.linalg.norm(stage_input.vehicle_position - global_path, axis=1)
        idx_closest_point_to_path = distance.argmin()

        # Roll the closed path so that the closest point sits a third of the
        # way along it, then keep the points within range of the car.
        roll_value = -idx_closest_point_to_path + len(global_path) // 3
        path_rolled = np.roll(global_path, roll_value, axis=0)
        distance_rolled = np.roll(distance, roll_value)
        mask_distance = distance_rolled < calculator.config.known_map_window_radius
        return path_rolled[mask_distance]


class KeepPrevious:
    """With fewer than three cones on each side, keep the previous path."""

    def basis(
        self, calculator: CalculatePath, stage_input: PathCalculationInput
    ) -> FloatArray | None:
        if len(stage_input.left_cones) >= 3 or len(stage_input.right_cones) >= 3:
            return None
        if len(calculator.previous_paths) > 0:
            # x, y of the previous parameterized path
            return calculator.previous_paths[-1][:, 1:3]
        return calculator.calculate_trivial_path()


class MatchCenterline:
    """Midpoints between the cones of the better side and their matches."""

    def basis(
        self, calculator: CalculatePath, stage_input: PathCalculationInput
    ) -> FloatArray | None:
        side_to_use, matches_to_other_side, other_side_cones = (
            calculator.select_side_to_use()
        )
        match_on_other_side = other_side_cones[matches_to_other_side]
        return calculator.calculate_centerline_points_of_matches(
            side_to_use, matches_to_other_side, match_on_other_side
        )


DEFAULT_BASIS_STRATEGIES: tuple[PathBasisStrategy, ...] = (
    KnownMapWindow(),
    KeepPrevious(),
    MatchCenterline(),
)


class SplineThroughBasis:
    """A smoothing spline through the basis; the previous path if it fails."""

    def fit(
        self,
        calculator: CalculatePath,
        stage_input: PathCalculationInput,
        basis: FloatArray,
    ) -> FloatArray:
        try:
            return calculator.spline_fitter.fit(basis).predict(der=0)
        except ValueError:
            return calculator.spline_fitter.fit(
                calculator.previous_paths[-1][:, 1:3]
            ).predict(der=0)


class FollowKnownMap:
    """Follow the known global path ahead of the car, ignoring the basis.

    The point to start from is searched only within
    `PathConfig.known_map_search_distance` of where the car was on the
    previous frame, so that where the path crosses itself the car keeps to
    the branch it is on.
    """

    def __init__(self) -> None:
        self.index_along_path = 0
        """Index into the global path of the car's last position."""

    def fit(
        self,
        calculator: CalculatePath,
        stage_input: PathCalculationInput,
        basis: FloatArray,  # noqa: ARG002
    ) -> FloatArray:
        global_path = stage_input.global_path
        if global_path is None:
            return calculator.calculate_trivial_path()

        # estimate sampling rate
        mean_distance = np.mean(trace_distance_to_next(global_path[:10]))
        config = calculator.config
        max_allowed_change = int(config.known_map_search_distance / mean_distance)

        min_index = max(self.index_along_path - max_allowed_change, 0)
        max_index = min(self.index_along_path + max_allowed_change, len(global_path))
        relevant_path = global_path[min_index:max_index]
        costs = calculator.cost_mpc_path_start(relevant_path)

        index_to_use = int(np.argmin(costs)) + min_index
        self.index_along_path = index_to_use

        final_index = index_to_use + int(config.known_map_length_ahead / mean_distance)

        return global_path[index_to_use:final_index]
