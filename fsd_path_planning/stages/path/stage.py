#!/usr/bin/env python3
"""
Path calculation class.

Description: Last step in Pathing pipeline
Project: fsd_path_planning
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from fsd_path_planning._deprecation import FsdPathPlanningDeprecationWarning
from fsd_path_planning.common_types import (
    FloatArray,
    IntArray,
    PathOutcome,
    PathResult,
)
from fsd_path_planning.config_dataclasses import PathConfig
from fsd_path_planning.geometry.angles import angle_from_2d_vector
from fsd_path_planning.geometry.splines import SplineEvaluator, SplineFitter
from fsd_path_planning.geometry.transforms import rotate
from fsd_path_planning.health import RecoveryHandler
from fsd_path_planning.stages.path.centerline import (
    calculate_centerline_points,
)
from fsd_path_planning.stages.path.centerline import (
    select_side_to_use as _select_side,
)
from fsd_path_planning.stages.path.centerline import (
    side_score as _side_score,
)
from fsd_path_planning.stages.path.extension import (
    connect_path_to_car as _connect_path,
)
from fsd_path_planning.stages.path.extension import (
    extend_path as _extend_path,
)
from fsd_path_planning.stages.path.extension import (
    remove_path_behind_car as _remove_behind,
)
from fsd_path_planning.stages.path.extension import (
    remove_path_not_in_prediction_horizon as _remove_not_in_horizon,
)
from fsd_path_planning.stages.path.helpers import (
    PathCalculatorHelpers,
)
from fsd_path_planning.stages.path.parameterization import PathParameterizer
from fsd_path_planning.stages.path.strategies import (
    DEFAULT_BASIS_STRATEGIES,
    PathBasisStrategy,
    PathFitStrategy,
    SplineThroughBasis,
)
from fsd_path_planning.utils.cone_types import ConeTypes

logger = logging.getLogger(__name__)

SplineEvalByType = list[SplineEvaluator]


@dataclass
class PathCalculationInput:
    """Dataclass holding calculation variables."""

    # pylint: disable=too-many-instance-attributes
    left_cones: FloatArray = field(default_factory=lambda: np.zeros((0, 2)))
    right_cones: FloatArray = field(default_factory=lambda: np.zeros((0, 2)))
    left_to_right_matches: IntArray = field(
        default_factory=lambda: np.zeros(0, dtype=int)
    )
    right_to_left_matches: IntArray = field(
        default_factory=lambda: np.zeros(0, dtype=int)
    )
    vehicle_position: FloatArray = field(default_factory=lambda: np.zeros((0, 2)))
    vehicle_direction: FloatArray = field(default_factory=lambda: np.array([1, 0]))
    global_path: FloatArray | None = field(default=None)


@dataclass
class PathCalculationScalarValues:
    """Class holding scalar values of a path calculator."""

    maximal_distance_for_valid_path: float
    mpc_path_length: float = 30
    mpc_prediction_horizon: int = 40


class CalculatePath:
    """
    Class that takes all path calculation responsibilities after the cones have been
    matched.
    """

    def __init__(
        self,
        config: PathConfig | None = None,
        *,
        # Legacy parameters (deprecated, use config instead)
        smoothing: float | None = None,
        predict_every: float | None = None,
        maximal_distance_for_valid_path: float | None = None,
        max_deg: int | None = None,
        mpc_path_length: float | None = None,
        mpc_prediction_horizon: int | None = None,
        basis_strategies: Sequence[PathBasisStrategy] | None = None,
        fit_strategy: PathFitStrategy | None = None,
    ):
        """
        Args:
            config: The path calculation settings.
            basis_strategies: How the points the path is built through are
                chosen, tried in order until one applies; see
                `calculate_path.strategies`. Defaults to the known map, then
                the previous path when there are too few cones, then the
                centreline of the matched cones.
            fit_strategy: How the path is fitted through those points. Defaults
                to a smoothing spline.
        """
        self.basis_strategies: tuple[PathBasisStrategy, ...] = (
            tuple(basis_strategies)
            if basis_strategies is not None
            else DEFAULT_BASIS_STRATEGIES
        )
        self.fit_strategy: PathFitStrategy = (
            fit_strategy if fit_strategy is not None else SplineThroughBasis()
        )
        self.recovery = RecoveryHandler()
        """Applies the error policy at the fallbacks. PathPlanner replaces it
        with its own, so that recoveries reach the planner's health record."""
        self._basis_outcome = PathOutcome.FRESH
        self._replaced_too_far = False
        if config is not None:
            self.config = config
        else:
            legacy_params = {
                "smoothing": smoothing,
                "predict_every": predict_every,
                "maximal_distance_for_valid_path": maximal_distance_for_valid_path,
                "max_deg": max_deg,
                "mpc_path_length": mpc_path_length,
                "mpc_prediction_horizon": mpc_prediction_horizon,
            }
            provided = {k: v for k, v in legacy_params.items() if v is not None}
            if provided:
                warnings.warn(
                    "Passing individual parameters to CalculatePath is deprecated. "
                    "Use CalculatePath(config=PathConfig(...)) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if "mpc_path_length" in provided and "path_length" not in provided:
                provided["path_length"] = provided.pop("mpc_path_length")
            if (
                "mpc_prediction_horizon" in provided
                and "number_of_samples" not in provided
            ):
                provided["number_of_samples"] = provided.pop("mpc_prediction_horizon")
            self.config = PathConfig(**provided)

        self.input = PathCalculationInput()
        self.scalars = PathCalculationScalarValues(
            maximal_distance_for_valid_path=self.config.maximal_distance_for_valid_path,
            mpc_path_length=self.config.path_length,
            mpc_prediction_horizon=self.config.number_of_samples,
        )
        self.path_calculator_helpers = PathCalculatorHelpers()
        self.spline_fitter = SplineFitter(
            self.config.smoothing, self.config.predict_every, self.config.max_deg
        )

        path_parameterizer = PathParameterizer(
            prediction_horizon=self.scalars.mpc_prediction_horizon
        )

        self.previous_paths = [
            path_parameterizer.parameterize_path(
                self.calculate_initial_path(), None, None, False
            )
        ]
        self._mpc_paths: list[FloatArray] = []
        self._path_is_trivial_list: list[bool] = []
        self._path_updates: list[FloatArray] = []

    @property
    def spline_fitter_factory(self) -> SplineFitter:
        """Deprecated name of `spline_fitter`."""
        warnings.warn(
            "CalculatePath.spline_fitter_factory has been renamed to"
            " spline_fitter; the old name will be removed in 0.7.0",
            FsdPathPlanningDeprecationWarning,
            stacklevel=2,
        )
        return self.spline_fitter

    def calculate_initial_path(self) -> FloatArray:
        """
        Calculate the initial path.
        """

        # calculate first path
        initial_path = self.spline_fitter.fit(
            self.path_calculator_helpers.calculate_almost_straight_path(
                radius=self.config.initial_path_radius,
                maximum_angle=self.config.initial_path_angle,
                number_of_points=self.config.initial_path_points,
            )
        ).predict(der=0)
        return initial_path

    def set_new_input(self, new_input: PathCalculationInput) -> None:
        """Update the state of the calculation.

        .. deprecated::
            Pass input directly to :meth:`run_path_calculation` instead.
        """
        warnings.warn(
            "set_new_input() is deprecated. Pass input directly to "
            "run_path_calculation().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.input = new_input

    def calculate_trivial_path(self) -> FloatArray:
        "Calculate a path that points straight from the car position and direction"
        origin_path = self.path_calculator_helpers.calculate_almost_straight_path()[1:]
        yaw = angle_from_2d_vector(self.input.vehicle_direction)
        path_rotated: FloatArray = rotate(origin_path, yaw)  # type: ignore

        final_trivial_path: FloatArray = path_rotated + self.input.vehicle_position
        return final_trivial_path

    def side_score(self, side: ConeTypes) -> tuple:
        matches_of_side = (
            self.input.left_to_right_matches
            if side == ConeTypes.LEFT
            else self.input.right_to_left_matches
        )
        return _side_score(matches_of_side)

    def select_side_to_use(self) -> tuple[FloatArray, IntArray, FloatArray]:
        "Select the main side to use for path calculation"
        return _select_side(
            self.input.left_cones,
            self.input.right_cones,
            self.input.left_to_right_matches,
            self.input.right_to_left_matches,
        )

    def calculate_centerline_points_of_matches(
        self,
        side_to_use: FloatArray,
        matches_to_other_side: IntArray,
        match_on_other_side: FloatArray,
    ) -> FloatArray:
        return calculate_centerline_points(
            side_to_use,
            matches_to_other_side,
            match_on_other_side,
            self.previous_paths[-1][:, 1:3],
        )

    def fit_matches_as_spline(
        self, center_along_match_connection: FloatArray
    ) -> FloatArray:
        """Fit the path update through the basis, with the fit strategy."""
        return self.fit_strategy.fit(self, self.input, center_along_match_connection)

    def select_path_basis(self) -> FloatArray:
        """The points the path is built through: from the first basis strategy
        that applies to this frame."""
        for strategy in self.basis_strategies:
            basis = strategy.basis(self, self.input)
            if basis is not None:
                self._basis_outcome = getattr(strategy, "outcome", PathOutcome.FRESH)
                return basis
        raise RuntimeError("no path basis strategy applies to this frame")

    def overwrite_path_if_it_is_too_far_away(
        self, path_update: FloatArray
    ) -> FloatArray:
        """
        If for some reason the calculated path is too far away from the position of the
        car (e.g. because of a bad sorting), the previously calculated path is used
        """
        min_distance_to_path = np.linalg.norm(
            self.input.vehicle_position - path_update, axis=-1
        ).min()
        if min_distance_to_path > self.scalars.maximal_distance_for_valid_path:
            path_update = self.previous_paths[-1][:, 1:3]
            self._replaced_too_far = True
        return path_update

    def refit_path_for_mpc_with_safety_factor(
        self, final_path: FloatArray
    ) -> FloatArray:
        """
        Refit the path for MPC with a safety factor. The length of the path is 1.5 times
        the length of the path required by MPC. The path will be trimmed to
        the correct length
        in another step
        """
        try:
            path_length_fixed = self.spline_fitter.fit(final_path).predict(
                der=0, max_u=self.scalars.mpc_path_length * 1.5
            )
        except Exception:
            mask = np.all(final_path[:-1] == final_path[1:], axis=1)
            logger.debug(
                "Spline refit failed. Duplicate points at indices: %s",
                np.where(mask),
            )
            raise

        return path_length_fixed

    def extend_path(self, path_update: FloatArray) -> FloatArray:
        """
        If the path is not long enough, extend it with a circular arc or
        straight line.
        """
        return _extend_path(
            path_update,
            self.input.vehicle_position,
            self.input.vehicle_direction,
            self.scalars.mpc_path_length,
            self.config.circle_fit_tail_points,
            self.config.min_extension_radius,
            self.config.max_extension_radius,
            self.config.circular_arc_threshold,
            self.config.straight_extension_points,
        )

    def create_path_for_mpc_from_path_update(
        self, path_update: FloatArray
    ) -> FloatArray:
        """
        Calculate the path for MPC from the path update. The path update is the basis of
        the new path.

        First a linear path is added at the end of the path update. This ensures that
        the path is long enough for MPC. Otherwise we would have to use
        spline extrapolation to get a path that is long enough, however
        polynomial extrapolation is not stable
        enough for our purposes.

        Then the path is fitted again as a spline. Because we have now added the linear
        part we can be sure that no polynomial extrapolation will be used.

        Then any path behind the car is removed.

        Finally the path is trimmed to the correct length, as desired from MPC.

        Args:
            path_update: The basis of the new path

        Returns:
            The path for MPC
        """
        path_connected_to_car = self.connect_path_to_car(path_update)
        path_with_enough_length = self.extend_path(path_connected_to_car)
        path_with_no_path_behind_car = self.remove_path_behind_car(
            path_with_enough_length
        )
        try:
            path_length_fixed = self.refit_path_for_mpc_with_safety_factor(
                path_with_no_path_behind_car
            )
        except Exception:
            logger.debug("Spline refit failed during MPC path creation")
            raise

        path_with_length_for_mpc = self.remove_path_not_in_prediction_horizon(
            path_length_fixed
        )

        return path_with_length_for_mpc

    def do_all_mpc_parameter_calculations(self, path_update: FloatArray) -> FloatArray:
        """
        Calculate the path that will be sent to the MPC. The general path that is
        calculated is based on the cones around the track and is also based on the
        surroundings (also cones from behind the car), which means that this path
        has an undefined length and starts behind the car. MPC expects the path to
        start where the car is and for it to have a specific length (both in meters,
        but also in the number of elements it is composed of). This method extrapolates
        the path if the length is not enough, removes the parts of the path that are
        behind the car and finally samples the path so that it has exactly as many
        elements as MPC needs.

        Args:
            path_update: The basis of the new path

        Returns:
            The parameterized path as a Nx4 array, where each column is:
                theta (spline parameter)
                x (x coordinate)
                y (y coordinate)
                curvature (curvature of the path at that point)
        """

        path_with_length_for_mpc = self.create_path_for_mpc_from_path_update(
            path_update
        )

        path_parameterizer = PathParameterizer(
            prediction_horizon=self.scalars.mpc_prediction_horizon
        )
        path_parameterized = path_parameterizer.parameterize_path(
            path_with_length_for_mpc,
            self.input.vehicle_position,
            self.input.vehicle_direction,
            path_is_closed=False,
        )

        return path_parameterized

    def cost_mpc_path_start(self, path_length_fixed: FloatArray) -> FloatArray:
        """Cost function for start of MPC path."""
        distance_cost: FloatArray = np.linalg.norm(
            self.input.vehicle_position - path_length_fixed, axis=1
        )
        return distance_cost

    def connect_path_to_car(self, path_update: FloatArray) -> FloatArray:
        """Connect the path update to the current position of the car."""
        return _connect_path(
            path_update, self.input.vehicle_position, self.input.vehicle_direction
        )

    def remove_path_behind_car(self, path_length_fixed: FloatArray) -> FloatArray:
        """Remove part of the path that is behind the car."""
        return _remove_behind(path_length_fixed, self.input.vehicle_position)

    def remove_path_not_in_prediction_horizon(
        self, path_length_fixed_forward: FloatArray
    ) -> FloatArray:
        """Truncate the path to the MPC prediction horizon length."""
        return _remove_not_in_horizon(
            path_length_fixed_forward,
            self.scalars.mpc_path_length,
            # The previous path is stored parameterized as (theta, x, y,
            # curvature); the caller needs the (x, y) points it is about to
            # parameterize again.
            self.previous_paths[-1][:, 1:3],
        )

    def store_paths(
        self,
        path_update: FloatArray,
        path_with_length_for_mpc: FloatArray,
        path_is_trivial: bool,
    ) -> None:
        """
        Store the calculated paths, in case they are need in the next calculation.
        """
        self._path_updates = self._path_updates[-10:] + [path_update]
        self._mpc_paths = self._mpc_paths[-10:] + [path_with_length_for_mpc]
        self._path_is_trivial_list = self._path_is_trivial_list[-10:] + [
            path_is_trivial
        ]

    def _warn_history(self, name: str) -> None:
        warnings.warn(
            f"CalculatePath.{name} is deprecated and will be removed in 0.7.0;"
            " the planner never reads it. previous_paths holds the paths it"
            " uses.",
            FsdPathPlanningDeprecationWarning,
            stacklevel=3,
        )

    @property
    def path_updates(self) -> list[FloatArray]:
        """Deprecated: the last path updates. Nothing reads them."""
        self._warn_history("path_updates")
        return self._path_updates

    @property
    def mpc_paths(self) -> list[FloatArray]:
        """Deprecated: the last parameterized paths; see previous_paths."""
        self._warn_history("mpc_paths")
        return self._mpc_paths

    @property
    def path_is_trivial_list(self) -> list[bool]:
        """Deprecated: always False. Nothing reads it."""
        self._warn_history("path_is_trivial_list")
        return self._path_is_trivial_list

    def run_path_calculation(
        self, input: PathCalculationInput | None = None
    ) -> PathResult:
        """Calculate path.

        Args:
            input: The path calculation input. If not provided, uses
                previously set input.
        """
        if input is not None:
            self.input = input
        self._replaced_too_far = False
        center_along_match_connection = self.select_path_basis()

        path_update_too_far_away = self.fit_matches_as_spline(
            center_along_match_connection
        )

        path_update = self.overwrite_path_if_it_is_too_far_away(
            path_update_too_far_away
        )

        try:
            path_parameterization = self.do_all_mpc_parameter_calculations(path_update)
        except ValueError as error:
            # there is a bug with the path extrapolation which leads to the spline
            # fit failing, in this case we just use the previous path
            self.recovery.recover("path", "parameterization_failed", error)
            path_parameterization = self.do_all_mpc_parameter_calculations(
                self.previous_paths[-1][:, 1:3]
            )

        self.store_paths(path_update, path_parameterization, False)
        self.previous_paths = self.previous_paths[-10:] + [path_parameterization]

        if self._replaced_too_far:
            outcome = PathOutcome.TOO_FAR_FROM_CAR
        else:
            outcome = self._basis_outcome

        return PathResult(
            final_path=path_parameterization,
            centerline_basis=center_along_match_connection,
            outcome=outcome,
        )

    def run(self, stage_input: PathCalculationInput) -> PathResult:
        """Calculate the path for `stage_input`; the `Stage` entry point."""
        return self.run_path_calculation(stage_input)
