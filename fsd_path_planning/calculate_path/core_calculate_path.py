#!/usr/bin/env python3
"""
Path calculation class.

Description: Last step in Pathing pipeline
Project: fsd_path_planning
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from fsd_path_planning.calculate_path.path_basis_selector import (
    calculate_centerline_points,
)
from fsd_path_planning.calculate_path.path_basis_selector import (
    select_side_to_use as _select_side,
)
from fsd_path_planning.calculate_path.path_basis_selector import (
    side_score as _side_score,
)
from fsd_path_planning.calculate_path.path_calculator_helpers import (
    PathCalculatorHelpers,
)
from fsd_path_planning.calculate_path.path_extender import (
    connect_path_to_car as _connect_path,
)
from fsd_path_planning.calculate_path.path_extender import (
    extend_path as _extend_path,
)
from fsd_path_planning.calculate_path.path_extender import (
    remove_path_behind_car as _remove_behind,
)
from fsd_path_planning.calculate_path.path_extender import (
    remove_path_not_in_prediction_horizon as _remove_not_in_horizon,
)
from fsd_path_planning.calculate_path.path_parameterization import PathParameterizer
from fsd_path_planning.config_dataclasses import PathConfig
from fsd_path_planning.types import FloatArray, IntArray, PathResult
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    rotate,
)
from fsd_path_planning.utils.spline_fit import SplineEvaluator, SplineFitterFactory

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
    ):
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
                provided["number_of_samples"] = provided.pop(
                    "mpc_prediction_horizon"
                )
            self.config = PathConfig(**provided)

        self.input = PathCalculationInput()
        self.scalars = PathCalculationScalarValues(
            maximal_distance_for_valid_path=self.config.maximal_distance_for_valid_path,
            mpc_path_length=self.config.path_length,
            mpc_prediction_horizon=self.config.number_of_samples,
        )
        self.path_calculator_helpers = PathCalculatorHelpers()
        self.spline_fitter_factory = SplineFitterFactory(
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
        self.mpc_paths = []
        self.path_is_trivial_list = []
        self.path_updates = []

    def calculate_initial_path(self) -> FloatArray:
        """
        Calculate the initial path.
        """

        # calculate first path
        initial_path = self.spline_fitter_factory.fit(
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

    def number_of_matches_on_one_side(self, side: ConeTypes) -> int:
        assert side in (ConeTypes.LEFT, ConeTypes.RIGHT)
        matches_of_side = (
            self.input.left_to_right_matches
            if side == ConeTypes.LEFT
            else self.input.right_to_left_matches
        )
        return_value: int = np.sum(matches_of_side != -1)
        return return_value

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
        """
        Fit the calculated basis path as a spline. If the computation fails, use the
        path calculated in the previous step
        """
        try:
            path_update = self.spline_fitter_factory.fit(
                center_along_match_connection
            ).predict(der=0)
        except ValueError:
            path_update = self.spline_fitter_factory.fit(
                self.previous_paths[-1][:, 1:3]
            ).predict(der=0)

        return path_update

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
        return path_update

    def refit_path_for_mpc_with_safety_factor(
        self, final_path: FloatArray
    ) -> FloatArray:
        """
        Refit the path for MPC with a safety factor. The length of the path is 1.5 times
        the length of the path required by MPC. The path will be trimmed to the correct length
        in another step
        """
        try:
            path_length_fixed = self.spline_fitter_factory.fit(final_path).predict(
                der=0, max_u=self.scalars.mpc_path_length * 1.5
            )
        except Exception:
            mask = np.all(final_path[:-1] == final_path[1:], axis=1)
            logger.debug(
                "Spline refit failed. Duplicate points at indices: {}",
                np.where(mask),
            )
            raise

        return path_length_fixed

    def extend_path(self, path_update: FloatArray) -> FloatArray:
        """If the path is not long enough, extend it with a circular arc or straight line."""
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
        the path is long enough for MPC. Otherwise we would have to use spline extrapolation
        to get a path that is long enough, however polynomial extrapolation is not stable
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
            self.previous_paths[-1],
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
        self.path_updates = self.path_updates[-10:] + [path_update]
        self.mpc_paths = self.mpc_paths[-10:] + [path_with_length_for_mpc]
        self.path_is_trivial_list = self.path_is_trivial_list[-10:] + [path_is_trivial]

    def run_path_calculation(
        self, input: PathCalculationInput | None = None
    ) -> PathResult:
        """Calculate path.

        Args:
            input: The path calculation input. If not provided, uses previously set input.
        """
        if input is not None:
            self.input = input
        if self.input.global_path is not None:
            distance = np.linalg.norm(
                self.input.vehicle_position - self.input.global_path, axis=1
            )

            idx_closest_point_to_path = distance.argmin()

            roll_value = -idx_closest_point_to_path + len(self.input.global_path) // 3

            path_rolled = np.roll(self.input.global_path, roll_value, axis=0)
            distance_rolled = np.roll(distance, roll_value)
            mask_distance = distance_rolled < 30
            path_rolled = path_rolled[mask_distance]
            center_along_match_connection = path_rolled

        elif len(self.input.left_cones) < 3 and len(self.input.right_cones) < 3:
            if len(self.previous_paths) > 0:
                # extract x, y from previously calculated path
                center_along_match_connection = self.previous_paths[-1][:, 1:3]
            else:
                center_along_match_connection = self.calculate_trivial_path()
        elif self.input.global_path is None:
            (
                side_to_use,
                matches_to_other_side,
                other_side_cones,
            ) = self.select_side_to_use()

            match_on_other_side = other_side_cones[matches_to_other_side]

            center_along_match_connection = self.calculate_centerline_points_of_matches(
                side_to_use, matches_to_other_side, match_on_other_side
            )

        path_update_too_far_away = self.fit_matches_as_spline(
            center_along_match_connection
        )

        path_update = self.overwrite_path_if_it_is_too_far_away(
            path_update_too_far_away
        )

        try:
            path_parameterization = self.do_all_mpc_parameter_calculations(path_update)
        except ValueError:
            # there is a bug with the path extrapolation which leads to the spline
            # fit failing, in this case we just use the previous path
            path_parameterization = self.do_all_mpc_parameter_calculations(
                self.previous_paths[-1][:, 1:3]
            )

        self.store_paths(path_update, path_parameterization, False)
        self.previous_paths = self.previous_paths[-10:] + [path_parameterization]

        return PathResult(
            final_path=path_parameterization,
            centerline_basis=center_along_match_connection,
        )
