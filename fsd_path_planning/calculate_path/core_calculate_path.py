#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Path calculation class.

Description: Last step in Pathing pipeline
Project: fsd_path_planning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, cast

import numpy as np
from icecream import ic  # pylint: disable=unused-import

from fsd_path_planning.calculate_path.path_calculator_helpers import (
    PathCalculatorHelpers,
)
from fsd_path_planning.calculate_path.path_parameterization import PathParameterizer
from fsd_path_planning.types import BoolArray, FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    circle_fit,
    normalize_last_axis,
    rotate,
    trace_distance_to_next,
    unit_2d_vector_from_angle,
    vec_angle_between,
)
from fsd_path_planning.utils.spline_fit import SplineEvaluator, SplineFitterFactory

SplineEvalByType = List[SplineEvaluator]


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
    position_global: FloatArray = field(default_factory=lambda: np.zeros((0, 2)))
    direction_global: FloatArray = field(default_factory=lambda: np.array([1, 0]))
    global_path: Optional[FloatArray] = field(default=None)


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
        smoothing: float,
        predict_every: float,
        maximal_distance_for_valid_path: float,
        max_deg: int,
        mpc_path_length: float,
        mpc_prediction_horizon: int,
    ):
        """
        Init method.

        Args:
            smoothing, predict_every, max_deg: Arguments for cone fitting.
            maximal_distance_for_valid_path: Maximum distance for a valid path. If the
                calculated path has a minimum distance from the car that is larger than
                this value, the path is not valid, and the previously calculated path is
                used.
        """
        self.input = PathCalculationInput()
        self.scalars = PathCalculationScalarValues(
            maximal_distance_for_valid_path=maximal_distance_for_valid_path,
            mpc_path_length=mpc_path_length,
            mpc_prediction_horizon=mpc_prediction_horizon,
        )
        self.path_calculator_helpers = PathCalculatorHelpers()
        self.spline_fitter_factory = SplineFitterFactory(
            smoothing, predict_every, max_deg
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
            self.path_calculator_helpers.calculate_almost_straight_path()
        ).predict(der=0)
        return initial_path

    def set_new_input(self, new_input: PathCalculationInput) -> None:
        """Update the state of the calculation."""
        self.input = new_input

    def calculate_trivial_path(self) -> FloatArray:
        "Calculate a path that points straight from the car position and direction"
        origin_path = self.path_calculator_helpers.calculate_almost_straight_path()[1:]
        yaw = angle_from_2d_vector(self.input.direction_global)
        path_rotated: FloatArray = rotate(origin_path, yaw)  # type: ignore
        final_trivial_path: FloatArray = path_rotated + self.input.position_global
        return final_trivial_path

    def number_of_matches_on_one_side(self, side: ConeTypes) -> int:
        """
        The matches array contains the index of the matched cone of the other side.
        If a cone does not have a match the index is set -1. This method finds how
        many cones actually have a match (the index of the match is not -1)
        """
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
        matches_of_side_filtered = matches_of_side[matches_of_side != -1]
        n_matches = len(matches_of_side_filtered)
        n_indices_sum = matches_of_side_filtered.sum()

        # first pick side with most matches, if both same number of matches, pick side
        # where the indices increase the most
        return n_matches, n_indices_sum

    def select_side_to_use(self) -> Tuple[FloatArray, IntArray, FloatArray]:
        "Select the main side to use for path calculation"

        side_to_pick = max([ConeTypes.LEFT, ConeTypes.RIGHT], key=self.side_score)

        side_to_use, matches_to_other_side, other_side_cones = (
            (
                self.input.left_cones,
                self.input.left_to_right_matches,
                self.input.right_cones,
            )
            if side_to_pick == ConeTypes.LEFT
            else (
                self.input.right_cones,
                self.input.right_to_left_matches,
                self.input.left_cones,
            )
        )
        return side_to_use, matches_to_other_side, other_side_cones

    def calculate_centerline_points_of_matches(
        self,
        side_to_use: FloatArray,
        matches_to_other_side: IntArray,
        match_on_other_side: FloatArray,
    ) -> FloatArray:
        """
        Calculate the basis of the new path by computing the middle between one side of
        the track and its corresponding match. If there are not enough cones with
        matches, the path from the previous calculation is used.
        """
        center_along_match_connection = (side_to_use + match_on_other_side) / 2
        center_along_match_connection = center_along_match_connection[
            matches_to_other_side != -1
        ]

        # need at least 2 points for path calculation
        if len(center_along_match_connection) < 2:
            center_along_match_connection = self.previous_paths[-1][:, 1:3]

        return center_along_match_connection

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
            self.input.position_global - path_update, axis=-1
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
        except Exception as e:
            print(e)
            mask = np.all(final_path[:-1] == final_path[1:], axis=1)
            print(np.where(mask))
            # print(repr(final_path))
            # print(repr(self.input))
            raise

        return path_length_fixed

    def extend_path(self, path_update: FloatArray) -> FloatArray:
        """
        If the path is not long enough, extend it with the path with a circular arc
        """

        ## find the length of the path in front of the car

        # find angle to each point in the path
        car_to_path = path_update - self.input.position_global
        mask_path_is_in_front_of_car = (
            np.dot(car_to_path, self.input.direction_global) > 0
        )
        # as soon as we find a point that is in front of the car, we can mark all the
        # points after it as being in front of the car
        for i, value in enumerate(mask_path_is_in_front_of_car.copy()):
            if value:
                mask_path_is_in_front_of_car[i:] = True
                break

        mask_path_is_in_front_of_car[-20:] = True

        if not mask_path_is_in_front_of_car.any():
            return path_update

        path_infront_of_car = path_update[mask_path_is_in_front_of_car]

        cum_path_length = trace_distance_to_next(path_infront_of_car).cumsum()
        # finally we get the length of the path in front of the car
        path_length = cum_path_length[-1]

        if path_length > self.scalars.mpc_path_length:
            return path_update

        # select n last points of the path and estimate the circle they form
        relevant_path = path_infront_of_car[-20:]
        center_x, center_y, radius = circle_fit(relevant_path)
        center = np.array([center_x, center_y])

        radius_to_use = min(max(radius, 10), 100)

        if radius_to_use < 80:
            # ic(center_x, center_y, radius_to_use)
            relevant_path_centered = relevant_path - center
            # find the orientation of the path part, to know if the circular arc should be
            # clockwise or counterclockwise
            three_points = relevant_path_centered[
                [0, int(len(relevant_path_centered) / 2), -1]
            ]

            # https://en.wikipedia.org/wiki/Curve_orientation#Orientation_of_a_simple_polygon
            homogeneous_points = np.column_stack((np.ones(3), three_points))
            orientation = np.linalg.det(homogeneous_points)
            orientation_sign = np.sign(orientation)

            # create the circular arc
            start_angle = float(angle_from_2d_vector(three_points[0]))
            end_angle = start_angle + orientation_sign * np.pi
            new_points_angles = np.linspace(start_angle, end_angle)
            new_points_raw = (
                unit_2d_vector_from_angle(new_points_angles) * radius_to_use
            )

            new_points = new_points_raw - new_points_raw[0] + path_update[-1]
            # ic(new_points)
            # to avoid overlapping when spline fitting, we need to first n points
        else:
            second_last_point = path_update[-2]
            last_point = path_update[-1]
            direction = last_point - second_last_point
            direction = direction / np.linalg.norm(direction)
            new_points = last_point + direction * np.arange(30)[:, None]

        new_points = new_points[1:]
        return np.row_stack((path_update, new_points))

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
            print("path update")
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
            self.input.position_global,
            self.input.direction_global,
            path_is_closed=False,
        )

        return path_parameterized

    def cost_mpc_path_start(self, path_length_fixed: FloatArray) -> FloatArray:
        """
        Cost function for start of MPC path. The cost is based on the distance from the
        car to the calculated path. Mission specific cost functions can be added here.
        """

        distance_cost: FloatArray = np.linalg.norm(
            self.input.position_global - path_length_fixed, axis=1
        )
        return distance_cost

    def connect_path_to_car(self, path_update: FloatArray) -> FloatArray:
        """
        Connect the path update to the current path of the car. This is done by
        calculating the distance between the last point of the path update and the
        current position of the car. The path update is then shifted by this distance.
        """
        distance_to_first_point = np.linalg.norm(
            self.input.position_global - path_update[0]
        )

        car_to_first_point = path_update[0] - self.input.position_global

        angle_to_first_point = vec_angle_between(
            car_to_first_point, self.input.direction_global
        )

        # there is path behind car or start is close enough
        if distance_to_first_point < 0.5 or angle_to_first_point > np.pi / 2:
            return path_update

        new_point = (
            self.input.position_global
            + normalize_last_axis(car_to_first_point[None])[0] * 0.2
        )

        path_update = np.row_stack((new_point, path_update))

        return path_update

    def remove_path_behind_car(self, path_length_fixed: FloatArray) -> FloatArray:
        """
        Remove part of the path that is behind the car.
        """
        idx_start_mpc_path = int(self.cost_mpc_path_start(path_length_fixed).argmin())
        path_length_fixed_forward: FloatArray = path_length_fixed[idx_start_mpc_path:]
        return path_length_fixed_forward

    def remove_path_not_in_prediction_horizon(
        self, path_length_fixed_forward: FloatArray
    ) -> FloatArray:
        """
        If the path with fixed length is too long, for the needs of MPC, it is
        truncated to the desired length.
        """
        distances = trace_distance_to_next(path_length_fixed_forward)
        cum_dist = np.cumsum(distances)
        # the code crashes if cum_dist is smaller than mpc_path_length -->
        # atm mpc_path_length has to be long enough so that doesn't happen
        # TODO: change it so that it is not dependent on mpc_path_length
        mask_cum_distance_over_mcp_path_length: BoolArray = (
            cum_dist > self.scalars.mpc_path_length
        )
        if len(mask_cum_distance_over_mcp_path_length) <= 1:
            return self.previous_paths[-1]

        first_point_over_distance = cast(
            int, mask_cum_distance_over_mcp_path_length.argmax()
        )

        # if all the elements in the mask are false then argmax will return 0, we need
        # to detect this case and use the whole path when this happens
        if (
            first_point_over_distance == 0
            and not mask_cum_distance_over_mcp_path_length[0]
        ):
            first_point_over_distance = len(cum_dist)
        path_with_length_for_mpc: FloatArray = path_length_fixed_forward[
            :first_point_over_distance
        ]
        return path_with_length_for_mpc

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

    def run_path_calculation(self) -> Tuple[FloatArray, FloatArray]:
        """Calculate path."""
        if len(self.input.left_cones) < 3 and len(self.input.right_cones) < 3:
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
        else:
            distance = np.linalg.norm(
                self.input.position_global - self.input.global_path, axis=1
            )

            idx_closest_point_to_path = distance.argmin()

            roll_value = -idx_closest_point_to_path + len(self.input.global_path) // 3

            path_rolled = np.roll(self.input.global_path, roll_value, axis=0)
            distance_rolled = np.roll(distance, roll_value)
            mask_distance = distance_rolled < 30
            path_rolled = path_rolled[mask_distance]
            center_along_match_connection = path_rolled

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

        return path_parameterization, center_along_match_connection
