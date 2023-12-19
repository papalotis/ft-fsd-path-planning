#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Special case of path calculation for skidpad.

Description: Skidpad requires a special case for path calculation. This file provides
a class that overwrites the path calculation for the skidpad track.
Project: fsd_path_planning
"""
from typing import Tuple

import numpy as np
from icecream import ic

from fsd_path_planning.calculate_path.core_calculate_path import (
    CalculatePath,
    PathCalculationInput,
)
from fsd_path_planning.skidpad.skidpad_path_data import BASE_SKIDPAD_PATH
from fsd_path_planning.types import FloatArray
from fsd_path_planning.utils.math_utils import trace_distance_to_next


class SkidpadCalculatePath(CalculatePath):
    """
    Skidpad needs a special case for path calculation. This class subclasses the general
    path calculation and injects skidpad specific logic.
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
        super().__init__(
            smoothing,
            predict_every,
            maximal_distance_for_valid_path,
            max_deg,
            mpc_path_length,
            mpc_prediction_horizon,
        )
        self.index_along_path = 0

    def set_new_input(self, new_input: PathCalculationInput) -> None:
        super().set_new_input(new_input)

    def fit_matches_as_spline(
        self,
        center_along_match_connection: FloatArray,  # pylint: disable=unused-argument
    ) -> FloatArray:
        global_path = self.input.global_path
        if global_path is None:
            return self.calculate_trivial_path()

        # estimate sampling rate
        mean_distance = np.mean(trace_distance_to_next(global_path[:10]))
        max_allowed_change = int(20 / mean_distance)

        min_index = max(self.index_along_path - max_allowed_change, 0)
        max_index = min(self.index_along_path + max_allowed_change, len(global_path))
        relevant_path = global_path[min_index:max_index]
        costs = self.cost_mpc_path_start(relevant_path)

        index_to_use = int(np.argmin(costs)) + min_index
        self.index_along_path = index_to_use

        final_index = index_to_use + int(25 / mean_distance)

        return global_path[index_to_use:final_index]
