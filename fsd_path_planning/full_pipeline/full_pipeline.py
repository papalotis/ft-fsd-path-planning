#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: A class that runs the whole path planning pipeline.

- Cone sorting
- Cone Matching
- Path Calculation

Project: fsd_path_planning
"""
from typing import List, Union

import numpy as np

from fsd_path_planning.calculate_path.core_calculate_path import PathCalculationInput
from fsd_path_planning.cone_matching.core_cone_matching import ConeMatchingInput
from fsd_path_planning.config import (
    create_default_cone_matching_with_non_monotonic_matches,
    create_default_pathing,
    create_default_sorting,
)
from fsd_path_planning.sorting_cones.utils.cone_sorting_dataclasses import (
    ConeSortingInput,
)
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.mission_types import MissionTypes
from fsd_path_planning.utils.utils import Timer


class PathPlanner:
    def __init__(self, mission: MissionTypes):
        self.cone_sorting = create_default_sorting(mission)
        self.cone_matching = create_default_cone_matching_with_non_monotonic_matches(
            mission
        )
        self.pathing = create_default_pathing(mission)

    def calculate_path_in_global_frame(
        self,
        cones: List[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
        return_intermediate_results: bool = False,
    ) -> Union[
        FloatArray,
        tuple[
            FloatArray,
            FloatArray,
            FloatArray,
            FloatArray,
            FloatArray,
            IntArray,
            IntArray,
        ],
    ]:
        """
        Runs the whole path planning pipeline.

        Args:
            cones: List of cones in global frame. Position of Nx2 arrays based on
            `ConeTypes`.
            vehicle_position: Vehicle position in global frame. 2 element array (x,y).
            vehicle_direction: Vehicle direction in global frame. 2 element array
            (dir_x,dir_y).
            return_intermediate_results: If True, returns intermediate results (sorting)
            and matching).

        Returns:
            A Nx4 array of waypoints in global frame. Each waypoint is a 4 element array
            (spline_parameter, path_x, path_y, curvature).
        """
        noprint = True

        # run cone sorting
        with Timer("Cone sorting", noprint=noprint):
            cone_sorting_input = ConeSortingInput(
                cones, vehicle_position, vehicle_direction
            )
            self.cone_sorting.set_new_input(cone_sorting_input)
            sorted_left, sorted_right = self.cone_sorting.run_cone_sorting()

        # run cone matching
        with Timer("Cone matching", noprint=noprint):
            matched_cones_input = [np.zeros((0, 2)) for _ in ConeTypes]
            matched_cones_input[ConeTypes.LEFT] = sorted_left
            matched_cones_input[ConeTypes.RIGHT] = sorted_right

            cone_matching_input = ConeMatchingInput(
                matched_cones_input, vehicle_position, vehicle_direction
            )
            self.cone_matching.set_new_input(cone_matching_input)
            (
                left_cones_with_virtual,
                right_cones_with_virtual,
                left_to_right_match,
                right_to_left_match,
            ) = self.cone_matching.run_cone_matching()

        # run path calculation
        with Timer("Path calculation", noprint=noprint):
            path_calculation_input = PathCalculationInput(
                left_cones_with_virtual,
                right_cones_with_virtual,
                left_to_right_match,
                right_to_left_match,
                vehicle_position,
                vehicle_direction,
            )
            self.pathing.set_new_input(path_calculation_input)
            final_path, _ = self.pathing.run_path_calculation()

        if return_intermediate_results:
            return (
                final_path,
                sorted_left,
                sorted_right,
                left_cones_with_virtual,
                right_cones_with_virtual,
                left_to_right_match,
                right_to_left_match,
            )

        return final_path
