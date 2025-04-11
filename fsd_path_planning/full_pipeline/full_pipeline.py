#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: A class that runs the whole path planning pipeline.

- Cone sorting
- Cone Matching
- Path Calculation

Project: fsd_path_planning
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from fsd_path_planning.calculate_path.core_calculate_path import PathCalculationInput
from fsd_path_planning.cone_matching.core_cone_matching import ConeMatchingInput
from fsd_path_planning.config import (
    create_default_cone_matching_with_non_monotonic_matches,
    create_default_pathing,
    create_default_sorting,
)
from fsd_path_planning.relocalization.acceleration.acceleration_relocalization import (
    AccelerationRelocalizer,
)
from fsd_path_planning.relocalization.relocalization_base_class import Relocalizer
from fsd_path_planning.relocalization.relocalization_information import (
    RelocalizationInformation,
)
from fsd_path_planning.relocalization.skidpad.skidpad_relocalizer import (
    SkidpadRelocalizer,
)
from fsd_path_planning.sorting_cones.core_cone_sorting import ConeSortingInput
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    unit_2d_vector_from_angle,
)
from fsd_path_planning.utils.mission_types import MissionTypes
from fsd_path_planning.utils.utils import Timer

MissionToRelocalizer: dict[MissionTypes, Relocalizer] = {
    MissionTypes.acceleration: AccelerationRelocalizer,
    MissionTypes.ebs_test: AccelerationRelocalizer,
    MissionTypes.skidpad: SkidpadRelocalizer,
}


class PathPlanner:
    def __init__(self, mission: MissionTypes, experimental_performance_improvements: bool = False) -> None:
        self.mission = mission

        self.relocalizer: Relocalizer | None = None
        relocalizer_class = MissionToRelocalizer.get(mission)

        if relocalizer_class is not None:
            self.relocalizer = relocalizer_class()

        self.cone_sorting = create_default_sorting(mission, experimental_performance_improvements)

        self.cone_matching = create_default_cone_matching_with_non_monotonic_matches(mission)
        self.pathing = create_default_pathing(mission)
        self.global_path: Optional[FloatArray] = None

        self.experimental_performance_improvements = experimental_performance_improvements

    def _convert_direction_to_array(self, direction: Any) -> FloatArray:
        direction = np.squeeze(np.array(direction))
        if direction.shape == (2,):
            return direction

        if direction.shape in [(1,), ()]:
            return unit_2d_vector_from_angle(direction)

        raise ValueError("direction must be a float or a 2 element array")

    def set_global_path(self, global_path):
        self.global_path = global_path

    def calculate_path_in_global_frame(
        self,
        cones: List[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: Union[FloatArray, float],
        return_intermediate_results: bool = False,
    ) -> Union[
        FloatArray,
        Tuple[
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
        vehicle_direction = self._convert_direction_to_array(vehicle_direction)

        noprint = True

        if self.relocalizer is not None:
            # attempt to relocalize
            with Timer("Relocalization", noprint=noprint):
                self.relocalizer.attempt_relocalization_calculation(cones, vehicle_position, vehicle_direction)

            if self.relocalizer.is_relocalized:
                vehicle_yaw = angle_from_2d_vector(vehicle_direction)
                (
                    vehicle_position,
                    vehicle_yaw,
                ) = self.relocalizer.transform_to_known_map_frame(vehicle_position, vehicle_yaw)
                vehicle_direction = unit_2d_vector_from_angle(vehicle_yaw)
                self.global_path = self.relocalizer.get_known_global_path()

                # print(vehicle_position, vehicle_yaw)

            sorted_left, sorted_right = np.zeros((2, 0, 2))
            left_cones_with_virtual, right_cones_with_virtual = np.zeros((2, 0, 2))
            left_to_right_match, right_to_left_match = np.zeros((2, 0), dtype=int)

        else:
            # run cone sorting
            with Timer("Cone sorting", noprint=noprint):
                cone_sorting_input = ConeSortingInput(cones, vehicle_position, vehicle_direction)
                self.cone_sorting.set_new_input(cone_sorting_input)
                sorted_left, sorted_right = self.cone_sorting.run_cone_sorting()

            # run cone matching
            with Timer("Cone matching", noprint=noprint):
                matched_cones_input = [np.zeros((0, 2)) for _ in ConeTypes]
                matched_cones_input[ConeTypes.LEFT] = sorted_left
                matched_cones_input[ConeTypes.RIGHT] = sorted_right

                cone_matching_input = ConeMatchingInput(matched_cones_input, vehicle_position, vehicle_direction)
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
                self.global_path,
            )
            self.pathing.set_new_input(path_calculation_input)
            final_path, _ = self.pathing.run_path_calculation()

        if self.relocalization_info is not None and self.relocalizer.is_relocalized:
            final_path = final_path.copy()
            # convert path points back to global frame
            path_xy = final_path[:, 1:3]
            fake_yaw = np.zeros(len(path_xy))

            # print("prev", path_xy)

            path_xy, _ = self.relocalizer.transform_to_original_frame(path_xy, fake_yaw)

            # print("trans", path_xy)

            # assert 0

            final_path = final_path.copy()

            final_path[:, 1:3] = path_xy

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

    @property
    def relocalization_info(self) -> RelocalizationInformation | None:
        if self.relocalizer is None:
            return None

        if not self.relocalizer.is_relocalized:
            return None

        return RelocalizationInformation.from_transform_function(self.relocalizer.transform_to_known_map_frame)
