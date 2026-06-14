#!/usr/bin/env python3
"""
Description: A class that runs the whole path planning pipeline.

- Cone sorting
- Cone Matching
- Path Calculation

Project: fsd_path_planning
"""

from __future__ import annotations

import numpy as np

from fsd_path_planning.calculate_path.core_calculate_path import (
    CalculatePath,
    PathCalculationInput,
)
from fsd_path_planning.cone_matching.core_cone_matching import (
    ConeMatching,
    ConeMatchingInput,
)
from fsd_path_planning.config import (
    create_default_cone_matching_with_non_monotonic_matches,
    create_default_pathing,
    create_default_sorting,
)
from fsd_path_planning.config_dataclasses import PipelineConfig, default_config
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
from fsd_path_planning.sorting_cones.core_cone_sorting import (
    ConeSorting,
    ConeSortingInput,
)
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.input_validation import (
    convert_direction_to_array,
    validate_and_normalize_cones,
    validate_vehicle_position,
)
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    unit_2d_vector_from_angle,
)
from fsd_path_planning.utils.mission_types import MissionTypes
from fsd_path_planning.utils.utils import Timer

MissionToRelocalizer: dict[MissionTypes, type[Relocalizer]] = {
    MissionTypes.acceleration: AccelerationRelocalizer,
    MissionTypes.ebs_test: AccelerationRelocalizer,
    MissionTypes.skidpad: SkidpadRelocalizer,
}


class PathPlanner:
    def __init__(
        self,
        mission: MissionTypes,
        experimental_performance_improvements: bool = False,
        config: PipelineConfig | None = None,
        *,
        cone_sorting: ConeSorting | None = None,
        cone_matching: ConeMatching | None = None,
        pathing: CalculatePath | None = None,
    ) -> None:
        self.mission = mission

        if config is None:
            config = default_config(mission)
        self.config = config

        self.relocalizer: Relocalizer | None = None
        relocalizer_class = MissionToRelocalizer.get(mission)

        if relocalizer_class is not None:
            self.relocalizer = relocalizer_class()

        self.cone_sorting = cone_sorting or create_default_sorting(
            mission, experimental_performance_improvements
        )
        self.cone_matching = (
            cone_matching
            or create_default_cone_matching_with_non_monotonic_matches(mission)
        )
        self.pathing = pathing or create_default_pathing(mission)
        self.global_path: FloatArray | None = None

        self.experimental_performance_improvements = (
            experimental_performance_improvements
        )

    def set_global_path(self, global_path):
        self.global_path = global_path

    def _run_relocalization(
        self,
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
        noprint: bool,
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        IntArray,
        IntArray,
    ]:
        """Run the relocalization path (skidpad/acceleration)."""
        with Timer("Relocalization", noprint=noprint):
            if self.relocalizer is None:
                raise ValueError("Relocalizer is not set for this mission")
            self.relocalizer.attempt_relocalization_calculation(
                cones, vehicle_position, vehicle_direction
            )

        if self.relocalizer.is_relocalized:
            vehicle_yaw = angle_from_2d_vector(vehicle_direction)
            (
                vehicle_position,
                vehicle_yaw,
            ) = self.relocalizer.transform_to_known_map_frame(
                vehicle_position, float(vehicle_yaw)
            )
            vehicle_direction = unit_2d_vector_from_angle(vehicle_yaw)
            self.global_path = self.relocalizer.get_known_global_path()

        sorted_left, sorted_right = np.zeros((2, 0, 2), dtype=float)
        left_cones_with_virtual, right_cones_with_virtual = np.zeros(
            (2, 0, 2), dtype=float
        )
        left_to_right_match, right_to_left_match = np.zeros((2, 0), dtype=int)

        return (
            vehicle_position,
            vehicle_direction,
            sorted_left,
            sorted_right,
            left_cones_with_virtual,
            right_cones_with_virtual,
            left_to_right_match,
            right_to_left_match,
        )

    def _run_sorting_and_matching(
        self,
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
        noprint: bool,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, IntArray, IntArray]:
        """Run the standard sorting → matching pipeline."""
        with Timer("Cone sorting", noprint=noprint):
            cone_sorting_input = ConeSortingInput(
                cones, vehicle_position, vehicle_direction
            )
            sorting_result = self.cone_sorting.run_cone_sorting(cone_sorting_input)
            sorted_left = sorting_result.left_cones
            sorted_right = sorting_result.right_cones

        with Timer("Cone matching", noprint=noprint):
            matched_cones_input = [np.zeros((0, 2), dtype=float) for _ in ConeTypes]
            matched_cones_input[ConeTypes.LEFT] = sorted_left
            matched_cones_input[ConeTypes.RIGHT] = sorted_right

            cone_matching_input = ConeMatchingInput(
                matched_cones_input, vehicle_position, vehicle_direction
            )
            matching_result = self.cone_matching.run_cone_matching(cone_matching_input)
            left_cones_with_virtual = matching_result.left_cones_with_virtual
            right_cones_with_virtual = matching_result.right_cones_with_virtual
            left_to_right_match = matching_result.left_to_right_matches
            right_to_left_match = matching_result.right_to_left_matches

        return (
            sorted_left,
            sorted_right,
            left_cones_with_virtual,
            right_cones_with_virtual,
            left_to_right_match,
            right_to_left_match,
        )

    def calculate_path_in_global_frame(
        self,
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray | float,
        return_intermediate_results: bool = False,
    ) -> (
        FloatArray
        | tuple[
            FloatArray,
            FloatArray,
            FloatArray,
            FloatArray,
            FloatArray,
            IntArray,
            IntArray,
        ]
    ):
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
        cones = validate_and_normalize_cones(cones)
        vehicle_position = validate_vehicle_position(vehicle_position)
        vehicle_direction = convert_direction_to_array(vehicle_direction)

        noprint = True

        if self.relocalizer is not None:
            (
                vehicle_position,
                vehicle_direction,
                sorted_left,
                sorted_right,
                left_cones_with_virtual,
                right_cones_with_virtual,
                left_to_right_match,
                right_to_left_match,
            ) = self._run_relocalization(
                cones, vehicle_position, vehicle_direction, noprint
            )
        else:
            (
                sorted_left,
                sorted_right,
                left_cones_with_virtual,
                right_cones_with_virtual,
                left_to_right_match,
                right_to_left_match,
            ) = self._run_sorting_and_matching(
                cones, vehicle_position, vehicle_direction, noprint
            )

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
            path_result = self.pathing.run_path_calculation(path_calculation_input)
            final_path = path_result.final_path

        if self.relocalizer is not None and self.relocalizer.is_relocalized:
            final_path = final_path.copy()
            path_xy = final_path[:, 1:3]
            fake_yaw = np.zeros(len(path_xy))
            path_xy, _ = self.relocalizer.transform_to_original_frame(path_xy, fake_yaw)
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

        return RelocalizationInformation.from_transform_function(
            self.relocalizer.transform_to_known_map_frame
        )
