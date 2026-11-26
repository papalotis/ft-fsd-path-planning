#!/usr/bin/env python3
"""
Description: A class that runs the whole path planning pipeline.

- Cone sorting
- Cone Matching
- Path Calculation

Project: fsd_path_planning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import numpy as np

from fsd_path_planning.calculate_path.core_calculate_path import (
    CalculatePath,
    PathCalculationInput,
)
from fsd_path_planning.common_types import (
    ConeInput,
    FloatArray,
    IntArray,
    MatchingResult,
    PathResult,
    SortingResult,
)
from fsd_path_planning.cone_matching.core_cone_matching import (
    ConeMatching,
    ConeMatchingInput,
)
from fsd_path_planning.config_dataclasses import PipelineConfig
from fsd_path_planning.geometry.angles import (
    angle_from_2d_vector,
    unit_2d_vector_from_angle,
)
from fsd_path_planning.health import (
    ErrorPolicy,
    HealthTracker,
    PathStatus,
    PlannerHealth,
    RecoveryHandler,
)
from fsd_path_planning.missions import MISSIONS, mission_spec
from fsd_path_planning.relocalization.relocalization_base_class import (
    RelocalizationInput,
    RelocalizationResult,
    Relocalizer,
)
from fsd_path_planning.relocalization.relocalization_information import (
    RelocalizationInformation,
)
from fsd_path_planning.sorting_cones.core_cone_sorting import (
    ConeSorting,
    ConeSortingInput,
)
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.input_validation import (
    convert_direction_to_array,
    normalize_cones,
    validate_vehicle_position,
)
from fsd_path_planning.utils.mission_types import MissionTypes

# Views of the mission registry, kept for code that imports them from here.
MissionToPathCalculation: dict[MissionTypes, type[CalculatePath]] = {
    mission: spec.path_calculation
    for mission, spec in MISSIONS.items()
    if spec.path_calculation is not CalculatePath
}

MissionToRelocalizer: dict[MissionTypes, type[Relocalizer]] = {
    mission: spec.relocalizer_class
    for mission, spec in MISSIONS.items()
    if spec.relocalizer_class is not None
}


@dataclass(frozen=True)
class PlannerResult:
    """Everything the planner produced for one frame."""

    path: FloatArray
    """(N, 4) waypoints in the global frame: spline parameter, x, y,
    curvature. Empty, (0, 4), when health.status is NO_VALID_PATH."""
    health: PlannerHealth
    """The planner's health after this frame."""
    relocalization: RelocalizationResult | None
    """None for missions without a relocalizer."""
    sorting: SortingResult | None
    """None for missions that follow a known map."""
    matching: MatchingResult | None
    """None for missions that follow a known map."""
    path_result: PathResult
    """What the path stage returned, in the frame it planned in."""

    def as_tuple(
        self,
    ) -> tuple[
        FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, IntArray, IntArray
    ]:
        """The 7-tuple that calculate_path_in_global_frame returns with
        return_intermediate_results=True."""
        empty_cones = np.zeros((0, 2), dtype=float)
        empty_matches = np.zeros(0, dtype=int)
        sorted_left, sorted_right = (
            self.sorting if self.sorting is not None else (empty_cones, empty_cones)
        )
        if self.matching is not None:
            left_virtual, right_virtual, left_to_right, right_to_left = self.matching
        else:
            left_virtual, right_virtual = empty_cones, empty_cones
            left_to_right, right_to_left = empty_matches, empty_matches
        return (
            self.path,
            sorted_left,
            sorted_right,
            left_virtual,
            right_virtual,
            left_to_right,
            right_to_left,
        )


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
        """Create a PathPlanner for the given mission.

        Args:
            mission: The mission type that controls default configuration and
                relocalization behaviour.  Use one of the :class:`MissionTypes`
                enum members (e.g. ``MissionTypes.trackdrive``).
            experimental_performance_improvements: Enable heuristics that may be
                faster but are less thoroughly validated.  Defaults to ``False``.
            config: Optional
                :class:`~fsd_path_planning.config_dataclasses.PipelineConfig`
                to override all algorithm parameters.  When ``None`` the defaults
                for the given *mission* are used.
            cone_sorting: Override the default :class:`ConeSorting` instance.
                Useful for dependency injection in tests or custom pipelines.
            cone_matching: Override the default :class:`ConeMatching` instance.
            pathing: Override the default :class:`CalculatePath` instance.
        """
        self.mission = mission
        spec = mission_spec(mission)

        if config is None:
            config = spec.default_config()
        self.config = config

        self.relocalizer: Relocalizer | None = (
            spec.make_relocalizer(self.config)
            if spec.make_relocalizer is not None
            else None
        )

        # Built from self.config, so that a config passed in actually reaches
        # the stages. It used to be stored and then ignored in favour of the
        # defaults.
        self.cone_sorting = cone_sorting or ConeSorting(
            config=self.config.sorting,
            experimental_performance_improvements=experimental_performance_improvements,
        )
        self.cone_matching = cone_matching or ConeMatching(config=self.config.matching)
        self.pathing = pathing or spec.path_calculation(config=self.config.path)
        self.global_path: FloatArray | None = None
        self.last_result: PlannerResult | None = None
        """Everything the planner produced for the most recent frame; None
        before the first."""

        # One handler for the whole planner, so that a recovery in any stage
        # reaches the health record.
        self._recovery = RecoveryHandler(self.config.on_error)
        self.pathing.recovery = self._recovery
        self._health = HealthTracker(
            max_consecutive_recoveries=self.config.max_consecutive_recoveries,
            max_distance_without_good_path=self.config.max_distance_without_good_path,
        )

        self.experimental_performance_improvements = (
            experimental_performance_improvements
        )

    def set_global_path(self, global_path):
        self.global_path = global_path

    @property
    def error_policy(self) -> ErrorPolicy:
        """The error policy in effect: the config's, unless the environment
        variable FSD_PATH_PLANNING_STRICT=1 asks for "raise"."""
        return self._recovery.policy

    @property
    def health(self) -> PlannerHealth:
        """The planner's health after its most recent frame; see
        `fsd_path_planning.health`."""
        if self._health.health is None:
            return PlannerHealth(
                frames=0,
                status=PathStatus.OK,
                recoveries_total=0,
                recoveries_by_kind={},
                consecutive_recoveries=0,
                frames_since_good_path=0,
                distance_since_good_path=0.0,
                last_recovery=None,
            )
        return self._health.health

    def _run_relocalization(
        self,
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> tuple[FloatArray, FloatArray, RelocalizationResult]:
        """Relocalize, and move the pose into the known map frame once done."""
        assert self.relocalizer is not None
        result = self.relocalizer.run(
            RelocalizationInput(cones, vehicle_position, vehicle_direction)
        )

        if result.is_relocalized:
            vehicle_yaw = angle_from_2d_vector(vehicle_direction)
            (
                vehicle_position,
                vehicle_yaw,
            ) = self.relocalizer.transform_to_known_map_frame(
                vehicle_position, float(vehicle_yaw)
            )
            vehicle_direction = unit_2d_vector_from_angle(vehicle_yaw)
            self.global_path = self.relocalizer.get_known_global_path()

        return vehicle_position, vehicle_direction, result

    def _run_sorting_and_matching(
        self,
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> tuple[SortingResult, MatchingResult]:
        """Run the standard sorting → matching pipeline."""
        sorting_result = self.cone_sorting.run(
            ConeSortingInput(cones, vehicle_position, vehicle_direction)
        )

        matched_cones_input: list[FloatArray] = [
            np.zeros((0, 2), dtype=float) for _ in ConeTypes
        ]
        matched_cones_input[ConeTypes.LEFT] = sorting_result.left_cones
        matched_cones_input[ConeTypes.RIGHT] = sorting_result.right_cones

        matching_result = self.cone_matching.run(
            ConeMatchingInput(matched_cones_input, vehicle_position, vehicle_direction)
        )
        return sorting_result, matching_result

    @overload
    def calculate_path_in_global_frame(
        self,
        cones: ConeInput,
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray | float,
        return_intermediate_results: Literal[False] = ...,
    ) -> FloatArray: ...

    @overload
    def calculate_path_in_global_frame(
        self,
        cones: ConeInput,
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray | float,
        return_intermediate_results: Literal[True],
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        IntArray,
        IntArray,
    ]: ...

    def calculate_path_in_global_frame(
        self,
        cones: ConeInput,
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
        """Calculate the planned path in the global frame.

        Args:
            cones: The cones in the global frame, in either accepted format.

                A sequence of **exactly 5** arrays, one per cone type ordered
                by :class:`~fsd_path_planning.utils.cone_types.ConeTypes`::

                    index 0 - UNKNOWN
                    index 1 - RIGHT  (yellow)
                    index 2 - LEFT   (blue)
                    index 3 - ORANGE_SMALL  (start/finish area)
                    index 4 - ORANGE_BIG   (start/finish line)

                Each array must be numeric, finite, and shaped ``(N, 2)`` where
                *N* may be zero.  Pass ``np.zeros((0, 2))`` for absent cone types.

                Or a single numeric, finite ``(N, 3)`` array whose rows are
                ``[x, y, cone_type]``, with the cone type as a whole number from
                the same enum. It is split into the per-type form on entry; see
                :func:`~fsd_path_planning.normalize_cones`.
            vehicle_position: Current vehicle position in the global frame.
                Must be a finite numeric array with shape ``(2,)`` — ``[x, y]``.
            vehicle_direction: Current vehicle heading in the global frame.
                Two accepted forms:

                * A finite numeric array with shape ``(2,)`` — ``[dir_x, dir_y]``.
                  The vector must be **non-zero**; it is normalized internally.
                * A finite scalar — the heading angle in **radians** measured
                  counter-clockwise from the positive x-axis.

            return_intermediate_results: When ``True`` the method returns a
                7-tuple instead of just the path array (see *Returns* below).

        Returns:
            When *return_intermediate_results* is ``False`` (default): a
            ``(N, 4)`` array of waypoints in the global frame.  Each row is
            ``[spline_parameter, path_x, path_y, curvature]``.

            When *return_intermediate_results* is ``True``: a 7-tuple
            ``(path, sorted_left, sorted_right, left_with_virtual,
            right_with_virtual, left_to_right_matches, right_to_left_matches)``
            where ``path`` is the array described above.

        Raises:
            TypeError: If any element of *cones* contains non-numeric values, or
                if *vehicle_position* or *vehicle_direction* are non-numeric.
            ValueError: If *cones* is neither 5 arrays nor an (N, 3) array; if a
                cone array has the wrong shape, holds non-finite values or an
                unknown cone type; if
                *vehicle_position* has the wrong shape or non-finite values; or if
                *vehicle_direction* is a zero vector or contains non-finite values.
        """
        result = self.plan(cones, vehicle_position, vehicle_direction)
        if return_intermediate_results:
            return result.as_tuple()
        return result.path

    def plan(
        self,
        cones: ConeInput,
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray | float,
    ) -> PlannerResult:
        """Plan one frame, and return everything the planner produced for it.

        Takes the same arguments as `calculate_path_in_global_frame`, which
        returns parts of this result. The result is also kept as
        `last_result`.
        """
        cones = normalize_cones(cones)
        vehicle_position = validate_vehicle_position(vehicle_position)
        vehicle_direction = convert_direction_to_array(vehicle_direction)
        global_frame_position = vehicle_position

        self._recovery.start_frame(self._health.frames)

        relocalization_result: RelocalizationResult | None = None
        sorting_result: SortingResult | None = None
        matching_result: MatchingResult | None = None

        if self.relocalizer is not None:
            vehicle_position, vehicle_direction, relocalization_result = (
                self._run_relocalization(cones, vehicle_position, vehicle_direction)
            )
            # missions on a known map follow it and do not sort cones
            left_with_virtual, right_with_virtual = np.zeros((2, 0, 2), dtype=float)
            left_to_right_match, right_to_left_match = np.zeros((2, 0), dtype=int)
        else:
            sorting_result, matching_result = self._run_sorting_and_matching(
                cones, vehicle_position, vehicle_direction
            )
            (
                left_with_virtual,
                right_with_virtual,
                left_to_right_match,
                right_to_left_match,
            ) = matching_result

        path_result = self.pathing.run(
            PathCalculationInput(
                left_with_virtual,
                right_with_virtual,
                left_to_right_match,
                right_to_left_match,
                vehicle_position,
                vehicle_direction,
                self.global_path,
            )
        )
        final_path = path_result.final_path

        if relocalization_result is not None and relocalization_result.is_relocalized:
            assert self.relocalizer is not None
            final_path = final_path.copy()
            path_xy, _ = self.relocalizer.transform_to_original_frame(
                final_path[:, 1:3], 0.0
            )
            final_path[:, 1:3] = path_xy

        health = self._health.update(
            global_frame_position,
            self._recovery.recoveries_this_frame,
            degraded=path_result.outcome.is_degraded,
        )
        if health.status is PathStatus.NO_VALID_PATH:
            final_path = np.zeros((0, 4), dtype=float)

        self.last_result = PlannerResult(
            path=final_path,
            health=health,
            relocalization=relocalization_result,
            sorting=sorting_result,
            matching=matching_result,
            path_result=path_result,
        )
        return self.last_result

    @property
    def relocalization_info(self) -> RelocalizationInformation | None:
        if self.relocalizer is None:
            return None

        if not self.relocalizer.is_relocalized:
            return None

        return RelocalizationInformation.from_transform_function(
            self.relocalizer.transform_to_known_map_frame
        )
