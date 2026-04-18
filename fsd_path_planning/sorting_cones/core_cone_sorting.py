#!/usr/bin/env python3
"""
Cone sorting class.
Description: Entry point for Pathing/ConeSorting
Project: fsd_path_planning
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from fsd_path_planning.config_dataclasses import SortingConfig
from fsd_path_planning.sorting_cones.trace_sorter.core_trace_sorter import TraceSorter
from fsd_path_planning.types import FloatArray, SortingResult
from fsd_path_planning.utils.cone_types import ConeTypes


@dataclass
class ConeSortingInput:
    """Dataclass holding inputs."""

    cones_by_type: list[FloatArray] = field(
        default_factory=lambda: [np.zeros((0, 2)) for _ in ConeTypes]
    )
    vehicle_position: FloatArray = field(default_factory=lambda: np.zeros(2))
    vehicle_direction: FloatArray = field(default_factory=lambda: np.zeros(2))


@dataclass
class ConeSortingState:
    """Dataclass holding calculation variables."""

    threshold_directional_angle: float
    threshold_absolute_angle: float
    max_n_neighbors: int
    max_dist: float
    max_dist_to_first: float
    max_length: int
    use_unknown_cones: bool
    position_global: FloatArray = field(default_factory=lambda: np.zeros(2))
    direction_global: FloatArray = field(default_factory=lambda: np.array([0, 1.0]))
    cones_by_type: list[FloatArray] = field(
        default_factory=lambda: [np.zeros((0, 2)) for _ in ConeTypes]
    )


class ConeSorting:
    """Class that takes all Pathing/ConeSorting responsibilities."""

    def __init__(
        self,
        config: SortingConfig | None = None,
        *,
        # Legacy parameters (deprecated, use config instead)
        max_n_neighbors: int | None = None,
        max_dist: float | None = None,
        max_dist_to_first: float | None = None,
        max_length: int | None = None,
        threshold_directional_angle: float | None = None,
        threshold_absolute_angle: float | None = None,
        use_unknown_cones: bool | None = None,
        experimental_performance_improvements: bool = False,
    ):
        if config is not None:
            self.config = config
        else:
            # Legacy path: build config from individual parameters
            legacy_params = {
                "max_n_neighbors": max_n_neighbors,
                "max_dist": max_dist,
                "max_dist_to_first": max_dist_to_first,
                "max_length": max_length,
                "threshold_directional_angle": threshold_directional_angle,
                "threshold_absolute_angle": threshold_absolute_angle,
                "use_unknown_cones": use_unknown_cones,
            }
            provided = {k: v for k, v in legacy_params.items() if v is not None}
            if provided:
                warnings.warn(
                    "Passing individual parameters to ConeSorting is deprecated. "
                    "Use ConeSorting(config=SortingConfig(...)) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self.config = SortingConfig(**provided)

        self.input = ConeSortingInput()

        self.state = ConeSortingState(
            max_n_neighbors=self.config.max_n_neighbors,
            max_dist=self.config.max_dist,
            max_dist_to_first=self.config.max_dist_to_first,
            max_length=self.config.max_length,
            threshold_directional_angle=self.config.threshold_directional_angle,
            threshold_absolute_angle=self.config.threshold_absolute_angle,
            use_unknown_cones=self.config.use_unknown_cones,
        )

        self.trace_sorter = TraceSorter(
            self.state.max_n_neighbors,
            self.state.max_dist,
            self.state.max_dist_to_first,
            self.state.max_length,
            self.state.threshold_directional_angle,
            self.state.threshold_absolute_angle,
            experimental_performance_improvements,
        )

    def set_new_input(self, slam_input: ConeSortingInput) -> None:
        """Save inputs from other software nodes in variable.

        .. deprecated::
            Pass input directly to :meth:`run_cone_sorting` instead.
        """
        warnings.warn(
            "set_new_input() is deprecated. Pass input directly to run_cone_sorting().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.input = slam_input

    def transition_input_to_state(self) -> None:
        """Parse and save the inputs in the state variable."""
        self.state.position_global, self.state.direction_global = (
            self.input.vehicle_position,
            self.input.vehicle_direction,
        )

        self.state.cones_by_type = self.input.cones_by_type.copy()
        if not self.state.use_unknown_cones:
            self.state.cones_by_type[ConeTypes.UNKNOWN] = np.zeros((0, 2))

    def run_cone_sorting(
        self,
        input: ConeSortingInput | None = None,
    ) -> SortingResult:
        """
        Calculate the sorted cones.

        Args:
            input: The sorting input. If not provided, uses previously set input.

        Returns:
            SortingResult with left_cones and right_cones arrays.
        """
        if input is not None:
            self.input = input
        # make transition from set inputs to usable state variables
        self.transition_input_to_state()

        left_cones, right_cones = self.trace_sorter.sort_left_right(
            self.state.cones_by_type,
            self.state.position_global,
            self.state.direction_global,
        )

        return SortingResult(left_cones=left_cones, right_cones=right_cones)
