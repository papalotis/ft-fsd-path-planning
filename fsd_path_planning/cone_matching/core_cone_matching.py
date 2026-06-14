#!/usr/bin/env python3
"""
Cone matching class.

Description: Provides class interface to functional cone matching.
Project: fsd_path_planning
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from fsd_path_planning.cone_matching.functional_cone_matching import (
    calculate_virtual_cones_for_both_sides,
)
from fsd_path_planning.config_dataclasses import MatchingConfig
from fsd_path_planning.types import FloatArray, IntArray, MatchingResult
from fsd_path_planning.utils.cone_types import ConeTypes

MatchedCones = tuple[FloatArray, FloatArray, IntArray, IntArray]


@dataclass
class ConeMatchingInput:
    """Dataclass holding inputs."""

    sorted_cones: list[FloatArray] = field(
        default_factory=lambda: [np.zeros((0, 2)) for _ in ConeTypes]
    )
    vehicle_position: FloatArray = field(default_factory=lambda: np.zeros(2))
    vehicle_direction: FloatArray = field(default_factory=lambda: np.zeros(2))


@dataclass
class ConeMatchingState:
    """Dataclass holding calculation variables."""

    min_track_width: float
    max_search_range: float
    max_search_angle: float
    matches_should_be_monotonic: bool
    sorted_left: FloatArray = field(default_factory=lambda: np.zeros((0, 2)))
    sorted_right: FloatArray = field(default_factory=lambda: np.zeros((0, 2)))
    position_global: FloatArray = field(init=False)
    direction_global: FloatArray = field(init=False)


class ConeMatching:
    """Class that takes all cone matching and virtual cone responsibilities."""

    def __init__(
        self,
        config: MatchingConfig | None = None,
        *,
        # Legacy parameters (deprecated, use config instead)
        min_track_width: float | None = None,
        max_search_range: float | None = None,
        max_search_angle: float | None = None,
        matches_should_be_monotonic: bool | None = None,
    ):
        if config is not None:
            self.config = config
        else:
            legacy_params = {
                "min_track_width": min_track_width,
                "max_search_range": max_search_range,
                "max_search_angle": max_search_angle,
                "matches_should_be_monotonic": matches_should_be_monotonic,
            }
            provided = {k: v for k, v in legacy_params.items() if v is not None}
            if provided:
                warnings.warn(
                    "Passing individual parameters to ConeMatching is deprecated. "
                    "Use ConeMatching(config=MatchingConfig(...)) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self.config = MatchingConfig(**provided)

        self.input = ConeMatchingInput()
        self.state = ConeMatchingState(
            min_track_width=self.config.min_track_width,
            max_search_range=self.config.max_search_range,
            max_search_angle=self.config.max_search_angle,
            matches_should_be_monotonic=self.config.matches_should_be_monotonic,
        )

    def set_new_input(self, cone_matching_input: ConeMatchingInput) -> None:
        """Save inputs from other software nodes in variable.

        .. deprecated::
            Pass input directly to :meth:`run_cone_matching` instead.
        """
        warnings.warn(
            "set_new_input() is deprecated. "
            "Pass input directly to run_cone_matching().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.input = cone_matching_input

    def transition_input_to_state(self) -> None:
        """Parse and save the inputs in state variable."""
        self.state.position_global, self.state.direction_global = (
            self.input.vehicle_position,
            self.input.vehicle_direction,
        )

        self.state.sorted_left = self.input.sorted_cones[ConeTypes.LEFT]
        self.state.sorted_right = self.input.sorted_cones[ConeTypes.RIGHT]

    def run_cone_matching(
        self, input: ConeMatchingInput | None = None
    ) -> MatchingResult:
        """
        Calculate matched cones.

        Args:
            input: The matching input. If not provided, uses previously set input.

        Returns:
            MatchingResult with left/right cones (including virtual) and match indices.
        """
        if input is not None:
            self.input = input
        self.transition_input_to_state()

        major_radius = self.state.max_search_range * self.config.search_range_multiplier
        minor_radius = self.state.min_track_width

        (
            (left_cones_with_virtual, _, left_to_right_match),
            (right_cones_with_virtual, _, right_to_left_match),
        ) = calculate_virtual_cones_for_both_sides(
            self.state.sorted_left,
            self.state.sorted_right,
            self.state.position_global,
            self.state.direction_global,
            self.state.min_track_width,
            major_radius,
            minor_radius,
            self.state.max_search_angle,
            self.state.matches_should_be_monotonic,
        )

        return MatchingResult(
            left_cones_with_virtual=left_cones_with_virtual,
            right_cones_with_virtual=right_cones_with_virtual,
            left_to_right_matches=left_to_right_match,
            right_to_left_matches=right_to_left_match,
        )
