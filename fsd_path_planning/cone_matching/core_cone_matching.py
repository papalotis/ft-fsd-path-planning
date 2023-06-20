#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Cone matching class.

Description: Provides class interface to functional cone matching.
Project: fsd_path_planning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from icecream import ic  # pylint: disable=unused-import

from fsd_path_planning.cone_matching.functional_cone_matching import (
    calculate_virtual_cones_for_both_sides,
)
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes

MatchedCones = Tuple[FloatArray, FloatArray, IntArray, IntArray]


@dataclass
class ConeMatchingInput:
    """Dataclass holding inputs."""

    sorted_cones: list[FloatArray] = field(
        default_factory=lambda: [np.zeros((0, 2)) for _ in ConeTypes]
    )
    slam_position: FloatArray = field(default_factory=lambda: np.zeros((2)))
    slam_direction: FloatArray = field(default_factory=lambda: np.zeros((2)))


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
        min_track_width: float,
        max_search_range: float,
        max_search_angle: float,
        matches_should_be_monotonic: bool,
    ):
        """
        Init method.

        Args:
        """

        self.input = ConeMatchingInput()
        self.state = ConeMatchingState(
            min_track_width=min_track_width,
            max_search_range=max_search_range,
            max_search_angle=max_search_angle,
            matches_should_be_monotonic=matches_should_be_monotonic,
        )

    def set_new_input(self, cone_matching_input: ConeMatchingInput) -> None:
        """Save inputs from other software nodes in variable."""
        self.input = cone_matching_input

    def transition_input_to_state(self) -> None:
        """Parse and save the inputs in state variable."""
        self.state.position_global, self.state.direction_global = (
            self.input.slam_position,
            self.input.slam_direction,
        )

        self.state.sorted_left = self.input.sorted_cones[ConeTypes.LEFT]
        self.state.sorted_right = self.input.sorted_cones[ConeTypes.RIGHT]

    def run_cone_matching(self) -> MatchedCones:
        """
        Calculate matched cones.

        Returns:
            Matched cones.
                The left cones with virtual cones.
                The right cones with virtual cones.
                The indices of the matches of the right cones for each left cone.
                The indices of the matches of the left cones for each right cone.

        """
        self.transition_input_to_state()

        major_radius = self.state.max_search_range * 1.5
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

        return (
            left_cones_with_virtual,
            right_cones_with_virtual,
            left_to_right_match,
            right_to_left_match,
        )
