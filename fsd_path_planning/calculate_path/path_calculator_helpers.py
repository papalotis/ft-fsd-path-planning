#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Core path calculation.

Description: A module for path update calculation that will be used in combination with
the existing path

Project: fsd_path_planning
"""
import numpy as np
from icecream import ic  # pylint: disable=unused-import
from typing_extensions import Literal

from fsd_path_planning.types import FloatArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import rotate, unit_2d_vector_from_angle

ConeTypesForPathCalculation = Literal[ConeTypes.LEFT, ConeTypes.RIGHT]


class PathCalculatorHelpers:
    """A class for calculating the update path that will be combined with the existing path."""

    HALF_PI = np.pi / 2

    def calculate_chord_path(
        self, radius: float, maximum_angle: float, number_of_points: int
    ) -> FloatArray:
        """
        Calculate a chord (part of circle) path with a specific radius.

        Args:
            radius: The radius of the chord path
            maximum_angle: The angle of the chord
            number_of_points: The number of points the path should be evaluated
            for
        Returns:
            The arc path
        """

        points: FloatArray = (
            # create points on a circle
            unit_2d_vector_from_angle(
                np.linspace(0, np.abs(maximum_angle), number_of_points)
            )
        )
        # rotate so initial points, point to the right
        points_centered: FloatArray = points - [1, 0]  # bring x axis to center
        points_centered_scaled: FloatArray = points_centered * radius  # scale
        points_centered_scaled_rotated = rotate(points_centered_scaled, -self.HALF_PI)

        # handle negative angles
        points_centered_scaled_rotated[:, 1] *= np.sign(maximum_angle)
        return points_centered_scaled_rotated

    def calculate_almost_straight_path(self) -> FloatArray:
        """
        Calculate a chord path with a very high radius and a very small chord angle.

        Returns:
            np.ndarray: The straight-like chord path update
        """
        return self.calculate_chord_path(
            # values for a slightly circular path to the right
            radius=1000,
            maximum_angle=np.pi / 50,
            number_of_points=40,
        )
