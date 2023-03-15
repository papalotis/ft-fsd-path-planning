#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Class that stores all dataclasses and custom defined datatypes.

Description: Entry point for Pathing/ConeSorting
Project: fsd_path_planning
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from fsd_path_planning.types import FloatArray
from fsd_path_planning.utils.cone_types import ConeTypes


@dataclass
class ConeSortingInput:
    """Dataclass holding inputs."""

    slam_cones: List[FloatArray] = field(
        default_factory=lambda: [np.zeros((0, 2)) for _ in ConeTypes]
    )
    slam_position: FloatArray = np.zeros(2)
    slam_direction: FloatArray = np.zeros(2)


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
    position_global: FloatArray = np.zeros((2,))
    direction_global: FloatArray = np.array([0, 1.0])
    cones_by_type_array: List[FloatArray] = field(
        default_factory=lambda: [np.zeros((0, 2)) for _ in ConeTypes]
    )
