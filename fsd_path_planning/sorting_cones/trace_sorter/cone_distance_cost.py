#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Calculate the cost of configurations according to the cost of the distance
between cones
Project: fsd_path_planning
"""
import numpy as np

from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.math_utils import trace_distance_to_next


def calc_distance_cost(
    points: FloatArray, configurations: IntArray, threshold_distance: float
) -> FloatArray:
    """
    Calculate the sum of the residual distances between consecutive cones. The residual
    distance is defined as the distance between two cones that is over
    `threshold_distance`. If two cones have a distance that is less than
    `threshold_distance`, then the residual distance is 0.
    """
    points_in_configurations = points[configurations]
    distances_to_next = trace_distance_to_next(points_in_configurations)

    distances_to_next_filtered = distances_to_next * (configurations != -1)[:, 1:]

    residual_distances = np.maximum(0, distances_to_next_filtered - threshold_distance)
    sum_of_residual_distances_for_configurations: FloatArray = residual_distances.sum(
        axis=-1
    )
    return sum_of_residual_distances_for_configurations
