#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Class creation config file.

Description: Config file to create instances of the pathing related classes.
Project: fsd_path_planning
"""
from typing import Any, Dict, Type

import numpy as np
from icecream import ic  # pylint: disable=unused-import

from fsd_path_planning.calculate_path.core_calculate_path import (
    CalculatePath as CalculatePath,
)

# for reexport
from fsd_path_planning.calculate_path.skidpad_calculate_path import SkidpadCalculatePath
from fsd_path_planning.cone_matching.core_cone_matching import (
    ConeMatching as ConeMatching,
)
from fsd_path_planning.sorting_cones.core_cone_sorting import ConeSorting
from fsd_path_planning.utils.mission_types import MissionTypes

KwargsType = Dict[str, Any]


def get_cone_sorting_config(
    mission: MissionTypes,  # pylint: disable=unused-argument
) -> KwargsType:
    """Create cone sorting kwargs."""

    return dict(
        max_n_neighbors=5,
        max_dist=6.5,
        max_dist_to_first=6.0,
        max_length=12,
        threshold_directional_angle=np.deg2rad(40),
        threshold_absolute_angle=np.deg2rad(65),
        use_unknown_cones=True,
    )


def get_cone_fitting_config(
    mission: MissionTypes,  # pylint: disable=unused-argument
) -> KwargsType:
    """Create cone fitting kwargs."""
    return dict(smoothing=0.2, predict_every=0.1, max_deg=3)


def get_path_calculation_config(
    mission: MissionTypes,  # pylint: disable=unused-argument
) -> KwargsType:
    """Create path calculation kwargs based on mission."""
    return dict(
        maximal_distance_for_valid_path=5,
        mpc_path_length=20,  # 20 meters
        mpc_prediction_horizon=40,  # 40 path points
    )


def create_default_pathing(mission: MissionTypes) -> CalculatePath:
    """
    Create a path calculation instance based on mission.

    Args:
        mission: The mission the Pathing instance should be
        configured for

    Returns:
        The created path calculation instance
    """
    path_calculation_kwargs = get_path_calculation_config(mission)
    cone_fitting_kwargs = get_cone_fitting_config(mission)

    possible_path_calculation_classes: Dict[MissionTypes, Type[CalculatePath]] = {
        MissionTypes.skidpad: SkidpadCalculatePath,
    }

    path_calculation_class = possible_path_calculation_classes.get(
        mission, CalculatePath
    )

    path_calculation = path_calculation_class(
        **path_calculation_kwargs,
        **cone_fitting_kwargs,
    )

    return path_calculation


def create_default_sorting(mission: MissionTypes) -> ConeSorting:
    """
    Create a cone sorting instance with default values.

    Args:
        mission: The mission the Pathing instance should be configured for

    Returns:
        cone_sorting: The created ConeSorting instance
    """
    cone_sorting_kwargs = get_cone_sorting_config(mission)

    cone_sorting = ConeSorting(**cone_sorting_kwargs)
    return cone_sorting


def get_default_matching_kwargs(mission: MissionTypes) -> KwargsType:
    """
    Create a cone matching kwargs based on mission.

    Args:
        mission: The mission the cone matching instance should be configured for

    Returns:
        The created cone matching kwargs
    """
    return dict(
        min_track_width=3,
        max_search_range=5,
        max_search_angle=np.deg2rad(50),
        matches_should_be_monotonic=True,
    )


def create_default_cone_matching(
    mission: MissionTypes,  # pylint: disable=unused-argument
) -> ConeMatching:
    """
    Create a cone matching instance based on mission.

    Args:
        mission: The mission the cone matching instance should be configured for

    Returns:
        The created ConeMatching instance
    """
    kwargs = get_default_matching_kwargs(mission)
    return ConeMatching(**kwargs)


def create_default_cone_matching_with_non_monotonic_matches(
    mission: MissionTypes,  # pylint: disable=unused-argument
) -> ConeMatching:
    """
    Create a cone matching instance based on mission.

    Args:
        mission: The mission the cone matching instance should be configured for

    Returns:
        The created ConeMatching instance
    """
    kwargs = get_default_matching_kwargs(mission)
    assert "matches_should_be_monotonic" in kwargs
    kwargs["matches_should_be_monotonic"] = False
    return ConeMatching(**kwargs)
