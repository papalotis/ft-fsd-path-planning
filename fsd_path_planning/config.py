#!/usr/bin/env python3
"""
Class creation config file.

Description: Config file to create instances of the pathing related classes.
Project: fsd_path_planning
"""

from typing import Any

from fsd_path_planning.calculate_path.core_calculate_path import (
    CalculatePath as CalculatePath,
)

# for reexport
from fsd_path_planning.calculate_path.skidpad_calculate_path import SkidpadCalculatePath
from fsd_path_planning.cone_matching.core_cone_matching import (
    ConeMatching as ConeMatching,
)
from fsd_path_planning.config_dataclasses import (
    PathConfig,
    SortingConfig,
    default_config,
)
from fsd_path_planning.sorting_cones.core_cone_sorting import ConeSorting
from fsd_path_planning.utils.mission_types import MissionTypes

KwargsType = dict[str, Any]


def get_cone_sorting_config(
    mission: MissionTypes,  # pylint: disable=unused-argument
) -> KwargsType:
    """Create cone sorting kwargs."""
    cfg = SortingConfig()
    return dict(
        max_n_neighbors=cfg.max_n_neighbors,
        max_dist=cfg.max_dist,
        max_dist_to_first=cfg.max_dist_to_first,
        max_length=cfg.max_length,
        threshold_directional_angle=cfg.threshold_directional_angle,
        threshold_absolute_angle=cfg.threshold_absolute_angle,
        use_unknown_cones=cfg.use_unknown_cones,
    )


def get_cone_fitting_config(
    mission: MissionTypes,  # pylint: disable=unused-argument
) -> KwargsType:
    """Create cone fitting kwargs."""
    cfg = PathConfig()
    return dict(
        smoothing=cfg.smoothing, predict_every=cfg.predict_every, max_deg=cfg.max_deg
    )


def get_path_calculation_config(
    mission: MissionTypes,  # pylint: disable=unused-argument
) -> KwargsType:
    """Create path calculation kwargs based on mission."""
    cfg = PathConfig()
    return dict(
        maximal_distance_for_valid_path=cfg.maximal_distance_for_valid_path,
        mpc_path_length=cfg.mpc_path_length,
        mpc_prediction_horizon=cfg.mpc_prediction_horizon,
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
    cfg = default_config(mission)

    possible_path_calculation_classes: dict[MissionTypes, type[CalculatePath]] = {
        MissionTypes.skidpad: SkidpadCalculatePath,
    }

    path_calculation_class = possible_path_calculation_classes.get(
        mission, CalculatePath
    )

    path_calculation = path_calculation_class(config=cfg.path)

    return path_calculation


def create_default_sorting(
    mission: MissionTypes, experiment_performance_improvements: bool = False
) -> ConeSorting:
    """
    Create a cone sorting instance with default values.

    Args:
        mission: The mission the Pathing instance should be configured for

    Returns:
        cone_sorting: The created ConeSorting instance
    """
    cfg = default_config(mission)
    cone_sorting = ConeSorting(
        config=cfg.sorting,
        experimental_performance_improvements=experiment_performance_improvements,
    )
    return cone_sorting


def get_default_matching_kwargs(mission: MissionTypes) -> KwargsType:
    """
    Create a cone matching kwargs based on mission.

    Args:
        mission: The mission the cone matching instance should be configured for

    Returns:
        The created cone matching kwargs
    """
    cfg = default_config(mission)
    return dict(
        min_track_width=cfg.matching.min_track_width,
        max_search_range=cfg.matching.max_search_range,
        max_search_angle=cfg.matching.max_search_angle,
        matches_should_be_monotonic=cfg.matching.matches_should_be_monotonic,
    )


def create_default_cone_matching(
    mission: MissionTypes,
) -> ConeMatching:
    """
    Create a cone matching instance based on mission.

    Args:
        mission: The mission the cone matching instance should be configured for

    Returns:
        The created ConeMatching instance
    """
    cfg = default_config(mission)
    return ConeMatching(config=cfg.matching)


def create_default_cone_matching_with_non_monotonic_matches(
    mission: MissionTypes,
) -> ConeMatching:
    """
    Create a cone matching instance based on mission.

    Args:
        mission: The mission the cone matching instance should be configured for

    Returns:
        The created ConeMatching instance
    """
    cfg = default_config(mission)
    cfg.matching.matches_should_be_monotonic = False
    return ConeMatching(config=cfg.matching)
