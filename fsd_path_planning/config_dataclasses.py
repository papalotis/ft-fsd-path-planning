#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Configuration dataclasses for the path planning pipeline.

Each module accepts its config dataclass directly. The config is the single
source of truth — no parameter duplication.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from fsd_path_planning.utils.mission_types import MissionTypes


@dataclass
class CostWeights:
    """Weights for the cone sorting cost function."""

    angle: float = 1000.0
    residual_distance: float = 200.0
    n_cones: float = 5000.0
    initial_direction: float = 1000.0
    direction_change: float = 0.0
    cones_on_side: float = 1000.0
    wrong_direction: float = 1000.0


@dataclass
class SortingConfig:
    """Configuration for cone sorting."""

    max_n_neighbors: int = 5
    max_dist: float = 6.5
    max_dist_to_first: float = 6.0
    max_length: int = 12
    threshold_directional_angle: float = field(
        default_factory=lambda: float(np.deg2rad(40))
    )
    threshold_absolute_angle: float = field(
        default_factory=lambda: float(np.deg2rad(65))
    )
    use_unknown_cones: bool = True

    # cost function weights
    cost_weights: CostWeights = field(default_factory=CostWeights)

    # threshold below which directional angle constraint is relaxed (meters)
    close_cone_threshold: float = 4.0

    # maximum distance between consecutive cones for residual distance cost
    residual_distance_threshold: float = 3.0

    # vehicle size used for between-cone skip detection (meters)
    car_size: float = 2.1


@dataclass
class MatchingConfig:
    """Configuration for cone matching."""

    min_track_width: float = 3.0
    max_search_range: float = 5.0
    max_search_angle: float = field(default_factory=lambda: float(np.deg2rad(50)))
    matches_should_be_monotonic: bool = True

    # multiplier applied to max_search_range for ellipse major radius
    search_range_multiplier: float = 1.5

    # angle threshold for removing sharp-angle cones after virtual insertion
    virtual_cone_angle_threshold: float = field(
        default_factory=lambda: float(np.deg2rad(85))
    )


@dataclass
class PathConfig:
    """Configuration for path calculation."""

    # spline fitting
    smoothing: float = 0.2
    predict_every: float = 0.1
    max_deg: int = 3

    # path validation
    maximal_distance_for_valid_path: float = 5.0

    # MPC parameters
    mpc_path_length: float = 20.0
    mpc_prediction_horizon: int = 40

    # path extension parameters
    circle_fit_tail_points: int = 20
    min_extension_radius: float = 10.0
    max_extension_radius: float = 100.0
    circular_arc_threshold: float = 80.0
    straight_extension_points: int = 30

    # initial path generation
    initial_path_radius: float = 1000.0
    initial_path_angle: float = field(default_factory=lambda: float(np.pi / 50))
    initial_path_points: int = 40


@dataclass
class PipelineConfig:
    """Top-level configuration for the full path planning pipeline."""

    sorting: SortingConfig = field(default_factory=SortingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    path: PathConfig = field(default_factory=PathConfig)

    def to_json(self) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> PipelineConfig:
        """Deserialize configuration from JSON string."""
        data = json.loads(json_str)

        cost_weights = CostWeights(**data.get("sorting", {}).pop("cost_weights", {}))
        sorting = SortingConfig(**data.get("sorting", {}), cost_weights=cost_weights)
        matching = MatchingConfig(**data.get("matching", {}))
        path = PathConfig(**data.get("path", {}))

        return cls(sorting=sorting, matching=matching, path=path)


@dataclass
class SkidpadConfig:
    """Configuration for skidpad relocalization. Separate from PipelineConfig."""

    # expected circle radius (meters, FSG rules)
    circle_radius: float = 7.625
    circle_radius_tolerance: float = 1.0

    # expected distance between circle centers (meters)
    center_distance: float = 18.25
    center_distance_tolerance: float = 0.5

    # DBSCAN clustering for circle center detection
    dbscan_eps: float = 3.0
    dbscan_min_samples: int = 1

    # mean cone distance from fitted circle
    mean_cone_distance: float = 2.4
    mean_cone_distance_tolerance: float = 1.5

    # maximum residual for circle fit acceptance
    max_residual: float = 0.4

    # maximum number of closest cones to use for relocalization
    max_cones_for_relocalization: int = 20

    # maximum powerset size for circle fitting
    max_powerset_size: int = 5


def default_config(mission: MissionTypes) -> PipelineConfig:
    """Return a PipelineConfig with per-mission defaults.

    Currently all missions use the same defaults, but this factory
    allows mission-specific overrides in the future.
    """
    config = PipelineConfig()

    if mission == MissionTypes.skidpad:
        config.matching.matches_should_be_monotonic = False

    return config
