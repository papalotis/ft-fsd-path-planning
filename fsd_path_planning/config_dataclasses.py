#!/usr/bin/env python3
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


def _ensure_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")

    float_value = float(value)
    if not np.isfinite(float_value):
        raise ValueError(f"{name} must be finite, got {value}")

    return float_value


def _validate_positive(name: str, value: object) -> float:
    float_value = _ensure_real(name, value)
    if float_value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return float_value


def _validate_non_negative(name: str, value: object) -> float:
    float_value = _ensure_real(name, value)
    if float_value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return float_value


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _validate_min_int(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _validate_angle(name: str, value: object) -> float:
    float_value = _validate_positive(name, value)
    if float_value >= np.pi:
        raise ValueError(f"{name} must be < pi, got {value}")
    return float_value


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

    def __post_init__(self) -> None:
        for name in (
            "angle",
            "residual_distance",
            "n_cones",
            "initial_direction",
            "direction_change",
            "cones_on_side",
            "wrong_direction",
        ):
            _ensure_real(name, getattr(self, name))


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

    def __post_init__(self) -> None:
        _validate_positive_int("max_n_neighbors", self.max_n_neighbors)
        _validate_positive("max_dist", self.max_dist)
        _validate_positive("max_dist_to_first", self.max_dist_to_first)
        _validate_positive_int("max_length", self.max_length)
        _validate_angle("threshold_directional_angle", self.threshold_directional_angle)
        _validate_angle("threshold_absolute_angle", self.threshold_absolute_angle)
        _validate_non_negative("close_cone_threshold", self.close_cone_threshold)
        _validate_positive(
            "residual_distance_threshold", self.residual_distance_threshold
        )
        _validate_positive("car_size", self.car_size)


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

    def __post_init__(self) -> None:
        _validate_positive("min_track_width", self.min_track_width)
        _validate_positive("max_search_range", self.max_search_range)
        _validate_angle("max_search_angle", self.max_search_angle)
        _validate_positive("search_range_multiplier", self.search_range_multiplier)
        _validate_angle(
            "virtual_cone_angle_threshold", self.virtual_cone_angle_threshold
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
    path_length: float = 20.0
    number_of_samples: int = 40

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

    def __post_init__(self) -> None:
        _validate_non_negative("smoothing", self.smoothing)
        _validate_positive("predict_every", self.predict_every)
        _validate_positive_int("max_deg", self.max_deg)
        _validate_non_negative(
            "maximal_distance_for_valid_path", self.maximal_distance_for_valid_path
        )
        _validate_positive("path_length", self.path_length)
        _validate_positive_int("number_of_samples", self.number_of_samples)
        _validate_min_int("circle_fit_tail_points", self.circle_fit_tail_points, 3)
        min_extension_radius = _validate_positive(
            "min_extension_radius", self.min_extension_radius
        )
        max_extension_radius = _validate_positive(
            "max_extension_radius", self.max_extension_radius
        )
        if max_extension_radius < min_extension_radius:
            raise ValueError(
                "max_extension_radius must be >= min_extension_radius, got "
                f"{self.max_extension_radius} < {self.min_extension_radius}"
            )
        _validate_positive("circular_arc_threshold", self.circular_arc_threshold)
        _validate_positive_int(
            "straight_extension_points", self.straight_extension_points
        )
        _validate_positive("initial_path_radius", self.initial_path_radius)
        _validate_angle("initial_path_angle", self.initial_path_angle)
        _validate_positive_int("initial_path_points", self.initial_path_points)

    @property
    def mpc_path_length(self) -> float:
        return self.path_length

    @mpc_path_length.setter
    def mpc_path_length(self, value: float) -> None:
        self.path_length = value

    @property
    def mpc_prediction_horizon(self) -> int:
        return self.number_of_samples

    @mpc_prediction_horizon.setter
    def mpc_prediction_horizon(self, value: int) -> None:
        self.number_of_samples = value


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
        path_data = data.get("path", {})

        if "mpc_path_length" in path_data and "path_length" not in path_data:
            path_data["path_length"] = path_data.pop("mpc_path_length")
        if (
            "mpc_prediction_horizon" in path_data
            and "number_of_samples" not in path_data
        ):
            path_data["number_of_samples"] = path_data.pop("mpc_prediction_horizon")

        cost_weights = CostWeights(**data.get("sorting", {}).pop("cost_weights", {}))
        sorting = SortingConfig(**data.get("sorting", {}), cost_weights=cost_weights)
        matching = MatchingConfig(**data.get("matching", {}))
        path = PathConfig(**path_data)

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

    def __post_init__(self) -> None:
        _validate_positive("circle_radius", self.circle_radius)
        _validate_non_negative("circle_radius_tolerance", self.circle_radius_tolerance)
        _validate_positive("center_distance", self.center_distance)
        _validate_non_negative(
            "center_distance_tolerance", self.center_distance_tolerance
        )
        _validate_positive("dbscan_eps", self.dbscan_eps)
        _validate_positive_int("dbscan_min_samples", self.dbscan_min_samples)
        _validate_positive("mean_cone_distance", self.mean_cone_distance)
        _validate_non_negative(
            "mean_cone_distance_tolerance", self.mean_cone_distance_tolerance
        )
        _validate_non_negative("max_residual", self.max_residual)
        _validate_positive_int(
            "max_cones_for_relocalization", self.max_cones_for_relocalization
        )
        _validate_min_int("max_powerset_size", self.max_powerset_size, 3)


def default_config(mission: MissionTypes) -> PipelineConfig:
    """Return a PipelineConfig with per-mission defaults.

    Currently all missions use the same defaults, but this factory
    allows mission-specific overrides in the future.
    """
    config = PipelineConfig()

    if mission == MissionTypes.skidpad:
        config.matching.matches_should_be_monotonic = False

    return config
