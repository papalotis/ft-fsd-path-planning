from __future__ import annotations

import json

import pytest

from fsd_path_planning.config_dataclasses import (
    MatchingConfig,
    PathConfig,
    PipelineConfig,
    SkidpadConfig,
    SortingConfig,
    default_config,
)
from fsd_path_planning.utils.mission_types import MissionTypes


class TestConfigDefaults:
    def test_default_pipeline_config_is_valid(self):
        config = PipelineConfig()

        assert config.sorting.max_dist > 0
        assert config.matching.max_search_range > 0
        assert config.path.path_length > 0
        assert config.path.number_of_samples > 0

    def test_default_config_skidpad_keeps_non_monotonic_matches(self):
        config = default_config(MissionTypes.skidpad)

        assert config.matching.matches_should_be_monotonic is False


class TestSortingConfigValidation:
    def test_negative_max_dist_raises(self):
        with pytest.raises(ValueError, match="max_dist must be > 0"):
            SortingConfig(max_dist=-1.0)

    def test_invalid_directional_angle_raises(self):
        with pytest.raises(
            ValueError, match="threshold_directional_angle must be < pi"
        ):
            SortingConfig(threshold_directional_angle=3.5)


class TestMatchingConfigValidation:
    def test_zero_search_multiplier_raises(self):
        with pytest.raises(ValueError, match="search_range_multiplier must be > 0"):
            MatchingConfig(search_range_multiplier=0.0)

    def test_non_numeric_track_width_raises(self):
        with pytest.raises(TypeError, match="min_track_width must be a real number"):
            MatchingConfig(min_track_width="wide")  # type: ignore[arg-type]


class TestPathConfigValidation:
    def test_negative_smoothing_raises(self):
        with pytest.raises(ValueError, match="smoothing must be >= 0"):
            PathConfig(smoothing=-0.1)

    def test_max_extension_radius_smaller_than_min_raises(self):
        with pytest.raises(
            ValueError,
            match="max_extension_radius must be >= min_extension_radius",
        ):
            PathConfig(min_extension_radius=20.0, max_extension_radius=10.0)


class TestSkidpadConfigValidation:
    def test_small_powerset_size_raises(self):
        with pytest.raises(ValueError, match="max_powerset_size must be >= 3"):
            SkidpadConfig(max_powerset_size=2)


class TestPipelineConfigJsonValidation:
    def test_from_json_invalid_value_raises(self):
        payload = {
            "path": {"path_length": -1.0},
        }

        with pytest.raises(ValueError, match="path_length must be > 0"):
            PipelineConfig.from_json(json.dumps(payload))

    def test_from_json_valid_payload_round_trips(self):
        payload = {
            "sorting": {"max_dist": 7.5},
            "matching": {"max_search_range": 6.0},
            "path": {"path_length": 25.0, "number_of_samples": 50},
        }

        config = PipelineConfig.from_json(json.dumps(payload))

        assert config.sorting.max_dist == 7.5
        assert config.matching.max_search_range == 6.0
        assert config.path.path_length == 25.0
        assert config.path.number_of_samples == 50

    def test_from_json_old_path_keys_still_work(self):
        payload = {
            "path": {"mpc_path_length": 21.0, "mpc_prediction_horizon": 45},
        }

        config = PipelineConfig.from_json(json.dumps(payload))

        assert config.path.path_length == 21.0
        assert config.path.number_of_samples == 45


class TestPathConfigBackwardCompatibleAliases:
    def test_old_property_aliases_reflect_new_names(self):
        config = PathConfig(path_length=22.0, number_of_samples=55)

        assert config.mpc_path_length == 22.0
        assert config.mpc_prediction_horizon == 55
