"""The contracts the stages share: frozen configs and results, the Stage
protocol, the mission registry, and expected outcomes reported as results.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner
from fsd_path_planning.calculate_path.core_calculate_path import CalculatePath
from fsd_path_planning.calculate_path.skidpad_calculate_path import (
    SkidpadCalculatePath,
)
from fsd_path_planning.common_types import (
    MatchingResult,
    PathResult,
    SortingResult,
    TraceOutcome,
)
from fsd_path_planning.cone_matching.core_cone_matching import ConeMatching
from fsd_path_planning.config_dataclasses import (
    CostWeights,
    MatchingConfig,
    PathConfig,
    PipelineConfig,
    SkidpadConfig,
    SortingConfig,
    default_config,
)
from fsd_path_planning.full_pipeline.full_pipeline import (
    MissionToPathCalculation,
    MissionToRelocalizer,
)
from fsd_path_planning.geometry.angles import unit_2d_vector_from_angle
from fsd_path_planning.missions import MISSIONS, mission_spec
from fsd_path_planning.relocalization.acceleration.acceleration_relocalization import (
    AccelerationRelocalizer,
)
from fsd_path_planning.relocalization.relocalization_base_class import (
    RelocalizationInput,
    RelocalizationOutcome,
)
from fsd_path_planning.relocalization.skidpad.skidpad_relocalizer import (
    SkidpadRelocalizer,
)
from fsd_path_planning.sorting_cones.core_cone_sorting import (
    ConeSorting,
    ConeSortingInput,
)
from fsd_path_planning.stages import Stage

POSITION = np.array([0.0, 0.0])
DIRECTION = np.array([1.0, 0.0])


def no_cones() -> list[np.ndarray]:
    return [np.zeros((0, 2)) for _ in ConeTypes]


def corner_cones() -> list[np.ndarray]:
    """A gentle left-hand corner ahead of a car at the origin heading along +x."""
    angles = np.linspace(0.0, 1.0, 10)
    radius = 20.0
    cones = no_cones()
    cones[ConeTypes.LEFT] = (
        unit_2d_vector_from_angle(angles - np.pi / 2) * (radius - 1.5)
    ) + [0.0, radius]
    cones[ConeTypes.RIGHT] = (
        unit_2d_vector_from_angle(angles - np.pi / 2) * (radius + 1.5)
    ) + [0.0, radius]
    return cones


class TestConfigsAreFrozen:
    @pytest.mark.parametrize(
        "config",
        [
            CostWeights(),
            SortingConfig(),
            MatchingConfig(),
            PathConfig(),
            SkidpadConfig(),
            PipelineConfig(),
        ],
        ids=lambda c: type(c).__name__,
    )
    def test_assignment_is_refused(self, config) -> None:
        name = next(iter(vars(config)))
        with pytest.raises(FrozenInstanceError):
            setattr(config, name, getattr(config, name))

    def test_replace_makes_a_changed_copy(self) -> None:
        original = PathConfig()
        changed = replace(original, smoothing=0.5)
        assert changed.smoothing == 0.5
        assert original.smoothing == PathConfig().smoothing

    def test_replace_still_validates(self) -> None:
        with pytest.raises(ValueError):
            replace(PathConfig(), smoothing=-1.0)

    def test_old_alias_names_still_read(self) -> None:
        config = PathConfig(path_length=30.0, number_of_samples=60)
        assert config.mpc_path_length == 30.0
        assert config.mpc_prediction_horizon == 60


class TestPipelineConfigJson:
    def test_round_trip_keeps_the_skidpad_settings(self) -> None:
        config = replace(
            PipelineConfig(),
            skidpad=replace(SkidpadConfig(), max_cones_for_relocalization=15),
        )
        assert PipelineConfig.from_json(config.to_json()) == config

    def test_json_written_before_skidpad_settings_still_loads(self) -> None:
        import json

        data = json.loads(PipelineConfig().to_json())
        del data["skidpad"]
        assert PipelineConfig.from_json(json.dumps(data)) == PipelineConfig()


class TestResults:
    def test_results_are_frozen(self) -> None:
        result = SortingResult(np.zeros((0, 2)), np.zeros((0, 2)))
        with pytest.raises(FrozenInstanceError):
            result.left_cones = np.zeros((1, 2))  # type: ignore[misc]

    def test_sorting_result_unpacks_to_the_two_arrays_only(self) -> None:
        left, right = np.ones((3, 2)), np.zeros((2, 2))
        result = SortingResult(
            left, right, TraceOutcome.FOUND, TraceOutcome.NO_START_CONE
        )
        unpacked_left, unpacked_right = result
        assert unpacked_left is left
        assert unpacked_right is right

    def test_matching_and_path_results_unpack_in_field_order(self) -> None:
        arrays = [np.full((1, 2), i) for i in range(4)]
        assert list(MatchingResult(*arrays)) == arrays
        assert list(PathResult(*arrays[:2])) == arrays[:2]


class TestStageProtocol:
    @pytest.mark.parametrize(
        "stage",
        [
            ConeSorting(),
            ConeMatching(),
            CalculatePath(),
            SkidpadCalculatePath(),
            AccelerationRelocalizer(),
            SkidpadRelocalizer(),
        ],
        ids=lambda s: type(s).__name__,
    )
    def test_every_stage_has_run(self, stage) -> None:
        assert isinstance(stage, Stage)

    def test_run_is_the_same_as_the_named_method(self) -> None:
        stage_input = ConeSortingInput(corner_cones(), POSITION, DIRECTION)
        by_run = ConeSorting().run(stage_input)
        by_name = ConeSorting().run_cone_sorting(stage_input)
        np.testing.assert_array_equal(by_run.left_cones, by_name.left_cones)
        np.testing.assert_array_equal(by_run.right_cones, by_name.right_cones)


class TestMissionRegistry:
    def test_every_mission_has_an_entry(self) -> None:
        assert set(MISSIONS) == set(MissionTypes)

    def test_relocalizer_map_is_unchanged(self) -> None:
        """The module-level map that 0.4 exposed keeps its contents."""
        assert MissionToRelocalizer == {
            MissionTypes.acceleration: AccelerationRelocalizer,
            MissionTypes.ebs_test: AccelerationRelocalizer,
            MissionTypes.skidpad: SkidpadRelocalizer,
        }
        assert MissionToPathCalculation == {MissionTypes.skidpad: SkidpadCalculatePath}

    def test_default_config_comes_from_the_registry(self) -> None:
        for mission in MissionTypes:
            assert default_config(mission) == mission_spec(mission).default_config()

    @pytest.mark.parametrize("mission", list(MissionTypes), ids=lambda m: m.name)
    def test_planner_is_built_from_the_registry(self, mission) -> None:
        planner = PathPlanner(mission)
        spec = mission_spec(mission)
        assert type(planner.pathing) is spec.path_calculation
        if spec.relocalizer_class is None:
            assert planner.relocalizer is None
        else:
            assert type(planner.relocalizer) is spec.relocalizer_class

    def test_skidpad_settings_reach_the_relocalizer(self) -> None:
        skidpad = replace(SkidpadConfig(), max_cones_for_relocalization=11)
        planner = PathPlanner(
            MissionTypes.skidpad, config=replace(PipelineConfig(), skidpad=skidpad)
        )
        assert isinstance(planner.relocalizer, SkidpadRelocalizer)
        assert planner.relocalizer.config is skidpad


class TestNoTraceIsAResult:
    def test_trace_found(self) -> None:
        result = ConeSorting().run(
            ConeSortingInput(corner_cones(), POSITION, DIRECTION)
        )
        assert result.left_outcome is TraceOutcome.FOUND
        assert result.right_outcome is TraceOutcome.FOUND

    def test_no_cones(self) -> None:
        result = ConeSorting().run(ConeSortingInput(no_cones(), POSITION, DIRECTION))
        assert len(result.left_cones) == len(result.right_cones) == 0
        assert result.left_outcome is TraceOutcome.TOO_FEW_CONES
        assert result.right_outcome is TraceOutcome.TOO_FEW_CONES

    def test_cones_too_far_away_to_start_from(self) -> None:
        far = [c + [100.0, 0.0] for c in corner_cones()]
        result = ConeSorting().run(ConeSortingInput(far, POSITION, DIRECTION))
        assert len(result.left_cones) == len(result.right_cones) == 0
        assert result.left_outcome is TraceOutcome.NO_START_CONE
        assert result.right_outcome is TraceOutcome.NO_START_CONE

    def test_facing_away_from_the_track(self) -> None:
        """The cones beside the car can start a trace, but none leads ahead."""
        result = ConeSorting().run(
            ConeSortingInput(corner_cones(), POSITION, -DIRECTION)
        )
        assert len(result.left_cones) == len(result.right_cones) == 0
        assert result.left_outcome is TraceOutcome.NO_VALID_TRACE
        assert result.right_outcome is TraceOutcome.NO_VALID_TRACE

    def test_the_pipeline_records_the_outcome(self) -> None:
        planner = PathPlanner(MissionTypes.trackdrive)
        planner.calculate_path_in_global_frame(no_cones(), POSITION, DIRECTION)
        results = planner.last_stage_results
        assert results is not None
        assert results.relocalization is None
        assert results.sorting is not None
        assert results.sorting.left_outcome is TraceOutcome.TOO_FEW_CONES


class TestNotRelocalizedIsAResult:
    def test_no_cones_is_not_yet(self) -> None:
        result = SkidpadRelocalizer().run(
            RelocalizationInput(no_cones(), POSITION, DIRECTION)
        )
        assert result.outcome is RelocalizationOutcome.NOT_YET
        assert not result.is_relocalized
        assert result.information is None

    def test_skidpad_run_relocalizes_once_and_then_keeps_it(self, skidpad_data) -> None:
        positions, directions, cones = skidpad_data
        planner = PathPlanner(MissionTypes.skidpad)
        outcomes = []
        for i in range(min(60, len(positions))):
            planner.calculate_path_in_global_frame(
                cones[i], positions[i], directions[i]
            )
            assert planner.last_stage_results is not None
            relocalization = planner.last_stage_results.relocalization
            assert relocalization is not None
            assert planner.last_stage_results.sorting is None
            outcomes.append(relocalization.outcome)

        assert outcomes.count(RelocalizationOutcome.RELOCALIZED) == 1
        first = outcomes.index(RelocalizationOutcome.RELOCALIZED)
        assert set(outcomes[:first]) <= {RelocalizationOutcome.NOT_YET}
        assert set(outcomes[first + 1 :]) == {RelocalizationOutcome.ALREADY_RELOCALIZED}

        # the result carries the same transform the planner reports
        information = relocalization.information
        expected = planner.relocalization_info
        assert information is not None and expected is not None
        np.testing.assert_array_equal(information.translation, expected.translation)
        assert information.rotation == expected.rotation


class TestFinalOutcome:
    def test_combining_can_drop_a_found_trace(self) -> None:
        from fsd_path_planning.sorting_cones.trace_sorter.core_trace_sorter import (
            _final_outcome,
        )

        assert _final_outcome(TraceOutcome.FOUND, 4) is TraceOutcome.FOUND
        assert (
            _final_outcome(TraceOutcome.FOUND, 0) is TraceOutcome.DROPPED_WHEN_COMBINED
        )
        assert (
            _final_outcome(TraceOutcome.NO_START_CONE, 0) is TraceOutcome.NO_START_CONE
        )
