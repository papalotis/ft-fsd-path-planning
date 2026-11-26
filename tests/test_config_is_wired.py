"""Every configurable field must actually reach the algorithm.

Several fields used to be declared on the config dataclasses while the code
went on using a literal, so changing them did nothing. These tests fail if a
field goes back to being decoration.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner
from fsd_path_planning.common_types import FloatArray
from fsd_path_planning.config_dataclasses import (
    CostWeights,
    MatchingConfig,
    PathConfig,
    PipelineConfig,
    SortingConfig,
)
from fsd_path_planning.geometry.angles import unit_2d_vector_from_angle
from fsd_path_planning.geometry.transforms import rotate
from fsd_path_planning.stages.sorting.costs import (
    cost_configurations,
)
from fsd_path_planning.stages.sorting.search import (
    neighbor_bool_mask_can_be_added_to_attempt,
)

POSITION = np.array([0.0, 0.0])
DIRECTION = np.array([1.0, 0.0])


def curved_cones() -> list[FloatArray]:
    """A right-hand corner, built the way the project's own notebook builds one."""
    inner = unit_2d_vector_from_angle(np.arange(0, np.pi / 2, np.pi / 15)) * 9
    outer = unit_2d_vector_from_angle(np.arange(0, np.pi / 2, np.pi / 20)) * 12

    center = np.mean((inner[:2] + outer[:2]) / 2, axis=0)
    inner = rotate(inner - center, -np.pi / 2)
    outer = rotate(outer - center, -np.pi / 2)

    cones: list[FloatArray] = [np.zeros((0, 2)) for _ in ConeTypes]
    cones[ConeTypes.LEFT] = inner
    cones[ConeTypes.RIGHT] = outer
    return cones


def plan(config: PipelineConfig | None = None) -> FloatArray:
    planner = PathPlanner(MissionTypes.trackdrive, config=config)
    return planner.calculate_path_in_global_frame(curved_cones(), POSITION, DIRECTION)


class TestConfigReachesTheStages:
    def test_sorting_config_reaches_the_trace_sorter(self) -> None:
        config = PipelineConfig(sorting=SortingConfig(max_dist=4.25))
        planner = PathPlanner(MissionTypes.trackdrive, config=config)
        assert planner.cone_sorting.config.max_dist == 4.25
        assert planner.cone_sorting.trace_sorter.max_dist == 4.25
        assert planner.cone_sorting.trace_sorter.config.max_dist == 4.25

    def test_matching_config_reaches_the_matcher(self) -> None:
        config = PipelineConfig(matching=MatchingConfig(min_track_width=2.25))
        planner = PathPlanner(MissionTypes.trackdrive, config=config)
        assert planner.cone_matching.config.min_track_width == 2.25

    def test_path_config_reaches_the_path_calculation(self) -> None:
        config = PipelineConfig(path=PathConfig(number_of_samples=17))
        planner = PathPlanner(MissionTypes.trackdrive, config=config)
        assert planner.pathing.config.number_of_samples == 17

    def test_number_of_samples_sets_the_number_of_waypoints(self) -> None:
        assert len(plan(PipelineConfig(path=PathConfig(number_of_samples=17)))) == 17
        assert len(plan(PipelineConfig(path=PathConfig(number_of_samples=55)))) == 55

    def test_path_length_sets_how_far_ahead_the_path_reaches(self) -> None:
        short = plan(PipelineConfig(path=PathConfig(path_length=8.0)))
        long = plan(PipelineConfig(path=PathConfig(path_length=18.0)))
        assert short[-1, 0] < long[-1, 0]


class TestFieldsThatReachTheWholePipeline:
    @pytest.mark.parametrize(("field", "value"), [("car_size", 9.0)])
    def test_changing_a_sorting_field_changes_the_result(
        self, field: str, value: float
    ) -> None:
        baseline = plan()
        changed = plan(
            PipelineConfig(sorting=replace(SortingConfig(), **{field: value}))
        )
        assert baseline.shape != changed.shape or not np.allclose(baseline, changed), (
            f"changing SortingConfig.{field} made no difference"
        )

    def test_matching_search_range_multiplier_changes_the_result(self) -> None:
        baseline = plan()
        changed = plan(
            PipelineConfig(matching=MatchingConfig(search_range_multiplier=0.2))
        )
        assert baseline.shape != changed.shape or not np.allclose(baseline, changed)


class TestCostFunctionFields:
    """The cost terms only separate traces when traces compete, so they are
    checked where they act rather than through a clean corner."""

    @staticmethod
    def competing_traces() -> tuple[FloatArray, np.ndarray]:
        xs = np.arange(0.0, 18.0, 3.0)
        straight = np.column_stack(
            [xs, np.full_like(xs, 2.0), np.full_like(xs, float(ConeTypes.LEFT))]
        )
        branch = np.array(
            [[7.0, 5.0, float(ConeTypes.LEFT)], [10.0, 6.0, float(ConeTypes.LEFT)]]
        )
        points = np.vstack([straight, branch])
        configurations = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7, -1]])
        return points, configurations

    def costs(self, config: SortingConfig | None = None) -> FloatArray:
        points, configurations = self.competing_traces()
        return cost_configurations(
            points=points,
            configurations=configurations,
            cone_type=ConeTypes.LEFT,
            vehicle_position=np.array([-1.0, 2.0]),
            vehicle_direction=np.array([1.0, 0.0]),
            return_individual_costs=True,
            config=config,
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("residual_distance_threshold", 0.5),
            ("cones_on_side_search_distance", 1.0),
            ("cones_on_side_search_angle", 0.05),
        ],
    )
    def test_changing_a_threshold_changes_the_costs(
        self, field: str, value: float
    ) -> None:
        changed = replace(SortingConfig(), **{field: value})
        assert not np.allclose(self.costs(), self.costs(changed)), (
            f"changing SortingConfig.{field} made no difference"
        )

    @pytest.mark.parametrize(
        ("weight", "column"),
        [
            ("angle", 0),
            ("residual_distance", 1),
            ("n_cones", 2),
            ("initial_direction", 3),
            ("direction_change", 4),
            ("cones_on_side", 5),
            ("wrong_direction", 6),
        ],
    )
    def test_each_weight_scales_its_own_column(self, weight: str, column: int) -> None:
        """A weight of zero must silence exactly its own term."""
        only_this = CostWeights(
            **{
                name: (1000.0 if name == weight else 0.0)
                for name in (
                    "angle",
                    "residual_distance",
                    "n_cones",
                    "initial_direction",
                    "direction_change",
                    "cones_on_side",
                    "wrong_direction",
                )
            }
        )
        costs = self.costs(SortingConfig(cost_weights=only_this))
        silenced = np.delete(costs, column, axis=1)
        assert np.allclose(silenced, 0.0)


class TestCloseConeThreshold:
    """Below the threshold a candidate may break the directional-angle limit."""

    @staticmethod
    def candidate_allowed(close_cone_threshold: float) -> bool:
        # Two cones straight ahead, then one that turns about 33 degrees left,
        # well past the 10 degree limit given below, 2.24 m from the last cone.
        trace = np.array([[0.0, 0.0], [3.0, 0.0], [4.0, 2.0]])
        attempt = np.array([0, 1, -1], dtype=np.int32)

        allowed = neighbor_bool_mask_can_be_added_to_attempt(
            trace,
            ConeTypes.LEFT,
            attempt,
            1,
            np.array([2]),
            np.deg2rad(10),
            np.deg2rad(65),
            np.array([-2.0, 0.0]),
            np.array([1.0, 0.0]),
            2.1,
            close_cone_threshold,
        )
        return bool(allowed[0])

    def test_a_close_cone_is_let_through(self) -> None:
        assert self.candidate_allowed(4.0)

    def test_the_same_cone_is_rejected_once_it_counts_as_far(self) -> None:
        assert not self.candidate_allowed(1.0)

    def test_the_shipped_default_lets_it_through(self) -> None:
        assert self.candidate_allowed(SortingConfig().close_cone_threshold)


class TestDefaultsAreTheShippedValues:
    def test_matching_is_non_monotonic_by_default(self) -> None:
        """The pipeline has always matched non-monotonically, for every mission."""
        for mission in (MissionTypes.trackdrive, MissionTypes.skidpad):
            planner = PathPlanner(mission)
            assert planner.cone_matching.config.matches_should_be_monotonic is False

    def test_sorting_defaults_match_the_released_literals(self) -> None:
        config = SortingConfig()
        assert config.car_size == 2.1
        assert config.close_cone_threshold == 4.0
        assert config.residual_distance_threshold == 3.0
        assert config.cones_on_side_search_distance == 6.0
        assert config.cones_on_side_search_angle == pytest.approx(np.pi / 1.5)

    def test_cost_weight_defaults_match_the_released_factors(self) -> None:
        weights = CostWeights()
        assert [
            weights.angle,
            weights.residual_distance,
            weights.n_cones,
            weights.initial_direction,
            weights.direction_change,
            weights.cones_on_side,
            weights.wrong_direction,
        ] == [1000.0, 200.0, 5000.0, 1000.0, 0.0, 1000.0, 1000.0]


class TestVirtualConeAngleThreshold:
    def test_it_reaches_the_insertion_of_virtual_cones(self, monkeypatch) -> None:
        from fsd_path_planning.stages.matching import virtual_cones as fcm
        from fsd_path_planning.stages.matching.stage import (
            ConeMatching,
            ConeMatchingInput,
        )

        seen: list[float | None] = []
        original = fcm.insert_virtual_cones_to_existing

        def spy(*args, **kwargs):
            seen.append(
                args[3] if len(args) > 3 else kwargs.get("virtual_cone_angle_threshold")
            )
            return original(*args, **kwargs)

        monkeypatch.setattr(fcm, "insert_virtual_cones_to_existing", spy)

        cones = curved_cones()
        # the far half of the right side is missing, so the left cones there
        # get virtual partners
        right = cones[ConeTypes.RIGHT]
        cones[ConeTypes.RIGHT] = right[: len(right) // 2 + 1]
        matcher = ConeMatching(
            config=replace(MatchingConfig(), virtual_cone_angle_threshold=1.0)
        )
        matcher.run(ConeMatchingInput(cones, POSITION, DIRECTION))
        assert seen and set(seen) == {1.0}

    def test_the_threshold_changes_which_cones_are_kept(self) -> None:
        from fsd_path_planning.stages.matching.virtual_cones import (
            insert_virtual_cones_to_existing,
        )

        # a straight row with a virtual cone that makes a sharp spike
        real = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
        virtual = np.array([[3.0, 2.5]])
        kept_loose, _ = insert_virtual_cones_to_existing(real, virtual, POSITION, 0.1)
        kept_strict, _ = insert_virtual_cones_to_existing(real, virtual, POSITION, 3.0)
        assert len(kept_strict) < len(kept_loose)
