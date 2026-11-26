"""The error policy and the health record: each fallback is forced to fire.

On the bundled data neither error fallback ever runs, so without forcing them
they would go untested; that is how a broken fallback once went unnoticed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import numpy as np
import pytest

from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner
from fsd_path_planning.common_types import PathOutcome
from fsd_path_planning.config_dataclasses import PathConfig, PipelineConfig
from fsd_path_planning.geometry.angles import unit_2d_vector_from_angle
from fsd_path_planning.health import STRICT_ENV_VAR, PathStatus, RecoveryHandler

DIRECTION = np.array([1.0, 0.0])


@pytest.fixture(autouse=True)
def not_strict(monkeypatch):
    """CI runs the suite with strict mode on; these tests choose their own."""
    monkeypatch.delenv(STRICT_ENV_VAR, raising=False)


def corner_cones(offset: float = 0.0) -> list[np.ndarray]:
    """A gentle left-hand corner starting at x = offset."""
    angles = np.linspace(0.0, 1.0, 10)
    radius = 20.0
    cones = [np.zeros((0, 2)) for _ in ConeTypes]
    arc = unit_2d_vector_from_angle(angles - np.pi / 2)
    cones[ConeTypes.LEFT] = arc * (radius - 1.5) + [offset, radius]
    cones[ConeTypes.RIGHT] = arc * (radius + 1.5) + [offset, radius]
    return cones


def no_cones() -> list[np.ndarray]:
    return [np.zeros((0, 2)) for _ in ConeTypes]


def planner(**policy) -> PathPlanner:
    return PathPlanner(
        MissionTypes.trackdrive, config=replace(PipelineConfig(), **policy)
    )


def plan_at(p: PathPlanner, x: float, cones=None):
    position = np.array([x, 0.0])
    if cones is None:
        cones = corner_cones(offset=x)
    return p.plan(cones, position, DIRECTION)


class FailingOnce:
    """Wraps a callable so that its next call raises ValueError."""

    def __init__(self, wrapped, times: int = 1) -> None:
        self.wrapped = wrapped
        self.remaining = times

    def __call__(self, *args, **kwargs):
        if self.remaining > 0:
            self.remaining -= 1
            raise ValueError("forced failure")
        return self.wrapped(*args, **kwargs)


class UnfittableBasis:
    """A basis strategy that, for a number of frames, returns points no spline
    can be fitted through (all the same point), so the real fit fails."""

    def __init__(self, times: int) -> None:
        self.remaining = times

    def basis(self, calculator, stage_input):
        if self.remaining <= 0:
            return None
        self.remaining -= 1
        return np.repeat(stage_input.vehicle_position[None], 5, axis=0)


def force_spline_fit_failure(p: PathPlanner, times: int = 1) -> None:
    p.pathing.basis_strategies = (
        UnfittableBasis(times),
        *p.pathing.basis_strategies,
    )


def force_parameterization_failure(p: PathPlanner, times: int = 1) -> None:
    p.pathing.do_all_mpc_parameter_calculations = FailingOnce(  # type: ignore[method-assign]
        p.pathing.do_all_mpc_parameter_calculations, times
    )


class TestPolicySelection:
    def test_recover_is_the_default(self) -> None:
        assert PathPlanner(MissionTypes.trackdrive).error_policy == "recover"

    def test_config_can_ask_for_raise(self) -> None:
        assert planner(on_error="raise").error_policy == "raise"

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_environment_variable_forces_raise(self, monkeypatch, value) -> None:
        monkeypatch.setenv(STRICT_ENV_VAR, value)
        assert planner(on_error="recover").error_policy == "raise"

    @pytest.mark.parametrize("value", ["", "0", "false"])
    def test_environment_variable_off(self, monkeypatch, value) -> None:
        monkeypatch.setenv(STRICT_ENV_VAR, value)
        assert planner().error_policy == "recover"

    def test_unknown_policy_is_refused(self) -> None:
        with pytest.raises(ValueError):
            PipelineConfig(on_error="ignore")  # type: ignore[arg-type]

    def test_policy_survives_json(self) -> None:
        config = replace(
            PipelineConfig(),
            on_error="raise",
            max_consecutive_recoveries=3,
            max_distance_without_good_path=12.5,
        )
        assert PipelineConfig.from_json(config.to_json()) == config
        data = json.loads(config.to_json())
        for key in (
            "on_error",
            "max_consecutive_recoveries",
            "max_distance_without_good_path",
        ):
            del data[key]
        assert PipelineConfig.from_json(json.dumps(data)) == PipelineConfig()


@pytest.mark.parametrize(
    ("force", "kind"),
    [
        (force_spline_fit_failure, "spline_fit_failed"),
        (force_parameterization_failure, "parameterization_failed"),
    ],
)
class TestEachFallback:
    def test_recover_runs_the_fallback_and_records_it(
        self, force, kind, caplog
    ) -> None:
        p = planner()
        plan_at(p, 0.0)
        force(p)
        with caplog.at_level(logging.WARNING, logger="fsd_path_planning"):
            result = plan_at(p, 1.0)

        assert result.path.shape == (PathConfig().number_of_samples, 4)
        assert np.isfinite(result.path).all()
        health = result.health
        assert health.status is PathStatus.RECOVERED
        assert health.recoveries_by_kind == {kind: 1}
        assert health.consecutive_recoveries == 1
        assert health.last_recovery is not None
        assert health.last_recovery.kind == kind
        assert health.last_recovery.stage == "path"
        assert health.last_recovery.frame == 1
        assert "ValueError" in health.last_recovery.traceback
        assert any(kind in r.getMessage() for r in caplog.records)
        assert any(r.exc_info for r in caplog.records)

    def test_raise_lets_the_error_through(self, force, kind) -> None:
        p = planner(on_error="raise")
        plan_at(p, 0.0)
        force(p)
        with pytest.raises(ValueError):
            plan_at(p, 1.0)

    def test_the_next_good_frame_is_ok_again(self, force, kind) -> None:
        p = planner()
        plan_at(p, 0.0)
        force(p)
        plan_at(p, 1.0)
        result = plan_at(p, 2.0)
        assert result.health.status is PathStatus.OK
        assert result.health.consecutive_recoveries == 0
        assert result.health.recoveries_total == 1


class TestDegradedIsNotAnError:
    def test_too_few_cones(self, monkeypatch) -> None:
        monkeypatch.setenv(STRICT_ENV_VAR, "1")
        p = planner()
        plan_at(p, 0.0)
        result = plan_at(p, 1.0, cones=no_cones())
        assert result.path_result.outcome is PathOutcome.TOO_FEW_CONES
        assert result.health.status is PathStatus.DEGRADED
        assert result.health.recoveries_total == 0
        assert len(result.path) > 0

    def test_path_too_far_from_the_car(self, monkeypatch) -> None:
        monkeypatch.setenv(STRICT_ENV_VAR, "1")
        p = PathPlanner(
            MissionTypes.trackdrive,
            config=replace(
                PipelineConfig(),
                path=replace(PathConfig(), maximal_distance_for_valid_path=0.0),
            ),
        )
        result = plan_at(p, 0.0)
        assert result.path_result.outcome is PathOutcome.TOO_FAR_FROM_CAR
        assert result.health.status is PathStatus.DEGRADED


class TestBoundedRecovery:
    def test_off_by_default(self) -> None:
        p = planner()
        plan_at(p, 0.0)
        force_spline_fit_failure(p, times=10)
        for i in range(1, 6):
            result = plan_at(p, float(i))
        assert result.health.status is PathStatus.RECOVERED
        assert result.health.consecutive_recoveries == 5
        assert len(result.path) > 0

    def test_consecutive_recoveries_bound(self) -> None:
        p = planner(max_consecutive_recoveries=2)
        plan_at(p, 0.0)
        force_spline_fit_failure(p, times=3)
        statuses = [plan_at(p, float(i)).health.status for i in range(1, 5)]
        assert statuses == [
            PathStatus.RECOVERED,
            PathStatus.RECOVERED,
            PathStatus.NO_VALID_PATH,
            PathStatus.OK,
        ]
        assert p.last_result is not None
        assert p.last_result.path.shape == (PathConfig().number_of_samples, 4)

    def test_no_valid_path_returns_an_empty_path(self) -> None:
        p = planner(max_consecutive_recoveries=0)
        plan_at(p, 0.0)
        force_spline_fit_failure(p)
        result = plan_at(p, 1.0)
        assert result.health.status is PathStatus.NO_VALID_PATH
        assert result.path.shape == (0, 4)
        path = p.calculate_path_in_global_frame(
            corner_cones(2.0), np.array([2.0, 0.0]), DIRECTION
        )
        assert path.shape == (PathConfig().number_of_samples, 4)

    def test_distance_bound(self) -> None:
        p = planner(max_distance_without_good_path=5.0)
        plan_at(p, 0.0)
        statuses = []
        for x in (2.0, 4.0, 6.0):
            statuses.append(plan_at(p, x, cones=no_cones()).health.status)
        assert statuses == [
            PathStatus.DEGRADED,
            PathStatus.DEGRADED,
            PathStatus.NO_VALID_PATH,
        ]
        assert p.health.distance_since_good_path == pytest.approx(6.0)
        assert plan_at(p, 8.0).health.status is PathStatus.OK


class TestHealthRecord:
    def test_before_the_first_frame(self) -> None:
        health = planner().health
        assert health.frames == 0
        assert health.status is PathStatus.OK

    def test_counts_frames(self) -> None:
        p = planner()
        for x in range(3):
            plan_at(p, float(x))
        assert p.health.frames == 3
        assert p.health is p.last_result.health  # type: ignore[union-attr]

    def test_recovery_handler_alone_defaults_to_recover(self) -> None:
        handler = RecoveryHandler()
        handler.recover("path", "test", ValueError("x"))
        assert [r.kind for r in handler.recoveries_this_frame] == ["test"]


class TestStraightTrackNeedsNoRecovery:
    """Strict mode found this: an exactly straight track made the path
    extension repeat its last point, so the refit failed on every frame and
    the parameterization fallback ran."""

    def test_extension_of_a_straight_tail_has_no_repeated_points(self) -> None:
        from fsd_path_planning.stages.path.extension import extend_path

        path = np.column_stack([np.arange(-0.8, 18.0, 0.1), np.zeros(188)])
        extended = extend_path(
            path, np.array([-1.0, 0.0]), DIRECTION, 20.0, 20, 10.0, 100.0, 80.0, 30
        )
        steps = np.linalg.norm(np.diff(extended, axis=0), axis=1)
        assert steps.min() > 0
        assert extended[-1, 0] > path[-1, 0] + 20

    def test_planner_on_a_straight_track(self, monkeypatch) -> None:
        monkeypatch.setenv(STRICT_ENV_VAR, "1")
        xs = np.arange(0.0, 21.0, 3.0)
        cones = no_cones()
        cones[ConeTypes.LEFT] = np.column_stack([xs, np.full_like(xs, 2.0)])
        cones[ConeTypes.RIGHT] = np.column_stack([xs, np.full_like(xs, -2.0)])
        result = PathPlanner(MissionTypes.trackdrive).plan(
            cones, np.array([-1.0, 0.0]), DIRECTION
        )
        assert result.health.status is PathStatus.OK
        np.testing.assert_allclose(result.path[:, 2], 0.0, atol=1e-6)
