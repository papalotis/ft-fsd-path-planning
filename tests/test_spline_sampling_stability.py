"""The number of spline samples must not depend on floating-point noise."""

from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning.utils.spline_fit import SplineEvaluator

DUMMY_TCK = (0, 0, 0)


def evaluator(max_u: float, predict_every: float) -> SplineEvaluator:
    return SplineEvaluator(max_u=max_u, tck=DUMMY_TCK, predict_every=predict_every)


def nudge(value: float, steps: int) -> float:
    """Move `value` by a few units in the last place."""
    out = value
    direction = np.inf if steps > 0 else -np.inf
    for _ in range(abs(steps)):
        out = float(np.nextafter(out, direction))
    return out


class TestSampleCountIsStable:
    @pytest.mark.parametrize("n_steps", [40, 120, 121, 400])
    def test_a_whole_number_of_steps_is_not_a_coin_flip(self, n_steps: int) -> None:
        """The path parameterizer puts the ratio exactly on a whole number."""
        predict_every = 20.0 / n_steps
        exact = n_steps * predict_every

        counts = {
            len(evaluator(nudge(exact, steps), predict_every).calculate_u_eval())
            for steps in range(-8, 9)
        }
        assert counts == {n_steps}

    def test_the_last_sample_stays_below_max_u(self) -> None:
        samples = evaluator(20.0, 0.5).calculate_u_eval()
        assert samples[-1] < 20.0
        assert len(samples) == 40

    @pytest.mark.parametrize(
        ("max_u", "predict_every"),
        [(2.5, 1.0), (19.9, 0.25), (1.0, 0.3), (7.0, 2.0)],
    )
    def test_matches_arange_away_from_the_boundary(
        self, max_u: float, predict_every: float
    ) -> None:
        np.testing.assert_allclose(
            evaluator(max_u, predict_every).calculate_u_eval(),
            np.arange(0, max_u, predict_every),
        )

    def test_explicit_max_u_is_honoured(self) -> None:
        samples = evaluator(20.0, 0.5).calculate_u_eval(max_u=5.0)
        assert len(samples) == 10
        assert samples[-1] == pytest.approx(4.5)

    def test_zero_length_gives_no_samples(self) -> None:
        assert len(evaluator(0.0, 0.5).calculate_u_eval()) == 0


class TestPipelineIsStableUnderNoise:
    def test_a_tiny_input_change_makes_a_tiny_output_change(self) -> None:
        """Before this fix a relative 1e-13 nudge moved the path by 0.17 m."""
        from fsd_path_planning import MissionTypes, PathPlanner
        from fsd_path_planning.utils.cone_types import ConeTypes

        angles = np.linspace(0.0, 1.2, 14)
        inner = np.column_stack([np.cos(angles) * 9, np.sin(angles) * 9])
        outer = np.column_stack([np.cos(angles) * 12, np.sin(angles) * 12])

        def run(epsilon: float) -> np.ndarray:
            cones = [np.zeros((0, 2)) for _ in ConeTypes]
            cones[ConeTypes.RIGHT] = inner * (1 + epsilon) - [11.0, 0.0]
            cones[ConeTypes.LEFT] = outer * (1 + epsilon) - [11.0, 0.0]
            return PathPlanner(MissionTypes.trackdrive).calculate_path_in_global_frame(
                cones, np.array([-1.0, 0.0]), np.array([1.0, 0.0])
            )

        baseline, nudged = run(0.0), run(1e-13)
        assert baseline.shape == nudged.shape
        assert np.abs(baseline - nudged).max() < 1e-6
