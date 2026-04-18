"""Unit tests for spline fitting utilities."""

from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning.utils.spline_fit import (
    NullSplineEvaluator,
    SplineEvaluator,
    SplineFitterFactory,
)


class TestSplineFitterFactory:
    @pytest.fixture()
    def factory(self):
        return SplineFitterFactory(smoothing=0.0, predict_every=0.1, max_deg=3)

    def test_fit_straight_line(self, factory):
        trace = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        evaluator = factory.fit(trace)
        pts = evaluator.predict(der=0)
        # All y-values should be ~0
        np.testing.assert_allclose(pts[:, 1], 0.0, atol=1e-6)
        # x should be monotonically increasing
        assert np.all(np.diff(pts[:, 0]) > 0)

    def test_fit_circular_arc(self, factory):
        angles = np.linspace(0, np.pi, 30)
        trace = np.column_stack([np.cos(angles), np.sin(angles)])
        evaluator = factory.fit(trace)
        pts = evaluator.predict(der=0)
        # Points should be approximately on a unit circle
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        np.testing.assert_allclose(radii, 1.0, atol=0.05)

    def test_derivative_tangent_direction(self, factory):
        # Straight horizontal line → derivative should be ~(1, 0)
        trace = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        evaluator = factory.fit(trace)
        deriv = evaluator.predict(der=1)
        # Normalize derivatives
        norms = np.linalg.norm(deriv, axis=1, keepdims=True)
        norms[norms == 0] = 1
        dirs = deriv / norms
        # Should point in the +x direction
        np.testing.assert_allclose(dirs[:, 0], 1.0, atol=0.1)
        np.testing.assert_allclose(dirs[:, 1], 0.0, atol=0.1)

    def test_periodic_spline(self, factory):
        # Closed loop (circle)
        n = 40
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        trace = np.column_stack([np.cos(angles), np.sin(angles)])
        evaluator = factory.fit(trace, periodic=True)
        pts = evaluator.predict(der=0)
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        np.testing.assert_allclose(radii, 1.0, atol=0.05)

    def test_fit_returns_evaluator_type(self, factory):
        trace = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        evaluator = factory.fit(trace)
        assert isinstance(evaluator, SplineEvaluator)

    def test_fit_single_point_returns_null(self, factory):
        trace = np.array([[0.0, 0.0]])
        evaluator = factory.fit(trace)
        assert isinstance(evaluator, NullSplineEvaluator)

    def test_fit_empty_returns_null(self, factory):
        trace = np.zeros((0, 2))
        evaluator = factory.fit(trace)
        assert isinstance(evaluator, NullSplineEvaluator)


class TestNullSplineEvaluator:
    def test_predict_returns_empty(self):
        evaluator = NullSplineEvaluator(0, (0, 0, 0), 0)
        result = evaluator.predict(der=0)
        assert result.shape == (0, 2)

    def test_predict_derivative_returns_empty(self):
        evaluator = NullSplineEvaluator(0, (0, 0, 0), 0)
        result = evaluator.predict(der=1)
        assert result.shape == (0, 2)


class TestSplineEvaluatorCalculateUEval:
    def test_u_eval_spacing(self):
        # Create a real evaluator via the factory
        factory = SplineFitterFactory(smoothing=0.0, predict_every=0.5, max_deg=3)
        trace = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        evaluator = factory.fit(trace)
        u_eval = evaluator.calculate_u_eval()
        # Check spacing is ~0.5
        diffs = np.diff(u_eval)
        np.testing.assert_allclose(diffs, 0.5, atol=1e-12)

    def test_u_eval_with_custom_max(self):
        factory = SplineFitterFactory(smoothing=0.0, predict_every=0.5, max_deg=3)
        trace = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        evaluator = factory.fit(trace)
        u_eval_default = evaluator.calculate_u_eval()
        u_eval_short = evaluator.calculate_u_eval(max_u=1.0)
        assert len(u_eval_short) <= len(u_eval_default)
