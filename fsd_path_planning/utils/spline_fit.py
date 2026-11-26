#!/usr/bin/env python3
"""
Description:

Project: fsd_path_planning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import splev, splprep

from fsd_path_planning.utils.math_utils import trace_distance_to_next

logger = logging.getLogger(__name__)

# Decimal places the number of sampling steps is rounded to before the sample
# positions are built. Ten is far finer than any real path resolution and far
# coarser than the float noise that used to decide the count.
ARANGE_DECIMALS = 10


@dataclass
class SplineEvaluator:
    """
    A class for evaluating a spline.
    """

    max_u: float
    tck: tuple[Any, Any, int]
    predict_every: float

    def calculate_u_eval(self, max_u: float | None = None) -> np.ndarray:
        """
        Calculate the u_eval values for the spline.

        Args:
            max_u (Optional[float], optional): The maximum u value. Defaults to None. If
                None, the maximum u value used during fitting is taken.

        Returns:
            np.ndarray: The values for which the spline should be evaluated.
        """

        if max_u is None:
            max_u = self.max_u

        # np.arange(0, max_u, predict_every) returns one more point when
        # max_u / predict_every lands a hair above a whole number instead of a
        # hair below. The path parameterizer divides the path length by a point
        # count to get predict_every, so the ratio is a whole number by
        # construction and the sampling sat exactly on that edge: a change far
        # below a micrometre anywhere upstream, or a different BLAS or SciPy
        # build, flipped the number of samples and moved the sampled path by up
        # to a third of a step. Round the count to the nearest whole number of
        # steps before taking the half-open range, so the edge resolves the
        # same way everywhere.
        n_steps = int(np.ceil(round(max_u / self.predict_every, ARANGE_DECIMALS)))
        return np.arange(n_steps) * self.predict_every

    def predict(self, der: int, max_u: float | None = None) -> np.ndarray:
        """
        Predict the spline. If der is 0, the function returns the spline. If der is 1,
        the function returns the first derivative of the spline and so on.

        Args:
            der (int): The derivative to predict.
            max_u (Optional[float], optional): The maximum u value. Defaults to None. If
                None, the maximum u value used during fitting is taken.

        Returns:
            np.ndarray: The predicted spline.
        """

        u_eval = self.calculate_u_eval(max_u)
        values = np.array(splev(u_eval, tck=self.tck, der=der)).T

        return values


class NullSplineEvaluator(SplineEvaluator):
    """
    A dummy spline evaluator used for when an empty list is attempted to be fitted
    """

    def predict(self, der: int, max_u: float | None = None) -> np.ndarray:
        points = np.zeros((0, 2))
        return points


class SplineFitterFactory:
    """
    Wrapper class for `splev`, `splprep` functions
    """

    def __init__(self, smoothing: float, predict_every: float, max_deg: int):
        """
        Constructor for SplineFitter class

        Args:
            smoothing (float): The smoothing factor. 0 means no smoothing
            predict_every (float): The approximate distance along the fitted
                trace to calculate a point for
            max_deg (int): The maximum degree of the fitted splines
        """
        self.smoothing = smoothing
        self.predict_every = predict_every
        self.max_deg = max_deg

    def fit(self, trace: np.ndarray, periodic: bool = False) -> SplineEvaluator:
        """
        Fit a trace and returns a SplineEvaluator that can evaluate the fitted spline at
        different positions.

        Args:
            trace (np.ndarray): The trace to fit

        Returns:
            A instance of SplineEvaluator that can be used to evaluate the spline
        """
        if len(trace) < 2:
            return NullSplineEvaluator(
                # dummy values
                0,
                (0, 0, 0),
                0,
            )
        k = np.clip(len(trace) - 1, 1, self.max_deg)
        distance_to_next = trace_distance_to_next(trace)
        u_fit = np.concatenate(([0], np.cumsum(distance_to_next)))
        try:
            tck, _ = splprep(  # pylint: disable=unbalanced-tuple-unpacking
                trace.T, s=self.smoothing, k=k, u=u_fit, per=periodic
            )
        except ValueError:
            logger.debug(
                "Spline fitting failed: smoothing=%s, max_deg=%s, trace shape=%s",
                self.smoothing,
                self.max_deg,
                trace.shape,
            )
            raise

        max_u = float(u_fit[-1])

        return SplineEvaluator(max_u, tck, self.predict_every)

    def fit_then_evaluate_trace_and_derivative(
        self, trace: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit a provided trace, then evaluates it, and its derivative in `n_predict`
        evenly spaced positions

        Args:
            trace (np.ndarray): The trace to fit

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the evaluated trace and
            the evaluated derivative
        """
        if len(trace) < 2:
            return trace.copy(), trace.copy()

        fitted_func = self.fit(trace)

        evaluated_trace = fitted_func.predict(der=0)
        evaluated_derivative = fitted_func.predict(der=1)

        return evaluated_trace, evaluated_derivative
