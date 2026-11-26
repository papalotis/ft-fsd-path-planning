#!/usr/bin/env python3
"""
Description:

Project: fsd_path_planning
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from typing import IO, Any

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

    def evaluate_at(self, u_eval: np.ndarray, der: int = 0) -> np.ndarray:
        """
        Evaluate the spline at given parameter values, ignoring `predict_every`.

        Args:
            u_eval: The parameter values to evaluate at.
            der: The derivative to evaluate.

        Returns:
            np.ndarray: The evaluated spline, one row per requested value.
        """
        return np.array(splev(np.asarray(u_eval, dtype=float), tck=self.tck, der=der)).T

    def to_arrays(self) -> dict[str, np.ndarray]:
        """
        The spline as plain arrays, ready to be stored in an .npz file.

        Returns:
            A mapping that `from_arrays` turns back into an evaluator.
        """
        knots, coefficients, degree = self.tck
        return {
            "knots": np.asarray(knots, dtype=float),
            "coefficients": np.asarray(coefficients, dtype=float),
            "degree": np.asarray(int(degree)),
            "max_u": np.asarray(float(self.max_u)),
            "predict_every": np.asarray(float(self.predict_every)),
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> SplineEvaluator:
        """Rebuild an evaluator from the arrays produced by `to_arrays`."""
        coefficients = np.asarray(arrays["coefficients"], dtype=float)
        tck = (
            np.asarray(arrays["knots"], dtype=float),
            [np.asarray(row, dtype=float) for row in coefficients],
            int(arrays["degree"]),
        )
        return cls(
            max_u=float(arrays["max_u"]),
            tck=tck,
            predict_every=float(arrays["predict_every"]),
        )

    def save(self, path: str | PathLike[str]) -> None:
        """
        Write the spline to an .npz file.

        Storing the spline instead of the points it was fitted to keeps a dense
        path small: a few hundred coefficients stand in for thousands of points.
        """
        np.savez_compressed(path, **self.to_arrays())

    @classmethod
    def load(cls, path: str | PathLike[str] | IO[bytes]) -> SplineEvaluator:
        """Read a spline written by `save`."""
        with np.load(path) as arrays:
            return cls.from_arrays(arrays)


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

    def fit(
        self,
        trace: np.ndarray,
        periodic: bool = False,
        u_fit: np.ndarray | None = None,
    ) -> SplineEvaluator:
        """
        Fit a trace and returns a SplineEvaluator that can evaluate the fitted spline at
        different positions.

        Args:
            trace (np.ndarray): The trace to fit
            periodic (bool): Whether the trace closes on itself
            u_fit (np.ndarray | None): The parameter value of each point. By
                default the cumulative distance along the trace, which makes the
                parameter an arc length. Pass the point indices instead to be
                able to recover the original points by index later.

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
        if u_fit is None:
            distance_to_next = trace_distance_to_next(trace)
            u_fit = np.concatenate(([0], np.cumsum(distance_to_next)))
        else:
            u_fit = np.asarray(u_fit, dtype=float)
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
