"""
Deprecated: the spline fitting moved to fsd_path_planning.geometry.splines.

Forwards to it with a warning until 0.7.0.

Project: fsd_path_planning
"""

from fsd_path_planning._deprecation import moved_module

__getattr__ = moved_module(
    "fsd_path_planning.utils.spline_fit",
    "fsd_path_planning.geometry.splines",
    (
        "SplineEvaluator",
        "NullSplineEvaluator",
        "SplineFitterFactory",
        "ARANGE_DECIMALS",
    ),
    removed_in="0.7.0",
)
