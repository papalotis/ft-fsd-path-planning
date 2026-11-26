"""
Deprecated: the spline fitting moved to fsd_path_planning.geometry.splines.

Forwards to it with a warning until 0.7.0.

Project: fsd_path_planning
"""

from fsd_path_planning._deprecation import moved_names

_NEW = "fsd_path_planning.geometry.splines"

__getattr__ = moved_names(
    "fsd_path_planning.utils.spline_fit",
    {
        "SplineEvaluator": _NEW,
        "NullSplineEvaluator": _NEW,
        "SplineFitterFactory": f"{_NEW}:SplineFitter",
        "ARANGE_DECIMALS": _NEW,
    },
    removed_in="0.7.0",
)
