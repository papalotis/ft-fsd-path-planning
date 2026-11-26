"""
Quantities along a trace: an ordered (n, 2) array of points.

Project: fsd_path_planning
"""

from __future__ import annotations

import numpy as np

from fsd_path_planning.geometry.angles import vec_angle_between


def trace_distance_to_next(trace: np.ndarray) -> np.ndarray:
    """
    Calculates the distance of one point in the trace to the next. Obviously the last
    point doesn't have any distance associated

    Args:
        trace (np.array): The points of the trace

    Returns:
        np.array: A vector containing the distances from one point to the next
    """
    return np.linalg.norm(np.diff(trace, axis=-2), axis=-1)


def trace_angles_between(trace: np.ndarray) -> np.ndarray:
    """
    Calculates the angles in a trace from each point to its next

    Args:
        trace (np.array): The trace containing a series of 2d vectors

    Returns:
        np.array: The angle from each vector to its next, with `len(return_value) ==
        len(trace) - 1`
    """
    all_to_next = np.diff(trace, axis=-2)
    # Numba compiles one version per array layout. Whether this slice is
    # contiguous depends on the data (with a single row it is), so without the
    # copy a frame could hit a layout the warm-up never compiled and stall.
    from_middle_to_next = np.ascontiguousarray(all_to_next[..., 1:, :])
    from_middle_to_prev = -all_to_next[..., :-1, :]
    angles = vec_angle_between(from_middle_to_next, from_middle_to_prev)
    return angles
