"""Path basis selection: side selection and centerline computation."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes


def side_score(matches: IntArray) -> tuple:
    """Score a side by number of matches and sum of match indices."""
    filtered = matches[matches != -1]
    return len(filtered), filtered.sum()


def select_side_to_use(
    left_cones: FloatArray,
    right_cones: FloatArray,
    left_to_right_matches: IntArray,
    right_to_left_matches: IntArray,
) -> Tuple[FloatArray, IntArray, FloatArray]:
    """Select the main side to use for path calculation."""
    left_score = side_score(left_to_right_matches)
    right_score = side_score(right_to_left_matches)

    if left_score >= right_score:
        return left_cones, left_to_right_matches, right_cones
    return right_cones, right_to_left_matches, left_cones


def calculate_centerline_points(
    side_to_use: FloatArray,
    matches_to_other_side: IntArray,
    match_on_other_side: FloatArray,
    fallback_path: FloatArray,
) -> FloatArray:
    """Compute centerline between matched cone pairs.

    Args:
        side_to_use: Cones on the selected side.
        matches_to_other_side: Match indices (-1 for unmatched).
        match_on_other_side: Matched cone positions on the other side.
        fallback_path: Previous path xy to use if too few matches.

    Returns:
        Centerline points (N, 2).
    """
    center = (side_to_use + match_on_other_side) / 2
    center = center[matches_to_other_side != -1]

    if len(center) < 2:
        return fallback_path

    return center
