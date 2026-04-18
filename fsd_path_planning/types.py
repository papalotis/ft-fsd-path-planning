#!/usr/bin/env python3
"""
Description: Define types that are used commonly in the whole package
Project: fsd_path_planning
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
from numpy.typing import NDArray

from fsd_path_planning.utils.cone_types import ConeTypes

if TYPE_CHECKING:
    GenericArray = NDArray[Any]
    FloatArray = NDArray[np.float64]
    IntArray = NDArray[np.signedinteger[Any]]
    BoolArray = NDArray[np.bool_]
    SortableConeTypes = Literal[
        ConeTypes.LEFT,
        ConeTypes.BLUE,
        ConeTypes.RIGHT,
        ConeTypes.YELLOW,
    ]
else:
    GenericArray = None  # pylint: disable=invalid-name
    FloatArray = None  # pylint: disable=invalid-name
    IntArray = None  # pylint: disable=invalid-name
    BoolArray = None  # pylint: disable=invalid-name
    SortableConeTypes = ConeTypes


@dataclass
class SortingResult:
    """Result of cone sorting."""

    left_cones: FloatArray
    right_cones: FloatArray

    def __iter__(self) -> Iterator:
        return (getattr(self, f.name) for f in fields(self))


@dataclass
class MatchingResult:
    """Result of cone matching."""

    left_cones_with_virtual: FloatArray
    right_cones_with_virtual: FloatArray
    left_to_right_matches: IntArray
    right_to_left_matches: IntArray

    def __iter__(self) -> Iterator:
        return (getattr(self, f.name) for f in fields(self))


@dataclass
class PathResult:
    """Result of path calculation."""

    final_path: FloatArray  # (N, 4): spline_parameter, x, y, curvature
    centerline_basis: FloatArray  # (M, 2): the centerline points used as basis

    def __iter__(self) -> Iterator:
        return (getattr(self, f.name) for f in fields(self))
