"""
Types used across the package: array aliases, and the result each stage
returns.

Project: fsd_path_planning
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from fsd_path_planning.utils.cone_types import ConeTypes

GenericArray: TypeAlias = NDArray[Any]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.signedinteger[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]
SortableConeTypes: TypeAlias = Literal[
    ConeTypes.LEFT,
    ConeTypes.BLUE,
    ConeTypes.RIGHT,
    ConeTypes.YELLOW,
]

# Either accepted way of passing cones in: one (N, 2) array per cone type, or a
# single (N, 3) array of [x, y, cone_type] rows. normalize_cones turns the
# second into the first.
ConeInput: TypeAlias = "Sequence[FloatArray] | FloatArray"


class TraceOutcome(Enum):
    """How the search for one side's trace ended.

    No trace is an expected outcome, for example with too few cones in view,
    and is reported here rather than raised.
    """

    FOUND = "found"
    """The side has a sorted trace."""
    TOO_FEW_CONES = "too_few_cones"
    """Fewer than three cones in view, so no trace could be formed."""
    NO_START_CONE = "no_start_cone"
    """No cone was close enough to the car, and in front of it, to start from."""
    NO_VALID_TRACE = "no_valid_trace"
    """Starting cones were found, but no trace of at least three cones."""
    DROPPED_WHEN_COMBINED = "dropped_when_combined"
    """A trace was found, but lost when the two sides were made consistent."""


class _TupleCompatible:
    """Results that still unpack like the tuples they replaced.

    Only the fields named in ``_tuple_fields`` take part, in that order, so a
    field added later does not change what ``a, b = result`` gives.
    """

    _tuple_fields: ClassVar[tuple[str, ...]]

    def __iter__(self) -> Iterator[Any]:
        return (getattr(self, name) for name in self._tuple_fields)


@dataclass(frozen=True)
class SortingResult(_TupleCompatible):
    """Result of cone sorting."""

    left_cones: FloatArray
    right_cones: FloatArray
    left_outcome: TraceOutcome = TraceOutcome.FOUND
    right_outcome: TraceOutcome = TraceOutcome.FOUND

    _tuple_fields: ClassVar[tuple[str, ...]] = ("left_cones", "right_cones")


@dataclass(frozen=True)
class MatchingResult(_TupleCompatible):
    """Result of cone matching."""

    left_cones_with_virtual: FloatArray
    right_cones_with_virtual: FloatArray
    left_to_right_matches: IntArray
    right_to_left_matches: IntArray

    _tuple_fields: ClassVar[tuple[str, ...]] = (
        "left_cones_with_virtual",
        "right_cones_with_virtual",
        "left_to_right_matches",
        "right_to_left_matches",
    )


@dataclass(frozen=True)
class PathResult(_TupleCompatible):
    """Result of path calculation."""

    final_path: FloatArray  # (N, 4): spline_parameter, x, y, curvature
    centerline_basis: FloatArray  # (M, 2): the centerline points used as basis

    _tuple_fields: ClassVar[tuple[str, ...]] = ("final_path", "centerline_basis")
