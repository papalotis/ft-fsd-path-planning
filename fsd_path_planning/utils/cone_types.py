"""
Cone types (yellow, blue, ...) and the two sides of the track.

Project: fsd_path_planning
"""

from __future__ import annotations

from enum import Enum, IntEnum


class ConeTypes(IntEnum):
    """
    Enum for all possible cone types
    """

    UNKNOWN = 0
    RIGHT = YELLOW = 1
    LEFT = BLUE = 2
    START_FINISH_AREA = ORANGE_SMALL = 3
    START_FINISH_LINE = ORANGE_BIG = 4


def invert_cone_type(cone_type: ConeTypes) -> ConeTypes:
    """
    Inverts the cone type. E.g. LEFT -> RIGHT
    Args:
        cone_type: The cone type to invert
    Returns:
        ConeTypes: The inverted cone type
    """
    if cone_type in (ConeTypes.LEFT, ConeTypes.RIGHT):
        return Side.of(cone_type).other.cone_type
    return cone_type


class Side(Enum):
    """A side of the track.

    The sorter, matcher and path calculation all work per side. Side names
    that explicitly; `cone_type` gives the ConeTypes value the side's cones
    carry, which is what arrays and compiled code use. Side is deliberately
    not an IntEnum: Numba compiles a separate version for each enum type it
    is given, so only ConeTypes values reach compiled code.
    """

    LEFT = "left"
    RIGHT = "right"

    @property
    def cone_type(self) -> ConeTypes:
        return ConeTypes.LEFT if self is Side.LEFT else ConeTypes.RIGHT

    @property
    def other(self) -> Side:
        return Side.RIGHT if self is Side.LEFT else Side.LEFT

    @classmethod
    def of(cls, cone_type: ConeTypes) -> Side:
        """The side whose cones have `cone_type`; LEFT/BLUE or RIGHT/YELLOW."""
        if cone_type == ConeTypes.LEFT:
            return cls.LEFT
        if cone_type == ConeTypes.RIGHT:
            return cls.RIGHT
        raise ValueError(f"{cone_type!r} is not the cone type of a side")
