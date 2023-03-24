#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Enum for cone types (yellow, blue, etc...)
Project: fsd_path_planning
"""
from enum import IntEnum


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
    if cone_type == ConeTypes.LEFT:
        return ConeTypes.RIGHT
    if cone_type == ConeTypes.RIGHT:
        return ConeTypes.LEFT
    return cone_type
