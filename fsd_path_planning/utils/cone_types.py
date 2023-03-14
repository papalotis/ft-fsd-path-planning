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
    YELLOW = 1
    RIGHT = 1
    BLUE = 2
    LEFT = 2
    ORANGE_SMALL = 3
    START_FINISH_AREA = 3
    ORANGE_BIG = 4
    START_FINISH_LINE = 4


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
