#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Enum for cone types (yellow, blue, etc...)

Taken directly from ft-as-utils

Project: FaSTTUBe Chabo Common
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
