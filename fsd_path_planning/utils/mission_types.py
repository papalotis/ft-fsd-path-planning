#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Provides an enum for mission types (acceleration, skidpad, etc...)

Project: FaSTTUBe Chabo Common
"""
from enum import IntEnum


class MissionTypes(IntEnum):
    """
    Enum for each mission type
    """

    (
        none,
        acceleration,
        skidpad,
        autocross,
        trackdrive,
        ebs_test,
        inspection,
        manual_driving,
    ) = range(8)
