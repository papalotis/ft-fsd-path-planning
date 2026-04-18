"""Unit tests for cone types and mission types enums."""

from __future__ import annotations

from fsd_path_planning.utils.cone_types import ConeTypes, invert_cone_type
from fsd_path_planning.utils.mission_types import MissionTypes


class TestConeTypes:
    def test_integer_values(self):
        assert ConeTypes.UNKNOWN == 0
        assert ConeTypes.RIGHT == 1
        assert ConeTypes.YELLOW == 1
        assert ConeTypes.LEFT == 2
        assert ConeTypes.BLUE == 2
        assert ConeTypes.START_FINISH_AREA == 3
        assert ConeTypes.ORANGE_SMALL == 3
        assert ConeTypes.START_FINISH_LINE == 4
        assert ConeTypes.ORANGE_BIG == 4

    def test_aliases_are_equal(self):
        assert ConeTypes.RIGHT is ConeTypes.YELLOW
        assert ConeTypes.LEFT is ConeTypes.BLUE
        assert ConeTypes.START_FINISH_AREA is ConeTypes.ORANGE_SMALL
        assert ConeTypes.START_FINISH_LINE is ConeTypes.ORANGE_BIG

    def test_len(self):
        # 5 unique values: 0, 1, 2, 3, 4
        assert len(ConeTypes) == 5

    def test_iterable(self):
        values = list(ConeTypes)
        assert len(values) == 5


class TestInvertConeType:
    def test_left_to_right(self):
        assert invert_cone_type(ConeTypes.LEFT) == ConeTypes.RIGHT

    def test_right_to_left(self):
        assert invert_cone_type(ConeTypes.RIGHT) == ConeTypes.LEFT

    def test_unknown_unchanged(self):
        assert invert_cone_type(ConeTypes.UNKNOWN) == ConeTypes.UNKNOWN

    def test_orange_small_unchanged(self):
        assert invert_cone_type(ConeTypes.ORANGE_SMALL) == ConeTypes.ORANGE_SMALL

    def test_orange_big_unchanged(self):
        assert invert_cone_type(ConeTypes.ORANGE_BIG) == ConeTypes.ORANGE_BIG


class TestMissionTypes:
    def test_integer_values(self):
        assert MissionTypes.none == 0
        assert MissionTypes.acceleration == 1
        assert MissionTypes.skidpad == 2
        assert MissionTypes.autocross == 3
        assert MissionTypes.trackdrive == 4
        assert MissionTypes.ebs_test == 5
        assert MissionTypes.inspection == 6
        assert MissionTypes.manual_driving == 7

    def test_len(self):
        assert len(MissionTypes) == 8
