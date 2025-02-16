#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Place the car in the known map and relocalize it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from typing_extensions import Protocol

from fsd_path_planning.types import FloatArray


class RelocalizationCallable(Protocol):
    def __call__(self, position_2d: FloatArray, direction_yaw: float) -> Tuple[FloatArray, float]:
        pass


class Relocalizer(ABC):
    def __init__(self):
        self._original_vehicle_position: FloatArray | None = None
        self._original_vehicle_direction: FloatArray | None = None

        self._original_to_known_map_transformer: RelocalizationCallable | None = None
        self._known_map_to_original_transformer: RelocalizationCallable | None = None

    @abstractmethod
    def do_relocalization_once(
        self,
        cones: List[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> Tuple[RelocalizationCallable, RelocalizationCallable] | None:
        """Does the actual relocalization calculation

        First callable should be original frame to known
        Second callable should be known frame to original

        If relocalization fails return None
        """
        pass

    @abstractmethod
    def get_known_global_path(self) -> FloatArray:
        pass

    def attempt_relocalization_calculation(
        self,
        cones: List[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> None:
        if self.is_relocalized:
            return

        is_vehicle_position_none = self._original_vehicle_position is None
        is_vehicle_direction_none = self._original_vehicle_direction is None

        assert is_vehicle_position_none == is_vehicle_direction_none, (
            f"One of position or direction is not None but the other is {is_vehicle_position_none=!r} | {is_vehicle_direction_none=!r}"
        )

        if is_vehicle_position_none and is_vehicle_direction_none:
            self._original_vehicle_position = vehicle_position
            self._original_vehicle_direction = vehicle_direction

        result = self.do_relocalization_once(cones, vehicle_position, vehicle_direction)
        if result is not None:
            (
                self._original_to_known_map_transformer,
                self._known_map_to_original_transformer,
            ) = result

    def transform_to_known_map_frame(
        self, position_2d: FloatArray, yaw: float
    ) -> Tuple[RelocalizationCallable, RelocalizationCallable]:
        if self._original_to_known_map_transformer is None:
            raise ValueError("No transformation calculated yet")

        return self._original_to_known_map_transformer(position_2d, yaw)

    def transform_to_original_frame(self, position_2d: FloatArray, yaw: float) -> Tuple[FloatArray, float]:
        if self._known_map_to_original_transformer is None:
            raise ValueError("No transformation calculated yet")

        return self._known_map_to_original_transformer(position_2d, yaw)

    @property
    def is_relocalized(self):
        return (
            self._original_to_known_map_transformer is not None and self._known_map_to_original_transformer is not None
        )
