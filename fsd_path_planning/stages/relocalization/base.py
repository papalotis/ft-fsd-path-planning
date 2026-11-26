"""
Place the car in a known map: the common base of the mission relocalizers.

Project: fsd_path_planning
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from typing_extensions import Protocol

from fsd_path_planning.common_types import FloatArray
from fsd_path_planning.stages.relocalization.information import (
    RelocalizationInformation,
)


class RelocalizationOutcome(Enum):
    """Where relocalization stands after a frame.

    Not yet relocalized is an expected outcome, for example while the car
    waits at the start and has seen too little of the track, and is reported
    here rather than raised.
    """

    RELOCALIZED = "relocalized"
    """Relocalized on this frame."""
    ALREADY_RELOCALIZED = "already_relocalized"
    """Relocalized on an earlier frame; the transform is kept."""
    NOT_YET = "not_yet"
    """Not enough of the known map recognised yet; tried again next frame."""


@dataclass
class RelocalizationInput:
    """What a relocalizer needs for one frame."""

    cones_by_type: list[FloatArray]
    vehicle_position: FloatArray
    vehicle_direction: FloatArray


@dataclass(frozen=True)
class RelocalizationResult:
    """Result of one relocalization attempt."""

    outcome: RelocalizationOutcome
    information: RelocalizationInformation | None
    """The transform into the known map frame, once relocalized."""

    @property
    def is_relocalized(self) -> bool:
        return self.outcome is not RelocalizationOutcome.NOT_YET


class RelocalizationCallable(Protocol):
    def __call__(
        self, position_2d: FloatArray, direction_yaw: float
    ) -> tuple[FloatArray, float]: ...


class Relocalizer(ABC):
    def __init__(self):
        self._original_vehicle_position: FloatArray | None = None
        self._original_vehicle_direction: FloatArray | None = None

        self._original_to_known_map_transformer: RelocalizationCallable | None = None
        self._known_map_to_original_transformer: RelocalizationCallable | None = None

    @abstractmethod
    def do_relocalization_once(
        self,
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> tuple[RelocalizationCallable, RelocalizationCallable] | None:
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
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> None:
        if self.is_relocalized:
            return

        is_vehicle_position_none = self._original_vehicle_position is None
        is_vehicle_direction_none = self._original_vehicle_direction is None

        assert is_vehicle_position_none == is_vehicle_direction_none, (
            f"One of position or direction is not None but the other is "
            f"{is_vehicle_position_none=!r} | {is_vehicle_direction_none=!r}"
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
    ) -> tuple[FloatArray, float]:
        if self._original_to_known_map_transformer is None:
            raise ValueError("No transformation calculated yet")

        return self._original_to_known_map_transformer(position_2d, yaw)

    def transform_to_original_frame(
        self, position_2d: FloatArray, yaw: float
    ) -> tuple[FloatArray, float]:
        if self._known_map_to_original_transformer is None:
            raise ValueError("No transformation calculated yet")

        return self._known_map_to_original_transformer(position_2d, yaw)

    def run(self, stage_input: RelocalizationInput) -> RelocalizationResult:
        """Try to relocalize on this frame; the `Stage` entry point.

        Once relocalized, later frames keep the transform and do not recompute
        it.
        """
        was_relocalized = self.is_relocalized
        self.attempt_relocalization_calculation(
            stage_input.cones_by_type,
            stage_input.vehicle_position,
            stage_input.vehicle_direction,
        )
        if not self.is_relocalized:
            return RelocalizationResult(RelocalizationOutcome.NOT_YET, None)

        # computed once, when relocalization succeeds; the transform is fixed
        # from then on
        information = getattr(self, "_information", None)
        if information is None or not was_relocalized:
            assert self._original_to_known_map_transformer is not None
            information = RelocalizationInformation.from_transform_function(
                self._original_to_known_map_transformer
            )
            self._information = information
        outcome = (
            RelocalizationOutcome.ALREADY_RELOCALIZED
            if was_relocalized
            else RelocalizationOutcome.RELOCALIZED
        )
        return RelocalizationResult(outcome, information)

    @property
    def is_relocalized(self) -> bool:
        return (
            self._original_to_known_map_transformer is not None
            and self._known_map_to_original_transformer is not None
        )
