"""
What each mission needs from the planner, in one place.

Every MissionTypes member has an entry, so a new mission is added here and
nowhere else. PathPlanner and the factory functions in `config` read from it.

Project: fsd_path_planning
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from fsd_path_planning.config_dataclasses import PipelineConfig
from fsd_path_planning.stages.path.skidpad import (
    SkidpadCalculatePath,
)
from fsd_path_planning.stages.path.stage import CalculatePath
from fsd_path_planning.stages.relocalization.acceleration import (
    AccelerationRelocalizer,
)
from fsd_path_planning.stages.relocalization.base import Relocalizer
from fsd_path_planning.stages.relocalization.skidpad import (
    SkidpadRelocalizer,
)
from fsd_path_planning.utils.mission_types import MissionTypes

RelocalizerFactory = Callable[[PipelineConfig], Relocalizer]


@dataclass(frozen=True)
class MissionSpec:
    """How the planner is put together for one mission."""

    path_calculation: type[CalculatePath] = CalculatePath
    """The path calculation stage class."""

    relocalizer_class: type[Relocalizer] | None = None
    """The relocalizer class, for missions on a known map; None otherwise."""

    make_relocalizer: RelocalizerFactory | None = None
    """Builds the relocalizer from the pipeline config. None for missions
    without one."""

    default_config: Callable[[], PipelineConfig] = PipelineConfig
    """The mission's default settings. Every mission uses the same values
    today; this is where a mission-specific default goes."""

    @property
    def sorts_cones(self) -> bool:
        """Whether the mission sorts and matches cones, rather than following
        a known map once relocalized."""
        return self.relocalizer_class is None


def _acceleration_relocalizer(_: PipelineConfig) -> Relocalizer:
    return AccelerationRelocalizer()


def _skidpad_relocalizer(config: PipelineConfig) -> Relocalizer:
    return SkidpadRelocalizer(config.skidpad)


_ACCELERATION = MissionSpec(
    relocalizer_class=AccelerationRelocalizer,
    make_relocalizer=_acceleration_relocalizer,
)
_SKIDPAD = MissionSpec(
    path_calculation=SkidpadCalculatePath,
    relocalizer_class=SkidpadRelocalizer,
    make_relocalizer=_skidpad_relocalizer,
)
_SORTING = MissionSpec()

MISSIONS: Mapping[MissionTypes, MissionSpec] = MappingProxyType(
    {
        MissionTypes.none: _SORTING,
        MissionTypes.acceleration: _ACCELERATION,
        MissionTypes.skidpad: _SKIDPAD,
        MissionTypes.autocross: _SORTING,
        MissionTypes.trackdrive: _SORTING,
        MissionTypes.ebs_test: _ACCELERATION,
        MissionTypes.inspection: _SORTING,
        MissionTypes.manual_driving: _SORTING,
    }
)


def mission_spec(mission: MissionTypes) -> MissionSpec:
    """The spec for `mission`."""
    return MISSIONS[MissionTypes(mission)]
