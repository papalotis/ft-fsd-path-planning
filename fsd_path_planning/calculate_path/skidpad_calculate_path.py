#!/usr/bin/env python3
"""
Special case of path calculation for skidpad.

Description: Skidpad requires a special case for path calculation. This file provides
a class that overwrites the path calculation for the skidpad track.
Project: fsd_path_planning
"""

from fsd_path_planning.calculate_path.core_calculate_path import CalculatePath
from fsd_path_planning.calculate_path.strategies import FollowKnownMap
from fsd_path_planning.config_dataclasses import PathConfig


class SkidpadCalculatePath(CalculatePath):
    """
    Path calculation for skidpad: follow the known figure-of-eight
    (`FollowKnownMap`) instead of fitting a spline through the basis.
    """

    def __init__(
        self,
        config: PathConfig | None = None,
        *,
        # Legacy parameters (deprecated, use config instead)
        smoothing: float | None = None,
        predict_every: float | None = None,
        maximal_distance_for_valid_path: float | None = None,
        max_deg: int | None = None,
        mpc_path_length: float | None = None,
        mpc_prediction_horizon: int | None = None,
    ):
        self._follow_known_map = FollowKnownMap()
        super().__init__(
            config=config,
            smoothing=smoothing,
            predict_every=predict_every,
            maximal_distance_for_valid_path=maximal_distance_for_valid_path,
            max_deg=max_deg,
            mpc_path_length=mpc_path_length,
            mpc_prediction_horizon=mpc_prediction_horizon,
            fit_strategy=self._follow_known_map,
        )

    @property
    def index_along_path(self) -> int:
        """Index into the global path of the car's last position."""
        return self._follow_known_map.index_along_path

    @index_along_path.setter
    def index_along_path(self, value: int) -> None:
        self._follow_known_map.index_along_path = value
