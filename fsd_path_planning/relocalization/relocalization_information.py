from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from typing_extensions import (
    Self,
)

from fsd_path_planning.types import FloatArray


@dataclass
class RelocalizationInformation:
    translation: FloatArray  # [x, y]
    rotation: float  # [orientation in radians]

    @classmethod
    def from_transform_function(
        cls, transform_function: Callable[[FloatArray, float], Tuple[FloatArray, float]]
    ) -> Self:
        origin_xy = np.array([0.0, 0.0])
        one_zero = np.array([1.0, 0.0])

        origin_straight = 0

        origin_xy_transformed, _ = transform_function(origin_xy, origin_straight)
        one_zero_transformed, _ = transform_function(one_zero, origin_straight)

        translation = np.array([origin_xy_transformed[0], origin_xy_transformed[1]])
        rotation = np.arctan2(
            one_zero_transformed[1] - origin_xy_transformed[1],
            one_zero_transformed[0] - origin_xy_transformed[0],
        )

        return cls(translation, rotation)
