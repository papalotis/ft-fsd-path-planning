#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Define types that are used commonly in the whole package
Project: fsd_path_planning
"""

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from fsd_path_planning.utils.cone_types import ConeTypes

if TYPE_CHECKING:
    GenericArray = NDArray[Any]
    FloatArray = NDArray[np.float_]
    IntArray = NDArray[np.int_]
    BoolArray = NDArray[np.bool_]
    SortableConeTypes = Literal[
        ConeTypes.LEFT,
        ConeTypes.BLUE,
        ConeTypes.RIGHT,
        ConeTypes.YELLOW,
    ]
else:
    GenericArray = None  # pylint: disable=invalid-name
    FloatArray = None  # pylint: disable=invalid-name
    IntArray = None  # pylint: disable=invalid-name
    BoolArray = None  # pylint: disable=invalid-name
    SortableConeTypes = ConeTypes
