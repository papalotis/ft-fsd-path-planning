"""
Array helpers that Numba compiles.

Project: fsd_path_planning
"""

from __future__ import annotations

import numpy as np

from fsd_path_planning._numba import njit


@njit
def in1d(test_values: np.ndarray, source_container: np.ndarray) -> np.ndarray:
    """
    Calculate a boolean mask for a 1d array indicating if an element in `test_values` is
    present in `source container` which is also 1d

    Args:
        test_values (np.ndarray): The values to test if they are inside the container
        source_container (np.ndarray): The container

    Returns:
        np.ndarray: A boolean array with the same length as `test_values`. If
        `return_value[i]` is `True` then `test_value[i]` is in `source_container`
    """
    source_sorted = np.sort(source_container)
    is_in = np.zeros(test_values.shape[0], dtype=np.bool_)
    for i, test_val in enumerate(test_values):
        for source_val in source_sorted:
            if test_val == source_val:
                is_in[i] = True
                break

            if source_val > test_val:
                break

    return is_in
