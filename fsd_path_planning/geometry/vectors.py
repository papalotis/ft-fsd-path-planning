"""
Dot products, norms and normalisation along the last axis.

Project: fsd_path_planning
"""

from __future__ import annotations

import numpy as np

from fsd_path_planning._numba import my_njit


@my_njit
def vec_dot(vecs1: np.ndarray, vecs2: np.ndarray) -> np.ndarray:
    """
    Mutliplies vectors in an array elementwise

    Args:
        vecs1 (np.array): The first "list" of vectors
        vecs2 (np.array): The second "list" of vectors

    Returns:
        np.array: The results
    """
    return np.sum(vecs1 * vecs2, axis=-1)


@my_njit
def norm_of_last_axis(arr: np.ndarray) -> np.ndarray:
    original_shape = arr.shape
    arr_row_col = np.ascontiguousarray(arr).reshape(-1, arr.shape[-1])
    result = np.empty(arr_row_col.shape[0])
    for i in range(arr_row_col.shape[0]):
        vec = arr_row_col[i]
        result[i] = np.sqrt(vec_dot(vec, vec))

    result = result.reshape(original_shape[:-1])

    return result


@my_njit
def normalize_last_axis(vecs: np.ndarray) -> np.ndarray:
    """
    Returns a normalized version of vecs

    Args:
        vecs (np.ndarray): The vectors to normalize
    Returns:
        np.ndarray: The normalized vectors
    """
    vecs_flat = vecs.reshape(-1, vecs.shape[-1])
    out = np.zeros(vecs.shape, dtype=vecs.dtype)
    for i, vec in enumerate(vecs_flat):
        out[i] = vec / np.linalg.norm(vec)

    return out.reshape(vecs.shape)
