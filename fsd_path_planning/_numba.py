"""
The Numba decorator every compiled function in the package uses.

Project: fsd_path_planning
"""

from __future__ import annotations

import os
from typing import TypeVar

try:
    from numba import jit
except ImportError:

    def jit(**kwargs):
        def wrapper(func):
            return func

        return wrapper


T = TypeVar("T")


def _is_coverage_running() -> bool:
    if os.getenv("COVERAGE_PROCESS_START"):
        return True

    try:
        from coverage import Coverage
    except ImportError:
        return False

    return Coverage.current() is not None


def njit(func: T) -> T:
    """
    numba.njit is an untyped decorator. This wrapper helps type checkers keep the
    type information after applying the decorator. Furthermore, it sets some performance
    flags

    Args:
        func (T): The function to jit

    Returns:
        T: The jitted function
    """
    if _is_coverage_running():
        return func

    jit_func: T = jit(
        nopython=True,
        cache=True,
        nogil=True,
        fastmath=True,
    )(func)

    return jit_func


def njit_exact(func: T) -> T:
    """Like `njit`, but with IEEE floating point (no ``fastmath``).

    For kernels that replace NumPy code whose results must stay bit for bit
    the same: ``fastmath`` lets the compiler reorder and fuse arithmetic,
    which changes the last bits, and a comparison against a threshold can
    then come out the other way.
    """
    if _is_coverage_running():
        return func

    jit_func: T = jit(
        nopython=True,
        cache=True,
        nogil=True,
        fastmath=False,
    )(func)

    return jit_func
