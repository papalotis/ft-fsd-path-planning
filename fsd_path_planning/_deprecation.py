"""
Deprecation support for names that moved.

Project: fsd_path_planning
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from typing import Any


class FsdPathPlanningDeprecationWarning(UserWarning):
    """Something used here is deprecated and will be removed.

    A UserWarning rather than a DeprecationWarning, so that Python shows it
    wherever the calling code lives, for example in a ROS node, not only in
    `__main__` and under test runners. Silence exactly these warnings with::

        warnings.filterwarnings("ignore", category=FsdPathPlanningDeprecationWarning)
    """


def moved_module(
    old_module: str, new_module: str, names: tuple[str, ...], removed_in: str
) -> Callable[[str], Any]:
    """A module ``__getattr__`` that forwards moved names and warns on each use.

    PEP 562: the old module defines nothing itself, so every name looked up on
    it reaches this function, which warns and returns the object from its new
    home. The warning points at the caller's import line.
    """

    def __getattr__(name: str) -> Any:
        if name not in names:
            raise AttributeError(f"module {old_module!r} has no attribute {name!r}")
        warnings.warn(
            f"{old_module}.{name} has moved to {new_module}.{name}; the old path"
            f" will be removed in {removed_in}",
            FsdPathPlanningDeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(new_module), name)

    return __getattr__
