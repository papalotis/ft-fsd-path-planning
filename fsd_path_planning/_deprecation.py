"""
Deprecation support for names that moved.

Project: fsd_path_planning
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable, Mapping
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
    return moved_names(old_module, dict.fromkeys(names, new_module), removed_in)


def moved_names(
    old_module: str,
    new_homes: Mapping[str, str],
    removed_in: str,
    retired: Mapping[str, str] | None = None,
) -> Callable[[str], Any]:
    """Like `moved_module`, for a module whose names went to different places.

    ``new_homes`` maps each moved name to the module it lives in now, or to
    ``"module:new_name"`` when it was also renamed.
    ``retired`` maps names that have no new public home, because the planner
    no longer uses them, to the private module that still holds them until
    ``removed_in``; their warning says so.
    """
    retired = dict(retired or {})

    def __getattr__(name: str) -> Any:
        if name in new_homes:
            home, _, new_name = new_homes[name].partition(":")
            new_name = new_name or name
            message = (
                f"{old_module}.{name} has moved to {home}.{new_name}; the old"
                f" path will be removed in {removed_in}"
            )
        elif name in retired:
            home, new_name = retired[name], name
            message = (
                f"{old_module}.{name} is no longer used by fsd_path_planning and"
                f" will be removed in {removed_in}; copy it into your own code if"
                " you need it"
            )
        else:
            raise AttributeError(f"module {old_module!r} has no attribute {name!r}")
        warnings.warn(message, FsdPathPlanningDeprecationWarning, stacklevel=2)
        return getattr(importlib.import_module(home), new_name)

    return __getattr__


def moved_module_entirely(
    old_module: str, new_module: str, removed_in: str
) -> Callable[[str], Any]:
    """A module ``__getattr__`` for a module that moved as a whole.

    Every name the new module has, public or not, is forwarded with a
    warning, so that whatever code imported from the old path keeps working.
    """

    def __getattr__(name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"module {old_module!r} has no attribute {name!r}")
        target = importlib.import_module(new_module)
        if not hasattr(target, name):
            raise AttributeError(f"module {old_module!r} has no attribute {name!r}")
        warnings.warn(
            f"{old_module} has moved to {new_module}; the old path will be"
            f" removed in {removed_in}",
            FsdPathPlanningDeprecationWarning,
            stacklevel=2,
        )
        return getattr(target, name)

    return __getattr__
