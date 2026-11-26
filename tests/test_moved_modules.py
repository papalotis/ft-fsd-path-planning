"""Old import paths keep working for one release, with a warning always shown."""

from __future__ import annotations

import importlib
import warnings

import pytest

from fsd_path_planning import common_types
from fsd_path_planning._deprecation import FsdPathPlanningDeprecationWarning


class TestTypesModuleMoved:
    def test_old_import_still_works_and_warns(self) -> None:
        with pytest.warns(FsdPathPlanningDeprecationWarning, match="common_types"):
            from fsd_path_planning.types import FloatArray
        assert FloatArray is common_types.FloatArray

    def test_every_moved_name_resolves_to_the_same_object(self) -> None:
        old = importlib.import_module("fsd_path_planning.types")
        for name in (
            "GenericArray",
            "FloatArray",
            "IntArray",
            "BoolArray",
            "SortableConeTypes",
            "ConeInput",
            "SortingResult",
            "MatchingResult",
            "PathResult",
        ):
            with pytest.warns(FsdPathPlanningDeprecationWarning):
                assert getattr(old, name) is getattr(common_types, name)

    def test_unknown_name_is_an_attribute_error(self) -> None:
        old = importlib.import_module("fsd_path_planning.types")
        with pytest.raises(AttributeError):
            old.DoesNotExist  # noqa: B018

    def test_the_warning_names_the_removal_version(self) -> None:
        with pytest.warns(FsdPathPlanningDeprecationWarning, match="0.7.0"):
            from fsd_path_planning.types import IntArray  # noqa: F401


class TestWarningIsAlwaysShown:
    def test_it_is_a_user_warning_not_a_deprecation_warning(self) -> None:
        """Python hides DeprecationWarning outside __main__ by default."""
        assert issubclass(FsdPathPlanningDeprecationWarning, UserWarning)
        assert not issubclass(FsdPathPlanningDeprecationWarning, DeprecationWarning)

    def test_default_filters_show_it(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()
            warnings.simplefilter("default")
            from fsd_path_planning.types import BoolArray  # noqa: F401
        assert any(
            issubclass(w.category, FsdPathPlanningDeprecationWarning) for w in caught
        )
