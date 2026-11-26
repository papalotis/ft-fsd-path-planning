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


GEOMETRY_SHIMS = (
    "fsd_path_planning.utils.math_utils",
    "fsd_path_planning.utils.spline_fit",
    "fsd_path_planning.sorting_cones.trace_sorter.line_segment_intersection",
)


def _forwarded_names(module_name: str) -> dict[str, str]:
    """Each name a shim forwards, mapped to the module it now lives in."""
    shim = importlib.import_module(module_name)
    mapping: dict[str, str] = {}
    for cell in shim.__getattr__.__closure__ or ():
        if isinstance(cell.cell_contents, dict):
            mapping.update(cell.cell_contents)
    return mapping


class TestGeometryMoved:
    @pytest.mark.parametrize("module_name", GEOMETRY_SHIMS)
    def test_every_old_name_resolves_to_the_new_object(self, module_name) -> None:
        old = importlib.import_module(module_name)
        forwarded = _forwarded_names(module_name)
        assert forwarded
        for name, home in forwarded.items():
            with pytest.warns(FsdPathPlanningDeprecationWarning, match="0.7.0"):
                assert getattr(old, name) is getattr(
                    importlib.import_module(home), name
                )

    def test_moved_names_point_at_their_new_home(self) -> None:
        with pytest.warns(
            FsdPathPlanningDeprecationWarning,
            match=r"fsd_path_planning\.geometry\.transforms\.rotate",
        ):
            from fsd_path_planning.utils.math_utils import rotate  # noqa: F401

    def test_retired_helpers_say_they_have_no_replacement(self) -> None:
        with pytest.warns(FsdPathPlanningDeprecationWarning, match="no longer used"):
            from fsd_path_planning.utils.math_utils import lerp  # noqa: F401

    def test_the_package_itself_uses_no_deprecated_path(self) -> None:
        """Importing and running the planner must not trigger any shim."""
        import subprocess
        import sys

        code = (
            "import warnings\n"
            "from fsd_path_planning._deprecation import "
            "FsdPathPlanningDeprecationWarning as W\n"
            "warnings.simplefilter('error', W)\n"
            "import fsd_path_planning, fsd_path_planning.geometry\n"
            "from fsd_path_planning.warmup import warmup\n"
            "warmup()\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
