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
        for name, target in forwarded.items():
            home, _, new_name = target.partition(":")
            with pytest.warns(FsdPathPlanningDeprecationWarning, match="0.7.0"):
                assert getattr(old, name) is getattr(
                    importlib.import_module(home), new_name or name
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


class TestRenamed:
    @pytest.mark.parametrize(
        ("module_name", "old_name", "new_path"),
        [
            (
                "fsd_path_planning.utils.math_utils",
                "my_cdist_sq_euclidean",
                "fsd_path_planning.geometry.distances.cdist_sq_euclidean",
            ),
            (
                "fsd_path_planning.utils.math_utils",
                "my_in1d",
                "fsd_path_planning.utils.arrays.in1d",
            ),
            (
                "fsd_path_planning.utils.math_utils",
                "my_njit",
                "fsd_path_planning._numba.njit",
            ),
            (
                "fsd_path_planning.utils.spline_fit",
                "SplineFitterFactory",
                "fsd_path_planning.geometry.splines.SplineFitter",
            ),
        ],
    )
    def test_old_name_gives_the_renamed_object(
        self, module_name, old_name, new_path
    ) -> None:
        new_module, _, new_name = new_path.rpartition(".")
        with pytest.warns(FsdPathPlanningDeprecationWarning, match=new_path):
            obj = getattr(importlib.import_module(module_name), old_name)
        assert obj is getattr(importlib.import_module(new_module), new_name)

    def test_spline_fitter_factory_attribute(self) -> None:
        from fsd_path_planning.stages.path.stage import (
            CalculatePath,
        )

        calculator = CalculatePath()
        with pytest.warns(FsdPathPlanningDeprecationWarning, match="spline_fitter"):
            assert calculator.spline_fitter_factory is calculator.spline_fitter


class TestPathHistoryLists:
    @pytest.mark.parametrize(
        "name", ["path_updates", "mpc_paths", "path_is_trivial_list"]
    )
    def test_still_readable_with_a_warning(self, name) -> None:
        from fsd_path_planning.stages.path.stage import (
            CalculatePath,
        )

        calculator = CalculatePath()
        with pytest.warns(FsdPathPlanningDeprecationWarning, match=name):
            assert getattr(calculator, name) == []


def _whole_module_shims() -> list[tuple[str, str]]:
    """(old module, new module) for every module that moved as a whole."""
    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parents[1] / "fsd_path_planning"
    pairs = []
    for path in sorted(package.rglob("*.py")):
        source = path.read_text()
        if "moved_module_entirely(" not in source or path.name == "_deprecation.py":
            continue
        call = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "moved_module_entirely"
        )
        old, new = (arg.value for arg in call.args[:2])  # type: ignore[attr-defined]
        pairs.append((old, new))
    return pairs


WHOLE_MODULE_SHIMS = _whole_module_shims()


class TestStageModulesMoved:
    def test_every_stage_module_moved(self) -> None:
        assert len(WHOLE_MODULE_SHIMS) == 26
        assert all(
            new.startswith("fsd_path_planning.stages.")
            or new == "fsd_path_planning.pipeline"
            for _, new in WHOLE_MODULE_SHIMS
        )

    @pytest.mark.parametrize(("old", "new"), WHOLE_MODULE_SHIMS)
    def test_old_path_forwards_every_name(self, old: str, new: str) -> None:
        old_module = importlib.import_module(old)
        new_module = importlib.import_module(new)
        names = [n for n in vars(new_module) if not n.startswith("__")]
        assert names
        for name in names:
            with pytest.warns(FsdPathPlanningDeprecationWarning, match=new):
                assert getattr(old_module, name) is getattr(new_module, name)

    @pytest.mark.parametrize(("old", "new"), WHOLE_MODULE_SHIMS[:3])
    def test_unknown_names_are_attribute_errors(self, old: str, new: str) -> None:
        with pytest.raises(AttributeError):
            importlib.import_module(old).does_not_exist  # noqa: B018

    def test_the_0_4_import_path_of_mission_to_relocalizer(self) -> None:
        with pytest.warns(FsdPathPlanningDeprecationWarning, match="pipeline"):
            from fsd_path_planning.full_pipeline.full_pipeline import (
                MissionToRelocalizer,
            )
        from fsd_path_planning.pipeline import MissionToRelocalizer as new

        assert MissionToRelocalizer is new
