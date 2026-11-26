"""The warm-up must leave nothing for Numba to compile while the car is driving."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from numba.core.dispatcher import Dispatcher

from fsd_path_planning import PathPlanner, warmup
from fsd_path_planning.geometry.angles import angle_from_2d_vector
from tests.conftest import DEMO_DIR, GOLDEN_RUNS, load_data_json, strip_colour


def compiled_signatures() -> dict[str, set[str]]:
    """Every compiled signature of every Numba function the package has loaded."""
    found: dict[str, set[str]] = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith("fsd_path_planning") or ".demo" in name:
            continue
        for value in vars(module).values():
            if isinstance(value, Dispatcher):
                key = f"{value.__module__}.{value.__name__}"
                found[key] = set(map(str, value.signatures))
    return found


class TestWarmupRuns:
    def test_returns_the_time_it_took(self) -> None:
        assert warmup() >= 0.0

    def test_a_second_call_is_harmless(self) -> None:
        warmup()
        assert warmup() >= 0.0


class TestAngleFromVector:
    def test_a_single_vector_gives_a_numpy_scalar(self) -> None:
        angle = angle_from_2d_vector(np.array([0.0, 1.0]))
        assert isinstance(angle, np.float64)
        assert angle == pytest.approx(np.pi / 2)

    def test_several_vectors_give_an_array(self) -> None:
        angles = angle_from_2d_vector(np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
        assert isinstance(angles, np.ndarray)
        np.testing.assert_allclose(angles, [0.0, np.pi / 4, np.pi / 2])

    def test_keeps_the_leading_shape(self) -> None:
        assert angle_from_2d_vector(np.ones((4, 3, 2))).shape == (4, 3)

    def test_integer_input_is_accepted(self) -> None:
        assert angle_from_2d_vector(np.array([1, 1])) == pytest.approx(np.pi / 4)

    def test_wrong_shape_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="2d vector"):
            angle_from_2d_vector(np.ones(3))


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("NUMBA_DISABLE_JIT") == "1", reason="needs the JIT enabled"
)
def test_nothing_compiles_while_driving_after_warmup() -> None:
    """On a fresh install a single mid-run compile stalled a frame for seconds."""
    warmup()
    before = compiled_signatures()

    late = []
    for stem, mission, colour in GOLDEN_RUNS:
        positions, directions, observations = load_data_json(DEMO_DIR / f"{stem}.json")
        if not colour:
            observations = [strip_colour(cones) for cones in observations]
        for experimental_performance_improvements in (False, True):
            planner = PathPlanner(mission, experimental_performance_improvements)
            for frame, (position, direction, cones) in enumerate(
                zip(positions, directions, observations, strict=True)
            ):
                planner.calculate_path_in_global_frame(cones, position, direction)
                now = compiled_signatures()
                for function, signatures in now.items():
                    added = signatures - before.get(function, set())
                    if added:
                        late.append(f"{stem} frame {frame}: {function} {sorted(added)}")
                before = now

    assert not late, "compiled after the warm-up:\n" + "\n".join(late)
