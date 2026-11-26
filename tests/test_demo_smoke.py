"""The demos still run: the Streamlit app, headless, and the command line demo.

Skipped unless the demo extra is installed (pip install .[demo]); CI runs them
in their own job.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.demo

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA = REPO_ROOT / "fsd_path_planning" / "demo"

TRACK_CONFIGURATIONS = [
    "Straight",
    "Simple Corner",
    "Corner Missing Blue",
    "Corner Missing Blue Alt",
    "Hairpin",
    "Hairpin Extreme",
]


@pytest.mark.parametrize("configuration", TRACK_CONFIGURATIONS)
def test_streamlit_app_renders_every_page(configuration: str) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(REPO_ROOT / "streamlit_main.py"), default_timeout=600)
    app.run()
    assert not app.exception, [e.value for e in app.exception]

    app.sidebar.radio[0].set_value(configuration).run()
    assert not app.exception, [e.value for e in app.exception]
    # the welcome tab and one tab per algorithm
    assert len(app.tabs) == 4


@pytest.mark.parametrize("dataset", ["skidpad.json", "fsg_19_2_laps.json"])
@pytest.mark.parametrize("remove_color_info", [False, True])
def test_command_line_demo_runs(dataset: str, remove_color_info: bool) -> None:
    pytest.importorskip("typer")
    pytest.importorskip("matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from typer.testing import CliRunner

    from fsd_path_planning.demo.json_demo import app

    args = ["--data-path", str(DEMO_DATA / dataset), "--disable-visualization"]
    if remove_color_info:
        args.append("--remove-color-info")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
