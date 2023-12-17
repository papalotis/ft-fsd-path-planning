from pathlib import Path

import streamlit as st

from fsd_path_planning.demo.streamlit_demo.cone_matching import run as run_matching
from fsd_path_planning.demo.streamlit_demo.cone_sorting import run as run_sorting
from fsd_path_planning.demo.streamlit_demo.path_calculation import (
    run as run_path_calculation,
)
from fsd_path_planning.demo.streamlit_demo.skidpad_relocalization import (
    run as run_skidpad,
)

st.set_page_config(page_title="FT Path Planning", page_icon="🏎️")


@st.cache  # type: ignore
def load_ed_slides_as_bytes() -> bytes:
    return Path("test.pdf").read_bytes()


def run_welcome() -> None:
    st.markdown(
        """
# FaSTTUBe Formula Student Driverless Path Planning Algorithm
## Path Planning Demo

Welcome to the path planning app. The goal of this app is to visualize the algorithms
that were used in the 2022/23 formula student season by FaSTTUBe (Formula Student Team TU Berlin).

The path planning algorithm is split into three parts:

- Cone Sorting
- Cone Matching
- Path Calculation

There is a special page for visualizing the algorithms that run when the Skidpad mission is selected.

At the top of the page you can select the algorithm you want to explore.
""".strip()
    )


STRING_TO_FUNCTION = {
    "Welcome": run_welcome,
    "Sorting": run_sorting,
    "Matching": run_matching,
    "Path calculation": run_path_calculation,
}

STRING_TO_FUNCTION_SKIDPAD = {
    "Welcome": run_welcome,
    "Relocalization": run_skidpad,
    "Path calculation": run_path_calculation,
}


with st.sidebar:
    st.session_state.track_configuration = st.radio(
        "Configuration",
        (
            "Straight",
            "Simple Corner",
            "Corner Missing Blue",
            "Corner Missing Blue Alt",
            "Hairpin",
            "Hairpin Extreme",
            "Skidpad",
            # "FS Spain 19 Full",
            # "Wrong sort",
            "Custom",
        ),
    )

    # only show text area if custom is selected
    if st.session_state.track_configuration == "Custom":
        json_text = st.text_area(
            "JSON Input",
            help='Put a string that represents a JSON dict here. The dict should have the keys "vehicle_position", "vehicle_direction", "cones_left" and "cones_right". The vehicle position and direction should be a list of two numbers, the cones should be a list of lists of two numbers.',
        ).strip()
        st.session_state.json_text = json_text

if st.session_state.track_configuration == "Custom":
    st.warning(
        "You can input arbitrary track configurations here. The path planner has not been thoroughly"
        " tested so it is possible that it will not work for all configurations. It is not difficult"
        " to find edge cases that will break the algorithm."
    )


string_to_function_to_use = (
    STRING_TO_FUNCTION_SKIDPAD
    if st.session_state.track_configuration == "Skidpad"
    else STRING_TO_FUNCTION
)

tabs = st.tabs(list(string_to_function_to_use.keys()))
for tab, page_function in zip(tabs, string_to_function_to_use.values()):
    with tab:
        page_function()
