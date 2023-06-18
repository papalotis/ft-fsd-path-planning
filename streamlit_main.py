from pathlib import Path

import streamlit as st

from fsd_path_planning.demo.streamlit_demo.cone_matching import run as run_matching
from fsd_path_planning.demo.streamlit_demo.cone_sorting import run as run_sorting
from fsd_path_planning.demo.streamlit_demo.path_calculation import (
    run as run_path_calculation,
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
that are used in the 2021/22 formula student season by FaSTTUBe (Formula Student Team TU Berlin).

The path planning algorithm is split ito three parts:

- Cone Sorting
- Cone Matching
- Path Calculation

In the sidebar of the app you can select the algorithm you want to explore.
""".strip()
    )


STRING_TO_FUNCTION = {
    "Welcome": run_welcome,
    "Sorting": run_sorting,
    "Matching": run_matching,
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
            # "FS Spain 19 Full",
            # "Wrong sort",
        ),
    )

tabs = st.tabs(list(STRING_TO_FUNCTION.keys()))
for tab, page_function in zip(tabs, STRING_TO_FUNCTION.values()):
    with tab:
        page_function()
