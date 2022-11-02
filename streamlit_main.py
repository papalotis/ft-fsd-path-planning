from itertools import chain
from pathlib import Path

import streamlit as st

from fsd_path_planning.demo.streamlit_demo.cone_matching import run as run_matching
from fsd_path_planning.demo.streamlit_demo.cone_sorting import run as run_sorting
from fsd_path_planning.demo.streamlit_demo.path_calculation import (
    run as run_path_calculation,
)

st.set_page_config(page_title="Chabo AS Pathing", page_icon="🏎")


@st.cache  # type: ignore
def load_ed_slides_as_bytes() -> bytes:
    return Path("test.pdf").read_bytes()


def run_welcome() -> None:
    st.markdown(
        """
# Chabo AS
## Pathing

Welcome to the Chabo AS Pathing app. The goal of this app is to visualize the algorithms
that are used for the path planning of the Chabo Autonomous System.

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
    st.markdown("# Chabo AS Pathing")

    modes = list(chain(STRING_TO_FUNCTION, map(str.lower, STRING_TO_FUNCTION)))
    try:
        mode = st.experimental_get_query_params()["mode"][0]
        index_mode = list(modes).index(mode) % len(STRING_TO_FUNCTION)
    except (KeyError, ValueError):
        index_mode = 0

    page_function = STRING_TO_FUNCTION[
        st.selectbox("Mode", STRING_TO_FUNCTION, index=index_mode)
    ]

    st.markdown("---")
    st.session_state.track_configuration = st.selectbox(
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


page_function()
