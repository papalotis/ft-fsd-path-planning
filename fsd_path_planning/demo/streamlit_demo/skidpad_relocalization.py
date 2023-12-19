from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from fsd_path_planning.demo.streamlit_demo.common import (
    get_cones_for_configuration,
    visualize_configuration,
)
from fsd_path_planning.skidpad.skidpad_path_data import BASE_SKIDPAD_PATH
from fsd_path_planning.skidpad.skidpad_relocalizer import (
    PowersetCirceFitResult,
    circle_fit_powerset,
)
from fsd_path_planning.types import FloatArray
from fsd_path_planning.utils.cone_types import ConeTypes

import plotly.graph_objects as go


def show_powerset(
    cones_by_type: list[FloatArray], position: FloatArray, direction: FloatArray
) -> PowersetCirceFitResult:
    all_cones = np.row_stack(cones_by_type)
    r = circle_fit_powerset(all_cones)

    ax = visualize_configuration(
        position,
        direction,
        cones_by_type,
        with_cone_index=False,
        with_lines=False,
        do_show=False,
    )

    for (cx, cy, cr), idxs in r:
        circle = plt.Circle((cx, cy), cr, fill=False)
        ax.add_artist(circle)

    st.pyplot(ax.figure)


def show_path() -> None:
    path = BASE_SKIDPAD_PATH[::75]
    idxs = np.arange(len(path))

    # 3d plot using plotly
    fig = go.Figure(data=[go.Scatter3d(x=path[:, 0], y=path[:, 1], z=idxs, mode="lines")])

    # make sure that the axis are equal
    # orthographic projection
    fig.update_layout(
        scene=dict(aspectmode="data")
    )

    st.plotly_chart(fig)
    


def run() -> None:
    st.markdown(
        """
# Skidpad Relocalization

The Skidpad mission presents a completely different challenge than the other missions. Rather than facing an unknown environment, the Skidpad mission is always the same. However, the way the track needs to be driven is not obvious at first glance, not even for human drivers.

The Skidpad mission requires therefore a completely different approach. Since the track is always the same, a precomputed path can be used to drive the track. The precomputed path can be seen below:
""".strip()
    )
    show_path()

    st.markdown(
        """
However, the car needs to know where it is along the precomputed path. This is where relocalization comes into play. The path planning algorithm already assumes that reliable localization is available. The problem that arises is that the frame of reference for the precomputed path and the frame of reference of the localization will not be the same. Furthermore, when the car is placed on the track, it will not be on the exact same location each time, so we cannot hardcode the transformation. Therefore, we need to calculate the transformation between the two frames of reference.

The relocalization algorithm is split into the following parts:

- Powerset circle fitting
- Center clustering
- Transformation calculation

The Skidpad track looks like this:
"""
    )

    position, direction, cones_by_type = get_cones_for_configuration(
        st.session_state.track_configuration, do_shuffle=True
    )

    n_cones_to_keep_default = 10
    n_cones_to_keep = st.slider(
        "Number of cones to keep",
        min_value=1,
        max_value=20,
        value=n_cones_to_keep_default,
        step=1,
        help="In order to simulate different detection ranges, only the n closest cones are kept for any given cone type."
    )

    copy_cones_by_type = deepcopy(cones_by_type)

    left_keep_idxs = np.linalg.norm(cones_by_type[ConeTypes.LEFT] - position, axis=1)
    left_keep_idxs = left_keep_idxs.argsort()[:n_cones_to_keep]
    copy_cones_by_type[ConeTypes.LEFT] = copy_cones_by_type[ConeTypes.LEFT][
        left_keep_idxs
    ]

    right_keep_idxs = np.linalg.norm(cones_by_type[ConeTypes.RIGHT] - position, axis=1)
    right_keep_idxs = right_keep_idxs.argsort()[:n_cones_to_keep]
    copy_cones_by_type[ConeTypes.RIGHT] = copy_cones_by_type[ConeTypes.RIGHT][
        right_keep_idxs
    ]

    visualize_configuration(
        position,
        direction,
        copy_cones_by_type,
        with_cone_index=True,
        with_lines=False,
        do_show=True,
    )

    st.markdown(
        """
The first step is to fit circles to the cones. The algorithm uses the powerset of the cones to fit circles. The powerset is the set of all subsets of the cones. The algorithm iterates over all subsets of the cones and fits a circle to each subset. Circles that fit appropriate criteria that could plausibly represent one of the two circles of the skidpad track are kept. The criteria for keeping circles are:

- The error of the circle fit itself
- The radius of the estimated circle
- The mean distance between cones that comprise the circle

Below you can see the circles that were kept after the powerset circle fitting:
"""
    )


    show_powerset(copy_cones_by_type, position, direction)
