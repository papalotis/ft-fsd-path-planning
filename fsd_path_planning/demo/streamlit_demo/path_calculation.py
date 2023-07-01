from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from fsd_path_planning import MissionTypes
from fsd_path_planning.calculate_path.core_calculate_path import PathCalculationInput
from fsd_path_planning.cone_matching.core_cone_matching import ConeMatchingInput
from fsd_path_planning.config import (
    create_default_cone_matching,
    create_default_pathing,
)
from fsd_path_planning.demo.streamlit_demo.common import (
    CONE_TYPE_TO_COLOR,
    create_animation,
    get_cones_for_configuration,
    visualize_configuration,
)
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    euler_angles_to_quaternion,
)


def show_base_points(
    left_to_right_matches: IntArray,
    position: FloatArray,
    direction: FloatArray,
    cones_by_type: list[FloatArray],
) -> None:
    middle = []

    left_cones = cones_by_type[ConeTypes.LEFT]
    right_cones = cones_by_type[ConeTypes.RIGHT]

    left_base_points = []
    right_base_points = []

    for left_cone, right_match_index in zip(left_cones, left_to_right_matches):
        if right_match_index == -1:
            continue
        right_cone = right_cones[right_match_index]
        middle.append(((left_cone + right_cone) / 2).tolist())

        left_base_points.append(left_cone)
        right_base_points.append(right_cone)

    middle_array = np.array(middle)
    plt.subplots()
    visualize_configuration(
        position,
        direction,
        cones_by_type,
        with_cone_index=False,
        with_lines=False,
        do_show=False,
    )
    plt.plot(*middle_array.T, "o", color="green", label="Path base points")
    plt.legend()

    for left_base_point, right_base_point in zip(left_base_points, right_base_points):
        plt.plot(
            [left_base_point[0], right_base_point[0]],
            [left_base_point[1], right_base_point[1]],
            "-",
            color="k",
        )

    st.pyplot(plt.gcf())


def run() -> None:
    st.markdown(
        """
    # Path Calculation

    Path calculation is the last step of the path planning process. It receives the matched
    cones and is responsible for calculating the centerline of the track part.

    The algorithm has one parameter, the desired path length. This is because the control
    module needs a fixed length path, so for the path calculation we need to have a strategy
    for when the path that we can calculate using the cone information is too short.

    The calculation is split into three parts:

    - Base centerpoint calculation
    - Base path calculation
    - Fine path and metric calculation
    """
    )

    position, direction, cones_by_type = get_cones_for_configuration(
        st.session_state.track_configuration, do_shuffle=False
    )

    left_cones = st.session_state["left_with_virtual"]
    right_cones = st.session_state["right_with_virtual"]
    left_to_right_index = st.session_state["left_to_right_matches"]
    right_to_left_index = st.session_state["right_to_left_matches"]

    pathing = create_default_pathing(MissionTypes.trackdrive)
    desired_path_length = st.slider(
        "Desired path length",
        min_value=5.0,
        max_value=50.0,
        value=20.0,
        step=0.5,
    )

    number_of_elements = st.slider(
        "Number of samples",
        min_value=10,
        max_value=100,
        value=40,
        step=1,
    )

    pathing.scalars.mpc_path_length = desired_path_length
    pathing.scalars.mpc_prediction_horizon = number_of_elements

    pathing_input = PathCalculationInput(
        left_cones,
        right_cones,
        left_to_right_index,
        right_to_left_index,
        position,
        direction,
    )
    pathing.set_new_input(pathing_input)

    final_path, path_update = pathing.run_path_calculation()

    cone_by_type_w_virtual = [np.zeros((0, 2)) for _ in ConeTypes]
    cone_by_type_w_virtual[ConeTypes.LEFT] = left_cones
    cone_by_type_w_virtual[ConeTypes.RIGHT] = right_cones

    st.markdown(
        """
    ## Base centerpoint calculation

    The base centerpoint calculation begins the path planning process. It calculates the
    middle between the cones and their matches.
    """
    )
    plt.subplots()
    visualize_configuration(
        position,
        direction,
        cone_by_type_w_virtual,
        with_cone_index=False,
        with_lines=False,
        do_show=False,
    )

    show_base_points(left_to_right_index, position, direction, cone_by_type_w_virtual)

    st.markdown(
        """
    ## Base path calculation

    The next step is to apply a parametric spline fit on the base centerpoints. This is done
    to get a much finer path. That way we can calculate metrics of the path (e.g. curvature)
    with much finer resolution.
    """
    )
    plt.subplots()
    visualize_configuration(
        position,
        direction,
        cone_by_type_w_virtual,
        with_cone_index=False,
        with_lines=False,
        do_show=False,
    )

    plt.plot(*path_update.T, "-", color="k", label="Base path")
    plt.legend()
    st.pyplot(plt.gcf())

    st.markdown(
        """
    ## Fine path calculation

    The final step of the path calculation is to fine tune the path so that it can be
    used by the vehicle control module. For this 3 steps are performed:

    - The path is trimmed so that it is starts at the car pose
    - The path is extrapolated if it shorter than the desired path length or then trimmed 
    to that length if it is longer
    - The curvature of the path is calculated as a function of the arc length of the path

    The path is trimmed so that it starts at the car pose. This is done by finding the closest
    point on the path to the car pose and dropping all points before that.

    The path is extrapolated if it shorter than the desired path length. This is done by
    calculating the radius at the end of the path. Then a circle is drawn at the end of the
    path so that its length is the desired path length.

    The curvature of the path is calculated as a function of the arc length of the path. This
    is done by calculating the second derivative of the path. A uniform filter is applied to
    the second derivative to smooth it out.
    """
    )
    plt.subplots()
    visualize_configuration(
        position,
        direction,
        cone_by_type_w_virtual,
        with_cone_index=False,
        with_lines=False,
        do_show=False,
    )

    plt.plot(final_path[:, 1], final_path[:, 2], "-", color="k", label="Fine path")
    plt.legend()

    st.pyplot(plt.gcf())

    plt.figure()

    plt.plot(
        final_path[:, 0],
        final_path[:, 3],
        "-",
        color="k",
        label="Curvature",
    )
    plt.title("Path arclength vs curvature")
    plt.ylim([-0.3, 0.3])
    # plt.xticks(np.arange(0, path.path_arclength[-1], 0.5))
    plt.yticks(np.arange(-0.3, 0.3, 0.1))
    plt.grid()
    plt.legend()
    st.pyplot(plt.gcf())
