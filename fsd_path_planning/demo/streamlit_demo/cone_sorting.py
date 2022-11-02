from itertools import zip_longest
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import angle_from_2d_vector, rotate
from fsd_path_planning.utils.utils import Timer
from fsd_path_planning.demo.streamlit_demo.common import (
    CONE_TYPE_TO_COLOR,
    create_animation,
    get_cones_for_configuration,
    visualize_configuration,
)
from matplotlib.axes import Axes
from matplotlib.patches import Polygon

from fsd_path_planning.sorting_cones.trace_sorter.adjacency_matrix import (
    create_adjacency_matrix,
)
from fsd_path_planning.sorting_cones.trace_sorter.core_trace_sorter import TraceSorter
from fsd_path_planning.sorting_cones.trace_sorter.cost_function import (
    cost_configurations,
)
from fsd_path_planning.sorting_cones.trace_sorter.end_configurations import (
    find_all_end_configurations,
)
from fsd_path_planning.types import BoolArray, FloatArray, IntArray


def get_points_on_ellipse(thetas: FloatArray, a: float, b: float) -> np.ndarray:
    x = a * np.cos(thetas)
    y = b * np.sin(thetas)
    return np.column_stack([x, y])


def show_starting_cone(
    position: FloatArray,
    direction: FloatArray,
    cones_by_type: FloatArray,
    max_distance: float,
) -> list[Optional[int]]:
    fig, ax = visualize_configuration(
        position,
        direction,
        cones_by_type,
        with_cone_index=False,
        with_lines=False,
        do_show=False,
    )

    fov_points = get_points_on_ellipse(
        np.arange(0, 2 * np.pi, 0.01),
        max_distance * 1.3,
        max_distance / 1.3,
    )

    angle_direction = float(angle_from_2d_vector(direction))
    final_fov_points = rotate(fov_points, angle_direction) + position

    poly = Polygon(final_fov_points, closed=True, fill=True, facecolor="red", alpha=0.3)

    sorter = TraceSorter(
        3,
        3,
        max_dist_to_first=max_distance,
        max_length=10,
        threshold_absolute_angle=0,
        threshold_directional_angle=0.5,
    )
    out: list[Optional[int]] = [None for _ in ConeTypes]
    for cone_type in (ConeTypes.LEFT, ConeTypes.RIGHT):
        cones = cones_by_type[cone_type]
        car_to_cone = cones - position
        angle_to_car = angle_from_2d_vector(rotate(car_to_cone, -angle_direction))

        distance_to_car = np.linalg.norm(car_to_cone, axis=1)
        idx = sorter.select_starting_cone(
            position, direction, cones, angle_to_car, distance_to_car
        )
        if idx is None:
            st.warning(f"No starting cone found for {cone_type}")
        else:
            ax.plot(cones[idx, 0], cones[idx, 1], "x", color="black")

        out[cone_type] = idx

    ax.add_patch(poly)

    st.pyplot(fig)  # type: ignore

    return out


def plot_adjacency_matrix(
    adjacency_matrix: BoolArray, cones: FloatArray, ax: Axes
) -> None:

    ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.plot(*cones.T, ".k")
    for cone, adj_mask in zip(cones, adjacency_matrix):
        for is_neighbor, neighbor in zip(adj_mask, cones):
            if is_neighbor:
                ax.plot(*np.array([cone, neighbor]).T, "-k", alpha=0.2)


def show_adjacency_matrix(
    cones_by_type: list[FloatArray],
    start_indices: list[Optional[int]],
    n_neighbors: int,
    max_distance: float,
) -> list[Optional[BoolArray]]:
    show_two_plots = st.checkbox("Show each side in a separate plot")

    adjacency_matrices: list[Optional[BoolArray]] = [None for _ in ConeTypes]

    fig, ax = plt.subplots(1, 2 if show_two_plots else 1)
    for i, cone_type in enumerate((ConeTypes.LEFT, ConeTypes.RIGHT)):
        cones = cones_by_type[cone_type]
        start_idx = start_indices[cone_type]
        adjacency_matrix, reachable_nodes = create_adjacency_matrix(
            cones, n_neighbors, start_idx, max_distance
        )
        adjacency_matrices[cone_type] = adjacency_matrix

        ax_to_use = ax[i] if show_two_plots else ax
        plot_adjacency_matrix(adjacency_matrix, cones, ax_to_use)

    st.pyplot(fig)  # type: ignore

    return adjacency_matrices


def show_graph_search(
    cones_by_type: list[FloatArray],
    adjacency_matrices: list[Optional[BoolArray]],
    start_indices: list[Optional[int]],
    target_length: int,
    threshold_directional_angle: float,
    threshold_absolute_angle: float,
) -> list[Optional[IntArray]]:

    cols = st.columns(2)
    all_end_configs: list[Optional[IntArray]] = [None for _ in ConeTypes]
    for cone_type, col in zip((ConeTypes.LEFT, ConeTypes.RIGHT), cols):

        cones = cones_by_type[cone_type]

        adjacency_matrix = adjacency_matrices[cone_type]
        start_idx = start_indices[cone_type]
        with Timer() as timer:
            end_configurations, (
                all_configurations,
                configuration_is_end,
            ) = find_all_end_configurations(
                cones,
                cone_type,
                start_idx,
                adjacency_matrix,
                target_length,
                threshold_directional_angle,
                threshold_absolute_angle,
                np.zeros(0, dtype=np.int32),
                store_all_end_configurations=True,
            )

        all_end_configs[cone_type] = end_configurations

        frames = []
        for config, is_end_configuration in zip(
            all_configurations, configuration_is_end
        ):

            config = config[config != -1]
            points = cones[config]
            scatter_lines = go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="lines+text",
                marker_color="black" if is_end_configuration else "red",
                line_dash="solid" if is_end_configuration else "dash",
                text=list(range(1, len(points) + 1)),
            )
            scatter_cones = go.Scatter(
                x=cones[:, 0],
                y=cones[:, 1],
                mode="markers",
            )
            frame = go.Frame(
                data=[
                    scatter_cones,
                    scatter_lines,
                ]
            )
            frames.append(frame)

        xm = np.min(cones[:, 0]) - 1
        xM = np.max(cones[:, 0]) + 1
        ym = np.min(cones[:, 1]) - 1
        yM = np.max(cones[:, 1]) + 1

        fig = create_animation(frames)
        fig.update_layout(
            title_text=f"{cone_type.name} configurations",
            plot_bgcolor="white",
            xaxis=dict(range=[xm, xM], autorange=False, showticklabels=False),
            yaxis=dict(
                range=[ym, yM],
                autorange=False,
                scaleanchor="x",
                scaleratio=1,
                showticklabels=False,
            ),
            showlegend=False,
        )

        fig.update_traces(textposition="top left")

        with col:
            st.plotly_chart(
                fig,
                config={"displayModeBar": False},
                use_container_width=True,
            )

    return all_end_configs


def show_costs(
    cones_by_type: list[Optional[FloatArray]],
    end_configurations_by_type: list[Optional[IntArray]],
    direction: FloatArray,
) -> list[FloatArray]:
    final_out = [np.zeros((0, 2)) for _ in ConeTypes]
    for cone_type in (ConeTypes.LEFT, ConeTypes.RIGHT):

        cones = cones_by_type[cone_type]
        end_configurations = end_configurations_by_type[cone_type]
        assert cones is not None
        assert end_configurations is not None

        costs = cost_configurations(
            cones,
            end_configurations,
            cone_type,
            direction,
            return_individual_costs=True,
        )

        n_configurations = len(end_configurations)
        n_cols = min(n_configurations, 4)
        n_rows = int(np.ceil(n_configurations / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        names = [
            "Mean angle",
            "Intersection",
            "Distance",
            "#Cones",
            "Direction",
            "Total",
        ]

        idx_sort_cost = costs.sum(axis=1).argsort()

        border = 1.5
        mx, Mx = np.min(cones[:, 0]) - border, np.max(cones[:, 0]) + border
        my, My = np.min(cones[:, 1]) - border, np.max(cones[:, 1]) + border

        sm, sM = max([(mx, Mx), (my, My)], key=lambda x: x[1] - x[0])

        best_config = end_configurations[idx_sort_cost[0]]
        final_out[cone_type] = cones[best_config[best_config != -1]]

        for config_costs, configuration, ax in zip_longest(
            costs[idx_sort_cost], end_configurations[idx_sort_cost], axes.flatten()
        ):
            if config_costs is not None and configuration is not None:
                configuration = configuration[configuration != -1]
                ax.plot(
                    *cones[configuration].T, "o-", color=CONE_TYPE_TO_COLOR[cone_type]
                )

                all_costs_config = [*config_costs, config_costs.sum()]
                ax.set_title(
                    "\n".join(
                        f"{name}: {value:.3f}"
                        for name, value in zip(names, all_costs_config)
                        if "Distance" != name
                    )
                )
                text_offset = 0.5
                for i, (x, y) in enumerate(cones[configuration], start=1):
                    ax.text(x - text_offset, y - text_offset, str(i))

                ax.set_xlim(sm, sM)
                ax.set_ylim(sm, sM)
                ax.set_aspect("equal")
            else:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()

        with st.expander(f"{cone_type.name} costs"):
            st.pyplot(fig)  # type: ignore

    return final_out


def run() -> None:
    st.markdown(
        """
# Cone Sorting

Cone sorting is the process of sorting a set of cones into a sequence of cones. The goal
of this process is to put the SLAM cones in the order of the track direction.

Left and right sides are sorted independently from each other.

The sorting algorithm runs on only a subset of all available cones. The only cones within
a specific range from the vehicle pose are considered. This is done to reduce computation
time.

Like the rest of the algorithms, the sorting algorithm has been designed to be completely
stateless.

The core cone sorting algorithm consists of the following steps:

- Pick a starting cone
- Construct graph of consecutive cones
- Apply graph search to find all possible configurations
- Calculate the cost of each configuration
- Select the configuration with the lowest cost
"""
    )

    st.markdown(
        """
## Inputs

In the bellow graph you can see the input for our algorithm. The inputs are:
- The vehicle pose
- The cones close to the vehicle

Our goal is that after the algorithm is finished, the cones are sorted in the order of the
track direction. We will be able to verify based on the index of each cone in the final
result.
"""
    )
    position, direction, cones_by_type = get_cones_for_configuration(
        st.session_state.track_configuration, do_shuffle=True
    )

    show_lines = st.checkbox("Show lines", help="Show lines between consecutive cones")
    visualize_configuration(
        position,
        direction,
        cones_by_type,
        with_cone_index=True,
        with_lines=show_lines,
        do_show=True,
    )

    st.markdown(
        """
    ## Picking a starting cone

    Our goal is to apply a graph search to find all possible configurations. For this we need
    to know the starting cone. The starting cone is picked as the one closest to the vehicle
    that is in front of the vehicle.

    ### Parameters

    - **Maximum distance**: The maximum distance for a cone to be considered a starting cone.
    We do not use the maximum distance directly, but we use it as the basis for the radii
    of an ellipse where the major axis is $1.3$ times the maximum distance and the minor axis is
    $1.3^{-1}$ times the maximum distance.
    """
    )

    maximum_distance = st.slider(
        "Maximum distance",
        4.0,
        10.0,
        help="Maximum distance for a cone to be considered a starting cone",
    )

    start_indices = show_starting_cone(
        position, direction, cones_by_type, maximum_distance
    )

    st.markdown(
        """
## Graph construction

Now that we know what our starting cone is, we can construct the graph. The main assumption
that we make when constructing the graph that we make is that cones that are far away from 
each other should not be connected. Intuitively, we can say that if two cones are, e.g. 20m apart,
then the likelihood of them being connected is very low.

### Parameters
- **Maximum number of neighbors**: The maximum number of neighbors that a cone can have.
- **Maximum distance**: The maximum distance for a cone to be considered a neighbor.
"""
    )
    col_neighbors, col_distance = st.columns(2)
    with col_neighbors:
        n_neighbors = st.slider(
            "Max number of neighbors", 2, 5, 3, step=1, help="Max number of neighbors"
        )

    with col_distance:
        maximum_distance = st.slider(
            "Max distance between neighbors",
            2.0,
            10.0,
            5.0,
            step=0.2,
            help="Max distance between neighbors",
        )
    if n_neighbors > 3:
        st.warning(
            "The number of neighbors can drastically increase the computation time. While it is possible to use a higher number of neighbors, it is not recommended. A value of 3 should be sufficient. If you want to experiment with a larger number keep the exponential nature of the algorithm in mind. If you want to avoid the exponential nature, set the max depth to a low number like 4."
        )
    adjacency_matrices = show_adjacency_matrix(
        cones_by_type, start_indices, n_neighbors, maximum_distance
    )
    st.markdown(
        """
## Graph search

After constructing the graph, we can now apply a graph search to find all possible configurations.
We apply depth first search to find all possible configurations, however the type of search
is not important.

### Parameters
- **Maximum depth**: The maximum depth of the graph search. The depth is the number of cones that can be visited before the search is stopped.
- **Directional angle threshold**: When sorting the cones, rotating to one side is more complicated. When sorting yellow cones (which are on the right side) rotating clockwise is more risky, since we might connect cones that should not be connected. To address this we add this parameter which limits the angle maximum angle in the direction (clockwise for yellow cones, counterclockwise for blue cones).
"""
    )

    max_maximum_depth = 15
    if n_neighbors >= 4:
        max_maximum_depth = 5
        st.warning(
            "The upper limit for maximum depth is set to 5 when maximum number of neighbors is over 3"
        )

    maximum_depth = st.slider(
        "Maximum depth",
        3,
        max_maximum_depth,
        None,
        step=1,
        help="Maximum depth of the graph search",
    )

    threshold_directional_angle = np.deg2rad(
        st.slider("Threshold directional angle", 20, 90, 40, step=1)
    )
    threshold_absolute_angle = np.deg2rad(
        st.slider("Threshold absolute angle", 20, 90, 70, step=1)
    )

    end_configurations_by_type = show_graph_search(
        cones_by_type,
        adjacency_matrices,
        start_indices,
        maximum_depth,
        threshold_directional_angle,
        threshold_absolute_angle,
    )

    st.markdown(
        """
## Cost calculation

Now that we have all the candidates, we need to calculate the cost of each configuration.
The cost configuration is consists of the following:
- Mean angle between consecutive cones
- Number of edge intersections ($2^{\#=intersections}-1$)
- Number of nodes (cones) in the configuration (more nodes are preferred) ($1/{\#nodes}$)

The final cost function is a weighted sum of the above cost functions.
"""
    )
    sorted_cones_by_type = show_costs(
        cones_by_type, end_configurations_by_type, direction
    )

    st.markdown(
        """
## Result

Finally we can visualize the final output of the algorithm.
"""
    )

    visualize_configuration(
        position,
        direction,
        sorted_cones_by_type,
        with_cone_index=True,
        with_lines=True,
        do_show=True,
    )
