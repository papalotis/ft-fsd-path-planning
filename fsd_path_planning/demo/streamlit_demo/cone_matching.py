from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from fsd_path_planning.cone_matching.functional_cone_matching import (
    calculate_match_search_direction,
    calculate_matches_for_side,
    calculate_positions_of_virtual_cones,
    combine_and_sort_virtual_with_real,
    find_boolean_mask_of_all_potential_matches,
    select_best_match_candidate,
)
from fsd_path_planning.demo.streamlit_demo.common import (
    CONE_TYPE_TO_COLOR,
    create_animation,
    get_cones_for_configuration,
    visualize_configuration,
)
from fsd_path_planning.types import BoolArray, FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import normalize_last_axis, rotate


def naive_search_directions(
    left_cones: FloatArray, right_cones: FloatArray
) -> tuple[FloatArray, FloatArray]:
    if len(left_cones) > 0:
        left_rotated = rotate(
            normalize_last_axis(np.diff(left_cones, axis=0)), -np.pi / 2
        )
        left_rotated = np.row_stack((left_rotated, left_rotated[-1]))
    else:
        left_rotated = np.zeros((0, 2))

    if len(right_cones) > 0:
        right_rotated = rotate(
            normalize_last_axis(np.diff(right_cones, axis=0)), np.pi / 2
        )
        right_rotated = np.row_stack((right_rotated, right_rotated[-1]))
    else:
        right_rotated = np.zeros((0, 2))

    return left_rotated, right_rotated


def show_search_direction(
    left_cones: FloatArray,
    right_cones: FloatArray,
    use_naive_direction: bool,
) -> tuple[FloatArray, FloatArray]:
    if use_naive_direction:
        left_rotated, right_rotated = naive_search_directions(left_cones, right_cones)
    else:
        if len(left_cones) > 1:
            left_rotated = calculate_match_search_direction(left_cones, ConeTypes.LEFT)
        else:
            left_rotated = np.zeros((0, 2))
        if len(right_cones) > 1:
            right_rotated = calculate_match_search_direction(
                right_cones, ConeTypes.RIGHT
            )
        else:
            right_rotated = np.zeros((0, 2))

    left_search_direction_ends = left_cones + left_rotated
    right_search_direction_ends = right_cones + right_rotated

    cones_by_type = [np.zeros((0, 2)) for _ in ConeTypes]
    cones_by_type[ConeTypes.LEFT] = left_cones
    cones_by_type[ConeTypes.RIGHT] = right_cones

    plt.subplots()
    visualize_configuration(
        np.full(2, np.inf),
        np.array([1.0, 0]),
        cones_by_type,
        with_cone_index=False,
        with_lines=False,
        do_show=False,
    )

    for left_start, left_end in zip(left_cones, left_search_direction_ends):
        plt.plot([left_start[0], left_end[0]], [left_start[1], left_end[1]], "b-")

    for right_start, right_end in zip(right_cones, right_search_direction_ends):
        plt.plot([right_start[0], right_end[0]], [right_start[1], right_end[1]], "r-")

    title_clarification = "Naive" if use_naive_direction else "Adapted"
    title = f"Search Direction using {title_clarification} Search Direction"
    plt.title(title)

    fig = plt.gcf()
    st.pyplot(fig)  # type: ignore

    return left_rotated, right_rotated


def _do_show_potential_matches(
    left_cones: FloatArray,
    right_cones: FloatArray,
    left_mask: BoolArray,
    right_mask: BoolArray,
    show_left: bool,
    show_right: bool,
) -> None:
    if show_left:
        for left_cone, left_mask_cone in zip(left_cones, left_mask):
            right_potential_matches = right_cones[left_mask_cone]
            for right_cone in right_potential_matches:
                dx, dy = right_cone - left_cone

                plt.arrow(
                    left_cone[0],
                    left_cone[1],
                    dx,
                    dy,
                    length_includes_head=True,
                    head_width=0.5,
                    facecolor="cyan",
                )

    if show_right:
        for right_cone, right_mask_cone in zip(right_cones, right_mask):
            left_potential_matches = left_cones[right_mask_cone]
            for left_cone in left_potential_matches:
                dx, dy = left_cone - right_cone

                plt.arrow(
                    right_cone[0],
                    right_cone[1],
                    dx,
                    dy,
                    length_includes_head=True,
                    head_width=0.5,
                    facecolor="gold",
                )


def show_potential_matches(
    left_cones: FloatArray,
    right_cones: FloatArray,
    left_directions: FloatArray,
    right_directions: FloatArray,
    major_radius: float,
    minor_radius: float,
    max_search_angle: float,
    focus_cone: Optional[tuple[ConeTypes, int]],
    side_to_show: str,
) -> tuple[FloatArray, FloatArray]:
    left_mask = find_boolean_mask_of_all_potential_matches(
        left_cones,
        left_directions,
        right_cones,
        right_directions,
        major_radius,
        minor_radius,
        max_search_angle,
    )

    right_mask = find_boolean_mask_of_all_potential_matches(
        right_cones,
        right_directions,
        left_cones,
        left_directions,
        major_radius,
        minor_radius,
        max_search_angle,
    )

    cones_by_type = [np.zeros((0, 2)) for _ in ConeTypes]
    cones_by_type[ConeTypes.LEFT] = left_cones
    cones_by_type[ConeTypes.RIGHT] = right_cones

    plt.subplots()
    visualize_configuration(
        np.full(2, np.inf),
        np.array([1.0, 0.0]),
        cones_by_type,
        with_cone_index=False,
        with_lines=False,
        do_show=False,
    )

    cone_with_focus_bool = focus_cone is not None

    show_left = side_to_show in ["left", "both"]
    show_right = side_to_show in ["right", "both"]
    _do_show_potential_matches(
        left_cones, right_cones, left_mask, right_mask, show_left, show_right
    )

    title_clarification = (
        "for both sides"
        if show_left and show_right
        else "for left side"
        if show_left
        else "for right side"
    )
    title = f"Potential Matches {title_clarification}"
    plt.title(title)

    fig = plt.gcf()
    st.pyplot(fig)  # type: ignore

    return left_mask, right_mask


def show_best_match_candidate(
    left_cones: FloatArray,
    right_cones: FloatArray,
    left_directions: FloatArray,
    right_directions: FloatArray,
    left_potential_matches_mask: BoolArray,
    right_potential_matches_mask: BoolArray,
) -> tuple[FloatArray, FloatArray]:
    matches_from_left_to_right = select_best_match_candidate(
        left_cones,
        left_directions,
        left_potential_matches_mask,
        right_cones,
        matches_should_be_monotonic=False,
    )

    new_mask_left = np.zeros_like(left_potential_matches_mask)
    if 0 not in new_mask_left.shape:
        new_mask_left[np.arange(len(left_cones)), matches_from_left_to_right] = 1
        new_mask_left *= left_potential_matches_mask

    matches_from_right_to_left = select_best_match_candidate(
        right_cones,
        right_directions,
        right_potential_matches_mask,
        left_cones,
        matches_should_be_monotonic=False,
    )

    new_mask_right = np.zeros_like(right_potential_matches_mask)

    if 0 not in new_mask_right.shape:
        new_mask_right[np.arange(len(right_cones)), matches_from_right_to_left] = 1
        new_mask_right *= right_potential_matches_mask

    cones_by_type = [np.zeros((0, 2)) for _ in ConeTypes]
    cones_by_type[ConeTypes.LEFT] = left_cones
    cones_by_type[ConeTypes.RIGHT] = right_cones

    plt.subplots()
    visualize_configuration(
        np.full(2, np.inf),
        np.array([1.0, 0.0]),
        cones_by_type,
        with_cone_index=False,
        with_lines=False,
        do_show=False,
    )

    _do_show_potential_matches(
        left_cones, right_cones, new_mask_left, new_mask_right, True, True
    )

    title = "Best Match Candidate"
    plt.title(title)

    fig = plt.gcf()
    st.pyplot(fig)  # type: ignore

    return matches_from_left_to_right, matches_from_right_to_left


def show_virtual_cones(
    left_cones: FloatArray,
    right_cones: FloatArray,
    left_directions: FloatArray,
    right_directions: FloatArray,
    left_to_right_matches: IntArray,
    right_to_left_matches: IntArray,
    min_track_width: float,
) -> tuple[FloatArray, FloatArray]:
    left_idx_no_match = np.where(left_to_right_matches == -1)[0]
    # the variable is correct, we use the left cones to calculate the virtual cones
    # of the right side
    right_virtual_cones = calculate_positions_of_virtual_cones(
        left_cones, left_idx_no_match, left_directions, min_track_width
    )

    right_idx_no_match = np.where(right_to_left_matches == -1)[0]
    left_virtual_cones = calculate_positions_of_virtual_cones(
        right_cones, right_idx_no_match, right_directions, min_track_width
    )

    cones_by_type = [np.zeros((0, 2)) for _ in ConeTypes]
    cones_by_type[ConeTypes.LEFT] = left_cones
    cones_by_type[ConeTypes.RIGHT] = right_cones

    plt.subplots()
    visualize_configuration(
        np.full(2, np.inf),
        np.array([1.0, 0]),
        cones_by_type,
        with_cone_index=False,
        with_lines=False,
        do_show=False,
    )

    for left_cone, left_direction in zip(
        left_cones[left_idx_no_match], left_directions[left_idx_no_match]
    ):
        plt.plot(
            [left_cone[0], left_cone[0] + left_direction[0]],
            [left_cone[1], left_cone[1] + left_direction[1]],
            "-",
        )

    for right_cone, right_direction in zip(
        right_cones[right_idx_no_match], right_directions[right_idx_no_match]
    ):
        plt.plot(
            [right_cone[0], right_cone[0] + right_direction[0]],
            [right_cone[1], right_cone[1] + right_direction[1]],
            "k-",
        )

    plt.plot(
        *left_virtual_cones.T,
        "x",
        color=CONE_TYPE_TO_COLOR[ConeTypes.LEFT],
        label="LEFT VIRTUAL",
    )

    plt.plot(
        *right_virtual_cones.T,
        "x",
        color=CONE_TYPE_TO_COLOR[ConeTypes.RIGHT],
        label="RIGHT VIRTUAL",
    )

    title = "Virtual Cones"
    plt.title(title)
    plt.legend()

    fig = plt.gcf()
    st.pyplot(fig)  # type: ignore

    return left_virtual_cones, right_virtual_cones


def show_merging(
    left_cones: FloatArray,
    right_cones: FloatArray,
    left_cones_virtual: FloatArray,
    right_cones_virtual: FloatArray,
    position: FloatArray,
    direction: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    combined_left, left_is_virtual, left_history = combine_and_sort_virtual_with_real(
        left_cones,
        left_cones_virtual,
        ConeTypes.LEFT,
        position,
        direction,
    )

    (
        combined_right,
        right_is_virtual,
        right_history,
    ) = combine_and_sort_virtual_with_real(
        right_cones,
        right_cones_virtual,
        ConeTypes.RIGHT,
        position,
        direction,
    )

    cones_by_type = [np.zeros((0, 2)) for _ in ConeTypes]
    cones_by_type[ConeTypes.LEFT] = combined_left
    cones_by_type[ConeTypes.RIGHT] = combined_right

    plt.subplots()
    visualize_configuration(
        position,
        direction,
        cones_by_type,
        with_cone_index=True,
        with_lines=False,
        do_show=False,
    )

    title = "Merged Cones"
    plt.title(title)

    fig = plt.gcf()
    st.pyplot(fig)  # type: ignore

    return combined_left, combined_right


def show_final_matching(
    left_cones_with_virtual: FloatArray,
    right_cones_with_virtual: FloatArray,
    major_radius: float,
    minor_radius: float,
    max_search_angle: float,
) -> None:
    _, left_to_right_match, _ = calculate_matches_for_side(
        left_cones_with_virtual,
        ConeTypes.LEFT,
        right_cones_with_virtual,
        major_radius,
        minor_radius,
        max_search_angle,
        matches_should_be_monotonic=False,
    )
    left_is_match_mask = np.zeros(
        (len(left_cones_with_virtual), len(right_cones_with_virtual)), dtype=np.bool_
    )

    left_has_match = left_to_right_match != -1
    idx_source = np.arange(len(left_cones_with_virtual))[left_has_match]
    idx_target = left_to_right_match[left_has_match]
    left_is_match_mask[idx_source, idx_target] = True

    _, right_to_left_match, _ = calculate_matches_for_side(
        right_cones_with_virtual,
        ConeTypes.RIGHT,
        left_cones_with_virtual,
        major_radius,
        minor_radius,
        max_search_angle,
        matches_should_be_monotonic=False,
    )

    right_is_match_mask = np.zeros(
        (len(right_cones_with_virtual), len(left_cones_with_virtual)), dtype=np.bool_
    )
    right_has_match = right_to_left_match != -1
    idx_source = np.arange(len(right_cones_with_virtual))[right_has_match]
    idx_target = right_to_left_match[right_has_match]
    right_is_match_mask[idx_source, idx_target] = True

    _do_show_potential_matches(
        left_cones_with_virtual,
        right_cones_with_virtual,
        left_is_match_mask,
        right_is_match_mask,
        True,
        True,
    )

    plt.title("Final Matches")

    fig = plt.gcf()
    st.pyplot(fig)  # type: ignore

    return left_to_right_match, right_to_left_match


def run() -> None:
    st.markdown(
        """
    # Cone Matching

    Cone matching is the algorithm that combines the sorted traces of the two track sides.
    It is the next step of the path calculation pipeline after cone sorting.

    The core of the cone matching algorithm is the following:
    - Find the search direction for each cone (normal of the cone trace)
    - Find the potential matches for each cone
    - Find best match for each cone
    - For any cone that does not have a match, create a virtual cone, based on the search direction
    - Merge the "virtual" with the "real" cones

    ## Inputs

    The inputs for the cone matching algorithm are the sorted traces of the two track sides
    as well as the vehicle pose.
    """
    )

    position, direction, cones_by_type = get_cones_for_configuration(
        st.session_state.track_configuration, do_shuffle=False
    )

    cones_by_type[ConeTypes.LEFT] = st.session_state.left_sorted_cones
    cones_by_type[ConeTypes.RIGHT] = st.session_state.right_sorted_cones

    plt.subplots()
    visualize_configuration(
        position,
        direction,
        cones_by_type,
        with_cone_index=True,
        with_lines=False,
        do_show=True,
    )

    st.markdown(
        """
        ## Search Direction

        The search direction is the direction in which the cone is looking for a match. It is
        effectively the normal of the cone trace at the cone's position.

        A naive solution, might be to get the vector from each cone to its next cone and 
        rotate it by 90 degrees. While this is a simple solution, it sometimes results in
        in a search direction which misses the true direction of the potential match by some
        degrees.

        Therefore, we use an adapted version of the above method. Instead of taking the
        vector from each cone to its next, we take the vector from the previous cone to the
        next cone. This way we can get a better search direction.
        """
    )
    use_naive_direction = st.checkbox(
        "Use naive search direction algorithm", value=False
    )
    left_search_directions, right_search_directions = show_search_direction(
        cones_by_type[ConeTypes.LEFT],
        cones_by_type[ConeTypes.RIGHT],
        use_naive_direction,
    )

    st.markdown(
        """
    ## Potential matches

    For each cone we want to find its potential matches. We do this by considering all
    cones that are in a specific range (inside an ellipse with a specific radius) and angle
    from the search direction. Furthermore, if the search directions of two cones that
    can be matched point in the same direction, then this match is discarded.
    """
    )
    major_radius = st.slider("Major radius", 5.0, 10.0, 8.0, 0.2)
    minor_radius = st.slider("Minor radius", 3.0, 10.0, 4.0, 0.2)

    if major_radius < minor_radius:
        st.error("Major radius must be larger than minor radius")
        return

    max_search_angle_deg = st.slider("Max search angle", 20, 80, 50, step=1)
    max_search_angle = np.deg2rad(max_search_angle_deg)

    side_to_show = st.radio(
        "Side to show", ["Both", "Left", "Right"], horizontal=True
    ).lower()

    left_mask, right_mask = show_potential_matches(
        cones_by_type[ConeTypes.LEFT],
        cones_by_type[ConeTypes.RIGHT],
        left_search_directions,
        right_search_directions,
        major_radius,
        minor_radius,
        max_search_angle,
        None,
        side_to_show,
    )

    st.markdown(
        """
    ## Best matches

    After finding the potential matches, we want to find the best match for each cone.
    We do this by picking the candidate with the smallest distance to the cone.

    While it would make other tasks easier, we cannot rely on a 1-1 matching between
    left and right cones. Therefore, we allow many cones to match with the same cone.
    """
    )

    left_to_right_matches, right_to_left_matches = show_best_match_candidate(
        cones_by_type[ConeTypes.LEFT],
        cones_by_type[ConeTypes.RIGHT],
        left_search_directions,
        right_search_directions,
        left_mask,
        right_mask,
    )

    st.markdown(
        """
    ## Virtual cones

    At this point we have found a match for each cone that could be matched. However,
    we do not necessarily have a match for all cones. One solution would be to accept 
    this fact and only use the cones that have a match. This is not a good solution.
    For example, we would not be able to drive if only one side of the track were available.

    In order to solve this problem, we need to create virtual cones. By using the search
    direction, we can place virtual cones where we would expect to have a match.
    The only remaining issue that remains is deciding at what distance the virtual cone
    should be placed. Here we take advantage of rule D 8.1.1

    > **The minimum track width is 3 m**

    This way we can place the virtual cone at a distance of 3 m from the cone, and know
    that there is no way that we will overestimate the actual track width at that point
    of the track


        """
    )

    n_without_match = (left_to_right_matches == -1).sum() + (
        right_to_left_matches == -1
    ).sum()

    if n_without_match == 0:
        st.info(
            "In this instance, all cones have a match, so no virtual cones will be"
            " computed"
        )

    minimum_track_width = st.slider("Minimum track width", 2.5, 6.0, 3.0, step=0.1)

    left_virtual_cones, right_virtual_cones = show_virtual_cones(
        cones_by_type[ConeTypes.LEFT],
        cones_by_type[ConeTypes.RIGHT],
        left_search_directions,
        right_search_directions,
        left_to_right_matches,
        right_to_left_matches,
        minimum_track_width,
    )

    st.markdown(
        """
    ## Merging virtual cones with real cones

    After creating the virtual cones, we need to merge them with the real cones. Naively,
    one could combine the real and virtual cones into one list, and apply the sorting
    algorithm again. While this can work, it is computationally expensive. Futhermore,
    it does not make use of one aspect regarding our virtual and real cones. They are *already*
    sorted. This is the same situation as the one in the merge step of the mergesort algorithm.
    In our case, we would like to merge two sorted cone list into one. We need an algorithm
    that will be able to merge two sorted cone lists into one.

    We use an iterative algorithm to merge the cones. We start with one base cone list,
    and iteratively merge cones from the other cone list into the base cone list.

    The algorithm is as follows:
    - Start with the base cone list
    - For each cone in the other cone list
        - Find the two closest cones in the base cone list
        - If the two cones are not consecutive, skip this candidate
        - If the angle formed by putting the cone between the two closest cones, is large enough,
            add it to the base cone list, between the two closest cones
        - If not add it before both of them or after both of them, depending on the
            configuration.

        """
    )

    left_with_virtual, right_with_virtual = show_merging(
        cones_by_type[ConeTypes.LEFT],
        cones_by_type[ConeTypes.RIGHT],
        left_virtual_cones,
        right_virtual_cones,
        position,
        direction,
    )

    st.markdown(
        """
    ## Final matching

    Now that we have added the virtual cones, we can apply our matching algorithm
    (compute search directions, find potential match candidates, find final match), once
    again, on the combined left and right cones. Since we have now added the virtual cones,
    we expect that almost all cones will have a match.
    """
    )

    left_to_right_matches, right_to_left_matches = show_final_matching(
        left_with_virtual,
        right_with_virtual,
        major_radius,
        minor_radius,
        max_search_angle,
    )

    st.session_state["left_with_virtual"] = left_with_virtual
    st.session_state["left_to_right_matches"] = left_to_right_matches
    st.session_state["right_with_virtual"] = right_with_virtual
    st.session_state["right_to_left_matches"] = right_to_left_matches
    st.session_state["match_track"] = st.session_state.track_configuration
