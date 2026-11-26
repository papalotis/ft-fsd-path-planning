"""
The cost the sorter uses to rank candidate traces: one function per term, and
COST_TERMS, which lists them with the weight that applies to each.

Project: fsd_path_planning
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fsd_path_planning.common_types import (
    BoolArray,
    FloatArray,
    IntArray,
    SortableConeTypes,
)
from fsd_path_planning.config_dataclasses import SortingConfig
from fsd_path_planning.geometry.angles import angle_difference, vec_angle_between
from fsd_path_planning.sorting_cones.trace_sorter.common import get_configurations_diff
from fsd_path_planning.sorting_cones.trace_sorter.cone_distance_cost import (
    calc_distance_cost,
)
from fsd_path_planning.sorting_cones.trace_sorter.nearby_cone_search import (
    NearbyConeSearcher,
)
from fsd_path_planning.utils.cone_types import ConeTypes


def calc_angle_to_next(points: FloatArray, configurations: IntArray) -> FloatArray:
    """
    Calculate the angle from one cone to the previous and the next one for all
    the provided configurations
    """
    all_to_next = get_configurations_diff(points, configurations)

    mask_should_overwrite = (configurations == -1)[:, 1:]
    all_to_next[mask_should_overwrite] = 100

    # Numba compiles one version per array layout. Whether this slice is
    # contiguous depends on the data (with a single row it is), so without the
    # copy a frame could hit a layout the warm-up never compiled and stall.
    from_middle_to_next = np.ascontiguousarray(all_to_next[..., 1:, :])
    from_prev_to_middle = all_to_next[..., :-1, :]
    from_middle_to_prev = -from_prev_to_middle

    angles = vec_angle_between(from_middle_to_next, from_middle_to_prev)
    return angles


def calc_angle_cost_for_configuration(
    points: FloatArray,
    configurations: IntArray,
    cone_type: SortableConeTypes,  # pylint: disable=unused-argument
) -> FloatArray:
    """
    Calculate the angle cost of cone configurations given a set of points and many index
    lists defining the configurations
    Args:
        points: The points to be used
        configurations: An array of indices defining the configurations
        cone_type: The type of cones. It is currently unused
    Returns:
        np.array: The score of each configuration
    """

    angles = calc_angle_to_next(points, configurations)

    is_part_of_configuration = (configurations != -1)[:, 2:]

    # invert angles and normalize between [0-1]
    angles_as_cost = (np.pi - angles) / np.pi

    angles_as_cost_filtered = angles_as_cost * is_part_of_configuration

    angles_are_under_threshold = np.logical_and(
        angles < np.deg2rad(40), is_part_of_configuration
    )

    # we will multiply the score by the number of angles that are under the threshold
    cost_factors = angles_are_under_threshold.sum(axis=-1) + 1

    # get sum of costs
    costs: FloatArray = angles_as_cost_filtered.sum(
        axis=-1
    ) / is_part_of_configuration.sum(axis=-1)

    costs = costs * cost_factors
    return costs


def calc_number_of_cones_cost(configurations: IntArray) -> FloatArray:
    """
    Calculates the number of cones in each configuration
    Args:
        configurations: An array of indices defining a configuration of the
        provided points
    Returns:
        A cost for each configuration
    """
    mask: BoolArray = configurations != -1
    number_of_cones: IntArray = mask.sum(axis=-1)

    # we prefer longer configurations
    cost = 1 / number_of_cones
    return cost


def calc_initial_direction_cost(
    points: FloatArray, configurations: IntArray, vehicle_direction: FloatArray
) -> FloatArray:
    points_configs_first_two = np.diff(points[configurations][:, :2], axis=1)[:, 0]

    return vec_angle_between(points_configs_first_two, vehicle_direction)


def _heading_changes(trace: FloatArray) -> FloatArray:
    """The change of heading at each inner point of a trace, in (-pi, pi]."""
    steps = np.diff(trace, axis=0)
    headings = np.arctan2(steps[:, 1], steps[:, 0])
    return angle_difference(headings[:-1], headings[1:])


def calc_change_of_direction_cost(
    points: FloatArray, configurations: IntArray
) -> FloatArray:
    """
    Calculates the change of direction cost in each configuration. This is done for each
    configuration using the following steps:
    1. Calculate the empiric first derivative of the configuration
    2. Calculate the angle of the first derivative
    3. Calculate the zero crossings of the angle along the configuration
    4. Calculate the sum of the change in the angle between the zero crossings

    Args:
        points: The underlying points
        configurations: An array of indices defining a configuration of the
        provided points
    Returns:
        A cost for each configuration
    """
    out = np.zeros(configurations.shape[0])
    for i, c in enumerate(configurations):
        c = c[c != -1]
        if len(c) == 3:
            continue

        difference = _heading_changes(points[c])

        mask_zero_crossing = np.sign(difference[:-1]) != np.sign(difference[1:])
        raw_cost_values = np.abs(difference[:-1] - difference[1:])

        cost_values = raw_cost_values * mask_zero_crossing
        out[i] = np.sum(cost_values)

    return out


def calc_wrong_direction_cost(
    points: FloatArray, configurations: IntArray, cone_type: ConeTypes
) -> FloatArray:
    """

    Args:
        points: The underlying points
        configurations: An array of indices defining a configuration of the
        provided points
    Returns:
        A cost for each configuration
    """
    out = np.zeros(configurations.shape[0])

    unwanted_direction_sign = 1 if cone_type == ConeTypes.LEFT else -1

    for i, c in enumerate(configurations):
        c = c[c != -1]
        if len(c) == 3:
            continue

        difference = _heading_changes(points[c])

        mask_wrong_direction = np.sign(difference) == unwanted_direction_sign
        mask_threshold = np.abs(difference) > np.deg2rad(40)

        mask = mask_wrong_direction & mask_threshold

        cost_values = np.abs(difference[mask].sum())

        out[i] = cost_values

    return out


def calc_cones_on_either_cost(
    points: FloatArray,
    configurations: IntArray,
    cone_type: SortableConeTypes,
    nearby_searcher: NearbyConeSearcher | None = None,
    search_distance: float = 6.0,
    search_angle: float = np.pi / 1.5,
) -> FloatArray:
    # Without a searcher from the caller the counts are computed uncached,
    # rather than through a cache shared with every other planner.
    searcher = nearby_searcher if nearby_searcher is not None else NearbyConeSearcher()
    n_good, n_bad = searcher.number_of_cones_on_each_side_for_each_config(
        points,
        configurations,
        cone_type,
        search_distance,
        search_angle,
    )

    diff = n_good - n_bad
    m_value = diff.min()

    diff += np.abs(m_value) + 1

    return 1 / diff


@dataclass(frozen=True)
class CostContext:
    """Everything a cost term may look at, for one side and one frame."""

    points_xy: FloatArray
    configurations: IntArray
    cone_type: SortableConeTypes
    vehicle_direction: FloatArray
    config: SortingConfig
    nearby_searcher: NearbyConeSearcher | None


@dataclass(frozen=True)
class CostTerm:
    """One term of the sorting cost.

    `name` is the CostWeights field that weights it. Every term returns one
    value per configuration; lower is better.
    """

    name: str
    compute: Callable[[CostContext], FloatArray]


def _cones_on_side_term(ctx: CostContext) -> FloatArray:
    return calc_cones_on_either_cost(
        ctx.points_xy,
        ctx.configurations,
        ctx.cone_type,
        nearby_searcher=ctx.nearby_searcher,
        search_distance=ctx.config.cones_on_side_search_distance,
        search_angle=ctx.config.cones_on_side_search_angle,
    )


COST_TERMS: tuple[CostTerm, ...] = (
    CostTerm(
        "angle",
        lambda ctx: calc_angle_cost_for_configuration(
            ctx.points_xy, ctx.configurations, ctx.cone_type
        ),
    ),
    CostTerm(
        "residual_distance",
        lambda ctx: calc_distance_cost(
            ctx.points_xy, ctx.configurations, ctx.config.residual_distance_threshold
        ),
    ),
    CostTerm("n_cones", lambda ctx: calc_number_of_cones_cost(ctx.configurations)),
    CostTerm(
        "initial_direction",
        lambda ctx: calc_initial_direction_cost(
            ctx.points_xy, ctx.configurations, ctx.vehicle_direction
        ),
    ),
    CostTerm(
        "direction_change",
        lambda ctx: calc_change_of_direction_cost(ctx.points_xy, ctx.configurations),
    ),
    CostTerm("cones_on_side", _cones_on_side_term),
    CostTerm(
        "wrong_direction",
        lambda ctx: calc_wrong_direction_cost(
            ctx.points_xy, ctx.configurations, ctx.cone_type
        ),
    ),
)
"""The terms of the sorting cost, in the column order of
cost_configurations(return_individual_costs=True)."""


def cost_configurations(
    points: FloatArray,
    configurations: IntArray,
    cone_type: SortableConeTypes,
    vehicle_position: FloatArray,  # pylint: disable=unused-argument (future proofing, incase we want to use it)
    vehicle_direction: FloatArray,
    *,
    return_individual_costs: bool,
    nearby_searcher: NearbyConeSearcher | None = None,
    config: SortingConfig | None = None,
) -> FloatArray:
    """
    Calculates a cost for each provided configuration: the weighted sum of
    the COST_TERMS, with the weights normalised to sum to one.

    Args:
        points: The underlying points
        configurations: An array of indices defining a configuration of the
        provided points
        cone_type: The type of cone (left/right)
        return_individual_costs: Return the weighted value of each term, one
        column per term in COST_TERMS order, instead of their sum
    Returns:
        A cost for each configuration
    """
    if config is None:
        config = SortingConfig()

    if len(configurations) == 0:
        return np.zeros(0)

    ctx = CostContext(
        points_xy=points[:, :2],
        configurations=configurations,
        cone_type=cone_type,
        vehicle_direction=vehicle_direction,
        config=config,
        nearby_searcher=nearby_searcher,
    )
    weights = [getattr(config.cost_weights, term.name) for term in COST_TERMS]

    # A term weighted zero contributes nothing, so it is not computed; its
    # column stays, as zeros, so the columns keep their order.
    columns = [
        term.compute(ctx) if weight != 0 else np.zeros(len(configurations))
        for term, weight in zip(COST_TERMS, weights, strict=True)
    ]

    factors: FloatArray = np.array(weights)
    factors = factors / factors.sum()
    final_costs = np.column_stack(columns) * factors

    if return_individual_costs:
        return final_costs

    return final_costs.sum(axis=-1)
