"""
Pure 2D geometry: no cone types, missions or planner state.

    vectors     dot products, norms, normalisation
    angles      angles of and between vectors, angle differences
    transforms  rotation of points
    distances   distances between point sets
    traces      step lengths and turning angles along an ordered trace
    shapes      circles through points, circle fit, points inside an ellipse
    segments    line segment intersections
    splines     spline fitting and arc-length sampling

Every function is importable from its module and from this package.

Project: fsd_path_planning
"""

from fsd_path_planning.geometry.angles import (
    angle_difference,
    angle_from_2d_vector,
    unit_2d_vector_from_angle,
    vec_angle_between,
)
from fsd_path_planning.geometry.distances import (
    calc_pairwise_distances,
    my_cdist_sq_euclidean,
)
from fsd_path_planning.geometry.segments import (
    batch_lines_segments_intersect_indicator,
    calc_intersections,
    check_intersection_within_bounding_boxes,
    intersection_in_bounding_box,
    lines_segments_intersect_indicator,
    number_of_intersections,
    number_of_intersections_in_configurations,
    number_of_intersections_in_trace,
    pairwise_segment_intersection,
    trace_intersections,
)
from fsd_path_planning.geometry.shapes import (
    calculate_radius_from_points,
    center_of_circle_from_3_points,
    circle_fit,
    points_inside_ellipse,
    trace_calculate_consecutive_radii,
)
from fsd_path_planning.geometry.splines import (
    NullSplineEvaluator,
    SplineEvaluator,
    SplineFitterFactory,
)
from fsd_path_planning.geometry.traces import (
    trace_angles_between,
    trace_distance_to_next,
)
from fsd_path_planning.geometry.transforms import (
    rotate,
)
from fsd_path_planning.geometry.vectors import (
    norm_of_last_axis,
    normalize_last_axis,
    vec_dot,
)

__all__ = [
    "vec_dot",
    "norm_of_last_axis",
    "normalize_last_axis",
    "vec_angle_between",
    "unit_2d_vector_from_angle",
    "angle_from_2d_vector",
    "angle_difference",
    "rotate",
    "my_cdist_sq_euclidean",
    "calc_pairwise_distances",
    "trace_distance_to_next",
    "trace_angles_between",
    "calculate_radius_from_points",
    "trace_calculate_consecutive_radii",
    "points_inside_ellipse",
    "center_of_circle_from_3_points",
    "circle_fit",
    "calc_intersections",
    "intersection_in_bounding_box",
    "check_intersection_within_bounding_boxes",
    "lines_segments_intersect_indicator",
    "batch_lines_segments_intersect_indicator",
    "pairwise_segment_intersection",
    "number_of_intersections",
    "trace_intersections",
    "number_of_intersections_in_trace",
    "number_of_intersections_in_configurations",
    "SplineEvaluator",
    "NullSplineEvaluator",
    "SplineFitterFactory",
]
