"""
Deprecated: the segment intersection moved to fsd_path_planning.geometry.segments.

Forwards to it with a warning until 0.7.0.

Project: fsd_path_planning
"""

from fsd_path_planning._deprecation import moved_module

__getattr__ = moved_module(
    "fsd_path_planning.sorting_cones.trace_sorter.line_segment_intersection",
    "fsd_path_planning.geometry.segments",
    (
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
    ),
    removed_in="0.7.0",
)
