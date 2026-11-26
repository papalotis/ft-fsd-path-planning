"""
Deprecated: the geometry moved to fsd_path_planning.geometry.

Each name forwards to its new module with a warning until 0.7.0. The helpers
the planner no longer uses (lerp, the PT1 filter, the quaternion conversions)
have no new home and are removed in 0.7.0.

Project: fsd_path_planning
"""

from fsd_path_planning._deprecation import moved_names

__getattr__ = moved_names(
    "fsd_path_planning.utils.math_utils",
    {
        "vec_dot": "fsd_path_planning.geometry.vectors",
        "norm_of_last_axis": "fsd_path_planning.geometry.vectors",
        "normalize_last_axis": "fsd_path_planning.geometry.vectors",
        "vec_angle_between": "fsd_path_planning.geometry.angles",
        "unit_2d_vector_from_angle": "fsd_path_planning.geometry.angles",
        "angle_from_2d_vector": "fsd_path_planning.geometry.angles",
        "angle_difference": "fsd_path_planning.geometry.angles",
        "rotate": "fsd_path_planning.geometry.transforms",
        "my_cdist_sq_euclidean": "fsd_path_planning.geometry.distances",
        "calc_pairwise_distances": "fsd_path_planning.geometry.distances",
        "trace_distance_to_next": "fsd_path_planning.geometry.traces",
        "trace_angles_between": "fsd_path_planning.geometry.traces",
        "calculate_radius_from_points": "fsd_path_planning.geometry.shapes",
        "trace_calculate_consecutive_radii": "fsd_path_planning.geometry.shapes",
        "points_inside_ellipse": "fsd_path_planning.geometry.shapes",
        "center_of_circle_from_3_points": "fsd_path_planning.geometry.shapes",
        "circle_fit": "fsd_path_planning.geometry.shapes",
        "my_in1d": "fsd_path_planning.utils.arrays",
        "my_njit": "fsd_path_planning._numba",
    },
    removed_in="0.7.0",
    retired={
        "lerp": "fsd_path_planning.utils._retired_math",
        "linearly_combine_values_over_time": "fsd_path_planning.utils._retired_math",
        "odd_square": "fsd_path_planning.utils._retired_math",
        "euler_angles_to_quaternion": "fsd_path_planning.utils._retired_math",
        "quaternion_to_euler_angles": "fsd_path_planning.utils._retired_math",
    },
)
