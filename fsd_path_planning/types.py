"""
Deprecated: moved to fsd_path_planning.common_types.

A module called ``types`` inside the package shadowed the standard library's
``types`` whenever the package directory itself ended up on ``sys.path``, for
example when running a script from inside it, and the import of ``enum`` then
failed deep inside the standard library. The contents now live in
``common_types``; this module forwards to it with a warning until 0.7.0.

Project: fsd_path_planning
"""

from fsd_path_planning._deprecation import moved_module

__getattr__ = moved_module(
    "fsd_path_planning.types",
    "fsd_path_planning.common_types",
    (
        "GenericArray",
        "FloatArray",
        "IntArray",
        "BoolArray",
        "SortableConeTypes",
        "ConeInput",
        "SortingResult",
        "MatchingResult",
        "PathResult",
    ),
    removed_in="0.7.0",
)
