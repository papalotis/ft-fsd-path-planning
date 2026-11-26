"""
Deprecated: moved to fsd_path_planning.stages.sorting.trace_sorter.

Forwards every name to it with a warning until 0.7.0.

Project: fsd_path_planning
"""

from fsd_path_planning._deprecation import moved_module_entirely

__getattr__ = moved_module_entirely(
    "fsd_path_planning.sorting_cones.trace_sorter.core_trace_sorter",
    "fsd_path_planning.stages.sorting.trace_sorter",
    removed_in="0.7.0",
)
