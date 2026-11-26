"""
Deprecated: moved to fsd_path_planning.stages.matching.search_directions.

Forwards every name to it with a warning until 0.7.0.

Project: fsd_path_planning
"""

from fsd_path_planning._deprecation import moved_module_entirely

__getattr__ = moved_module_entirely(
    "fsd_path_planning.cone_matching.match_directions",
    "fsd_path_planning.stages.matching.search_directions",
    removed_in="0.7.0",
)
