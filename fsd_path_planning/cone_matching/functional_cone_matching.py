"""
Deprecated: moved to fsd_path_planning.stages.matching.virtual_cones.

Forwards every name to it with a warning until 0.7.0.

Project: fsd_path_planning
"""

from fsd_path_planning._deprecation import moved_module_entirely

__getattr__ = moved_module_entirely(
    "fsd_path_planning.cone_matching.functional_cone_matching",
    "fsd_path_planning.stages.matching.virtual_cones",
    removed_in="0.7.0",
)
