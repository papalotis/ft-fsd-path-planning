"""
Deprecated: moved to fsd_path_planning.stages.path.strategies.

Forwards every name to it with a warning until 0.7.0.

Project: fsd_path_planning
"""

from fsd_path_planning._deprecation import moved_module_entirely

__getattr__ = moved_module_entirely(
    "fsd_path_planning.calculate_path.strategies",
    "fsd_path_planning.stages.path.strategies",
    removed_in="0.7.0",
)
