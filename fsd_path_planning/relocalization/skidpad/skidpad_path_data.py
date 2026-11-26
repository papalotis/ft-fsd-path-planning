"""
Deprecated: moved to fsd_path_planning.stages.relocalization.skidpad_path_data.

Forwards every name to it with a warning until 0.7.0.

Project: fsd_path_planning
"""

from fsd_path_planning._deprecation import moved_module_entirely

__getattr__ = moved_module_entirely(
    "fsd_path_planning.relocalization.skidpad.skidpad_path_data",
    "fsd_path_planning.stages.relocalization.skidpad_path_data",
    removed_in="0.7.0",
)
