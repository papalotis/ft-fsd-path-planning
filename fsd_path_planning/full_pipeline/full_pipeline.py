"""
Deprecated: moved to fsd_path_planning.pipeline.

Forwards every name to it with a warning until 0.7.0.

Project: fsd_path_planning
"""

from fsd_path_planning._deprecation import moved_module_entirely

__getattr__ = moved_module_entirely(
    "fsd_path_planning.full_pipeline.full_pipeline",
    "fsd_path_planning.pipeline",
    removed_in="0.7.0",
)
