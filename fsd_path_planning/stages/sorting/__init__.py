"""
Cone sorting: order the cones of each side into a trace.

    stage          ConeSorting, the stage itself
    trace_sorter   the search for each side, and combining the two
    search         the depth-first search for candidate traces (Numba)
    scoring        candidate traces and their costs
    costs          the cost terms, COST_TERMS
    adjacency, nearby_cones, combine, distance_cost, common  helpers

Project: fsd_path_planning
"""
