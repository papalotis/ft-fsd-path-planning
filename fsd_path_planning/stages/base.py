"""
The contract every stage of the planner follows.

A stage is built once from its config and then called once per frame with an
input dataclass, returning a frozen result dataclass:

    ConeSorting       ConeSortingInput       -> SortingResult
    ConeMatching      ConeMatchingInput      -> MatchingResult
    CalculatePath     PathCalculationInput   -> PathResult
    Relocalizer       RelocalizationInput    -> RelocalizationResult

Expected outcomes, such as finding no trace or not being relocalized yet, are
part of the result, not exceptions. A stage may keep state between frames
(caches, the previous path); it owns that state, and nothing else shares it.

Project: fsd_path_planning
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

InputT_contra = TypeVar("InputT_contra", contravariant=True)
ResultT_co = TypeVar("ResultT_co", covariant=True)


@runtime_checkable
class Stage(Protocol[InputT_contra, ResultT_co]):
    """One step of the planner: an input for a frame in, a result out."""

    def run(self, stage_input: InputT_contra) -> ResultT_co:
        """Process one frame."""
        ...
