"""The sorting cost registry and the Side type."""

from __future__ import annotations

from dataclasses import fields

import pytest

from fsd_path_planning.config_dataclasses import CostWeights
from fsd_path_planning.sorting_cones.trace_sorter.cost_function import COST_TERMS
from fsd_path_planning.utils.cone_types import ConeTypes, Side, invert_cone_type


class TestCostTerms:
    def test_one_term_per_weight_in_the_same_order(self) -> None:
        """A weight without a term, or a term without a weight, fails here."""
        assert [t.name for t in COST_TERMS] == [f.name for f in fields(CostWeights)]


class TestSide:
    def test_cone_types(self) -> None:
        assert Side.LEFT.cone_type is ConeTypes.LEFT
        assert Side.RIGHT.cone_type is ConeTypes.RIGHT

    def test_other(self) -> None:
        assert Side.LEFT.other is Side.RIGHT
        assert Side.RIGHT.other is Side.LEFT

    @pytest.mark.parametrize("cone_type", [ConeTypes.BLUE, ConeTypes.YELLOW])
    def test_of_round_trips(self, cone_type) -> None:
        assert Side.of(cone_type).cone_type == cone_type

    @pytest.mark.parametrize(
        "cone_type",
        [ConeTypes.UNKNOWN, ConeTypes.ORANGE_SMALL, ConeTypes.ORANGE_BIG],
    )
    def test_of_refuses_cones_without_a_side(self, cone_type) -> None:
        with pytest.raises(ValueError):
            Side.of(cone_type)

    def test_invert_cone_type_is_unchanged(self) -> None:
        assert invert_cone_type(ConeTypes.LEFT) is ConeTypes.RIGHT
        assert invert_cone_type(ConeTypes.RIGHT) is ConeTypes.LEFT
        assert invert_cone_type(ConeTypes.UNKNOWN) is ConeTypes.UNKNOWN
        assert invert_cone_type(2) is ConeTypes.RIGHT
