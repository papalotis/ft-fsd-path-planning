from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning.sorting_cones.trace_sorter.common import NoPathError
from fsd_path_planning.sorting_cones.trace_sorter.end_configurations import (
    adjacency_matrix_to_borders_and_targets,
    angle_difference,
    find_all_end_configurations,
)
from fsd_path_planning.utils.cone_types import ConeTypes


class TestEndConfigurationHelpers:
    def test_adjacency_matrix_to_borders_and_targets(self):
        adjacency_matrix = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
            ],
            dtype=np.int32,
        )

        neighbors, borders = adjacency_matrix_to_borders_and_targets(adjacency_matrix)

        np.testing.assert_array_equal(neighbors, np.array([1, 2, 3, 2]))
        np.testing.assert_array_equal(borders, np.array([0, 1, 3, 3, 4]))

    def test_angle_difference_wraps_to_minus_pi_to_pi(self):
        difference = angle_difference(np.deg2rad(-170), np.deg2rad(170))

        np.testing.assert_allclose(difference, np.deg2rad(20))


class TestFindAllEndConfigurations:
    def test_finds_simple_left_cone_chain(self):
        points = np.array(
            [
                [0.0, 1.0, ConeTypes.LEFT],
                [1.0, 1.0, ConeTypes.LEFT],
                [2.0, 1.0, ConeTypes.LEFT],
                [3.0, 1.0, ConeTypes.LEFT],
            ]
        )
        adjacency_matrix = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        end_configurations, history = find_all_end_configurations(
            points=points,
            cone_type=ConeTypes.LEFT,
            start_idx=0,
            adjacency_matrix=adjacency_matrix,
            target_length=4,
            threshold_directional_angle=np.deg2rad(80),
            threshold_absolute_angle=np.deg2rad(100),
            first_k_indices_must_be=np.zeros(0, dtype=np.int32),
            car_position=np.array([0.0, 0.0]),
            car_direction=np.array([1.0, 0.0]),
            car_size=1.0,
            store_all_end_configurations=True,
        )

        np.testing.assert_array_equal(end_configurations, np.array([[0, 1, 2, 3]]))
        assert history is not None
        all_configurations, is_end = history
        assert len(all_configurations) == len(is_end)
        assert np.any(is_end)

    def test_raises_when_no_valid_path_can_be_built(self):
        points = np.array(
            [
                [0.0, 1.0, ConeTypes.LEFT],
                [0.0, -1.0, ConeTypes.LEFT],
                [-1.0, 1.0, ConeTypes.LEFT],
            ]
        )
        adjacency_matrix = np.array(
            [
                [0, 1, 1],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.int32,
        )

        with pytest.raises(NoPathError):
            find_all_end_configurations(
                points=points,
                cone_type=ConeTypes.LEFT,
                start_idx=0,
                adjacency_matrix=adjacency_matrix,
                target_length=3,
                threshold_directional_angle=np.deg2rad(80),
                threshold_absolute_angle=np.deg2rad(100),
                first_k_indices_must_be=np.zeros(0, dtype=np.int32),
                car_position=np.array([0.0, 0.0]),
                car_direction=np.array([1.0, 0.0]),
                car_size=1.0,
                store_all_end_configurations=False,
            )

    def test_filters_configurations_by_required_prefix(self):
        points = np.array(
            [
                [0.0, 1.0, ConeTypes.LEFT],
                [1.0, 1.0, ConeTypes.LEFT],
                [1.2, 1.2, ConeTypes.LEFT],
                [2.0, 1.0, ConeTypes.LEFT],
                [2.2, 1.2, ConeTypes.LEFT],
            ]
        )
        adjacency_matrix = np.array(
            [
                [0, 1, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        end_configurations, _ = find_all_end_configurations(
            points=points,
            cone_type=ConeTypes.LEFT,
            start_idx=0,
            adjacency_matrix=adjacency_matrix,
            target_length=3,
            threshold_directional_angle=np.deg2rad(80),
            threshold_absolute_angle=np.deg2rad(100),
            first_k_indices_must_be=np.array([0, 2], dtype=np.int32),
            car_position=np.array([0.0, 0.0]),
            car_direction=np.array([1.0, 0.0]),
            car_size=1.0,
            store_all_end_configurations=False,
        )

        np.testing.assert_array_equal(end_configurations, np.array([[0, 2, 4]]))

    def test_trims_terminal_cone_of_wrong_type(self):
        points = np.array(
            [
                [0.0, 1.0, ConeTypes.LEFT],
                [1.0, 1.0, ConeTypes.LEFT],
                [2.0, 1.0, ConeTypes.LEFT],
                [3.0, 1.0, ConeTypes.UNKNOWN],
            ]
        )
        adjacency_matrix = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        end_configurations, _ = find_all_end_configurations(
            points=points,
            cone_type=ConeTypes.LEFT,
            start_idx=0,
            adjacency_matrix=adjacency_matrix,
            target_length=4,
            threshold_directional_angle=np.deg2rad(80),
            threshold_absolute_angle=np.deg2rad(100),
            first_k_indices_must_be=np.zeros(0, dtype=np.int32),
            car_position=np.array([0.0, 0.0]),
            car_direction=np.array([1.0, 0.0]),
            car_size=1.0,
            store_all_end_configurations=False,
        )

        np.testing.assert_array_equal(end_configurations, np.array([[0, 1, 2, -1]]))