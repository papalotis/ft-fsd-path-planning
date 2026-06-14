from __future__ import annotations

import numpy as np

from fsd_path_planning.relocalization.acceleration.acceleration_relocalization import (
    BASE_ACCELERATION_PATH,
    AccelerationRelocalizer,
    best_fit,
    select_random_subset,
)


class TestAccelerationRelocalizationHelpers:
    def test_select_random_subset_returns_requested_points(self):
        np.random.seed(0)
        points = np.arange(20, dtype=float).reshape(10, 2)

        subset = select_random_subset(points, subset_size=4)

        assert subset.shape == (4, 2)
        assert len(np.unique(subset, axis=0)) == 4
        assert all(
            any(np.array_equal(point, candidate) for candidate in points)
            for point in subset
        )

    def test_best_fit_recovers_line_coefficients(self):
        np.random.seed(0)
        x_values = np.arange(6, dtype=float)
        points = np.column_stack((x_values, 2.0 * x_values + 1.0))

        coefficients = best_fit(points, subset_size=3, iterations=25)

        np.testing.assert_allclose(coefficients, np.array([2.0, 1.0]), atol=1e-12)


class TestAccelerationRelocalizer:
    def test_returns_none_before_original_pose_is_known(self):
        relocalizer = AccelerationRelocalizer()

        result = relocalizer.do_relocalization_once(
            cones=[np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])],
            vehicle_position=np.array([0.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )

        assert result is None

    def test_returns_none_when_too_few_expected_left_cones_are_visible(self):
        relocalizer = AccelerationRelocalizer()
        relocalizer._original_vehicle_position = np.array([0.0, 0.0])

        result = relocalizer.do_relocalization_once(
            cones=[np.array([[0.0, 1.0], [1.0, 1.1], [2.0, 0.9]])],
            vehicle_position=np.array([0.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )

        assert result is None

    def test_successful_relocalization_returns_inverse_transforms(self):
        np.random.seed(0)
        relocalizer = AccelerationRelocalizer()
        relocalizer._original_vehicle_position = np.array([5.0, -3.0])

        cones = [
            np.array(
                [
                    [0.0, 1.0],
                    [2.0, 1.1],
                    [4.0, 0.9],
                    [6.0, 1.05],
                ]
            ),
            np.zeros((0, 2)),
        ]

        result = relocalizer.do_relocalization_once(
            cones=cones,
            vehicle_position=np.array([0.0, 0.0]),
            vehicle_direction=np.array([1.0, 0.0]),
        )

        assert result is not None
        transform_to_known_frame, transform_to_base_frame = result

        original_position = np.array([7.5, -1.25])
        original_yaw = 0.3

        known_position, known_yaw = transform_to_known_frame(
            original_position, original_yaw
        )
        recovered_position, recovered_yaw = transform_to_base_frame(
            known_position, known_yaw
        )

        np.testing.assert_allclose(recovered_position, original_position, atol=1e-10)
        np.testing.assert_allclose(recovered_yaw, original_yaw, atol=1e-10)

    def test_known_global_path_matches_precomputed_path(self):
        relocalizer = AccelerationRelocalizer()

        known_path = relocalizer.get_known_global_path()

        np.testing.assert_allclose(known_path, BASE_ACCELERATION_PATH)
