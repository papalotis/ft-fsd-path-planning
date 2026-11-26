"""The skidpad centre clustering must match DBSCAN(eps, min_samples=1)."""

from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning.stages.relocalization.skidpad import (
    cluster_within_distance,
)


class TestClusterWithinDistance:
    def test_points_within_the_distance_share_a_cluster(self) -> None:
        points = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
        labels = cluster_within_distance(points, 3.0)
        assert labels[0] == labels[1] != labels[2]

    def test_a_chain_of_neighbours_is_one_cluster(self) -> None:
        """Single linkage: 0 and 2 are 5 m apart but linked through 1."""
        points = np.array([[0.0, 0.0], [2.5, 0.0], [5.0, 0.0]])
        assert len(set(cluster_within_distance(points, 3.0))) == 1

    def test_clusters_are_numbered_by_their_first_point(self) -> None:
        points = np.array([[50.0, 0.0], [0.0, 0.0], [50.5, 0.0], [0.5, 0.0]])
        np.testing.assert_array_equal(
            cluster_within_distance(points, 3.0), [0, 1, 0, 1]
        )

    def test_isolated_points_are_their_own_clusters(self) -> None:
        points = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        np.testing.assert_array_equal(cluster_within_distance(points, 3.0), [0, 1, 2])

    def test_no_points_gives_no_labels(self) -> None:
        assert len(cluster_within_distance(np.zeros((0, 2)), 3.0)) == 0

    def test_matches_dbscan_on_random_points(self) -> None:
        sklearn_cluster = pytest.importorskip("sklearn.cluster")
        rng = np.random.default_rng(0)
        for n_points in (1, 2, 5, 20, 100):
            for scale in (2.0, 10.0, 40.0):
                for _ in range(20):
                    points = rng.normal(0.0, scale, size=(n_points, 2))
                    expected = (
                        sklearn_cluster.DBSCAN(eps=3.0, min_samples=1)
                        .fit(points)
                        .labels_
                    )
                    np.testing.assert_array_equal(
                        cluster_within_distance(points, 3.0), expected
                    )
