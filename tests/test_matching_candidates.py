"""The compiled candidate search gives exactly what the NumPy loop gave."""

from __future__ import annotations

import numpy as np
import pytest

from fsd_path_planning.cone_matching.functional_cone_matching import (
    find_boolean_mask_of_all_potential_matches,
)
from fsd_path_planning.geometry.angles import angle_from_2d_vector, vec_angle_between
from fsd_path_planning.geometry.transforms import rotate


def reference_mask(
    start_points,
    directions,
    other_side_cones,
    other_side_directions,
    major_radius,
    minor_radius,
    max_search_angle,
):
    """The implementation before it was compiled, kept as the reference."""
    out = np.zeros((len(start_points), len(other_side_cones)), dtype=bool)
    if len(start_points) == 0 or len(other_side_cones) == 0:
        return out
    radii_square = np.array([major_radius, minor_radius]) ** 2
    angles = angle_from_2d_vector(directions)
    to_other = other_side_cones - start_points[:, None]
    angle_diff = vec_angle_between(directions[:, None], other_side_directions)
    for i in range(len(start_points)):
        rotated = rotate(to_other[i], -angles[i])
        out[i] = ((rotated**2) / radii_square).sum(axis=1) < 1
        out[i, np.abs(angle_from_2d_vector(rotated) / 2) > max_search_angle] = False
        out[i, angle_diff[i] < np.pi / 2] = False
    distances = np.linalg.norm(to_other, axis=-1)
    for i in range(len(start_points)):
        row = distances[i]
        row[~out[i]] = np.inf
        idxs = np.argsort(row)[:2]
        idxs = idxs[np.isfinite(row[idxs])]
        new = np.zeros_like(out[i])
        new[idxs] = True
        out[i] = new
    return out


def random_sides(rng, n_start, n_other):
    """Two noisy rows of cones about a track width apart, facing each other."""
    xs = np.sort(rng.uniform(0, 30, n_start))
    start = np.column_stack([xs, rng.normal(1.5, 0.4, n_start)])
    other = np.column_stack(
        [np.sort(rng.uniform(0, 30, n_other)), rng.normal(-1.5, 0.4, n_other)]
    )
    directions = np.column_stack([rng.normal(0, 0.3, n_start), -np.ones(n_start)])
    other_directions = np.column_stack([rng.normal(0, 0.3, n_other), np.ones(n_other)])
    return start, directions, other, other_directions


@pytest.mark.parametrize("seed", range(40))
def test_same_mask_as_the_numpy_loop(seed) -> None:
    rng = np.random.default_rng(seed)
    args = (
        *random_sides(rng, int(rng.integers(1, 15)), int(rng.integers(1, 15))),
        7.5,
        3.0,
        np.deg2rad(50),
    )
    np.testing.assert_array_equal(
        find_boolean_mask_of_all_potential_matches(*args), reference_mask(*args)
    )


def test_empty_sides() -> None:
    empty = np.zeros((0, 2))
    some = np.ones((3, 2))
    assert find_boolean_mask_of_all_potential_matches(
        empty, empty, some, some, 7.5, 3.0, 1.0
    ).shape == (0, 3)
    assert find_boolean_mask_of_all_potential_matches(
        some, some, empty, empty, 7.5, 3.0, 1.0
    ).shape == (3, 0)
