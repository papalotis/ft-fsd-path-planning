"""Unit tests for fsd_path_planning.utils.math_utils."""

from __future__ import annotations

import numpy as np
import pytest

import fsd_path_planning.utils.math_utils as math_utils_module
from fsd_path_planning.utils.math_utils import (
    angle_difference,
    angle_from_2d_vector,
    calc_pairwise_distances,
    calculate_radius_from_points,
    center_of_circle_from_3_points,
    circle_fit,
    euler_angles_to_quaternion,
    lerp,
    my_cdist_sq_euclidean,
    my_in1d,
    my_njit,
    norm_of_last_axis,
    normalize_last_axis,
    odd_square,
    points_inside_ellipse,
    quaternion_to_euler_angles,
    rotate,
    trace_angles_between,
    trace_distance_to_next,
    unit_2d_vector_from_angle,
    vec_angle_between,
    vec_dot,
)


class TestMyNjit:
    def test_returns_original_function_when_coverage_is_running(self, monkeypatch):
        def sample_function(value: float) -> float:
            return value + 1.0

        monkeypatch.setattr(math_utils_module, "_is_coverage_running", lambda: True)

        decorated = my_njit(sample_function)

        assert decorated is sample_function
        assert decorated(2.0) == 3.0

    def test_uses_jit_when_coverage_is_not_running(self, monkeypatch):
        jit_calls = []

        def fake_jit(**kwargs):
            jit_calls.append(kwargs)

            def wrapper(func):
                def compiled(*args, **inner_kwargs):
                    return func(*args, **inner_kwargs)

                return compiled

            return wrapper

        def sample_function(value: float) -> float:
            return value + 1.0

        monkeypatch.setattr(math_utils_module, "_is_coverage_running", lambda: False)
        monkeypatch.setattr(math_utils_module, "jit", fake_jit)

        decorated = my_njit(sample_function)

        assert decorated is not sample_function
        assert decorated(2.0) == 3.0
        assert jit_calls == [
            {
                "nopython": True,
                "cache": True,
                "nogil": True,
                "fastmath": True,
            }
        ]

# ── vec_dot ──────────────────────────────────────────────────────────────────


class TestVecDot:
    def test_orthogonal(self):
        v1 = np.array([[1.0, 0.0]])
        v2 = np.array([[0.0, 1.0]])
        result = vec_dot(v1, v2)
        np.testing.assert_allclose(result, [0.0])

    def test_parallel(self):
        v1 = np.array([[3.0, 4.0]])
        v2 = np.array([[3.0, 4.0]])
        result = vec_dot(v1, v2)
        np.testing.assert_allclose(result, [25.0])

    def test_batch(self):
        v1 = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        v2 = np.array([[1, 0], [1, 0], [1, 1]], dtype=float)
        result = vec_dot(v1, v2)
        np.testing.assert_allclose(result, [1.0, 0.0, 2.0])


# ── norm_of_last_axis ────────────────────────────────────────────────────────


class TestNormOfLastAxis:
    def test_unit_vectors(self):
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(norm_of_last_axis(vecs), [1.0, 1.0])

    def test_3_4_5(self):
        vecs = np.array([[3.0, 4.0]])
        np.testing.assert_allclose(norm_of_last_axis(vecs), [5.0])


# ── rotate ───────────────────────────────────────────────────────────────────


class TestRotate:
    def test_90_degrees(self):
        pts = np.array([[1.0, 0.0]])
        result = rotate(pts, np.pi / 2)
        np.testing.assert_allclose(result, [[0.0, 1.0]], atol=1e-12)

    def test_180_degrees(self):
        pts = np.array([[1.0, 0.0]])
        result = rotate(pts, np.pi)
        np.testing.assert_allclose(result, [[-1.0, 0.0]], atol=1e-12)

    def test_360_degrees(self):
        pts = np.array([[3.0, 4.0]])
        result = rotate(pts, 2 * np.pi)
        np.testing.assert_allclose(result, pts, atol=1e-12)

    def test_zero_rotation(self):
        pts = np.array([[5.0, 7.0], [1.0, 2.0]])
        result = rotate(pts, 0.0)
        np.testing.assert_allclose(result, pts, atol=1e-12)

    def test_batch_points(self):
        pts = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = rotate(pts, np.pi / 2)
        expected = np.array([[0.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ── vec_angle_between ────────────────────────────────────────────────────────


class TestVecAngleBetween:
    def test_same_direction(self):
        v = np.array([[1.0, 0.0]])
        result = vec_angle_between(v, v)
        np.testing.assert_allclose(result, [0.0], atol=1e-12)

    def test_perpendicular(self):
        v1 = np.array([[1.0, 0.0]])
        v2 = np.array([[0.0, 1.0]])
        result = vec_angle_between(v1, v2)
        np.testing.assert_allclose(result, [np.pi / 2], atol=1e-12)

    def test_opposite(self):
        v1 = np.array([[1.0, 0.0]])
        v2 = np.array([[-1.0, 0.0]])
        result = vec_angle_between(v1, v2)
        np.testing.assert_allclose(result, [np.pi], atol=1e-12)

    def test_invalid_vecs1_shape_raises_value_error(self):
        with pytest.raises(ValueError, match="vecs1 must contain 2d vectors"):
            vec_angle_between(np.array([[1.0, 0.0, 0.0]]), np.array([[1.0, 0.0]]))

    def test_invalid_vecs2_shape_raises_value_error(self):
        with pytest.raises(ValueError, match="vecs2 must contain 2d vectors"):
            vec_angle_between(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))


# ── my_cdist_sq_euclidean ────────────────────────────────────────────────────


class TestMyCdistSqEuclidean:
    def test_against_manual(self):
        a = np.array([[0.0, 0.0], [1.0, 0.0]])
        b = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        result = my_cdist_sq_euclidean(a, b)
        expected = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 1.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_single_point(self):
        a = np.array([[3.0, 4.0]])
        b = np.array([[0.0, 0.0]])
        result = my_cdist_sq_euclidean(a, b)
        np.testing.assert_allclose(result, [[25.0]], atol=1e-10)


# ── calc_pairwise_distances ──────────────────────────────────────────────────


class TestCalcPairwiseDistances:
    def test_diagonal_zero(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        result = calc_pairwise_distances(pts)
        np.testing.assert_allclose(np.diag(result), [0.0, 0.0], atol=1e-10)

    def test_symmetric(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        result = calc_pairwise_distances(pts)
        np.testing.assert_allclose(result, result.T, atol=1e-10)

    def test_dist_to_self(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = calc_pairwise_distances(pts, dist_to_self=99.0)
        np.testing.assert_allclose(np.diag(result), [99.0, 99.0], atol=1e-10)


# ── my_in1d ──────────────────────────────────────────────────────────────────


class TestMyIn1d:
    def test_basic(self):
        test = np.array([1, 2, 3, 4, 5])
        source = np.array([2, 4])
        result = my_in1d(test, source)
        np.testing.assert_array_equal(result, [False, True, False, True, False])

    def test_empty_source(self):
        test = np.array([1, 2])
        source = np.array([], dtype=int)
        result = my_in1d(test, source)
        np.testing.assert_array_equal(result, [False, False])


# ── unit_2d_vector_from_angle / angle_from_2d_vector ─────────────────────────


class TestAngleVectorRoundtrip:
    @pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3])
    def test_roundtrip_scalar(self, angle):
        angle_arr = np.array([angle])
        vec = unit_2d_vector_from_angle(angle_arr)
        recovered = angle_from_2d_vector(vec)
        np.testing.assert_allclose(recovered, angle_arr, atol=1e-12)

    def test_batch(self):
        angles = np.array([0.0, np.pi / 2, np.pi])
        vecs = unit_2d_vector_from_angle(angles)
        recovered = angle_from_2d_vector(vecs)
        np.testing.assert_allclose(recovered, angles, atol=1e-12)

    def test_unit_length(self):
        angles = np.linspace(-np.pi, np.pi, 20)
        vecs = unit_2d_vector_from_angle(angles)
        norms = np.linalg.norm(vecs, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_invalid_vector_shape_raises_value_error(self):
        with pytest.raises(ValueError, match="vecs must be a 2d vector"):
            angle_from_2d_vector(np.array([[1.0, 0.0, 0.0]]))


# ── normalize_last_axis ──────────────────────────────────────────────────────


class TestNormalizeLastAxis:
    def test_already_unit(self):
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = normalize_last_axis(vecs)
        np.testing.assert_allclose(result, vecs, atol=1e-12)

    def test_scaling(self):
        vecs = np.array([[3.0, 4.0]])
        result = normalize_last_axis(vecs)
        np.testing.assert_allclose(result, [[0.6, 0.8]], atol=1e-12)


# ── lerp ─────────────────────────────────────────────────────────────────────


class TestLerp:
    def test_identity_mapping(self):
        vals = np.array([0.0, 0.5, 1.0])
        result = lerp(vals, 0.0, 1.0, 0.0, 1.0)
        np.testing.assert_allclose(result, vals)

    def test_scaling(self):
        vals = np.array([1.0, 2.0, 3.0])
        result = lerp(vals, 0.0, 10.0, 30.0, 100.0)
        np.testing.assert_allclose(result, [37.0, 44.0, 51.0])

    def test_boundaries(self):
        result_start = lerp(np.array([0.0]), 0.0, 1.0, 10.0, 20.0)
        result_end = lerp(np.array([1.0]), 0.0, 1.0, 10.0, 20.0)
        np.testing.assert_allclose(result_start, [10.0])
        np.testing.assert_allclose(result_end, [20.0])


# ── angle_difference ─────────────────────────────────────────────────────────


class TestAngleDifference:
    def test_same_angle(self):
        result = angle_difference(np.array([1.0]), np.array([1.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-12)

    def test_wrap_around(self):
        # pi - (-pi) should be 0 (they represent the same angle)
        result = angle_difference(np.array([np.pi]), np.array([-np.pi]))
        np.testing.assert_allclose(result, [0.0], atol=1e-12)

    def test_positive_difference(self):
        result = angle_difference(np.array([np.pi / 2]), np.array([0.0]))
        np.testing.assert_allclose(result, [np.pi / 2], atol=1e-12)

    def test_negative_difference(self):
        result = angle_difference(np.array([0.0]), np.array([np.pi / 2]))
        np.testing.assert_allclose(result, [-np.pi / 2], atol=1e-12)


# ── circle_fit ───────────────────────────────────────────────────────────────


class TestCircleFit:
    def test_known_circle(self):
        # Points on a circle with center (2, 3) and radius 5
        n = 50
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cx, cy, r = 2.0, 3.0, 5.0
        coords = np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])
        result = circle_fit(coords)
        np.testing.assert_allclose(result[0], cx, atol=1e-6)
        np.testing.assert_allclose(result[1], cy, atol=1e-6)
        np.testing.assert_allclose(result[2], r, atol=1e-6)

    def test_unit_circle(self):
        n = 30
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        coords = np.column_stack([np.cos(angles), np.sin(angles)])
        result = circle_fit(coords)
        np.testing.assert_allclose(result[:2], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result[2], 1.0, atol=1e-6)


# ── points_inside_ellipse ────────────────────────────────────────────────────


class TestPointsInsideEllipse:
    def test_center_is_inside(self):
        center = np.array([0.0, 0.0])
        pts = np.array([[0.0, 0.0]])
        result = points_inside_ellipse(pts, center, np.array([1.0, 0.0]), 5.0, 3.0)
        assert result[0]

    def test_outside(self):
        center = np.array([0.0, 0.0])
        pts = np.array([[100.0, 100.0]])
        result = points_inside_ellipse(pts, center, np.array([1.0, 0.0]), 5.0, 3.0)
        assert not result[0]

    def test_along_major(self):
        center = np.array([0.0, 0.0])
        # Just inside the major radius
        pts = np.array([[4.9, 0.0]])
        result = points_inside_ellipse(pts, center, np.array([1.0, 0.0]), 5.0, 3.0)
        assert result[0]

    def test_along_minor_outside(self):
        center = np.array([0.0, 0.0])
        # Outside along minor axis
        pts = np.array([[0.0, 3.1]])
        result = points_inside_ellipse(pts, center, np.array([1.0, 0.0]), 5.0, 3.0)
        assert not result[0]


# ── center_of_circle_from_3_points ───────────────────────────────────────────


class TestCenterOfCircleFrom3Points:
    def test_unit_circle(self):
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.0, 1.0])
        p3 = np.array([-1.0, 0.0])
        center = center_of_circle_from_3_points(p1, p2, p3)
        np.testing.assert_allclose(center, [0.0, 0.0], atol=1e-10)

    def test_collinear_raises(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 1.0])
        p3 = np.array([2.0, 2.0])
        with pytest.raises(ValueError, match="colinear"):
            center_of_circle_from_3_points(p1, p2, p3)


# ── trace_distance_to_next ──────────────────────────────────────────────────


class TestTraceDistanceToNext:
    def test_simple(self):
        trace = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
        result = trace_distance_to_next(trace)
        np.testing.assert_allclose(result, [5.0, 0.0])


# ── trace_angles_between ────────────────────────────────────────────────────


class TestTraceAnglesBetween:
    def test_straight_line(self):
        trace = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        result = trace_angles_between(trace)
        np.testing.assert_allclose(result, [np.pi], atol=1e-12)

    def test_right_angle(self):
        trace = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        result = trace_angles_between(trace)
        np.testing.assert_allclose(result, [np.pi / 2], atol=1e-12)


# ── calculate_radius_from_points ─────────────────────────────────────────────


class TestCalculateRadiusFromPoints:
    def test_unit_circle_points(self):
        # 3 points on a unit circle
        angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        radius = calculate_radius_from_points(pts[np.newaxis])
        np.testing.assert_allclose(radius, [1.0], atol=1e-10)


# ── euler_angles / quaternion roundtrip ──────────────────────────────────────


class TestEulerQuaternionRoundtrip:
    @pytest.mark.parametrize(
        "euler",
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.2, 0.3]),
            np.array([0.0, 0.0, np.pi / 2]),
        ],
    )
    def test_roundtrip(self, euler):
        q = euler_angles_to_quaternion(euler)
        recovered = quaternion_to_euler_angles(q)
        np.testing.assert_allclose(recovered, euler, atol=1e-10)


# ── odd_square ───────────────────────────────────────────────────────────────


class TestOddSquare:
    def test_positive(self):
        np.testing.assert_allclose(odd_square(np.array([3.0])), [9.0])

    def test_negative(self):
        np.testing.assert_allclose(odd_square(np.array([-3.0])), [-9.0])

    def test_zero(self):
        np.testing.assert_allclose(odd_square(np.array([0.0])), [0.0])
