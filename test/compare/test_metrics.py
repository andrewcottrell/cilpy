# test/compare/test_metrics.py
"""Unit tests for cilpy.compare.metrics with hand-verifiable values."""

import math

import numpy as np
import pytest

from cilpy.problem import Evaluation
from cilpy.compare.metrics import (
    feasibility_rate,
    generational_distance,
    hypervolume,
    hypervolume_ratio,
    inverted_generational_distance,
    nondominated_filter,
    p_red,
    spacing,
    spread,
)


class TestNondominatedFilter:
    def test_removes_dominated(self):
        front = [[1.0, 1.0], [2.0, 2.0], [0.5, 3.0]]
        result = nondominated_filter(front)
        assert [2.0, 2.0] not in result.tolist()
        assert len(result) == 2

    def test_keeps_incomparable(self):
        front = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]
        assert len(nondominated_filter(front)) == 3


class TestDistances:
    def test_gd_zero_when_on_reference(self):
        reference = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        assert generational_distance(reference, reference) == pytest.approx(0.0)

    def test_igd_zero_when_covering_reference(self):
        reference = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        assert inverted_generational_distance(reference, reference) == pytest.approx(0.0)

    def test_gd_known_value(self):
        # One point at distance exactly 1 from the nearest reference point.
        assert generational_distance([[1.0, 1.0]], [[1.0, 0.0], [5.0, 5.0]]) \
            == pytest.approx(1.0)

    def test_igd_detects_coverage_gap_gd_does_not(self):
        reference = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        partial = [[0.0, 1.0]]  # on the front, but covers one point only
        assert generational_distance(partial, reference) == pytest.approx(0.0)
        assert inverted_generational_distance(partial, reference) > 0.3


class TestHypervolume:
    def test_single_point(self):
        # Point (1,1), reference (2,2): dominated area = 1*1 = 1.
        assert hypervolume([[1.0, 1.0]], [2.0, 2.0]) == pytest.approx(1.0)

    def test_two_points(self):
        # (0,1) and (1,0) with reference (2,2):
        # sorted by f1: (0,1) contributes (2-0)*(2-1) = 2;
        # (1,0) contributes (2-1)*(1-0) = 1; total 3.
        assert hypervolume([[0.0, 1.0], [1.0, 0.0]], [2.0, 2.0]) == pytest.approx(3.0)

    def test_dominated_point_adds_nothing(self):
        base = hypervolume([[0.0, 1.0], [1.0, 0.0]], [2.0, 2.0])
        with_dominated = hypervolume(
            [[0.0, 1.0], [1.0, 0.0], [1.5, 1.5]], [2.0, 2.0]
        )
        assert with_dominated == pytest.approx(base)

    def test_point_outside_reference_ignored(self):
        assert hypervolume([[3.0, 3.0]], [2.0, 2.0]) == pytest.approx(0.0)

    def test_better_front_has_larger_hv(self):
        worse = hypervolume([[1.0, 1.0]], [2.0, 2.0])
        better = hypervolume([[0.5, 0.5]], [2.0, 2.0])
        assert better > worse

    def test_three_objectives_not_implemented(self):
        with pytest.raises(NotImplementedError):
            hypervolume([[1.0, 1.0, 1.0]], [2.0, 2.0, 2.0])


class TestSpacingAndSpread:
    def test_spacing_zero_for_uniform_front(self):
        front = [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]
        assert spacing(front) == pytest.approx(0.0)

    def test_spacing_positive_for_nonuniform_front(self):
        front = [[0.0, 3.0], [0.1, 2.9], [3.0, 0.0]]
        assert spacing(front) > 0

    def test_spread_zero_for_perfect_front(self):
        # Obtained front == uniformly spaced reference with same extremes.
        reference = [[float(i), 3.0 - i] for i in range(4)]
        assert spread(reference, reference) == pytest.approx(0.0)

    def test_spread_penalizes_missing_extent(self):
        reference = [[float(i), 10.0 - i] for i in range(11)]
        clustered = [[4.0, 6.0], [5.0, 5.0], [6.0, 4.0]]
        assert spread(clustered, reference) > 0.5


class TestFeasibilityRate:
    def test_all_feasible(self):
        evals = [Evaluation(fitness=[1.0], constraints_inequality=[-1.0])] * 4
        assert feasibility_rate(evals) == pytest.approx(100.0)

    def test_half_feasible(self):
        evals = [
            Evaluation(fitness=[1.0], constraints_inequality=[-1.0]),
            Evaluation(fitness=[1.0], constraints_inequality=[0.5]),
        ]
        assert feasibility_rate(evals) == pytest.approx(50.0)

    def test_equality_tolerance(self):
        evals = [
            Evaluation(fitness=[1.0], constraints_equality=[1e-8]),
            Evaluation(fitness=[1.0], constraints_equality=[0.1]),
        ]
        assert feasibility_rate(evals) == pytest.approx(50.0)

    def test_unconstrained_counts_feasible(self):
        assert feasibility_rate([Evaluation(fitness=[1.0])]) == pytest.approx(100.0)

    def test_empty(self):
        assert feasibility_rate([]) == pytest.approx(0.0)


# --- hypervolume_ratio and p_red ---

REF = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])


class TestHypervolumeRatio:
    def test_ratio_is_one_for_the_reference_front_itself(self):
        assert hypervolume_ratio(REF, REF) == pytest.approx(1.0)

    def test_ratio_is_zero_for_empty_front(self):
        assert hypervolume_ratio([], REF) == 0.0

    def test_ratio_is_zero_when_front_is_past_reference_point(self):
        assert hypervolume_ratio([[100.0, 100.0]], REF) == 0.0

    def test_ratio_never_goes_above_one(self):
        better = REF - 0.05
        assert hypervolume_ratio(better, REF) == 1.0

    def test_ratio_ignores_scale_of_an_objective(self):
        front = np.array([[0.2, 0.9], [0.6, 0.6]])
        scaled_front = front * [100.0, 1.0]
        scaled_ref = REF * [100.0, 1.0]
        assert hypervolume_ratio(front, REF) == pytest.approx(
            hypervolume_ratio(scaled_front, scaled_ref)
        )


class TestPRed:
    def test_zero_when_always_optimal(self):
        assert p_red([1.0, 1.0, 1.0]) == 0.0

    def test_one_when_always_worst(self):
        assert p_red([0.0, 0.0]) == pytest.approx(1.0)

    def test_matches_worked_example(self):
        assert p_red([0.5, 0.75, 0.9, 1.0]) == pytest.approx(0.2839, abs=1e-4)
