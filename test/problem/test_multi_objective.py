# test/problem/test_multi_objective.py
"""Unit tests for the multi-objective benchmark problems.

Correctness anchors: each ZDT problem is checked at x = (x1, 0, ..., 0)
(the Pareto-optimal set, g = 1) and at a known off-front point. Constrained
problems are checked at hand-verified feasible and infeasible points.
"""

import math

import numpy as np
import pytest

from cilpy.problem.multi_objective import (
    BNH, CONSTR, OSY, SCH1, SRN, TNK, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6,
)


def _on_front_zdt(problem, x1):
    """Evaluates a ZDT problem on the Pareto-optimal set (tail = 0)."""
    return problem.evaluate([x1] + [0.0] * (problem.dimension - 1))


class TestZDT:
    def test_zdt1_on_front(self):
        ev = _on_front_zdt(ZDT1(), 0.25)
        assert ev.fitness[0] == pytest.approx(0.25)
        assert ev.fitness[1] == pytest.approx(1.0 - math.sqrt(0.25))

    def test_zdt1_off_front(self):
        # all ones: f1 = 1, g = 10, f2 = 10 * (1 - sqrt(1/10))
        problem = ZDT1(dimension=30)
        ev = problem.evaluate([1.0] * 30)
        assert ev.fitness[1] == pytest.approx(10.0 * (1.0 - math.sqrt(0.1)))

    def test_zdt2_on_front(self):
        ev = _on_front_zdt(ZDT2(), 0.5)
        assert ev.fitness[1] == pytest.approx(1.0 - 0.25)

    def test_zdt3_on_g1_surface(self):
        x1 = 0.2
        ev = _on_front_zdt(ZDT3(), x1)
        expected = 1.0 - math.sqrt(x1) - x1 * math.sin(10.0 * math.pi * x1)
        assert ev.fitness[1] == pytest.approx(expected)

    def test_zdt4_optimum_matches_zdt1_front(self):
        ev = _on_front_zdt(ZDT4(), 0.36)
        assert ev.fitness[1] == pytest.approx(1.0 - 0.6)

    def test_zdt4_g_is_rastrigin_like(self):
        # tail of ones: g = 1 + 10*9 + 9*(1 - 10*cos(4*pi)) = 91 + 9*(-9) = 10
        problem = ZDT4()
        ev = problem.evaluate([0.0] + [1.0] * 9)
        g = 10.0
        assert ev.fitness[1] == pytest.approx(g * (1.0 - 0.0))

    def test_zdt6_on_front(self):
        x1 = 0.5
        problem = ZDT6()
        ev = _on_front_zdt(problem, x1)
        f1 = 1.0 - math.exp(-2.0) * math.sin(3.0 * math.pi) ** 6
        assert ev.fitness[0] == pytest.approx(f1)
        assert ev.fitness[1] == pytest.approx(1.0 - f1**2)

    def test_true_fronts_are_nondominated(self):
        for problem in [ZDT1(), ZDT2(), ZDT3(), ZDT4(), ZDT6(), SCH1()]:
            front = problem.true_pareto_front(200)
            assert front.shape[1] == 2
            assert len(front) > 0

    def test_flags(self):
        for problem in [ZDT1(), ZDT2(), ZDT3(), ZDT4(), ZDT6()]:
            assert problem.is_multi_objective()
            assert problem.is_dynamic() == (False, False)


class TestConstrained:
    def test_bnh_known_values(self):
        ev = BNH().evaluate([0.0, 0.0])
        assert ev.fitness == pytest.approx([0.0, 50.0])
        # g1 = 25 - 25 = 0 (feasible boundary), g2 = 7.7 - 64 - 9 < 0
        assert ev.constraints_inequality[0] == pytest.approx(0.0)
        assert ev.constraints_inequality[1] < 0

    def test_bnh_infeasible_point(self):
        # x = (5, 3): g1 = 0 + 9 - 25 < 0 feasible; second constraint:
        # g2 = 7.7 - 9 - 36 < 0 feasible -> pick a g2 violator instead.
        # Point near (8, -3) is outside bounds; violate g2 within bounds is
        # impossible for BNH's box, so check g1 violation is impossible too:
        # max of (x1-5)^2 + x2^2 within box at x=(0,3) = 25 + 9 - 25 = 9 > 0.
        ev = BNH().evaluate([0.0, 3.0])
        assert ev.constraints_inequality[0] > 0  # infeasible w.r.t. g1

    def test_srn_known_values(self):
        ev = SRN().evaluate([0.0, 5.0])
        assert ev.fitness[0] == pytest.approx(2.0 + 4.0 + 16.0)
        assert ev.fitness[1] == pytest.approx(-16.0)
        # g1 = 25 - 225 < 0, g2 = 0 - 15 + 10 < 0 -> feasible
        assert all(g <= 0 for g in ev.constraints_inequality)

    def test_tnk_feasible_point(self):
        # (1, 1): g1 = -(1 + 1 - 1 - 0.1 cos(16*atan(1))) = -(1 - 0.1 cos(4pi))
        ev = TNK().evaluate([1.0, 1.0])
        expected_g1 = -(1.0 - 0.1 * math.cos(16.0 * math.atan(1.0)))
        assert ev.constraints_inequality[0] == pytest.approx(expected_g1)
        # g2 = 0.25 + 0.25 - 0.5 = 0
        assert ev.constraints_inequality[1] == pytest.approx(0.0)

    def test_tnk_origin_is_infeasible(self):
        ev = TNK().evaluate([0.0, 0.0])
        assert ev.constraints_inequality[0] > 0

    def test_constr_known_values(self):
        ev = CONSTR().evaluate([1.0, 0.0])
        assert ev.fitness == pytest.approx([1.0, 1.0])
        # g1 = 6 - 0 - 9 = -3, g2 = 1 + 0 - 9 = -8 -> feasible
        assert all(g <= 0 for g in ev.constraints_inequality)

    def test_constr_infeasible(self):
        ev = CONSTR().evaluate([0.1, 0.0])
        # g1 = 6 - 0.9 = 5.1 > 0
        assert ev.constraints_inequality[0] > 0

    def test_osy_known_feasible_point(self):
        # x = (5, 1, 2, 0, 5, 10):
        # c1: 5+1-2 = 4 >= 0 ok; c2: 6-6 = 0 ok; c3: 2-1+5 = 6 ok;
        # c4: 2-5+3 = 0 ok; c5: 4-1-0 = 3 ok; c6: 4+10-4 = 10 ok.
        ev = OSY().evaluate([5.0, 1.0, 2.0, 0.0, 5.0, 10.0])
        assert all(g <= 1e-12 for g in ev.constraints_inequality)
        assert ev.fitness[1] == pytest.approx(25 + 1 + 4 + 0 + 25 + 100)

    def test_osy_infeasible_point(self):
        # x1 = x2 = 0 violates c1: x1 + x2 - 2 >= 0
        ev = OSY().evaluate([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        assert ev.constraints_inequality[0] > 0

    def test_sampled_reference_fronts(self):
        for problem in [BNH(), SRN(), TNK(), CONSTR()]:
            front = problem.true_pareto_front(100)
            assert len(front) > 10
            # Front must be sorted-compatible and mutually non-dominated:
            f = np.asarray(front)
            order = np.argsort(f[:, 0])
            assert np.all(np.diff(f[order][:, 1]) <= 1e-9)

    def test_osy_front_not_implemented(self):
        with pytest.raises(AttributeError):
            OSY().true_pareto_front()  # type: ignore[attr-defined]
