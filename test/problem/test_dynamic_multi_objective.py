# test/problem/test_dynamic_multi_objective.py
"""Tests for the dynamic multi-objective problems and change response."""

import math

import numpy as np
import pytest

from cilpy.problem.dynamic_multi_objective import FDA1, FDA3, DTNK, DTNK2
from cilpy.solver.mgpso import MGPSO


def advance(problem, iterations):
    for _ in range(iterations):
        problem.begin_iteration()


class TestTimeMechanics:
    def test_time_steps_with_tau(self):
        p = FDA1(tau_t=10, n_t=10)
        assert p.t == 0.0
        advance(p, 9)
        assert p.t == 0.0          # still first environment
        advance(p, 1)
        assert p.t == pytest.approx(0.1)   # change after tau_t iterations
        advance(p, 10)
        assert p.t == pytest.approx(0.2)

    def test_reset_time(self):
        p = FDA1()
        advance(p, 25)
        p.reset_time()
        assert p.t == 0.0

    def test_dynamism_flags(self):
        assert FDA1().is_dynamic() == (True, False)
        assert FDA3().is_dynamic() == (True, False)
        assert DTNK().is_dynamic() == (False, True)
        assert DTNK2().is_dynamic() == (True, True)


class TestFDA1:
    def test_pareto_set_at_t0(self):
        p = FDA1(dimension=5)
        # At t=0, G=0: tail of zeros gives g=1, on-front point.
        ev = p.evaluate([0.25, 0.0, 0.0, 0.0, 0.0])
        assert ev.fitness[1] == pytest.approx(1.0 - math.sqrt(0.25))

    def test_pareto_set_moves(self):
        p = FDA1(dimension=5, tau_t=1, n_t=2)
        advance(p, 1)          # t = 0.5, G = sin(pi/4)
        G = math.sin(0.25 * math.pi)
        on_front = p.evaluate([0.25, G, G, G, G])
        stale = p.evaluate([0.25, 0.0, 0.0, 0.0, 0.0])
        assert on_front.fitness[1] == pytest.approx(1.0 - math.sqrt(0.25))
        assert stale.fitness[1] > on_front.fitness[1]  # old set now off-front

    def test_front_is_time_invariant(self):
        p = FDA1()
        f_before = p.true_pareto_front(100)
        advance(p, 50)
        f_after = p.true_pareto_front(100)
        assert np.allclose(f_before, f_after)


class TestFDA3:
    def test_front_moves_with_time(self):
        p = FDA3(tau_t=1, n_t=2)
        f0 = p.true_pareto_front(100)
        advance(p, 1)          # G jumps to |sin(pi/4)| > 0
        f1 = p.true_pareto_front(100)
        assert not np.allclose(f0, f1)
        # front moved up: f2 at f1=0 equals 1+G
        assert f1[0, 1] == pytest.approx(
            1.0 + abs(math.sin(0.25 * math.pi))
        )

    def test_on_front_at_t0(self):
        p = FDA3(dimension=5)
        ev = p.evaluate([0.49, 0.0, 0.0, 0.0, 0.0])   # F(0)=1, G(0)=0
        assert ev.fitness[0] == pytest.approx(0.49)
        assert ev.fitness[1] == pytest.approx(1.0 - math.sqrt(0.49))


class TestDTNK:
    def test_constraint_changes_objectives_do_not(self):
        p = DTNK(tau_t=1, n_t=2)
        x = [1.0, 1.0]
        before = p.evaluate(x)
        advance(p, 1)
        after = p.evaluate(x)
        assert before.fitness == pytest.approx(after.fitness)  # SO static
        assert before.constraints_inequality[0] != pytest.approx(
            after.constraints_inequality[0]
        )

    def test_front_cache_bounded_and_correct(self):
        p = DTNK(tau_t=1, n_t=2)
        f0 = p.true_pareto_front(50)
        advance(p, 1)
        f1 = p.true_pareto_front(50)
        assert not np.allclose(f0[:, :], f1[: len(f0), :]) or len(f0) != len(f1)
        # returning to an earlier environment reuses the cache
        p.reset_time()
        f0_again = p.true_pareto_front(50)
        assert np.allclose(f0, f0_again)
        assert len(p._front_cache) == 2


class TestDTNK2:
    def test_both_change(self):
        p = DTNK2(tau_t=1, n_t=2)
        x = [1.0, 1.0]
        before = p.evaluate(x)
        advance(p, 1)
        after = p.evaluate(x)
        assert before.fitness != pytest.approx(after.fitness)
        assert before.constraints_inequality[0] != pytest.approx(
            after.constraints_inequality[0]
        )


class TestChangeResponse:
    def test_sentinel_detects_change(self):
        p = FDA1(dimension=5, tau_t=1, n_t=2)
        s = MGPSO(problem=p, name="M", swarm_size=5)
        assert not s._environment_changed()   # nothing moved yet
        p.begin_iteration()                    # t changes
        assert s._environment_changed()
        assert not s._environment_changed()   # sentinel updated, stable now

    def test_pbests_rescored_after_change(self):
        np.random.seed(0)
        p = FDA1(dimension=5, tau_t=1, n_t=2)
        s = MGPSO(problem=p, name="M", swarm_size=5)
        s.step()
        old_pbest_fitness = [
            part.y_eval.fitness
            for sw in s._swarms for part in sw.particles
        ]
        p.begin_iteration()   # environment change
        s.step()              # triggers response: pbests re-evaluated
        new_pbest_fitness = [
            part.y_eval.fitness
            for sw in s._swarms for part in sw.particles
        ]
        assert any(
            o != pytest.approx(n)
            for o, n in zip(old_pbest_fitness, new_pbest_fitness)
        )

    def test_tracking_after_changes(self):
        """After several environment changes, the swarm should still hold a
        near-front archive (change response prevents collapse)."""
        np.random.seed(4)
        from cilpy.compare.metrics import inverted_generational_distance
        p = FDA1(dimension=10, tau_t=20, n_t=10)
        s = MGPSO(problem=p, name="M", swarm_size=30)
        for _ in range(120):   # 6 environments
            p.begin_iteration()
            s.step()
        front = [ev.fitness for _, ev in s.get_result()]
        igd = inverted_generational_distance(front, p.true_pareto_front(200))
        assert igd < 0.3, f"IGD after changes too high: {igd}"


class TestDTNK3:
    def test_dosc_flags(self):
        from cilpy.problem.dynamic_multi_objective import DTNK3
        assert DTNK3().is_dynamic() == (True, False)

    def test_objectives_change_constraints_do_not(self):
        from cilpy.problem.dynamic_multi_objective import DTNK3
        p = DTNK3(tau_t=1, n_t=2)
        x = [1.0, 1.0]
        before = p.evaluate(x)
        advance(p, 1)
        after = p.evaluate(x)
        assert before.fitness != pytest.approx(after.fitness)
        assert before.constraints_inequality == pytest.approx(
            after.constraints_inequality
        )

    def test_front_translates(self):
        from cilpy.problem.dynamic_multi_objective import DTNK3
        p = DTNK3(tau_t=1, n_t=2)
        f0 = p.true_pareto_front(50)
        advance(p, 1)
        f1 = p.true_pareto_front(50)
        assert not np.allclose(f0, f1[: len(f0)]) or len(f0) != len(f1)
