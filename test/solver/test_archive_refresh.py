# test/solver/test_archive_refresh.py
"""Tests for in-place archive refresh (rescore + prune)."""

import numpy as np
import pytest

from cilpy.problem import Evaluation
from cilpy.problem.multi_objective import SCH1, TNK
from cilpy.problem.dynamic_multi_objective import DTNK, FDA1
from cilpy.solver.mgpso import MGPSO, _Archive


def _ev(fitness, g=None):
    return Evaluation(fitness=list(fitness), constraints_inequality=g)


def _fill(archive, points, gs=None):
    for i, p in enumerate(points):
        g = None if gs is None else gs[i]
        archive.positions.append(np.array([float(i)]))
        archive.evaluations.append(_ev(p, g))
    archive._invalidate()


class TestPrune:
    def test_removes_dominated_only(self):
        a = _Archive(capacity=10)
        _fill(a, [[1.0, 1.0], [2.0, 2.0], [0.5, 3.0]])
        n_inf, n_dom = a.prune()
        assert (n_inf, n_dom) == (0, 1)
        fits = [e.fitness for e in a.evaluations]
        assert [2.0, 2.0] not in fits
        assert len(a) == 2

    def test_keeps_all_mutually_nondominated(self):
        a = _Archive(capacity=10)
        _fill(a, [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        assert a.prune() == (0, 0)
        assert len(a) == 3

    def test_duplicates_both_retained(self):
        """Equal objective vectors do not dominate each other."""
        a = _Archive(capacity=10)
        _fill(a, [[1.0, 1.0], [1.0, 1.0]])
        assert a.prune() == (0, 0)
        assert len(a) == 2

    def test_removes_infeasible_when_strict(self):
        a = _Archive(capacity=10, feasible_only=True)
        _fill(a, [[1.0, 3.0], [2.0, 2.0]], gs=[[-1.0], [0.5]])
        n_inf, n_dom = a.prune()
        assert n_inf == 1 and len(a) == 1
        assert a.evaluations[0].fitness == [1.0, 3.0]

    def test_keeps_infeasible_when_filter(self):
        a = _Archive(capacity=10, feasible_only=False)
        _fill(a, [[1.0, 3.0], [2.0, 2.0]], gs=[[-1.0], [0.5]])
        assert a.prune() == (0, 0)
        assert len(a) == 2

    def test_empty_archive(self):
        assert _Archive(capacity=5).prune() == (0, 0)

    def test_positions_preserved_for_survivors(self):
        a = _Archive(capacity=10)
        _fill(a, [[1.0, 1.0], [2.0, 2.0], [0.5, 3.0]])
        before = {tuple(p) for p, e in zip(a.positions, a.evaluations)
                  if e.fitness != [2.0, 2.0]}
        a.prune()
        after = {tuple(p) for p in a.positions}
        assert before == after


class TestRescore:
    def test_positions_unchanged_evaluations_replaced(self):
        a = _Archive(capacity=10)
        _fill(a, [[1.0, 1.0], [2.0, 0.5]])
        positions_before = [p.copy() for p in a.positions]
        a.rescore([_ev([9.0, 9.0]), _ev([8.0, 8.0])])
        assert all(np.array_equal(x, y)
                   for x, y in zip(positions_before, a.positions))
        assert [e.fitness for e in a.evaluations] == [[9.0, 9.0], [8.0, 8.0]]

    def test_cache_invalidated(self):
        a = _Archive(capacity=10)
        _fill(a, [[1.0, 3.0], [3.0, 1.0]])
        _ = a._matrix()
        a.rescore([_ev([5.0, 7.0]), _ev([7.0, 5.0])])
        assert a._matrix()[0].tolist() == [5.0, 7.0]


class TestRefreshModeClear:
    def test_default_is_clean(self):
        solver = MGPSO(problem=SCH1(), name="M", swarm_size=10)
        assert solver.refresh_mode == "clean"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            MGPSO(problem=SCH1(), name="M", swarm_size=10, refresh_mode="wipe")

    def test_clear_empties_archive(self):
        np.random.seed(0)
        problem = DTNK(tau_t=1, n_t=10)
        solver = MGPSO(problem=problem, name="M", swarm_size=20,
                       refresh_mode="clear")
        for _ in range(5):
            problem.begin_iteration()
            solver.step()
        assert len(solver._archive) > 0
        solver._refresh_archive()
        assert len(solver._archive) == 0

    def test_clean_retains_survivors_clear_does_not(self):
        """Same trajectory up to a change: clean keeps feasible/non-dominated
        members, clear discards everything regardless."""
        for mode, expect_survivors in [("clean", True), ("clear", False)]:
            np.random.seed(2)
            problem = DTNK(tau_t=10, n_t=10, )
            solver = MGPSO(problem=problem, name="M", swarm_size=30,
                           refresh_mode=mode, feasible_archive_only=True)
            for _ in range(29):
                problem.begin_iteration()
                solver.step()
            assert len(solver._archive) > 0
            problem.begin_iteration()
            solver._refresh_archive()
            if expect_survivors:
                assert len(solver._archive) > 0
            else:
                assert len(solver._archive) == 0

    def test_clear_archive_repopulates_next_step(self):
        np.random.seed(0)
        problem = DTNK(tau_t=1, n_t=10)
        solver = MGPSO(problem=problem, name="M", swarm_size=20,
                       refresh_mode="clear")
        for _ in range(5):
            problem.begin_iteration()
            solver.step()
        problem.begin_iteration()
        solver.step()   # triggers refresh (clear) then repopulates within step
        assert len(solver._archive) > 0


class TestRefreshIntegration:
    def test_archive_never_grows_and_stays_nondominated(self):
        np.random.seed(0)
        problem = DTNK(tau_t=5, n_t=10)
        solver = MGPSO(problem=problem, name="M", swarm_size=20)
        for _ in range(40):
            problem.begin_iteration()
            solver.step()
            fits = [e.fitness for e in solver._archive.evaluations]
            for i, fa in enumerate(fits):
                for j, fb in enumerate(fits):
                    if i != j:
                        assert not (all(x <= y for x, y in zip(fb, fa))
                                    and any(x < y for x, y in zip(fb, fa)))
        assert len(solver._archive) <= solver._archive.capacity

    def test_strict_archive_stays_feasible_across_changes(self):
        np.random.seed(0)
        problem = DTNK(tau_t=5, n_t=10)
        solver = MGPSO(problem=problem, name="M", swarm_size=20,
                       feasible_archive_only=True)
        for _ in range(40):
            problem.begin_iteration()
            solver.step()
            for e in solver._archive.evaluations:
                assert all(g <= 0 for g in e.constraints_inequality)

    def test_refresh_preserves_survivors_not_rebuild(self):
        """A member that stays feasible and non-dominated across a change
        must remain in the archive, at the same position."""
        np.random.seed(3)
        problem = DTNK(tau_t=1, n_t=10)
        solver = MGPSO(problem=problem, name="M", swarm_size=20)
        for _ in range(10):
            problem.begin_iteration()
            solver.step()
        before = {tuple(np.round(p, 12)) for p in solver._archive.positions}
        problem.begin_iteration()
        solver._refresh_archive()
        after = {tuple(np.round(p, 12)) for p in solver._archive.positions}
        assert after <= before          # only removals, never additions
        assert len(after) > 0

    def test_tracking_quality_maintained(self):
        from cilpy.compare.metrics import inverted_generational_distance
        np.random.seed(4)
        problem = FDA1(dimension=10, tau_t=20, n_t=10)
        solver = MGPSO(problem=problem, name="M", swarm_size=30)
        for _ in range(120):
            problem.begin_iteration()
            solver.step()
        front = [ev.fitness for _, ev in solver.get_result()]
        igd = inverted_generational_distance(front, problem.true_pareto_front(200))
        assert igd < 0.3, f"IGD after changes too high: {igd}"
