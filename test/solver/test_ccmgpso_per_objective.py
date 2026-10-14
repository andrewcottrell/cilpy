"""Tests for the per-objective penalty swarms and the separation of the
environment-change and multiplier-change triggers."""
import random

import numpy as np

from cilpy.problem.multi_objective import TNK
from cilpy.problem.dynamic_multi_objective import DTNK
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.solver.mgpso import MGPSO
from cilpy.solver.pso import PSO


def _solver(problem, refresh_mode="clean", **kwargs):
    random.seed(1)
    np.random.seed(1)
    return CoevolutionaryLagrangianSolver(
        name="t", problem=problem,
        objective_solver_class=MGPSO, multiplier_solver_class=PSO,
        objective_solver_params={"swarm_size": 20, "w": 0.72, "c1": 1.49,
                                 "c2": 1.49, "c3": 1.49,
                                 "refresh_mode": refresh_mode,
                                 "sentry_mode": "archive"},
        multiplier_solver_params={"swarm_size": 10, "w": 0.4, "c1": 1.2,
                                  "c2": 1.2},
        max_multiplier=1000.0, archive_strategy="strict",
        multiplier_mode="per_objective", multiplier_immigrants=0.3,
        multiplier_steps=5, **kwargs)


def test_one_penalty_swarm_per_objective():
    s = _solver(TNK())
    assert len(s.multiplier_solvers) == 2
    assert len(s.dual_problems) == 2


def test_multipliers_do_not_collapse_to_zero():
    s = _solver(TNK())
    for _ in range(60):
        s.step()
    mus = [max(m.get_result()[0][0]) for m in s.multiplier_solvers]
    assert max(mus) > 1e-3


def test_clear_policy_not_applied_on_multiplier_change():
    # Static problem: no environment change ever happens, so under the clear
    # policy the archive must never be emptied, however often the
    # multipliers change.
    s = _solver(TNK(), refresh_mode="clear")
    sizes = []
    for _ in range(40):
        s.step()
        sizes.append(len(s.objective_solver.get_result()))
    assert min(sizes[5:]) > 0
    assert s.objective_solver.n_changes_detected == 0


def test_environment_change_still_detected():
    problem = DTNK(tau_t=5, n_t=10)
    s = _solver(problem)
    for _ in range(30):
        problem.begin_iteration()
        s.step()
    assert s.objective_solver.n_changes_detected >= 4


def _count_evaluations(problem):
    """Wraps problem.evaluate and returns a one-element list holding the
    number of calls."""
    count = [0]
    evaluate = problem.evaluate

    def counted(solution):
        count[0] += 1
        return evaluate(solution)

    problem.evaluate = counted
    return count


def _trace(problem, reuse, iterations=100):
    """Archive positions and multipliers after every iteration, and the
    number of evaluations of the original problem."""
    count = _count_evaluations(problem)
    s = _solver(problem, reuse_evaluations=reuse)
    trace = []
    for _ in range(iterations):
        if hasattr(problem, "begin_iteration"):
            problem.begin_iteration()
        s.step()
        archive = [list(x) for x, _ in s.objective_solver.get_result()]
        multipliers = [m.get_result()[0][0] for m in s.multiplier_solvers]
        trace.append((archive, multipliers))
    return trace, count[0]


def test_reuse_evaluations_changes_nothing_but_the_count_static():
    from cilpy.problem.multi_objective import CONSTR
    off, n_off = _trace(CONSTR(), reuse=False)
    on, n_on = _trace(CONSTR(), reuse=True)
    assert on == off
    assert n_on < n_off / 2


def test_reuse_evaluations_changes_nothing_but_the_count_dynamic():
    off, n_off = _trace(DTNK(tau_t=10, n_t=10), reuse=False)
    on, n_on = _trace(DTNK(tau_t=10, n_t=10), reuse=True)
    assert on == off
    assert n_on < n_off / 2


def test_reuse_evaluations_matches_mgpso_budget():
    # With the cache, CCMGPSO evaluates only what MGPSO evaluates: the
    # particles, the archive sentries and the change response.
    problem = DTNK(tau_t=10, n_t=10)
    count = _count_evaluations(problem)
    s = _solver(problem, reuse_evaluations=True)
    for _ in range(100):
        problem.begin_iteration()
        s.step()
    n_cc = count[0]

    random.seed(1)
    np.random.seed(1)
    problem = DTNK(tau_t=10, n_t=10)
    count = _count_evaluations(problem)
    m = MGPSO(name="m", problem=problem, swarm_size=20, w=0.72, c1=1.49, c2=1.49,
              c3=1.49, feasible_archive_only=True, sentry_mode="archive")
    for _ in range(100):
        problem.begin_iteration()
        m.step()
    n_mgpso = count[0]
    # The archive sizes differ between the two runs, so the sentry counts
    # differ a little; the reference sets and re-scoring add nothing.
    assert n_cc < 1.5 * n_mgpso
