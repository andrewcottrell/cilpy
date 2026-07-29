# test/solver/test_ccpso_mo.py
"""Integration tests for the CoevolutionaryLagrangianSolver wrapping MGPSO.

These tests validate Milestone 4: constrained multi-objective optimization
via the co-evolutionary Lagrangian framework with MGPSO as the objective
solver, under both archive management strategies.
"""

import numpy as np
import pytest

from cilpy.problem.multi_objective import CONSTR, BNH, SRN
from cilpy.solver.mgpso import MGPSO
from cilpy.solver.pso import PSO
from cilpy.solver.ccls import (
    CoevolutionaryLagrangianSolver,
    _total_violation,
    _LagrangianMinProblem,
)
from cilpy.problem.constrained import G01


def _make_ccpso(problem, strategy, swarm_size=20):
    return CoevolutionaryLagrangianSolver(
        name="CCPSO_MO",
        problem=problem,
        objective_solver_class=MGPSO,
        multiplier_solver_class=PSO,
        objective_solver_params={"swarm_size": swarm_size},
        multiplier_solver_params={
            "swarm_size": 30, "w": 0.40, "c1": 1.20, "c2": 1.20,
        },
        penalty_rho=1.0,
        max_multiplier=1000.0,
        archive_strategy=strategy,
    )


class TestMinProxyMO:
    def test_mo_proxy_returns_vector_fitness(self):
        proxy = _LagrangianMinProblem(CONSTR(), penalty_rho=1.0)
        ev = proxy.evaluate([0.5, 1.0])
        assert isinstance(ev.fitness, list)
        assert len(ev.fitness) == 2

    def test_penalty_added_equally_to_all_objectives(self):
        problem = CONSTR()
        proxy = _LagrangianMinProblem(problem, penalty_rho=1.0)
        proxy.set_fixed_multipliers([2.0, 2.0], [])
        x = [0.15, 0.0]  # infeasible: g1 = 6 - 0 - 1.35 > 0
        raw = problem.evaluate(x)
        pen = proxy.evaluate(x)
        deltas = [p - r for p, r in zip(pen.fitness, raw.fitness)]
        assert deltas[0] == pytest.approx(deltas[1])
        assert deltas[0] > 0  # infeasible -> positive penalty

    def test_feasible_solution_penalty_matches_lagrangian_slack(self):
        problem = CONSTR()
        proxy = _LagrangianMinProblem(problem, penalty_rho=1.0)
        proxy.set_fixed_multipliers([0.0, 0.0], [])
        x = [1.0, 0.0]  # feasible, zero multipliers -> no penalty
        raw = problem.evaluate(x)
        pen = proxy.evaluate(x)
        assert pen.fitness == pytest.approx(raw.fitness)

    def test_mo_proxy_reports_objective_dynamic(self):
        proxy = _LagrangianMinProblem(CONSTR(), penalty_rho=1.0)
        assert proxy.is_dynamic()[0] is True

    def test_so_proxy_dynamism_unchanged(self):
        proxy = _LagrangianMinProblem(G01(), penalty_rho=1.0)
        assert proxy.is_dynamic() == G01().is_dynamic()

    def test_constraints_passed_through(self):
        proxy = _LagrangianMinProblem(CONSTR(), penalty_rho=1.0)
        ev = proxy.evaluate([0.15, 0.0])
        assert ev.constraints_inequality is not None
        assert len(ev.constraints_inequality) == 2


class TestCCPSOIntegration:
    @pytest.mark.parametrize("strategy", ["filter", "strict"])
    def test_runs_and_returns_front(self, strategy):
        np.random.seed(3)
        solver = _make_ccpso(CONSTR(), strategy)
        for _ in range(30):
            solver.step()
        result = solver.get_result()
        assert len(result) >= 1
        # results are re-evaluated on the original problem: fitness is the
        # TRUE objective vector (no penalty), constraints present
        for solution, ev in result:
            fresh = CONSTR().evaluate(solution)
            assert np.allclose(fresh.fitness, ev.fitness)

    def test_strict_archive_contains_only_feasible(self):
        np.random.seed(3)
        solver = _make_ccpso(CONSTR(), "strict")
        for _ in range(30):
            solver.step()
        # internal archive of the MGPSO must be all-feasible at all times
        for _, ev in solver.objective_solver.get_result():
            assert _total_violation(ev) == 0.0

    def test_filter_mode_final_front_feasible(self):
        np.random.seed(3)
        solver = _make_ccpso(CONSTR(), "filter")
        for _ in range(60):
            solver.step()
        result = solver.get_result()
        feasibility = [
            _total_violation(ev) == 0.0 for _, ev in result
        ]
        # after 60 co-evolutionary steps CONSTR should have a feasible front
        assert all(feasibility)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            _make_ccpso(CONSTR(), "banana")

    def test_single_objective_path_unaffected(self):
        np.random.seed(3)
        solver = CoevolutionaryLagrangianSolver(
            name="CCPSO",
            problem=G01(),
            objective_solver_class=PSO,
            multiplier_solver_class=PSO,
            objective_solver_params={
                "swarm_size": 20, "w": 0.72, "c1": 1.49, "c2": 1.49,
            },
            multiplier_solver_params={
                "swarm_size": 20, "w": 0.40, "c1": 1.20, "c2": 1.20,
            },
            penalty_rho=0.5,
            max_multiplier=10000.0,
        )
        for _ in range(10):
            solver.step()
        result = solver.get_result()
        assert len(result) == 1
        assert isinstance(result[0][1].fitness, float)


class TestAnchorSelection:
    def test_anchor_is_most_violating_archive_member(self):
        np.random.seed(3)
        solver = _make_ccpso(BNH(), "filter")
        solver.step()
        anchor = solver._select_multiplier_anchor()
        archive = solver.objective_solver.get_result()
        violations = [_total_violation(ev) for _, ev in archive]
        anchor_eval = BNH().evaluate(anchor)
        assert _total_violation(anchor_eval) == pytest.approx(
            max(violations), abs=1e-9
        )
