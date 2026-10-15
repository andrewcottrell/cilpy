# test/solver/test_pso.py
"""Tests for the PSO's sentry-based change detection (`detect_changes`)."""

import random

import numpy as np

from cilpy.problem import Evaluation, Problem
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.solver.pso import PSO


class ShiftingSphere(Problem):
    """Constrained sphere whose centre and constraint shift every `period`
    calls of `begin_iteration`. With `period=None` it never changes but still
    reports itself as dynamic."""

    def __init__(self, period=None, dynamic=True):
        super().__init__(2, ([-5.0, -5.0], [5.0, 5.0]), "ShiftingSphere")
        self.period = period
        self.dynamic = dynamic
        self.iteration = 0
        self.shift = 0.0

    def begin_iteration(self):
        self.iteration += 1
        if self.period and self.iteration % self.period == 0:
            self.shift += 1.0

    def evaluate(self, solution):
        x = np.asarray(solution, dtype=float)
        f = float(np.sum((x - self.shift) ** 2))
        g = float(x[0] - 1.0 - self.shift)  # x0 <= 1 + shift
        return Evaluation(fitness=f, constraints_inequality=[g])

    def get_fitness_bounds(self):
        return (0.0, 100.0)

    def is_dynamic(self):
        return (self.dynamic, self.dynamic)

    def is_multi_objective(self):
        return False


def _pso(problem, detect):
    return PSO(problem, "PSO", swarm_size=10, w=0.72, c1=1.49, c2=1.49,
               detect_changes=detect)


def _run(solver, problem, iterations):
    for _ in range(iterations):
        problem.begin_iteration()
        solver.step()


class TestPSOChangeDetection:
    def setup_method(self):
        random.seed(1)
        np.random.seed(1)

    def test_off_by_default(self):
        problem = ShiftingSphere(period=5)
        solver = PSO(problem, "PSO", swarm_size=10, w=0.72, c1=1.49, c2=1.49)
        _run(solver, problem, 50)
        assert solver.n_changes_detected == 0

    def test_static_problem_never_checks(self):
        problem = ShiftingSphere(period=None, dynamic=False)
        solver = _pso(problem, True)
        _run(solver, problem, 20)
        assert solver._sentry_x is None

    def test_detects_every_change(self):
        problem = ShiftingSphere(period=5)
        solver = _pso(problem, True)
        _run(solver, problem, 50)
        assert solver.n_changes_detected == 50 // 5

    def test_no_false_detection_without_change(self):
        problem = ShiftingSphere(period=None)
        solver = _pso(problem, True)
        _run(solver, problem, 50)
        assert solver.n_changes_detected == 0

    def test_pbests_rescored_after_change(self):
        problem = ShiftingSphere(period=5)
        solver = _pso(problem, True)
        _run(solver, problem, 5)  # the change happens at iteration 5
        solver.step()
        for pos, ev in zip(solver.pbest_positions, solver.pbest_evaluations):
            assert ev.fitness == problem.evaluate(pos).fitness

    def test_multiplier_changes_not_detected_in_ccls(self):
        """Inside the co-evolutionary solver the sentry uses the original
        problem, so new multipliers are not taken for an environment
        change."""
        problem = ShiftingSphere(period=None)
        solver = CoevolutionaryLagrangianSolver(
            name="CCLS",
            problem=problem,
            objective_solver_class=PSO,
            multiplier_solver_class=PSO,
            objective_solver_params={
                "swarm_size": 10, "w": 0.72, "c1": 1.49, "c2": 1.49,
                "detect_changes": True,
            },
            multiplier_solver_params={
                "swarm_size": 10, "w": 0.40, "c1": 1.20, "c2": 1.20,
            },
            penalty_rho=0.5,
            max_multiplier=100.0,
        )
        _run(solver, problem, 50)
        assert solver.objective_solver.n_changes_detected == 0

    def test_ccls_detects_environment_changes(self):
        problem = ShiftingSphere(period=10)
        solver = CoevolutionaryLagrangianSolver(
            name="CCLS",
            problem=problem,
            objective_solver_class=PSO,
            multiplier_solver_class=PSO,
            objective_solver_params={
                "swarm_size": 10, "w": 0.72, "c1": 1.49, "c2": 1.49,
                "detect_changes": True,
            },
            multiplier_solver_params={
                "swarm_size": 10, "w": 0.40, "c1": 1.20, "c2": 1.20,
            },
            penalty_rho=0.5,
            max_multiplier=100.0,
        )
        _run(solver, problem, 50)
        assert solver.objective_solver.n_changes_detected == 50 // 10
