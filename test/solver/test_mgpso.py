# test/solver/test_mgpso.py
"""Unit tests for the MGPSO solver."""

import numpy as np
import pytest

from cilpy.problem.multi_objective import SCH1, ZDT1
from cilpy.problem.unconstrained import Sphere
from cilpy.solver.mgpso import MGPSO, dominates
from cilpy.compare.metrics import inverted_generational_distance


def test_dominates():
    assert dominates([1.0, 1.0], [2.0, 2.0])
    assert dominates([1.0, 2.0], [1.0, 3.0])
    assert not dominates([1.0, 3.0], [2.0, 2.0])  # incomparable
    assert not dominates([1.0, 1.0], [1.0, 1.0])  # equal


def test_rejects_single_objective_problem():
    with pytest.raises(ValueError):
        MGPSO(problem=Sphere(dimension=5), name="MGPSO", swarm_size=10)


def test_subswarm_per_objective_and_population_size():
    solver = MGPSO(problem=SCH1(), name="MGPSO", swarm_size=15)
    assert solver.n_objectives == 2
    assert len(solver.get_population()) == 2 * 15
    assert len(solver.get_population_evaluations()) == 2 * 15


def test_archive_is_mutually_nondominated():
    solver = MGPSO(problem=SCH1(), name="MGPSO", swarm_size=20)
    for _ in range(50):
        solver.step()
    result = solver.get_result()
    fitnesses = [ev.fitness for _, ev in result]
    for i, fa in enumerate(fitnesses):
        for j, fb in enumerate(fitnesses):
            if i != j:
                assert not dominates(fa, fb)


def test_archive_capacity_respected():
    swarm_size = 10
    solver = MGPSO(problem=SCH1(), name="MGPSO", swarm_size=swarm_size)
    for _ in range(100):
        solver.step()
    assert len(solver.get_result()) <= 2 * swarm_size


def test_positions_stay_within_bounds():
    problem = ZDT1(dimension=10)
    solver = MGPSO(problem=problem, name="MGPSO", swarm_size=10)
    for _ in range(20):
        solver.step()
    lower, upper = problem.bounds
    for position in solver.get_population():
        assert all(l - 1e-12 <= x <= u + 1e-12
                   for x, l, u in zip(position, lower, upper))


def test_fixed_parameter_mode():
    solver = MGPSO(
        problem=SCH1(), name="MGPSO", swarm_size=10,
        w=0.72, c1=1.49, c2=1.49, c3=1.49,
    )
    solver.step()
    for swarm in solver._swarms:
        for particle in swarm.particles:
            assert particle.w == 0.72


def test_converges_on_sch1():
    """After a moderate run, the front should approximate f2 = (sqrt(f1)-2)^2."""
    np.random.seed(42)
    problem = SCH1()
    solver = MGPSO(problem=problem, name="MGPSO", swarm_size=25)
    for _ in range(200):
        solver.step()
    front = [ev.fitness for _, ev in solver.get_result()]
    igd = inverted_generational_distance(front, problem.true_pareto_front())
    assert igd < 0.1, f"IGD too high: {igd}"


def test_result_solutions_match_evaluations():
    """Re-evaluating a returned solution must reproduce its stored fitness."""
    problem = SCH1()
    solver = MGPSO(problem=problem, name="MGPSO", swarm_size=10)
    for _ in range(10):
        solver.step()
    for solution, evaluation in solver.get_result():
        fresh = problem.evaluate(solution)
        assert np.allclose(fresh.fitness, evaluation.fitness)
