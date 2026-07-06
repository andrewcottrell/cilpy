# examples/normal/ccpso_exa.py
"""Example: constrained multi-objective optimization with CCPSO.

CCPSO = CoevolutionaryLagrangianSolver wrapping MGPSO as the objective
solver. Demonstrates both archive strategies, usage through the
ExperimentRunner, and direct usage with metric computation.

Run from the repository root:
    python examples/normal/ccpso_exa.py
"""

import numpy as np

from cilpy.problem.multi_objective import TNK
from cilpy.solver.mgpso import MGPSO
from cilpy.solver.pso import PSO
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.runner import ExperimentRunner
from cilpy.compare.metrics import (
    inverted_generational_distance,
    hypervolume,
    feasibility_rate,
)


def ccpso_config(strategy: str) -> dict:
    """A CCPSO solver configuration for the ExperimentRunner."""
    return {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": f"CCPSO_{strategy}",
            "objective_solver_class": MGPSO,
            "multiplier_solver_class": PSO,
            "objective_solver_params": {
                "swarm_size": 50,
                # omit w/c1/c2/c3 for stability-guided sampling
            },
            "multiplier_solver_params": {
                "swarm_size": 30, "w": 0.40, "c1": 1.20, "c2": 1.20,
            },
            "penalty_rho": 1.0,
            "max_multiplier": 1000.0,
            "archive_strategy": strategy,   # "filter" or "strict"
        },
    }


def main():
    problem = TNK()

    # 1. Through the ExperimentRunner: both strategies side by side.
    #    Output CSVs land in out/TNK_CCPSO_filter.out.csv etc., with IGD,
    #    GD, hypervolume, spacing, spread, and feasibility per iteration.
    runner = ExperimentRunner(
        problems=[problem],
        solver_configurations=[ccpso_config("filter"), ccpso_config("strict")],
        num_runs=3,
        max_iterations=150,
    )
    runner.run_experiments()

    # 2. Direct usage with metric computation on the final front.
    np.random.seed(7)
    solver = CoevolutionaryLagrangianSolver(
        name="CCPSO",
        problem=problem,
        objective_solver_class=MGPSO,
        multiplier_solver_class=PSO,
        objective_solver_params={"swarm_size": 50},
        multiplier_solver_params={
            "swarm_size": 30, "w": 0.40, "c1": 1.20, "c2": 1.20,
        },
        penalty_rho=1.0,
        max_multiplier=1000.0,
        archive_strategy="filter",
    )
    for _ in range(300):
        problem.begin_iteration()
        solver.step()

    # get_result() re-evaluates against the ORIGINAL problem: fitness is
    # the true objective vector, constraints are the true violations.
    result = solver.get_result()
    front = np.array([evaluation.fitness for _, evaluation in result])
    reference = problem.true_pareto_front(500)

    print(f"\nFront size:       {len(front)}")
    print(f"Feasibility rate: {feasibility_rate([e for _, e in result]):.1f}%")
    print(f"IGD:              {inverted_generational_distance(front, reference):.4f}")
    print(f"Hypervolume:      {hypervolume(front, [1.3, 1.3]):.4f}")


if __name__ == "__main__":
    main()