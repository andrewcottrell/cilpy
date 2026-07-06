# examples/normal/mgpso_exa.py
"""Example: MGPSO on the ZDT1 benchmark, with post-hoc metric computation.

Run from the repository root:
    python examples/normal/mgpso_exa.py
"""

import numpy as np

from cilpy.problem.multi_objective import ZDT1
from cilpy.solver.mgpso import MGPSO
from cilpy.runner import ExperimentRunner
from cilpy.compare.metrics import (
    inverted_generational_distance,
    hypervolume,
    spacing,
    feasibility_rate,
)


def main():
    problem = ZDT1(dimension=30)

    runner = ExperimentRunner(
        problems=[problem],
        solver_configurations=[
            {
                "class": MGPSO,
                "params": {
                    "name": "MGPSO",
                    "swarm_size": 50,
                    "tournament_size": 3,
                    # Omit w/c1/c2/c3 to use stability-guided sampling
                    # (Scheepers et al., 2019); or fix them, e.g.:
                    # "w": 0.475, "c1": 1.80, "c2": 1.10, "c3": 1.80,
                },
            }
        ],
        num_runs=3,
        max_iterations=250,
    )
    runner.run_experiments()

    # metric computation on the final archive.
    solver = MGPSO(problem=problem, name="MGPSO", swarm_size=50)
    for _ in range(250):
        problem.begin_iteration()
        solver.step()

    result = solver.get_result()
    front = np.array([evaluation.fitness for _, evaluation in result])
    reference = problem.true_pareto_front(500)

    print(f"\nArchive size:       {len(front)}")
    print(f"IGD:                {inverted_generational_distance(front, reference):.4f}")
    print(f"Hypervolume(1.1^2): {hypervolume(front, [1.1, 1.1]):.4f}")
    print(f"Spacing:            {spacing(front):.4f}")
    print(f"Feasibility rate:   {feasibility_rate([e for _, e in result]):.1f}%")


if __name__ == "__main__":
    main()
