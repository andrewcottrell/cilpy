from cilpy.runner import ExperimentRunner
from cilpy.problem.constrained import G01
from cilpy.solver.pso import PSO
from cilpy.solver.chm.alpha_constraint import AlphaConstraintHandler

problems = [G01()]

solver_configs = [
    {
        "class": PSO,
        "params": {
            "name": "PSO_G01",
            "swarm_size": 30,
            "w": 0.7298,
            "c1": 1.49618,
            "c2": 1.49618,
        },
        "constraint_handler": {
            "class": AlphaConstraintHandler,
            "params": {
                "alpha": 0.5,
                "b_inequality": 5.0,
            }
        }
    }
]

runner = ExperimentRunner(
    problems=problems,
    solver_configurations=solver_configs,
    num_runs=30,
    max_iterations=1000,
)
runner.run_experiments()