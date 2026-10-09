# examples/pso_constrained_alpha_exa.py
from cilpy.runner import ExperimentRunner
from cilpy.problem.constrained import G01
from cilpy.solver.pso import PSO
from cilpy.solver.chm.alpha_constraint import AlphaConstraintHandler

# --- 1. Define the Problem ---
problems_to_run = [
    G01(),
]

# --- 2. Define the Solver and Constraint Handler ---
solver_configs = [
    # {
    #     "class": PSO,
    #     "params": {
    #         "name": "PSO_DefaultConstraint1",
    #         "swarm_size": 50,
    #         "w": 0.7, "c1": 1.5, "c2": 1.5,
    #     },
    # },
    {
        "class": PSO,
        "params": {
            "name": "PSO_AlphaConstraint1",
            "swarm_size": 50,
            "w": 0.7, "c1": 1.5, "c2": 1.5,
        },
        "constraint_handler": {
            "class": AlphaConstraintHandler,
            "params": {
                "alpha": 0.5,
                "b_inequality": 5.0
            }
        }
    },
    # {
    #     "class": PSO,
    #     "params": {
    #         "name": "PSO_AlphaConstraint2",
    #         "swarm_size": 50,
    #         "w": 0.7, "c1": 1.5, "c2": 1.5,
    #     },
    #     "constraint_handler": {
    #         "class": AlphaConstraintHandler,
    #         "params": {
    #             "alpha": 0.95,
    #             "b_inequality": 5.0
    #         }
    #     }
    # },
]

# --- 3. Define the Experiment parameters ---
number_of_runs = 3
max_iter = 1000

# --- 4. Create and run the experiments ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=number_of_runs,
    max_iterations=max_iter
)
runner.run_experiments()
