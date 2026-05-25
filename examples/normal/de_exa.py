from cilpy.runner import ExperimentRunner
from cilpy.problem.unconstrained import Sphere, Ackley
from cilpy.solver.de import DE

# --- 1. Define the Problems ---
dim = 3
dom = (-5.12, 5.12)
problems_to_run = [
    Sphere(dimension=dim, domain=dom),
    Ackley(dimension=dim, domain=dom)
]

# --- 2. Define the Solvers and their parameters ---
# Note: The 'problem' parameter is omitted here as the runner will assign it.
solver_configs = [
    {
        "class": DE,
        "params": {
            "name": "DE_HighCrossover",
            "population_size": 30,
            "crossover_rate": 0.2,
            "f_weight": 1,
            "tournament_size": 7,
        }
    },
    {
        "class": DE,
        "params": {
            "name": "DE_LowCrossover",
            "population_size": 30,
            "crossover_rate": 0.4,
            "f_weight": 1,
            "tournament_size": 7,
        }
    },
]

# --- 3. Define the Experiment parameters ---
number_of_runs = 1
max_iter = 10

# --- 4. Create and run the experiments ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=number_of_runs,
    max_iterations=max_iter
)
runner.run_experiments()
