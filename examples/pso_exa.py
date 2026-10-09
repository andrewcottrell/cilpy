# example/pso_exa.py
from cilpy.runner import ExperimentRunner
from cilpy.problem.mpb import MovingPeaksBenchmark
from cilpy.solver.pso import PSO

# --- 1. Define the Problems ---
problems_to_run = [
    MovingPeaksBenchmark(
        dimension = 2,
        num_peaks = 5,
        change_frequency = 10,
        lambda_param = 1.0,
        name="exp_1"
    ),
]

# --- 2. Define the Solvers and their parameters ---
# Note: The 'problem' parameter is omitted here as the runner will assign it.
solver_configs = [
    {
        "class": PSO,
        "params": {
            "name": "PSO",
            "swarm_size": 10,
            "w": 0.7298,
            "c1": 1.49618,
            "c2": 1.49618,
        }
    },
]

# --- 3. Define the Experiment parameters ---
number_of_runs = 10
max_iter = 1000

# --- 4. Create and run the experiments ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=number_of_runs,
    max_iterations=max_iter
)
runner.run_experiments()

# from cilpy.problem.unconstrained import Sphere, Quadratic, Ackley
# dim = 3
# dom = (-5.12, 5.12)
    # Sphere(dimension=dim, domain=dom),
    # Quadratic(dimension=dim, domain=dom),
    # Ackley(dimension=dim, domain=dom),