from cilpy.runner import ExperimentRunner
from cilpy.problem.mpb import MovingPeaksBenchmark
from cilpy.solver.pso import QPSO

# --- 1. Define the Problems ---
problems_to_run = [
    MovingPeaksBenchmark(
        dimension = 2,
        num_peaks = 5,
        change_frequency = 10,
        lambda_param = 0.0,
        name="exp_1"
    ),
    MovingPeaksBenchmark(
        dimension = 2,
        num_peaks = 5,
        change_frequency = 10,
        lambda_param = 1.0,
        name="exp_2"
    ),
]

# --- 2. Define the Solvers and their parameters ---
# Note: The 'problem' parameter is omitted here as the runner will assign it.
solver_configs = [
    {
        "class": QPSO,
        "params": {
            "name": "QPSO",
            "swarm_size": 30,
            "w": 0.7298,
            "c1": 1.49618,
            "c2": 1.49618,
            "split_ratio": 0.5,
            "r_cloud": 0.5,
        }
    },
]

# --- 3. Define the Experiment parameters ---
number_of_runs = 1
max_iter = 100

# --- 4. Create and run the experiments ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=number_of_runs,
    max_iterations=max_iter
)
runner.run_experiments()
