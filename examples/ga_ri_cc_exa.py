# examples/ga_ri_cc_exa.py
from cilpy.runner import ExperimentRunner
from cilpy.problem.cmpb import ConstrainedMovingPeaksBenchmark, generate_mpb_configs
from cilpy.solver.ga import RIGA
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver

# --- 1. Define the Problems ---
all_mpb_configs = generate_mpb_configs(dimension=5)
f_params = all_mpb_configs['A3L']
g_params = all_mpb_configs['P1R']

problems_to_run = [
    ConstrainedMovingPeaksBenchmark(
        f_params=f_params,
        g_params=g_params,
        name="CMPB_A3L_P1R"
    )
]

# --- 2. Define the Solvers and their parameters ---
solver_configs = [
    {
        # This is the co-evolutionary solver configuration
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCLS_with_RIGAs",
            "objective_solver_class": RIGA,
            "multiplier_solver_class": RIGA,
            "objective_solver_params": {
                "name": "ObjectiveGA",
                "population_size": 40,
                "crossover_rate": 0.1,
                "mutation_rate": 0.15,
                "immigrant_rate": 0.3,
                "tournament_size": 3,
            },
            "multiplier_solver_params": {
                "name": "MultiplierGA",
                "population_size": 40,
                "crossover_rate": 0.1,
                "mutation_rate": 0.15,
                "immigrant_rate": 0.3,
                "tournament_size": 3,
            }
        }
    },
]

# --- 3. Define the Experiment parameters ---
number_of_runs = 1
max_iter = 5

# --- 4. Create and run the experiments ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=number_of_runs,
    max_iterations=max_iter
)
runner.run_experiments()
