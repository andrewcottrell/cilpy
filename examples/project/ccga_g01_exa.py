from cilpy.runner import ExperimentRunner
from cilpy.problem.constrained import G01
from cilpy.solver.ga import GA
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver

problems = [G01()]

solver_configs = [
    {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCGA_G01",
            "objective_solver_class": GA,
            "multiplier_solver_class": GA,
            "max_multiplier": 10000.0,
            "penalty_rho": 0.5,
            "penalty_rho_equality": 0.5,
            "objective_solver_params": {
                "name": "obj_ga",
                "population_size": 30,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1,
                "tournament_size": 3,
            },
            "multiplier_solver_params": {
                "name": "mul_ga",
                "population_size": 30,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1,
                "tournament_size": 3,
            },
        },
    }
]

runner = ExperimentRunner(
    problems=problems,
    solver_configurations=solver_configs,
    num_runs=30,
    max_iterations=1000,
)
runner.run_experiments()
