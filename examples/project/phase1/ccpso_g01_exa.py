from cilpy.runner import ExperimentRunner
from cilpy.problem.constrained import G01
from cilpy.solver.pso import PSO
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver

problems = [G01()]

solver_configs = [
    {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCPSO_G01",
            "objective_solver_class": PSO,
            "multiplier_solver_class": PSO,
            "objective_solver_params": {
                "name": "obj_pso",
                "swarm_size": 30,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            },
            "multiplier_solver_params": {
                "name": "mul_pso",
                "swarm_size": 30,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            },
            # Direct penalty terms accelerate feasibility discovery
            # Balanced: 0.35 trades feasibility stability for better optimization
            "penalty_rho": 0.35,
            "penalty_rho_equality": 0.35,
        }
    }
]

runner = ExperimentRunner(
    problems=problems,
    solver_configurations=solver_configs,
    num_runs=30,
    max_iterations=3000,
)
runner.run_experiments()