from cilpy.problem.constrained import G01, G02, G03, G04, G05, G06
from cilpy.problem.cmpb import ConstrainedMovingPeaksBenchmark
from cilpy.problem.mpb import generate_mpb_configs
from concurrent.futures import ThreadPoolExecutor
import time

# load all named MPB configs
mpb_configs = generate_mpb_configs(dimension=2)

# SOSC — static objective, static constraint
cmpb_sosc = ConstrainedMovingPeaksBenchmark(
    f_params=mpb_configs["STA"],
    g_params=mpb_configs["STA"],
    name="CMPB_SOSC",
)

# DOSC — dynamic objective, static constraint
cmpb_dosc = ConstrainedMovingPeaksBenchmark(
    f_params=mpb_configs["A2R"],
    g_params=mpb_configs["STA"],
    name="CMPB_DOSC",
)

# SODC — static objective, dynamic constraint
cmpb_sodc = ConstrainedMovingPeaksBenchmark(
    f_params=mpb_configs["STA"],
    g_params=mpb_configs["A2R"],
    name="CMPB_SODC",
)

RUN_CCPSO = True
RUN_DIRECT_PSO = False
NUM_PARALLEL_THREADS = 4

if RUN_CCPSO:
    from cilpy.runner import ExperimentRunner
    from cilpy.solver.pso import PSO
    from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
    from cilpy.solver.chm.deb_feasibility import DebFeasibilityHandler

if RUN_DIRECT_PSO:
    from cilpy.runner import ExperimentRunner


def run_experiment(problem, solver_config, num_runs, max_iterations):
    """Run a single experiment (problem + solver config combination)."""
    runner = ExperimentRunner(
        problems=[problem],
        solver_configurations=[solver_config],
        num_runs=num_runs,
        max_iterations=max_iterations,
    )
    runner.run_experiments()
    return f"{problem.name} with {solver_config['params']['name']} complete"


def main() -> None:
    g_problems = [G01(), G02(), G04(), G05(), G06()]  # Skip G03 for now
    cmpb_problems = [cmpb_sosc]  # Focus on SOSC for Phase 2
    #all_problems = g_problems + cmpb_problems
    all_problems = cmpb_problems

    if RUN_CCPSO:
        solver_config = {
            "class": CoevolutionaryLagrangianSolver,
            "params": {
                "name": "CCPSO",
                "penalty_rho": 0.5,
                "penalty_rho_equality": 0.5,
                "max_multiplier": 100.0,
                "objective_solver_class": PSO,
                "multiplier_solver_class": PSO,
                "objective_solver_params": {
                    "name": "obj_pso",
                    "swarm_size": 80,
                    "w": 0.72,
                    "c1": 1.49,
                    "c2": 1.49,
                },
                "multiplier_solver_params": {
                    "name": "mul_pso",
                    "swarm_size": 40,
                    "w": 0.4,
                    "c1": 1.2,
                    "c2": 1.2,
                },
            },
        }

        print("=" * 80)
        print("Starting 30-run Phase 2 validation with parallel execution")
        print(f"Using {NUM_PARALLEL_THREADS} threads")
        print(f"Problems: {[p.name for p in all_problems]}")
        print("=" * 80)

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=NUM_PARALLEL_THREADS) as executor:
            futures = []
            for problem in all_problems:
                future = executor.submit(
                    run_experiment,
                    problem,
                    solver_config,
                    num_runs=30,
                    max_iterations=1000,
                )
                futures.append(future)

            for i, future in enumerate(futures, 1):
                try:
                    result = future.result()
                    print(f"[{i}/{len(futures)}] {result}")
                except Exception as e:
                    print(f"[{i}/{len(futures)}] Error: {e}")

        elapsed = time.time() - start_time
        print("=" * 80)
        print(f"All experiments finished in {elapsed:.2f}s")
        print("=" * 80)

    if RUN_DIRECT_PSO:
        solver_config = {
            "class": PSO,
            "params": {
                "name": "PSO",
                "swarm_size": 500,
                "w": 0.72,
                "c1": 1.49,
                "c2": 1.49,
            },
        }

        print("=" * 80)
        print("Starting 30-run direct PSO baseline with parallel execution")
        print(f"Using {NUM_PARALLEL_THREADS} threads")
        print("=" * 80)

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=NUM_PARALLEL_THREADS) as executor:
            futures = []
            for problem in all_problems:
                future = executor.submit(
                    run_experiment,
                    problem,
                    solver_config,
                    num_runs=30,
                    max_iterations=1000,
                )
                futures.append(future)

            for i, future in enumerate(futures, 1):
                try:
                    result = future.result()
                    print(f"[{i}/{len(futures)}] {result}")
                except Exception as e:
                    print(f"[{i}/{len(futures)}] Error: {e}")

        elapsed = time.time() - start_time
        print("=" * 80)
        print(f"All experiments finished in {elapsed:.2f}s")
        print("=" * 80)


if __name__ == "__main__":
    main()