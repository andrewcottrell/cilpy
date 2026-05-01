from typing import Callable, Dict, List, Tuple

import numpy as np

from cilpy.problem.constrained import G01

# Optional: keep the original CCGA benchmark as a toggle.
# It is off by default because this script is focused on validation.
RUN_CCGA = False
RUN_DIRECT_GA = False
RUN_DIRECT_PSO = True

if RUN_CCGA:
    from cilpy.runner import ExperimentRunner
    from cilpy.solver.ga import GA
    from cilpy.solver.ccls import CoevolutionaryLagrangianSolver

if RUN_DIRECT_GA or RUN_DIRECT_PSO:
    from cilpy.runner import ExperimentRunner
    from cilpy.solver.chm.alpha_constraint import AlphaConstraintHandler

if RUN_DIRECT_GA:
    from cilpy.solver.ga import GA

    class SeededGA(GA):
        """GA that injects a known feasible seed into the initial population."""

        def __init__(self, *args, seed_solution: List[float], **kwargs):
            self._seed_solution = seed_solution
            super().__init__(*args, **kwargs)

        def _initialize_population(self) -> List[List[float]]:
            population = super()._initialize_population()
            if population:
                population[0] = self._seed_solution[:]
            return population

if RUN_DIRECT_PSO:
    from cilpy.solver.pso import PSO

    class ConstraintSeededPSO(PSO):
        """PSO that injects a known feasible seed and uses a constraint handler."""

        def __init__(
            self,
            *args,
            seed_solution: List[float],
            constraint_handler: AlphaConstraintHandler | None = None,
            **kwargs,
        ):
            self._seed_solution = seed_solution
            self._constraint_handler = constraint_handler
            super().__init__(*args, **kwargs)
            if self._constraint_handler is not None:
                self.comparator = self._constraint_handler

        def _initialize_positions(self) -> List[List[float]]:
            positions = super()._initialize_positions()
            if positions:
                positions[0] = self._seed_solution[:]
            return positions


def _max_constraint_violation(evaluation) -> float:
    violations = []
    if evaluation.constraints_inequality:
        violations.extend(evaluation.constraints_inequality)
    if evaluation.constraints_equality:
        violations.extend(evaluation.constraints_equality)
    return max(violations) if violations else 0.0


def _validate_known_optimum(problem, x_star: List[float], f_star: float) -> None:
    evaluation = problem.evaluate(x_star)
    max_violation = _max_constraint_violation(evaluation)
    fitness_ok = abs(evaluation.fitness - f_star) <= 1e-3
    constraints_ok = max_violation <= 1e-6
    status = "PASS" if (fitness_ok and constraints_ok) else "FAIL"

    print(f"{problem.name} known optimum check: {status}")
    print(f"  fitness = {evaluation.fitness:.12f} (expected {f_star:.12f})")
    print(f"  max_violation = {max_violation:.6e}")


def _build_scipy_constraints(problem) -> List[Dict[str, Callable[[np.ndarray], float]]]:
    constraints = []
    sample = problem.evaluate(problem.bounds[0])
    num_ineq = len(sample.constraints_inequality or [])
    num_eq = len(sample.constraints_equality or [])

    for idx in range(num_ineq):
        def ineq_fun(x, i=idx):
            return -(problem.evaluate(x.tolist()).constraints_inequality or [])[i]

        constraints.append({"type": "ineq", "fun": ineq_fun})

    for idx in range(num_eq):
        def eq_fun(x, i=idx):
            return -(problem.evaluate(x.tolist()).constraints_equality or [])[i]

        constraints.append({"type": "ineq", "fun": eq_fun})

    return constraints


def _run_scipy_solver(problem, seed: int = 7, restarts: int = 5) -> None:
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("SciPy not available. Skipping deterministic solver run.")
        return

    bounds = list(zip(problem.bounds[0], problem.bounds[1]))
    constraints = _build_scipy_constraints(problem)

    def objective(x: np.ndarray) -> float:
        return problem.evaluate(x.tolist()).fitness

    rng = np.random.default_rng(seed)
    best = None

    for restart in range(restarts):
        if restart == 0:
            x0 = np.array([(l + u) / 2.0 for l, u in bounds], dtype=float)
        else:
            x0 = np.array([rng.uniform(l, u) for l, u in bounds], dtype=float)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 2000},
        )

        if best is None or result.fun < best.fun:
            best = result

    if best is None:
        print(f"{problem.name} solver: no result")
        return

    best_eval = problem.evaluate(best.x.tolist())
    max_violation = _max_constraint_violation(best_eval)
    print(f"{problem.name} solver: success={best.success}, iters={best.nit}")
    print(f"  fitness = {best_eval.fitness:.12f}")
    print(f"  max_violation = {max_violation:.6e}")


def main() -> None:
    problems = [G01()]

    known_optima: Dict[str, Tuple[List[float], float]] = {
        "G01": (
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1],
            -15.0,
        )
    }

    print("=== Known Optimum Validation ===")
    for problem in problems:
        x_star, f_star = known_optima[problem.name]
        _validate_known_optimum(problem, x_star, f_star)

    print("\n=== Deterministic Solver (SciPy SLSQP) ===")
    for problem in problems:
        _run_scipy_solver(problem)

    if RUN_CCGA:
        solver_configs = [
            {
                "class": CoevolutionaryLagrangianSolver,
                "params": {
                    "name": "CCGA",
                    "objective_solver_class": GA,
                    "multiplier_solver_class": GA,
                    "objective_solver_params": {
                        "name": "obj_ga",
                        "population_size": 40,
                        "crossover_rate": 0.8,
                        "mutation_rate": 0.1,
                        "tournament_size": 3,
                    },
                    "multiplier_solver_params": {
                        "name": "mul_ga",
                        "population_size": 40,
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
            num_runs=5,
            max_iterations=1000,
        )
        runner.run_experiments()

    if RUN_DIRECT_GA:
        solver_configs = [
            {
                "class": SeededGA,
                "params": {
                    "name": "GA_Alpha",
                    "population_size": 80,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1,
                    "tournament_size": 3,
                    "seed_solution": known_optima["G01"][0],
                },
                "constraint_handler": {
                    "class": AlphaConstraintHandler,
                    "params": {
                        "alpha": 1.0,
                        "b_inequality": 5,
                        "b_equality": 10.0,
                    },
                },
            }
        ]

        runner = ExperimentRunner(
            problems=problems,
            solver_configurations=solver_configs,
            num_runs=5,
            max_iterations=1000,
        )
        runner.run_experiments()

    if RUN_DIRECT_PSO:
        solver_configs = [
            {
                "class": ConstraintSeededPSO,
                "params": {
                    "name": "PSO_Alpha",
                    "swarm_size": 80,
                    "w": 0.72,
                    "c1": 1.49,
                    "c2": 1.49,
                    "seed_solution": known_optima["G01"][0],
                },
                "constraint_handler": {
                    "class": AlphaConstraintHandler,
                    "params": {
                        "alpha": 1.0,
                        "b_inequality": 5.0,
                        "b_equality": 10.0,
                    },
                },
            }
        ]

        runner = ExperimentRunner(
            problems=problems,
            solver_configurations=solver_configs,
            num_runs=5,
            max_iterations=1000,
        )
        runner.run_experiments()


if __name__ == "__main__":
    main()