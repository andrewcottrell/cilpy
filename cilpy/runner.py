# cilpy/runner.py
"""The experiment runner: Orchestrates computational intelligence experiments."""

import time
import csv
import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Sequence

from cilpy.problem import Problem, Evaluation
from cilpy.solver import Solver
from cilpy.compare.metrics import (
    inverted_generational_distance,
    generational_distance,
    hypervolume,
    spacing,
    spread,
)


class ExperimentRunner:
    """Orchestrates the execution of computational intelligence experiments.

    Output files
    ------------
    For each (problem, solver) pair the runner writes two CSV files to ``out/``.

    **<problem>_<solver>.out.csv** — one row per iteration per run.

    Single-objective columns:

    * ``run_id``                    — independent run number.
    * ``iteration``                 — iteration number within the run.
    * ``best_fitness``              — best objective value found so far.
    * ``population_feasibility_pct``— % of current population satisfying all
      constraints; empty when unavailable.
    * ``population_diversity``      — mean Euclidean distance of particles from
      the swarm centroid; collapses toward 0 at convergence.
    * ``relative_error``            — (f_max - best) / (f_max - f_min); 0 at
      the known optimum. Empty when bounds are unknown.

    Multi-objective columns:

    * ``run_id``, ``iteration``
    * ``archive_size``              — non-dominated solutions in the archive.
    * ``igd``                       — Inverted Generational Distance to the true
      Pareto front. Measures both convergence and coverage; 0 is ideal. Empty
      when the problem has no ``true_pareto_front()``.
    * ``gd``                        — Generational Distance. Measures convergence
      only (how close your solutions are to the true front); blind to gaps in
      coverage. Empty when ``true_pareto_front()`` unavailable.
    * ``hypervolume``               — Dominated area between the archive and a
      fixed reference point. Combines convergence and spread into one scalar;
      no true front required. Larger is better.
    * ``spacing``                   — Standard deviation of nearest-neighbour
      distances within the front. 0 = perfectly uniform distribution.
    * ``spread``                    — Deb's Delta metric; combines extent and
      uniformity. 0 = front spans the full true front evenly. Empty when
      ``true_pareto_front()`` unavailable.
    * ``population_feasibility_pct``— as above.
    * ``population_diversity``      — as above.
    * ``front``                     — the full Pareto archive as a list of
      objective vectors; useful for plotting convergence over iterations.

    **<problem>_<solver>.summary.out.csv** — one row per run.

    Single-objective: ``problem_name``, ``solver_name``, ``run_id``,
    ``final_best_fitness``, ``final_feasibility_pct``, ``p_red``, ``run_time_s``.

    Multi-objective: ``problem_name``, ``solver_name``, ``run_id``,
    ``final_archive_size``, ``final_igd``, ``final_gd``, ``final_hypervolume``,
    ``final_spacing``, ``final_spread``, ``final_feasibility_pct``, ``run_time_s``.
    """

    def __init__(
        self,
        problems: Sequence[Problem],
        solver_configurations: List[Dict[str, Any]],
        num_runs: int,
        max_iterations: int,
    ):
        self.problems = problems
        self.solver_configurations = solver_configurations
        self.num_runs = num_runs
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_experiments(self):
        """Executes the full suite of defined experiments."""
        total_start_time = time.time()
        print("======== Starting All Experiments ========")
        os.makedirs("out", exist_ok=True)

        for problem in self.problems:
            print(f"\n--- Processing Problem: {problem.name} ---")
            for config in self.solver_configurations:
                solver_class = config["class"]
                solver_params = config["params"].copy()
                constraint_handler_config = config.get("constraint_handler")

                solver_params["problem"] = problem
                solver_name = solver_params.get("name", solver_class.__name__)
                output_file_path = os.path.join(
                    "out", f"{problem.name}_{solver_name}.out.csv"
                )

                print(f"\n  -> Starting Experiment: {solver_name} on {problem.name}")
                print(
                    f"     Configuration: {self.num_runs} runs, "
                    f"{self.max_iterations} iterations/run."
                )
                print(f"     Results will be saved to: {output_file_path}")

                self._run_single_experiment(
                    solver_class,
                    solver_params,
                    output_file_path,
                    constraint_handler_config,
                )

        total_end_time = time.time()
        print("\n======== All Experiments Finished ========")
        print(f"Total execution time: {total_end_time - total_start_time:.2f}s")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _is_solution_feasible(self, evaluation: Evaluation, tolerance: float = 1e-6) -> bool:
        if evaluation is None:
            return False
        if evaluation.constraints_inequality:
            if any(v > 0 for v in evaluation.constraints_inequality):
                return False
        if evaluation.constraints_equality:
            if any(abs(v) > tolerance for v in evaluation.constraints_equality):
                return False
        return True

    def _measure_feasibility(self, solver: Solver) -> str:
        try:
            evals = solver.get_population_evaluations()
            if not evals:
                return ""
            n_feasible = sum(1 for e in evals if self._is_solution_feasible(e))
            return f"{100.0 * n_feasible / len(evals):.4f}"
        except NotImplementedError:
            return ""

    def _measure_diversity(self, solver: Solver) -> str:
        try:
            population = solver.get_population()
            if not population:
                return ""
            pop_array = np.array(population, dtype=float)
            centroid = np.mean(pop_array, axis=0)
            dists = np.sqrt(np.sum((pop_array - centroid) ** 2, axis=1))
            return f"{float(np.mean(dists)):.6f}"
        except NotImplementedError:
            return ""

    # ------------------------------------------------------------------
    # Single-objective columns
    # ------------------------------------------------------------------

    _SO_MAIN_HEADER = [
        "run_id",
        "iteration",
        "best_fitness                 [best objective value found so far in this run]",
        "population_feasibility_pct   [% of current population satisfying all constraints]",
        "population_diversity         [mean distance of particles from swarm centroid]",
        "relative_error               [how far best_fitness is from known optimum; 0=optimal]",
    ]

    _SO_SUMMARY_HEADER = [
        "problem_name",
        "solver_name",
        "run_id",
        "final_best_fitness           [best objective value at the last iteration]",
        "final_feasibility_pct        [% feasible in population at the last iteration]",
        "p_red                        [mean relative error distance across all iterations; 0=ideal]",
        "run_time_s",
    ]

    def _so_iteration_row(
        self,
        run_id: int,
        iteration: int,
        solver: Solver,
        bounds_known: bool,
        f_max: float,
        fitness_range: float,
        relative_error_history: List[float],
    ) -> list:
        result = solver.get_result()
        best_fitness = result[0][1].fitness if result else ""

        if bounds_known and isinstance(best_fitness, (int, float)):
            rel_err = (f_max - best_fitness) / fitness_range
            relative_error_history.append(rel_err)
            rel_err_str = f"{rel_err:.6f}"
        else:
            rel_err_str = ""

        return [
            run_id,
            iteration,
            f"{best_fitness:.6f}" if isinstance(best_fitness, (int, float)) else best_fitness,
            self._measure_feasibility(solver),
            self._measure_diversity(solver),
            rel_err_str,
        ]

    def _so_summary_row(
        self,
        problem_name: str,
        solver_name: str,
        run_id: int,
        solver: Solver,
        relative_error_history: List[float],
        run_time: float,
    ) -> list:
        result = solver.get_result()
        final_fitness = result[0][1].fitness if result else ""
        final_feasibility = self._measure_feasibility(solver)

        if relative_error_history:
            b = np.array(relative_error_history)
            p_red = float(np.sqrt(np.sum((1 - b) ** 2) / len(b)))
            p_red_str = f"{p_red:.6f}"
        else:
            p_red_str = ""

        return [
            problem_name,
            solver_name,
            run_id,
            f"{final_fitness:.6f}" if isinstance(final_fitness, (int, float)) else final_fitness,
            final_feasibility,
            p_red_str,
            f"{run_time:.2f}",
        ]

    # ------------------------------------------------------------------
    # Multi-objective columns
    # ------------------------------------------------------------------

    _MO_MAIN_HEADER = [
        "run_id",
        "iteration",
        "archive_size",
        "igd",
        "gd",
        "hypervolume",
        "spacing",
        "spread",
        "population_feasibility_pct",
        "population_diversity",
        "front",
    ]

    _MO_SUMMARY_HEADER = [
        "problem_name",
        "solver_name",
        "run_id",
        "final_archive_size",
        "final_igd",
        "final_gd",
        "final_hypervolume",
        "final_spacing",
        "final_spread",
        "final_feasibility_pct",
        "run_time_s",
    ]

    def _mo_iteration_row(
        self,
        run_id: int,
        iteration: int,
        solver: Solver,
        hv_ref: Optional[Tuple[float, float]],
        ref_front: Optional[np.ndarray],
    ) -> Tuple[list, Optional[Tuple[float, float]]]:
        """Returns (row, updated_hv_ref)."""
        result = solver.get_result()
        front_fitnesses = [ev.fitness for _, ev in result]

        archive_size = len(front_fitnesses)

        # --- Hypervolume: fix reference point after first populated archive ---
        if hv_ref is None and front_fitnesses:
            f1_max = max(f[0] for f in front_fitnesses)
            f2_max = max(f[1] for f in front_fitnesses)
            hv_ref = (f1_max * 1.1 + 0.1, f2_max * 1.1 + 0.1)

        hv_str = ""
        if hv_ref and front_fitnesses:
            try:
                hv_str = f"{hypervolume(front_fitnesses, hv_ref):.6f}"
            except Exception:
                hv_str = ""

        # --- Spacing: no reference front needed ---
        sp_str = ""
        if front_fitnesses:
            try:
                sp_str = f"{spacing(front_fitnesses):.6f}"
            except Exception:
                sp_str = ""

        # --- IGD, GD, Spread: only when reference front is available ---
        igd_str = gd_str = spread_str = ""
        if ref_front is not None and front_fitnesses:
            try:
                igd_str = f"{inverted_generational_distance(front_fitnesses, ref_front):.6f}"
            except Exception:
                pass
            try:
                gd_str = f"{generational_distance(front_fitnesses, ref_front):.6f}"
            except Exception:
                pass
            try:
                spread_str = f"{spread(front_fitnesses, ref_front):.6f}"
            except Exception:
                pass

        row = [
            run_id,
            iteration,
            archive_size,
            igd_str,
            gd_str,
            hv_str,
            sp_str,
            spread_str,
            self._measure_feasibility(solver),
            self._measure_diversity(solver),
            [[float(v) for v in f] for f in front_fitnesses],
        ]
        return row, hv_ref

    def _mo_summary_row(
        self,
        problem_name: str,
        solver_name: str,
        run_id: int,
        solver: Solver,
        hv_ref: Optional[Tuple[float, float]],
        ref_front: Optional[np.ndarray],
        run_time: float,
    ) -> list:
        result = solver.get_result()
        front_fitnesses = [ev.fitness for _, ev in result]

        hv_str = ""
        if hv_ref and front_fitnesses:
            try:
                hv_str = f"{hypervolume(front_fitnesses, hv_ref):.6f}"
            except Exception:
                hv_str = ""

        sp_str = ""
        if front_fitnesses:
            try:
                sp_str = f"{spacing(front_fitnesses):.6f}"
            except Exception:
                sp_str = ""

        igd_str = gd_str = spread_str = ""
        if ref_front is not None and front_fitnesses:
            try:
                igd_str = f"{inverted_generational_distance(front_fitnesses, ref_front):.6f}"
            except Exception:
                pass
            try:
                gd_str = f"{generational_distance(front_fitnesses, ref_front):.6f}"
            except Exception:
                pass
            try:
                spread_str = f"{spread(front_fitnesses, ref_front):.6f}"
            except Exception:
                pass

        return [
            problem_name,
            solver_name,
            run_id,
            len(front_fitnesses),
            igd_str,
            gd_str,
            hv_str,
            sp_str,
            spread_str,
            self._measure_feasibility(solver),
            f"{run_time:.2f}",
        ]

    # ------------------------------------------------------------------
    # Core run loop
    # ------------------------------------------------------------------

    def _run_single_run(
        self,
        run_id: int,
        constraint_handler_config: Optional[Dict],
        solver_params: Dict,
        solver_class: Type[Solver],
        main_writer,
        summary_file_path: str,
        is_multi_objective: bool,
        bounds_known: bool = False,
        f_max: float = 0.0,
        fitness_range: float = 0.0,
        ref_front: Optional[np.ndarray] = None,
    ):
        run_start_time = time.time()
        print(f"     --- Starting Run {run_id}/{self.num_runs} ---")

        current_params = solver_params.copy()

        # Only inject constraint_handler when explicitly provided. Solvers like
        # CoevolutionaryLagrangianSolver handle their own constraint logic via
        # **kwargs; injecting None conflicts with their sub-solver constructors.
        if constraint_handler_config:
            handler_class = constraint_handler_config["class"]
            handler_params = constraint_handler_config.get("params", {})
            current_params["constraint_handler"] = handler_class(**handler_params)

        solver = solver_class(**current_params)

        problem_name = solver.problem.name
        solver_name = solver.name

        relative_error_history: List[float] = []
        hv_ref: Optional[Tuple[float, float]] = None

        for iteration in range(1, self.max_iterations + 1):
            solver.problem.begin_iteration()
            solver.step()

            if is_multi_objective:
                row, hv_ref = self._mo_iteration_row(
                    run_id, iteration, solver, hv_ref, ref_front
                )
            else:
                row = self._so_iteration_row(
                    run_id, iteration, solver,
                    bounds_known, f_max, fitness_range, relative_error_history,
                )

            main_writer.writerow(row)

        run_time = time.time() - run_start_time

        # --- Console summary ---
        if is_multi_objective:
            result = solver.get_result()
            front_fitnesses = [ev.fitness for _, ev in result]
            igd_console = ""
            if ref_front is not None and front_fitnesses:
                try:
                    igd_console = f"  IGD={inverted_generational_distance(front_fitnesses, ref_front):.4f}"
                except Exception:
                    pass
            print(
                f"     Run {run_id} finished in {run_time:.2f}s. "
                f"Archive size: {len(result)}{igd_console}"
            )
        else:
            result = solver.get_result()
            final_fitness = result[0][1].fitness if result else "N/A"
            print(
                f"     Run {run_id} finished in {run_time:.2f}s. "
                f"Best fitness: {final_fitness}"
            )
            if relative_error_history:
                b = np.array(relative_error_history)
                p_red = float(np.sqrt(np.sum((1 - b) ** 2) / len(b)))
                print(f"     P_RED for Run {run_id}: {p_red:.6f}")
            else:
                print(f"     P_RED for Run {run_id}: N/A (fitness bounds unknown)")

        # --- Append to summary file ---
        with open(summary_file_path, "a", newline="") as f_sum:
            w = csv.writer(f_sum)
            if is_multi_objective:
                w.writerow(self._mo_summary_row(
                    problem_name, solver_name, run_id, solver,
                    hv_ref, ref_front, run_time
                ))
            else:
                w.writerow(self._so_summary_row(
                    problem_name, solver_name, run_id, solver,
                    relative_error_history, run_time
                ))

    def _run_single_experiment(
        self,
        solver_class: Type[Solver],
        solver_params: Dict,
        output_file: str,
        constraint_handler_config: Optional[Dict] = None,
    ):
        problem = solver_params["problem"]
        is_mo = problem.is_multi_objective()

        # --- Single-objective: resolve fitness bounds for relative error ---
        bounds_known = False
        f_max = 0.0
        fitness_range = 0.0
        if not is_mo:
            try:
                f_min, f_max = problem.get_fitness_bounds()
                fitness_range = f_max - f_min
                if fitness_range > 0:
                    bounds_known = True
                else:
                    print("     Warning: Fitness range is zero; relative error skipped.")
            except NotImplementedError:
                print("     Info: get_fitness_bounds() not implemented; relative error skipped.")

        # --- Multi-objective: generate reference front once, before runs ---
        ref_front: Optional[np.ndarray] = None
        if is_mo:
            try:
                print("     Generating reference Pareto front...", end=" ", flush=True)
                ref_front = problem.true_pareto_front()
                print(f"done ({len(ref_front)} points). IGD/GD/Spread will be reported.")
            except (NotImplementedError, AttributeError):
                print("     Info: true_pareto_front() not available; IGD/GD/Spread will be empty.")

        summary_file_path = output_file.replace(".out.csv", ".summary.out.csv")
        main_header = self._MO_MAIN_HEADER if is_mo else self._SO_MAIN_HEADER
        summary_header = self._MO_SUMMARY_HEADER if is_mo else self._SO_SUMMARY_HEADER

        with open(summary_file_path, "w", newline="") as f_sum:
            csv.writer(f_sum).writerow(summary_header)

        experiment_start_time = time.time()

        with open(output_file, "w", newline="") as f_main:
            writer = csv.writer(f_main)
            writer.writerow(main_header)

            for run_id in range(1, self.num_runs + 1):
                self._run_single_run(
                    run_id,
                    constraint_handler_config,
                    solver_params,
                    solver_class,
                    writer,
                    summary_file_path,
                    is_multi_objective=is_mo,
                    bounds_known=bounds_known,
                    f_max=f_max,
                    fitness_range=fitness_range,
                    ref_front=ref_front,
                )

        experiment_time = time.time() - experiment_start_time
        solver_name = solver_params.get("name", solver_class.__name__)
        print(
            f"  -> Experiment for {solver_name} on {problem.name} "
            f"finished in {experiment_time:.2f}s."
        )
        print(f"     Summary saved to: {summary_file_path}")


if __name__ == "__main__":
    from cilpy.problem.unconstrained import Sphere
    from cilpy.solver.pso import PSO

    runner = ExperimentRunner(
        problems=[Sphere(dimension=3)],
        solver_configurations=[{
            "class": PSO,
            "params": {
                "name": "PSO_Standard",
                "swarm_size": 30,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            },
        }],
        num_runs=5,
        max_iterations=1000,
    )
    runner.run_experiments()