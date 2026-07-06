# examples/project/run_static_mo_experiments.py
"""Static multi-objective experimental campaign (Phase 2).

Orchestrates the full static MO benchmark grid:

* Unconstrained problems (SCH1, ZDT1-4, ZDT6): plain MGPSO only. The
  co-evolutionary framework is meaningless without constraints (the
  multiplier search space would be zero-dimensional).

* Constrained problems (BNH, SRN, TNK, CONSTR, OSY): four algorithms --
    - MGPSO            : no constraint handling at all (lower baseline)
    - MGPSO_feasarch   : MGPSO with feasible-only archive admission but NO
                         co-evolution (ablation: isolates the contribution
                         of the co-evolutionary multipliers)
    - CCPSO_filter     : co-evolutionary Lagrangian + MGPSO, permissive
                         archive, feasible-filtered final front
    - CCPSO_strict     : co-evolutionary Lagrangian + MGPSO, feasible-only
                         archive admission

Reproducibility: a deterministic seed is derived per (problem, algorithm)
experiment from BASE_SEED, and BOTH random number generators are seeded
(cilpy's PSO uses the stdlib `random` module; MGPSO uses `numpy.random`).
Re-running any subset of the campaign reproduces its results exactly.

Per-iteration results land in out/<problem>_<algorithm>.out.csv via the
ExperimentRunner (IGD/GD/HV/spacing/spread/feasibility per iteration, plus
the full front for plotting). After all experiments finish, this script
aggregates the per-run summaries into results_static_mo.csv with
mean +/- std over runs. Final hypervolume is recomputed from the stored
final fronts using FIXED per-problem reference points, because the runner's
per-iteration HV uses a per-run auto-derived reference and is therefore not
comparable across algorithms.

Usage (from the repository root):
    python examples/project/run_static_mo_experiments.py            # full: 30 runs x 1000 iters
    python examples/project/run_static_mo_experiments.py --quick    # smoke test: 3 runs x 100 iters
    python examples/project/run_static_mo_experiments.py --suite constrained
    python examples/project/run_static_mo_experiments.py --runs 30 --iters 500
"""

import argparse
import ast
import csv
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

# Make the repository root importable regardless of invocation directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cilpy.problem.multi_objective import (
    SCH1, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6,
    BNH, SRN, TNK, CONSTR, OSY,
)
from cilpy.solver.mgpso import MGPSO
from cilpy.solver.pso import PSO
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.runner import ExperimentRunner
from cilpy.compare.metrics import hypervolume

BASE_SEED = 26989395  # student number: memorable and citable
SWARM_SIZE = 50

# Fixed hypervolume reference points, one per problem. These must NEVER
# change between algorithms or runs being compared. Chosen a comfortable
# margin beyond the nadir of each true/expected front.
HV_REF_POINTS = {
    "SCH1":   (110.0, 110.0),
    "ZDT1":   (1.1, 1.1),
    "ZDT2":   (1.1, 1.1),
    "ZDT3":   (1.1, 1.1),
    "ZDT4":   (1.1, 1.1),
    "ZDT6":   (1.1, 1.1),
    "BNH":    (150.0, 60.0),
    "SRN":    (250.0, 100.0),
    "TNK":    (1.3, 1.3),
    "CONSTR": (1.1, 10.0),
    "OSY":    (0.0, 100.0),   # f1 <= 0 by construction; no true front available
}


def unconstrained_problems():
    return [SCH1(), ZDT1(), ZDT2(), ZDT3(), ZDT4(), ZDT6()]


def constrained_problems():
    return [BNH(), SRN(), TNK(), CONSTR(), OSY()]


def mgpso_config(name="MGPSO", feasible_archive_only=False):
    return {
        "class": MGPSO,
        "params": {
            "name": name,
            "swarm_size": SWARM_SIZE,
            "feasible_archive_only": feasible_archive_only,
            "w": 0.72, "c1": 1.49, "c2": 1.49, "c3": 1.49,
        },
    }


def ccpso_config(strategy):
    return {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": f"CCPSO_{strategy}",
            "objective_solver_class": MGPSO,
            "multiplier_solver_class": PSO,
            "objective_solver_params": {
                "swarm_size": SWARM_SIZE,
                "w": 0.72, "c1": 1.49, "c2": 1.49, "c3": 1.49,
            },
            "multiplier_solver_params": {
                "swarm_size": 30, "w": 0.40, "c1": 1.20, "c2": 1.20,
            },
            "penalty_rho": 1.0,
            "max_multiplier": 1000.0,
            "archive_strategy": strategy,
        },
    }


def experiment_seed(problem_name: str, solver_name: str) -> int:
    """Deterministic per-experiment seed. Stable across Python sessions
    (unlike hash()), so any experiment can be reproduced in isolation."""
    key = f"{problem_name}:{solver_name}"
    return BASE_SEED + sum(ord(c) * (i + 1) for i, c in enumerate(key))


def run_campaign(problem_factories, configs, num_runs, max_iterations):
    """Runs each (problem, config) pair as its own seeded experiment."""
    for factory in problem_factories:
        for config in configs:
            problem = factory()  # fresh instance per experiment
            solver_name = config["params"]["name"]
            seed = experiment_seed(problem.name, solver_name)
            random.seed(seed)      # cilpy PSO uses stdlib random
            np.random.seed(seed)   # MGPSO uses numpy random
            print(f"\n### {problem.name} x {solver_name}  (seed={seed})")
            runner = ExperimentRunner(
                problems=[problem],
                solver_configurations=[config],
                num_runs=num_runs,
                max_iterations=max_iterations,
            )
            runner.run_experiments()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean_std(values):
    if not values:
        return "", ""
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.4f}", f"{arr.std():.4f}"


def _final_fronts_from_out_csv(path):
    """Extracts the final front of each run from a per-iteration out CSV."""
    last_row_per_run = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        front_idx = len(header) - 1  # 'front' is the last column
        for row in reader:
            run_id = row[0]
            last_row_per_run[run_id] = row[front_idx]
    fronts = []
    for raw in last_row_per_run.values():
        try:
            fronts.append(ast.literal_eval(raw))
        except (ValueError, SyntaxError):
            fronts.append([])
    return fronts


def aggregate(problem_names, solver_names, out_path="results_static_mo.csv"):
    """Aggregates per-run summaries into mean +/- std per experiment."""
    rows = []
    for problem_name in problem_names:
        for solver_name in solver_names:
            summary_path = f"out/{problem_name}_{solver_name}.summary.out.csv"
            iter_path = f"out/{problem_name}_{solver_name}.out.csv"
            if not os.path.exists(summary_path):
                continue

            columns = defaultdict(list)
            with open(summary_path, newline="") as f:
                reader = csv.reader(f)
                header = [h.split("[")[0].strip() for h in next(reader)]
                for row in reader:
                    for key, value in zip(header, row):
                        if value != "":
                            columns[key].append(value)

            igd_m, igd_s = _mean_std(columns.get("final_igd", []))
            spread_m, spread_s = _mean_std(columns.get("final_spread", []))
            feas_m, feas_s = _mean_std(
                columns.get("final_front_feasibility_pct", [])
            )
            size_m, _ = _mean_std(columns.get("final_archive_size", []))
            time_m, _ = _mean_std(columns.get("run_time_s", []))

            # HV from the FEASIBLE front only, with a FIXED reference point.
            # Using the raw front would flatter algorithms whose archives sit
            # in infeasible regions (infeasible points can dominate a large
            # area of objective space they have no right to claim).
            hv_m = hv_s = ""
            ref = HV_REF_POINTS.get(problem_name)
            if ref:
                hvs = []
                for raw in columns.get("final_feasible_front", []):
                    try:
                        front = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        continue
                    # empty feasible front contributes HV = 0 (honest)
                    hvs.append(hypervolume(front, ref) if front else 0.0)
                hv_m, hv_s = _mean_std(hvs)

            rows.append([
                problem_name, solver_name,
                igd_m, igd_s, hv_m, hv_s,
                spread_m, spread_s, feas_m, feas_s,
                size_m, time_m,
            ])

    header = [
        "problem", "algorithm",
        "igd_mean", "igd_std",
        "hv_feasible_mean (fixed ref)", "hv_std",
        "spread_mean", "spread_std",
        "front_feasibility_pct_mean", "front_feasibility_pct_std",
        "archive_size_mean", "run_time_s_mean",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Console table
    print(f"\n{'problem':8} {'algorithm':16} {'IGD':>18} {'HV(feas,fixed)':>22} {'front feas%':>16}")
    for r in rows:
        igd = f"{r[2]} ± {r[3]}" if r[2] else "n/a"
        hv = f"{r[4]} ± {r[5]}" if r[4] else "n/a"
        feas = f"{r[8]} ± {r[9]}" if r[8] else "n/a"
        print(f"{r[0]:8} {r[1]:16} {igd:>18} {hv:>22} {feas:>16}")
    print(f"\nAggregated results written to {out_path}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--suite", choices=["all", "unconstrained", "constrained"],
                        default="all")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 3 runs, 100 iterations")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="skip experiments, just re-aggregate existing out/ CSVs")
    args = parser.parse_args()

    num_runs = 3 if args.quick else args.runs
    max_iterations = 100 if args.quick else args.iters

    unconstrained_names = [p.name for p in unconstrained_problems()]
    constrained_names = [p.name for p in constrained_problems()]
    constrained_configs = [
        mgpso_config("MGPSO"),
        mgpso_config("MGPSO_feasarch", feasible_archive_only=True),
        ccpso_config("filter"),
        ccpso_config("strict"),
    ]

    start = time.time()
    if not args.aggregate_only:
        if args.suite in ("all", "unconstrained"):
            run_campaign(
                [SCH1, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6],
                [mgpso_config("MGPSO")],
                num_runs, max_iterations,
            )
        if args.suite in ("all", "constrained"):
            run_campaign(
                [BNH, SRN, TNK, CONSTR, OSY],
                constrained_configs,
                num_runs, max_iterations,
            )
        print(f"\nAll experiments done in {(time.time() - start) / 60:.1f} min")

    all_problems = []
    if args.suite in ("all", "unconstrained"):
        all_problems += unconstrained_names
    if args.suite in ("all", "constrained"):
        all_problems += constrained_names
    all_solvers = ["MGPSO", "MGPSO_feasarch", "CCPSO_filter", "CCPSO_strict"]
    aggregate(all_problems, all_solvers)


if __name__ == "__main__":
    main()