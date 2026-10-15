# examples/project/run_phase1.py
"""Phase 1 re-run: the single-objective co-evolutionary Lagrangian solver.

Re-runs the Phase 1 validation with the current code (the July 2026 runs
predate the fix that re-scores the multiplier swarm, and their P_RED came from
an older runner). Problems:

* CEC2006 G01, G02, G04, G05, G06 -- static objectives, static constraints.
* CMPB in all four classes: SOSC (f and g static), SODC (g moves),
  DOSC (f moves), DODC (f and g move). Dynamic landscapes use the "A2R"
  moving peaks configuration, static ones "STA".

Configuration as in the report (Appendix A, Phase 1): P1 and P2 are PSOs of
200 particles, P1 with w = 0.72, c1 = c2 = 1.49 and sentry change detection,
P2 with w = 0.4, c1 = c2 = 1.2; rho = rho_eq = 0.5; max_multiplier = 10000;
30 runs of 1000 iterations, seeded per experiment as in the other campaigns.

Usage (from the repository root):
    python examples/project/run_phase1.py --workers 4
    python examples/project/run_phase1.py --quick
    python examples/project/run_phase1.py --aggregate-only --out-dir raw/out_phase1

The runner writes to out/ in the working directory; the aggregate table goes
to results/results_phase1.csv.
"""

import argparse
import csv
import multiprocessing
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from paths import results_path
from configs import experiment_seed
from cilpy.problem.constrained import G01, G02, G04, G05, G06
from cilpy.problem.cmpb import ConstrainedMovingPeaksBenchmark
from cilpy.problem.mpb import generate_mpb_configs
from cilpy.runner import ExperimentRunner
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.solver.pso import PSO

SOLVER_NAME = "CCLS"

CEC_PROBLEMS = {"G01": G01, "G02": G02, "G04": G04, "G05": G05, "G06": G06}

# (objective landscape, constraint landscape) per CMPB class
CMPB_CLASSES = {
    "CMPB_SOSC": ("STA", "STA"),
    "CMPB_SODC": ("STA", "A2R"),
    "CMPB_DOSC": ("A2R", "STA"),
    "CMPB_DODC": ("A2R", "A2R"),
}

PROBLEM_NAMES = list(CEC_PROBLEMS) + list(CMPB_CLASSES)


def make_problem(name):
    if name in CEC_PROBLEMS:
        return CEC_PROBLEMS[name]()
    f_key, g_key = CMPB_CLASSES[name]
    mpb_configs = generate_mpb_configs(dimension=2)
    return ConstrainedMovingPeaksBenchmark(
        f_params=mpb_configs[f_key], g_params=mpb_configs[g_key], name=name)


def solver_config():
    return {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": SOLVER_NAME,
            "penalty_rho": 0.5,
            "penalty_rho_equality": 0.5,
            "max_multiplier": 10000.0,
            "objective_solver_class": PSO,
            "multiplier_solver_class": PSO,
            "objective_solver_params": {
                "swarm_size": 200, "w": 0.72, "c1": 1.49, "c2": 1.49,
                "detect_changes": True,
            },
            "multiplier_solver_params": {
                "swarm_size": 200, "w": 0.40, "c1": 1.20, "c2": 1.20,
            },
        },
    }


def _run_one(task):
    name, num_runs, max_iterations = task
    seed = experiment_seed(name, SOLVER_NAME)
    random.seed(seed)
    np.random.seed(seed)
    problem = make_problem(name)  # after seeding: CMPB peaks are random
    print(f"\n### {name} x {SOLVER_NAME}  (seed={seed})")
    ExperimentRunner(
        problems=[problem],
        solver_configurations=[solver_config()],
        num_runs=num_runs,
        max_iterations=max_iterations,
    ).run_experiments()
    return name


def run_campaign(tasks, workers):
    if workers <= 1:
        for task in tasks:
            _run_one(task)
        return
    with multiprocessing.Pool(processes=workers) as pool:
        for done, name in enumerate(pool.imap_unordered(_run_one, tasks), 1):
            print(f"\n>>> [{done}/{len(tasks)}] finished: {name}")


def _column(path, prefix):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.split("[")[0].strip() for h in next(reader)]
        idx = header.index(prefix)
        return [float(r[idx]) for r in reader if r[idx] != ""]


def aggregate(out_dir, out_path=None):
    out_path = out_path or results_path("results_phase1.csv")
    rows = []
    for name in PROBLEM_NAMES:
        path = os.path.join(out_dir, f"{name}_{SOLVER_NAME}.summary.out.csv")
        if not os.path.exists(path):
            continue
        f = _column(path, "final_best_fitness")
        feas = _column(path, "final_feasibility_pct")
        pred = _column(path, "p_red")
        evals = _column(path, "evaluations")
        rows.append({
            "problem": name,
            "runs": len(f),
            "fitness_mean": np.mean(f), "fitness_std": np.std(f),
            "feasibility_pct_mean": np.mean(feas),
            "feasibility_pct_std": np.std(feas),
            "p_red_mean": np.mean(pred) if pred else "",
            "p_red_std": np.std(pred) if pred else "",
            "evaluations_mean": np.mean(evals) if evals else "",
        })
    if not rows:
        print(f"No Phase 1 summaries found in {out_dir}")
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['problem']:<10} f={r['fitness_mean']:.4f} "
              f"feas={r['feasibility_pct_mean']:.1f}% P_RED={r['p_red_mean']}")
    print(f"\nPhase 1 table written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--problems", nargs="+", default=PROBLEM_NAMES,
                        choices=PROBLEM_NAMES)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--out-dir", default="out",
                        help="folder of runner CSVs to aggregate")
    args = parser.parse_args()

    num_runs = 2 if args.quick else args.runs
    max_iterations = 50 if args.quick else args.iters
    tasks = [(name, num_runs, max_iterations) for name in args.problems]

    if not args.aggregate_only:
        start = time.time()
        run_campaign(tasks, args.workers)
        print(f"\nPhase 1 done in {(time.time() - start) / 60:.1f} min")
    aggregate(args.out_dir)


if __name__ == "__main__":
    main()
