# examples/project/run_dynamic_mo_experiments.py
"""Dynamic multi-objective experimental campaign (Milestone 5).

Covers three dynamic categories (SOSC is the static campaign):

* Dynamic objectives (FDA1, FDA3): plain MGPSO only -- box constraints, so
  the co-evolutionary framework does not apply. Tests the change-response
  machinery (sentinel detection, archive + pbest re-evaluation).
* SODC (DTNK -- static objectives, oscillating constraint boundary),
  DOSC (DTNK3 -- moving objectives over the static TNK ring),
  DODC (DTNK2 -- translated objectives + oscillating radius), and
  DODC (DTNK4 -- orbiting shape-changing crescent + rotated objectives):
  the full four-algorithm grid, i.e. plain MGPSO, the feasible-archive
  ablation, and CCPSO filter/strict.

Reporting uses the standard dynamic-MO metrics, computed from the runner's
per-iteration IGD column (the runner refreshes the reference front to the
CURRENT environment each iteration, so per-iteration IGD is always measured
against the correct front):

* MIGD     -- mean IGD over ALL iterations of a run: overall tracking
  quality, penalising both slow recovery after changes and poor converged
  quality between them.
* MIGD_bc  -- mean IGD sampled at the LAST iteration of each environment
  (just before each change): converged quality per environment, ignoring
  the transient recovery spikes. The gap between MIGD and MIGD_bc measures
  recovery speed.

Seeding follows the static campaign exactly (per-experiment deterministic
seeds for both RNGs).

Usage (from the repository root):
    python examples/project/run_dynamic_mo_experiments.py             # 30 x 1000
    python examples/project/run_dynamic_mo_experiments.py --quick
    python examples/project/run_dynamic_mo_experiments.py --workers 4
    python examples/project/run_dynamic_mo_experiments.py --tau-t 25  # slower changes
"""

import argparse
import csv
import multiprocessing
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cilpy.problem.dynamic_multi_objective import FDA1, FDA3, DTNK, DTNK2, DTNK3, DTNK4
from cilpy.solver.mgpso import MGPSO
from cilpy.solver.pso import PSO
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.compare.metrics import p_red as compute_p_red
from cilpy.runner import ExperimentRunner

BASE_SEED = 26989395
SWARM_SIZE = 50

# Fixed MGPSO parameters, consistent with the static campaign / Phase 1.
FIXED_PARAMS = {"w": 0.72, "c1": 1.49, "c2": 1.49, "c3": 1.49}


def mgpso_config(name="MGPSO", feasible_archive_only=False,
                 sentry_mode="fixed", n_sentries=None):
    return {
        "class": MGPSO,
        "params": {
            "name": name,
            "swarm_size": SWARM_SIZE,
            "feasible_archive_only": feasible_archive_only,
            "sentry_mode": sentry_mode,
            "n_sentries": n_sentries,
            **FIXED_PARAMS,
        },
    }


def ccpso_config(strategy, sentry_mode="fixed", n_sentries=None):
    name = f"CCPSO_{strategy}"
    if sentry_mode == "archive":
        name += "_archsentry"
    return {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": name,
            "objective_solver_class": MGPSO,
            "multiplier_solver_class": PSO,
            "objective_solver_params": {
                "swarm_size": SWARM_SIZE,
                "sentry_mode": sentry_mode,
                "n_sentries": n_sentries,
                **FIXED_PARAMS,
            },
            "multiplier_solver_params": {
                "swarm_size": 30, "w": 0.40, "c1": 1.20, "c2": 1.20,
            },
            "penalty_rho": 1.0,
            "max_multiplier": 1000.0,
            "archive_strategy": strategy,
        },
    }


def experiment_seed(problem_name, solver_name):
    key = f"{problem_name}:{solver_name}"
    return BASE_SEED + sum(ord(c) * (i + 1) for i, c in enumerate(key))


def _run_one_experiment(task):
    factory, config, num_runs, max_iterations, tau_t, n_t = task
    problem = factory(tau_t=tau_t, n_t=n_t)
    solver_name = config["params"]["name"]
    seed = experiment_seed(problem.name, solver_name)
    random.seed(seed)
    np.random.seed(seed)
    print(f"\n### {problem.name} x {solver_name}  "
          f"(seed={seed}, tau_t={tau_t}, n_t={n_t})")
    ExperimentRunner(
        problems=[problem],
        solver_configurations=[config],
        num_runs=num_runs,
        max_iterations=max_iterations,
    ).run_experiments()
    return f"{problem.name} x {solver_name}"


def run_campaign(tasks, workers):
    if workers <= 1:
        for task in tasks:
            _run_one_experiment(task)
        return
    completed = 0
    with multiprocessing.Pool(processes=workers) as pool:
        for name in pool.imap_unordered(_run_one_experiment, tasks):
            completed += 1
            print(f"\n>>> [{completed}/{len(tasks)}] finished: {name}")


# ---------------------------------------------------------------------------
# MIGD aggregation
# ---------------------------------------------------------------------------

def _mean_std(values):
    if not values:
        return "", ""
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.4f}", f"{arr.std():.4f}"


def _per_run_series(path, column):
    """Extracts {run_id: [(iteration, value), ...]} for a metric column."""
    series = defaultdict(list)
    if not os.path.exists(path):
        return series
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.split("[")[0].strip() for h in next(reader)]
        try:
            idx = header.index(column)
        except ValueError:
            return series
        for row in reader:
            if row[idx] != "":
                series[row[0]].append((int(row[1]), float(row[idx])))
    return series


def aggregate(problem_names, solver_names, tau_t,
              out_path="results_dynamic_mo.csv"):
    rows = []
    for problem_name in problem_names:
        for solver_name in solver_names:
            iter_path = f"out/{problem_name}_{solver_name}.out.csv"
            summary_path = f"out/{problem_name}_{solver_name}.summary.out.csv"
            if not os.path.exists(iter_path):
                continue

            igd_series = _per_run_series(iter_path, "igd")
            migd_per_run, migd_bc_per_run = [], []
            for run_series in igd_series.values():
                values = [v for _, v in run_series]
                if values:
                    migd_per_run.append(float(np.mean(values)))
                # Environments change when tau crosses multiples of tau_t,
                # i.e. iteration k*tau_t is the FIRST iteration of a new
                # environment; the last iteration of the previous one is
                # k*tau_t - 1.
                before_change = [
                    v for it, v in run_series if it % tau_t == tau_t - 1
                ]
                if before_change:
                    migd_bc_per_run.append(float(np.mean(before_change)))

            # --- P_RED from summary CSV ---
            p_red_values = []
            feas = []
            if os.path.exists(summary_path):
                with open(summary_path, newline="") as f:
                    reader = csv.reader(f)
                    header = [h.split("[")[0].strip() for h in next(reader)]
                    try:
                        f_idx = header.index("final_front_feasibility_pct")
                    except ValueError:
                        f_idx = None
                    try:
                        pr_idx = header.index("p_red")
                    except ValueError:
                        pr_idx = None
                    for r in reader:
                        if f_idx is not None and r[f_idx] != "":
                            feas.append(float(r[f_idx]))
                        if pr_idx is not None and r[pr_idx] != "":
                            p_red_values.append(float(r[pr_idx]))

            # --- RED before-change (from per-iteration relative_error) ---
            red_series = _per_run_series(iter_path, "relative_error")
            red_bc_per_run = []
            for run_series in red_series.values():
                before_change = [
                    v for it, v in run_series if it % tau_t == tau_t - 1
                ]
                if before_change:
                    red_bc_per_run.append(compute_p_red(before_change))

            migd_m, migd_s = _mean_std(migd_per_run)
            bc_m, bc_s = _mean_std(migd_bc_per_run)
            feas_m, feas_s = _mean_std(feas)
            pr_m, pr_s = _mean_std(p_red_values)
            rbc_m, rbc_s = _mean_std(red_bc_per_run)
            rows.append([
                problem_name, solver_name,
                migd_m, migd_s, bc_m, bc_s, feas_m, feas_s,
                pr_m, pr_s, rbc_m, rbc_s,
            ])

    header = [
        "problem", "algorithm",
        "migd_mean", "migd_std",
        "migd_before_change_mean", "migd_before_change_std",
        "front_feasibility_pct_mean", "front_feasibility_pct_std",
        "p_red_mean", "p_red_std",
        "p_red_before_change_mean", "p_red_before_change_std",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n{'problem':8} {'algorithm':16} {'MIGD':>18} "
          f"{'MIGD(before chg)':>20} {'front feas%':>16} "
          f"{'P_RED':>18} {'P_RED(bc)':>18}")
    for r in rows:
        migd = f"{r[2]} ± {r[3]}" if r[2] else "n/a"
        bc = f"{r[4]} ± {r[5]}" if r[4] else "n/a"
        feas = f"{r[6]} ± {r[7]}" if r[6] else "n/a"
        pred = f"{r[8]} ± {r[9]}" if r[8] else "n/a"
        pred_bc = f"{r[10]} ± {r[11]}" if r[10] else "n/a"
        print(f"{r[0]:8} {r[1]:16} {migd:>18} {bc:>20} {feas:>16} "
              f"{pred:>18} {pred_bc:>18}")
    print(f"\nAggregated results written to {out_path}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--tau-t", type=int, default=10,
                        help="change frequency: iterations per environment")
    parser.add_argument("--n-t", type=int, default=10,
                        help="change severity: steps discretising t")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    num_runs = 3 if args.quick else args.runs
    max_iterations = 100 if args.quick else args.iters

    constrained_configs = [
        mgpso_config("MGPSO"),
        mgpso_config("MGPSO_feasarch", feasible_archive_only=True),
        ccpso_config("filter"),
        ccpso_config("strict"),
        mgpso_config("MGPSO_archsentry", sentry_mode="archive"),
        mgpso_config("MGPSO_feasarch_archsentry",
                     feasible_archive_only=True, sentry_mode="archive"),
        ccpso_config("filter", sentry_mode="archive"),
        ccpso_config("strict", sentry_mode="archive"),
    ]
    fda_configs = [
        mgpso_config("MGPSO"),
        mgpso_config("MGPSO_archsentry", sentry_mode="archive"),
    ]
    tasks = (
        [(f, c, num_runs, max_iterations, args.tau_t, args.n_t)
         for f in (FDA1, FDA3) for c in fda_configs]
        + [(f, c, num_runs, max_iterations, args.tau_t, args.n_t)
           for f in (DTNK, DTNK3, DTNK2, DTNK4) for c in constrained_configs]
    )

    start = time.time()
    if not args.aggregate_only:
        run_campaign(tasks, args.workers)
        print(f"\nAll experiments done in {(time.time() - start) / 60:.1f} min")

    aggregate(
        ["FDA1", "FDA3", "DTNK", "DTNK3", "DTNK2", "DTNK4"],
        ["MGPSO", "MGPSO_feasarch", "CCPSO_filter", "CCPSO_strict",
         "MGPSO_archsentry", "MGPSO_feasarch_archsentry",
         "CCPSO_filter_archsentry", "CCPSO_strict_archsentry"],
        args.tau_t,
    )


if __name__ == "__main__":
    main()
