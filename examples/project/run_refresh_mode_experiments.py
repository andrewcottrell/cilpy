# examples/project/run_refresh_mode_experiments.py
"""Archive refresh mode campaign: "clean" vs "clear".

Compares the two archive refresh policies on environment change:

* ``clean``  -- the archive is re-scored in place and only members that
  have become infeasible or dominated are removed, so surviving members
  (and the knowledge encoded in their positions) persist across changes.
* ``clear``  -- the archive is discarded on every change and rebuilt from
  scratch, as in a cold start.

PAIRED DESIGN
-------------
The two modes of a given (problem, algorithm) pair receive the SAME seed,
so run *i* of ``clean`` and run *i* of ``clear`` begin from identical
swarms and encounter identical random draws until the first environment
change. Differences are therefore attributable to the refresh policy
rather than to sampling. Aggregation exploits this with a Wilcoxon
signed-rank test (the paired counterpart of the Mann--Whitney U test used
elsewhere), which is markedly more sensitive than an unpaired test at the
same number of runs.

SCOPE
-----
The refresh mode is swept only for algorithms that actually maintain a
front worth retaining. The plain MGPSO on the DTNK family returns no
feasible solutions at any frequency, so retaining or discarding its
archive is meaningless; it is run once, under ``clean``, as the baseline
reference. On the FDA problems the plain MGPSO does track a genuine
front and is swept.

Output files carry the mode in the solver name
(e.g. ``DTNK_CCPSO_strict_clean.out.csv``) so that the two modes never
overwrite each other.

Usage (from the repository root):
    python examples/project/run_refresh_mode_experiments.py
    python examples/project/run_refresh_mode_experiments.py --quick
    python examples/project/run_refresh_mode_experiments.py --workers 4
    python examples/project/run_refresh_mode_experiments.py --tau-t 10
    python examples/project/run_refresh_mode_experiments.py --aggregate-only
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

from cilpy.problem.dynamic_multi_objective import FDA1, FDA3, DTNK, DTNK2, DTNK3
from cilpy.solver.mgpso import MGPSO
from cilpy.solver.pso import PSO
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.runner import ExperimentRunner

try:
    from scipy.stats import wilcoxon
except ImportError:  # analysis degrades gracefully without scipy
    wilcoxon = None

BASE_SEED = 26989395
SWARM_SIZE = 50
FIXED_PARAMS = {"w": 0.72, "c1": 1.49, "c2": 1.49, "c3": 1.49}
REFRESH_MODES = ("clean", "clear")


# ---------------------------------------------------------------------------
# Solver configurations
# ---------------------------------------------------------------------------

def mgpso_config(base_name, refresh_mode, feasible_archive_only=False):
    return {
        "class": MGPSO,
        "params": {
            "name": f"{base_name}_{refresh_mode}",
            "swarm_size": SWARM_SIZE,
            "feasible_archive_only": feasible_archive_only,
            "refresh_mode": refresh_mode,
            **FIXED_PARAMS,
        },
    }


def ccpso_config(strategy, refresh_mode):
    return {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": f"CCPSO_{strategy}_{refresh_mode}",
            "objective_solver_class": MGPSO,
            "multiplier_solver_class": PSO,
            "objective_solver_params": {
                "swarm_size": SWARM_SIZE,
                "refresh_mode": refresh_mode,
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


def base_algorithm_name(solver_name):
    """Strips the refresh-mode suffix, so paired experiments share a seed."""
    for mode in REFRESH_MODES:
        if solver_name.endswith(f"_{mode}"):
            return solver_name[: -(len(mode) + 1)]
    return solver_name


def experiment_seed(problem_name, solver_name):
    """Deterministic seed, IDENTICAL for the two refresh modes of a pair."""
    key = f"{problem_name}:{base_algorithm_name(solver_name)}"
    return BASE_SEED + sum(ord(c) * (i + 1) for i, c in enumerate(key))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _run_one_experiment(task):
    factory, config, num_runs, max_iterations, tau_t, n_t = task
    problem = factory(tau_t=tau_t, n_t=n_t)
    solver_name = config["params"]["name"]
    seed = experiment_seed(problem.name, solver_name)
    random.seed(seed)
    np.random.seed(seed)
    print(f"\n### {problem.name} x {solver_name}  (seed={seed}, tau_t={tau_t})")
    ExperimentRunner(
        problems=[problem],
        solver_configurations=[config],
        num_runs=num_runs,
        max_iterations=max_iterations,
    ).run_experiments()
    return f"{problem.name} x {solver_name}"


def build_tasks(num_runs, max_iterations, tau_t, n_t):
    tasks = []

    # FDA: plain MGPSO tracks a genuine front -> sweep both modes.
    for factory in (FDA1, FDA3):
        for mode in REFRESH_MODES:
            tasks.append((factory, mgpso_config("MGPSO", mode),
                          num_runs, max_iterations, tau_t, n_t))

    # DTNK family: sweep the algorithms that maintain a feasible front.
    for factory in (DTNK, DTNK3, DTNK2):
        for mode in REFRESH_MODES:
            tasks.append((factory,
                          mgpso_config("MGPSO_feasarch", mode,
                                       feasible_archive_only=True),
                          num_runs, max_iterations, tau_t, n_t))
            for strategy in ("filter", "strict"):
                tasks.append((factory, ccpso_config(strategy, mode),
                              num_runs, max_iterations, tau_t, n_t))
        # Plain MGPSO baseline: no feasible front to retain, run once.
        tasks.append((factory, mgpso_config("MGPSO", "clean"),
                      num_runs, max_iterations, tau_t, n_t))

    return tasks


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
# Aggregation with paired analysis
# ---------------------------------------------------------------------------

def _migd_per_run(path, tau_t):
    """Returns {run_id: (MIGD, MIGD_bc)} from a per-iteration CSV."""
    series = defaultdict(list)
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.split("[")[0].strip() for h in next(reader)]
        try:
            idx = header.index("igd")
        except ValueError:
            return {}
        for row in reader:
            if row[idx] != "":
                series[row[0]].append((int(row[1]), float(row[idx])))

    out = {}
    for run_id, pairs in series.items():
        values = [v for _, v in pairs]
        before = [v for it, v in pairs if it % tau_t == tau_t - 1]
        if values:
            out[run_id] = (
                float(np.mean(values)),
                float(np.mean(before)) if before else float("nan"),
            )
    return out


def aggregate(tau_t, out_path="results_refresh_mode.csv"):
    problems = ["FDA1", "FDA3", "DTNK", "DTNK3", "DTNK2"]
    algorithms = ["MGPSO", "MGPSO_feasarch", "CCPSO_filter", "CCPSO_strict"]

    rows = []
    print(f"\n{'problem':7} {'algorithm':16} {'clean MIGD':>18} "
          f"{'clear MIGD':>18} {'paired p':>10} {'better':>7}")

    for problem in problems:
        for algorithm in algorithms:
            per_mode = {}
            for mode in REFRESH_MODES:
                path = f"out/{problem}_{algorithm}_{mode}.out.csv"
                per_mode[mode] = _migd_per_run(path, tau_t)

            if not per_mode["clean"] or not per_mode["clear"]:
                continue

            # Pair by run_id: run i of clean against run i of clear.
            shared = sorted(set(per_mode["clean"]) & set(per_mode["clear"]),
                            key=int)
            clean = np.array([per_mode["clean"][r][0] for r in shared])
            clear = np.array([per_mode["clear"][r][0] for r in shared])
            clean_bc = np.array([per_mode["clean"][r][1] for r in shared])
            clear_bc = np.array([per_mode["clear"][r][1] for r in shared])

            p_value = float("nan")
            verdict = "n/a"
            if wilcoxon is not None and len(shared) >= 5:
                differences = clean - clear
                if np.any(differences != 0):
                    _, p_value = wilcoxon(clean, clear)
                    if p_value < 0.05:
                        verdict = ("clean" if np.median(clean)
                                   < np.median(clear) else "clear")
                    else:
                        verdict = "tie"
                else:
                    p_value = 1.0
                    verdict = "identical"

            rows.append([
                problem, algorithm, len(shared),
                f"{clean.mean():.4f}", f"{clean.std():.4f}",
                f"{clear.mean():.4f}", f"{clear.std():.4f}",
                f"{clean_bc.mean():.4f}", f"{clear_bc.mean():.4f}",
                f"{p_value:.4g}", verdict,
            ])
            print(f"{problem:7} {algorithm:16} "
                  f"{clean.mean():9.4f} ± {clean.std():6.4f} "
                  f"{clear.mean():9.4f} ± {clear.std():6.4f} "
                  f"{p_value:10.4g} {verdict:>7}")

    header = [
        "problem", "algorithm", "paired_runs",
        "clean_migd_mean", "clean_migd_std",
        "clear_migd_mean", "clear_migd_std",
        "clean_migd_bc_mean", "clear_migd_bc_mean",
        "wilcoxon_p", "better",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nAggregated results written to {out_path}")
    if wilcoxon is None:
        print("NOTE: scipy unavailable; paired tests skipped.")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--tau-t", type=int, default=10,
                        help="change frequency; the refresh policy matters "
                             "most at fast frequencies")
    parser.add_argument("--n-t", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 3 runs, 100 iterations")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    num_runs = 3 if args.quick else args.runs
    max_iterations = 100 if args.quick else args.iters

    if not args.aggregate_only:
        tasks = build_tasks(num_runs, max_iterations, args.tau_t, args.n_t)
        print(f"Refresh-mode campaign: {len(tasks)} experiments, "
              f"{num_runs} runs x {max_iterations} iterations, "
              f"tau_t={args.tau_t}")
        start = time.time()
        run_campaign(tasks, args.workers)
        print(f"\nAll experiments done in {(time.time() - start) / 60:.1f} min")

    aggregate(args.tau_t)


if __name__ == "__main__":
    main()
