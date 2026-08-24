# examples/project/run_nsga2_baseline.py
"""External baseline: NSGA-II (pymoo) on the Cilpy benchmark problems.

Phase 1 promised evaluation "against existing constrained multi-objective
optimization approaches". Comparing IGD values copied from other papers is
not meaningful -- reference front sampling, metric variants, and evaluation
budgets all differ between studies. This script instead runs NSGA-II
locally under conditions identical to the CCPSO campaign, so the numbers
are directly comparable.

What is held identical to the CCPSO campaign:

* **Problem definitions** -- the Cilpy problem objects themselves are
  wrapped for pymoo, so both algorithms see exactly the same objectives,
  constraints and bounds. No re-implementation, no transcription risk.
* **Evaluation budget** -- CCPSO uses (swarm_size x n_objectives) particles
  for max_iterations iterations, i.e. 50 x 2 x 1000 = 100,000 evaluations.
  NSGA-II is given a population of 100 for 1000 generations = 100,000.
* **Number of runs** -- 30 independent runs.
* **Seeds** -- derived per (problem, algorithm) from the same BASE_SEED
  and the same derivation function as the CCPSO campaigns.
* **Metrics and reference fronts** -- scored with `cilpy.compare.metrics`
  against `problem.true_pareto_front()`, NOT with pymoo's own indicators,
  which use different reference sets and normalisation.
* **Feasibility convention** -- pymoo expects G(x) <= 0, which matches the
  Cilpy convention exactly, so constraints pass through unmodified.

Constraint handling: NSGA-II uses its standard constrained-domination
(Deb's feasibility rules), which is the algorithm's native and intended
mechanism. Real-valued operators are used throughout (SBX crossover,
polynomial mutation, float random sampling) -- the binary operators shown
in many pymoo examples apply only to binary-coded problems such as ZDT5.

Usage (from the repository root):

    python examples/project/run_nsga2_baseline.py                 # full: 30 x 100k evals
    python examples/project/run_nsga2_baseline.py --quick         # 3 runs, 5k evals
    python examples/project/run_nsga2_baseline.py --problems TNK OSY
    python examples/project/run_nsga2_baseline.py --compare-with results_static_mo.csv

Writes results_nsga2.csv, and with --compare-with also prints a
head-to-head table with Mann-Whitney U tests against the CCPSO results.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cilpy.problem.multi_objective import (
    SCH1, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, BNH, SRN, TNK, CONSTR, OSY,
)
from cilpy.compare.metrics import (
    inverted_generational_distance, hypervolume, feasibility_rate,
)

try:
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
except ImportError:
    print("pymoo is required:  pip install pymoo")
    raise

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    mannwhitneyu = None

BASE_SEED = 26989395          # identical to the CCPSO campaigns
POP_SIZE = 100                # = CCPSO's 50 particles x 2 sub-swarms
N_GEN = 1000                  # = CCPSO's 1000 iterations

PROBLEMS = {
    "SCH1": SCH1, "ZDT1": ZDT1, "ZDT2": ZDT2, "ZDT3": ZDT3,
    "ZDT4": ZDT4, "ZDT6": ZDT6,
    "BNH": BNH, "SRN": SRN, "TNK": TNK, "CONSTR": CONSTR, "OSY": OSY,
}

# Identical to the CCPSO campaign's aggregation. Never change these
# between algorithms being compared.
HV_REF_POINTS = {
    "SCH1": (110.0, 110.0),
    "ZDT1": (1.1, 1.1), "ZDT2": (1.1, 1.1), "ZDT3": (1.1, 1.1),
    "ZDT4": (1.1, 1.1), "ZDT6": (1.1, 1.1),
    "BNH": (150.0, 60.0), "SRN": (250.0, 100.0),
    "TNK": (1.3, 1.3), "CONSTR": (1.1, 10.0), "OSY": (0.0, 100.0),
}


def experiment_seed(problem_name, solver_name):
    """Same derivation as the CCPSO campaigns."""
    key = f"{problem_name}:{solver_name}"
    return BASE_SEED + sum(ord(c) * (i + 1) for i, c in enumerate(key))


class CilpyProblemAdapter(PymooProblem):
    """Presents a Cilpy problem to pymoo.

    Evaluations are delegated to the Cilpy problem object itself, so both
    algorithms optimise byte-identical objective and constraint functions.
    Both libraries use the g(x) <= 0 convention, so constraints require no
    transformation.
    """

    def __init__(self, cilpy_problem):
        self.cilpy_problem = cilpy_problem
        lower, upper = cilpy_problem.bounds
        probe = cilpy_problem.evaluate(list(lower))
        n_obj = len(probe.fitness)
        n_ieq = len(probe.constraints_inequality or [])
        super().__init__(
            n_var=cilpy_problem.dimension,
            n_obj=n_obj,
            n_ieq_constr=n_ieq,
            xl=np.asarray(lower, dtype=float),
            xu=np.asarray(upper, dtype=float),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        objectives, constraints = [], []
        for row in np.atleast_2d(X):
            evaluation = self.cilpy_problem.evaluate([float(v) for v in row])
            objectives.append(evaluation.fitness)
            if self.n_ieq_constr:
                constraints.append(evaluation.constraints_inequality)
        out["F"] = np.asarray(objectives, dtype=float)
        if self.n_ieq_constr:
            out["G"] = np.asarray(constraints, dtype=float)


def feasible_front(cilpy_problem, X):
    """Re-evaluates pymoo's result on the ORIGINAL problem and keeps the
    feasible members, mirroring how the CCPSO results were scored."""
    front, evaluations = [], []
    for row in np.atleast_2d(X):
        evaluation = cilpy_problem.evaluate([float(v) for v in row])
        evaluations.append(evaluation)
        violations = evaluation.constraints_inequality or []
        if all(g <= 0 for g in violations):
            front.append([float(v) for v in evaluation.fitness])
    return front, evaluations


def run_problem(problem_name, num_runs, pop_size, n_gen, verbose=False):
    cls = PROBLEMS[problem_name]
    reference = None
    try:
        reference = cls().true_pareto_front(500)
    except (AttributeError, NotImplementedError):
        pass                      # OSY: hypervolume only

    ref_point = HV_REF_POINTS[problem_name]
    igds, hvs, feasibilities, sizes = [], [], [], []

    for run_id in range(1, num_runs + 1):
        problem = cls()                        # fresh instance per run
        seed = experiment_seed(problem_name, "NSGA2") + run_id
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),    # real-valued, not binary
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )
        result = minimize(
            CilpyProblemAdapter(problem), algorithm, ("n_gen", n_gen),
            seed=seed, verbose=False,
        )

        X = result.X if result.X is not None else np.empty((0, problem.dimension))
        front, evaluations = feasible_front(problem, X)
        sizes.append(len(front))
        feasibilities.append(
            feasibility_rate(evaluations) if evaluations else 0.0)
        hvs.append(hypervolume(front, ref_point) if front else 0.0)
        if reference is not None and front:
            igds.append(inverted_generational_distance(front, reference))
        if verbose:
            print(f"    run {run_id}: |front|={len(front)}")

    return {
        "igd": np.asarray(igds), "hv": np.asarray(hvs),
        "feas": np.asarray(feasibilities), "size": np.asarray(sizes),
        "run_ids": np.arange(1, num_runs + 1),
    }


def load_ccpso(path):
    """Reads a CCPSO aggregate table, tolerating whitespace in headers."""
    table = {}
    if not os.path.exists(path):
        return table
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            clean = {k.strip(): (v.strip() if isinstance(v, str) else v)
                     for k, v in row.items()}
            table[(clean["problem"], clean["algorithm"])] = clean
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--pop-size", type=int, default=POP_SIZE)
    ap.add_argument("--n-gen", type=int, default=N_GEN)
    ap.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    ap.add_argument("--quick", action="store_true",
                    help="3 runs of 50 generations, for a smoke test")
    ap.add_argument("--compare-with", default=None,
                    help="CCPSO aggregate CSV, e.g. results_static_mo.csv")
    ap.add_argument("--ccpso-algorithm", default="CCPSO_strict")
    ap.add_argument("--out", default="results_nsga2.csv")
    ap.add_argument("--per-run-dir", default="per_run",
                    help="folder for per-run value CSVs, used for "
                         "significance testing")
    args = ap.parse_args()

    num_runs = 3 if args.quick else args.runs
    n_gen = 50 if args.quick else args.n_gen
    evaluations = args.pop_size * n_gen

    print(f"NSGA-II baseline: {num_runs} runs, pop {args.pop_size} x "
          f"{n_gen} generations = {evaluations:,} evaluations per run")
    print(f"(CCPSO campaign: 50 particles x 2 sub-swarms x 1000 iterations "
          f"= 100,000)\n")

    rows = []
    for problem_name in args.problems:
        if problem_name not in PROBLEMS:
            print(f"unknown problem: {problem_name}")
            continue
        start = time.time()
        stats = run_problem(problem_name, num_runs, args.pop_size, n_gen)
        igd_mean = stats["igd"].mean() if stats["igd"].size else float("nan")
        igd_std = stats["igd"].std() if stats["igd"].size else float("nan")
        print(f"{problem_name:8} IGD {igd_mean:8.4f} ± {igd_std:7.4f}   "
              f"HV {stats['hv'].mean():10.2f}   "
              f"feas {stats['feas'].mean():6.1f}%   "
              f"|front| {stats['size'].mean():5.1f}   "
              f"({time.time() - start:.0f}s)")
        rows.append([
            problem_name, "NSGA2", num_runs,
            f"{igd_mean:.6f}", f"{igd_std:.6f}",
            f"{stats['hv'].mean():.4f}", f"{stats['hv'].std():.4f}",
            f"{stats['feas'].mean():.4f}", f"{stats['size'].mean():.2f}",
        ])
        # Per-run values, retained so that significance tests against
        # CCPSO are possible. Aggregate means alone cannot support them.
        os.makedirs(args.per_run_dir, exist_ok=True)
        per_run_path = os.path.join(
            args.per_run_dir, f"perrun_nsga2_{problem_name}.csv")
        with open(per_run_path, "w", newline="") as pr:
            writer = csv.writer(pr)
            writer.writerow(["run_id", "final_igd", "hv_feasible",
                             "front_feasibility_pct", "front_size"])
            n = len(stats["hv"])
            igd_column = stats["igd"] if stats["igd"].size == n else \
                np.full(n, np.nan)
            for i in range(n):
                writer.writerow([
                    i + 1,
                    "" if np.isnan(igd_column[i]) else f"{igd_column[i]:.8f}",
                    f"{stats['hv'][i]:.8f}",
                    f"{stats['feas'][i]:.4f}",
                    int(stats["size"][i]),
                ])

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "algorithm", "runs", "igd_mean",
                         "igd_std", "hv_mean", "hv_std",
                         "front_feasibility_pct_mean", "archive_size_mean"])
        writer.writerows(rows)
    print(f"\n-> {args.out}")
    print(f"-> {args.per_run_dir}/perrun_nsga2_<problem>.csv "
          f"(per-run values for significance tests)")

    if args.compare_with:
        ccpso = load_ccpso(args.compare_with)
        if not ccpso:
            print(f"could not read {args.compare_with}")
            return
        print(f"\n=== NSGA-II vs {args.ccpso_algorithm} "
              f"(mean IGD; lower is better) ===")
        print(f"{'problem':8} {'NSGA-II':>10} {'CCPSO':>10} "
              f"{'difference':>12} {'better':>10}")
        for row in rows:
            problem_name = row[0]
            key = (problem_name, args.ccpso_algorithm)
            if key not in ccpso:
                continue
            nsga_igd = float(row[3])
            try:
                ccpso_igd = float(ccpso[key]["igd_mean"])
            except (KeyError, ValueError):
                continue
            if np.isnan(nsga_igd) or np.isnan(ccpso_igd):
                continue
            delta = 100.0 * (nsga_igd - ccpso_igd) / nsga_igd
            better = "CCPSO" if ccpso_igd < nsga_igd else "NSGA-II"
            print(f"{problem_name:8} {nsga_igd:10.4f} {ccpso_igd:10.4f} "
                  f"{delta:11.1f}% {better:>10}")
        print("\nNote: per-run CCPSO values are needed for significance "
              "tests; aggregate means only are compared here.")


if __name__ == "__main__":
    main()
