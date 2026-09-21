# examples/project/run_dnsga2_baseline.py
"""External dynamic baseline: DNSGA-II (pymoo) on the DTNK/FDA problems.

Plain NSGA-II has no change-response mechanism: on a dynamic problem it
carries stale fitness values across environment changes and drifts. The
established dynamic variant is DNSGA-II (Deb, Rao and Karthik, 2007),
which adds exactly that: when a change is detected, the population is
re-evaluated and diversity is injected. Two variants are published and
both are implemented here:

* ``A`` -- a fraction zeta of the population is replaced by randomly
  generated solutions. Better on severe changes.
* ``B`` -- a fraction zeta is replaced by mutated copies of existing
  members. Better on mild changes.

Held identical to the CCPSO dynamic campaign so results are comparable:

* the Cilpy problem objects themselves are wrapped, so both algorithms
  see identical objectives, constraints and dynamics;
* evaluation budget: population 100 x 1000 generations = 100,000, the
  same as 50 particles x 2 sub-swarms x 1000 iterations;
* 30 runs, seeds derived with the same function and BASE_SEED;
* IGD computed every generation against the reference front of the
  CURRENT environment, using cilpy.compare.metrics -- not pymoo's
  indicators, which use different reference sets;
* MIGD = mean IGD over all generations; MIGD_bc = mean IGD at the last
  generation of each environment, sampled at ``it % tau_t == tau_t - 1``,
  matching the campaign's aggregation exactly;
* feasibility measured on the returned front, re-evaluated against the
  original problem.

Reference:
    K. Deb, U. B. Rao N., and S. Karthik, "Dynamic multi-objective
    optimization and decision-making using modified NSGA-II: A case study
    on hydro-thermal power scheduling," in Proc. EMO 2007, LNCS vol. 4403,
    pp. 803-817, doi: 10.1007/978-3-540-70928-2_60.

Usage (from the repository root):

    python examples/project/run_dnsga2_baseline.py --tau-t 10
    python examples/project/run_dnsga2_baseline.py --tau-t 10 --quick
    python examples/project/run_dnsga2_baseline.py --tau-t 25 --variant B
    python examples/project/run_dnsga2_baseline.py --tau-t 10 --problems DTNK DTNK3

Writes results_dnsga2_taut<TAU>.csv with the same column meanings as the
CCPSO dynamic tables, so the two can be placed side by side.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cilpy.problem.dynamic_multi_objective import (
    FDA1, FDA3, DTNK, DTNK2, DTNK3, DTNK4,
)
from cilpy.compare.metrics import (
    inverted_generational_distance, feasibility_rate,
)

try:
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.evaluator import Evaluator
    from pymoo.core.population import Population
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
except ImportError:
    print("pymoo is required:  pip install pymoo")
    raise

BASE_SEED = 26989395
POP_SIZE = 100
N_GEN = 1000
ZETA = 0.2          # fraction replaced on a change (Deb et al. use 0.2-0.3)

PROBLEMS = {
    "FDA1": FDA1, "FDA3": FDA3,
    "DTNK": DTNK, "DTNK3": DTNK3, "DTNK2": DTNK2, "DTNK4": DTNK4,
}


def experiment_seed(problem_name, solver_name):
    key = f"{problem_name}:{solver_name}"
    return BASE_SEED + sum(ord(c) * (i + 1) for i, c in enumerate(key))


class DynamicCilpyAdapter(PymooProblem):
    """Wraps a Cilpy dynamic problem for pymoo.

    Time is owned by the Cilpy problem: the driver loop calls
    ``begin_iteration()`` once per generation, exactly as the Cilpy
    ExperimentRunner does, so both algorithms experience the same
    environment schedule.
    """

    def __init__(self, cilpy_problem):
        self.cilpy_problem = cilpy_problem
        lower, upper = cilpy_problem.bounds
        probe = cilpy_problem.evaluate(list(lower))
        super().__init__(
            n_var=cilpy_problem.dimension,
            n_obj=len(probe.fitness),
            n_ieq_constr=len(probe.constraints_inequality or []),
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


def current_front(cilpy_problem, X):
    """Feasible objective vectors of a decision-space population."""
    front, evaluations = [], []
    for row in np.atleast_2d(X):
        evaluation = cilpy_problem.evaluate([float(v) for v in row])
        evaluations.append(evaluation)
        if all(g <= 0 for g in (evaluation.constraints_inequality or [])):
            front.append([float(v) for v in evaluation.fitness])
    return front, evaluations


def nondominated(front):
    """Non-dominated subset of a list of objective vectors."""
    if not front:
        return []
    matrix = np.asarray(front, dtype=float)
    keep = np.ones(len(matrix), dtype=bool)
    for i in range(len(matrix)):
        others = np.delete(matrix, i, axis=0)
        if np.any(np.all(others <= matrix[i], axis=1)
                  & np.any(others < matrix[i], axis=1)):
            keep[i] = False
    return matrix[keep].tolist()


def respond_to_change(algorithm, pymoo_problem, variant, zeta, rng):
    """DNSGA-II change response.

    Re-evaluates the whole population under the new environment, then
    replaces a fraction zeta with random (variant A) or mutated (variant
    B) solutions to restore diversity.

    The replacement individuals carry no rank or crowding distance, which
    NSGA-II's tournament selection requires, so the merged population is
    passed back through the algorithm's own survival operator to restore
    those attributes before the next generation.
    """
    population = algorithm.pop
    evaluator = Evaluator()
    evaluator.eval(pymoo_problem, population, skip_already_evaluated=False)

    n_replace = max(1, int(zeta * len(population)))
    indices = rng.choice(len(population), size=n_replace, replace=False)

    if variant.upper() == "A":
        replacements = FloatRandomSampling().do(pymoo_problem, n_replace)
    else:
        donors = Population.create(
            *[population[i] for i in
              rng.choice(len(population), size=n_replace, replace=True)])
        replacements = PM(eta=20, prob=1.0).do(pymoo_problem, donors)

    evaluator.eval(pymoo_problem, replacements, skip_already_evaluated=False)

    survivors = [population[i] for i in range(len(population))
                 if i not in set(indices.tolist())]
    merged = Population.create(*survivors, *replacements)

    # Restore rank and crowding distance via NSGA-II's survival operator.
    algorithm.pop = algorithm.survival.do(
        pymoo_problem, merged, n_survive=len(population))


def run_one(problem_cls, tau_t, n_t, pop_size, n_gen, seed, variant, zeta):
    """One run; returns (per-generation IGD list, final front, final evals)."""
    cilpy_problem = problem_cls(tau_t=tau_t, n_t=n_t)
    cilpy_problem.reset_time()
    pymoo_problem = DynamicCilpyAdapter(cilpy_problem)
    rng = np.random.default_rng(seed)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    algorithm.setup(pymoo_problem, termination=("n_gen", n_gen + 1),
                    seed=int(seed), verbose=False)

    igd_series = []
    previous_time = cilpy_problem.t

    for generation in range(1, n_gen + 1):
        cilpy_problem.begin_iteration()          # advance time, as the runner does
        if cilpy_problem.t != previous_time:     # environment changed
            respond_to_change(algorithm, pymoo_problem, variant, zeta, rng)
            previous_time = cilpy_problem.t

        algorithm.next()
        if algorithm.pop is None:
            break

        X = algorithm.pop.get("X")
        front, _ = current_front(cilpy_problem, X)
        front = nondominated(front)
        if front:
            reference = cilpy_problem.true_pareto_front(300)
            igd_series.append(
                (generation, inverted_generational_distance(front, reference)))

    X = algorithm.pop.get("X") if algorithm.pop is not None else np.empty((0, 1))
    final_front, final_evaluations = current_front(cilpy_problem, X)
    return igd_series, nondominated(final_front), final_evaluations


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--pop-size", type=int, default=POP_SIZE)
    ap.add_argument("--n-gen", type=int, default=N_GEN)
    ap.add_argument("--tau-t", type=int, required=True,
                    help="change frequency; must match the CCPSO campaign")
    ap.add_argument("--n-t", type=int, default=10)
    ap.add_argument("--variant", choices=["A", "B"], default="A",
                    help="A: random re-initialisation; B: mutation-based")
    ap.add_argument("--zeta", type=float, default=ZETA,
                    help="fraction of the population replaced on a change")
    ap.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    ap.add_argument("--quick", action="store_true",
                    help="3 runs of 100 generations, for a smoke test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-run-dir", default="per_run",
                    help="folder for per-run value CSVs, used for "
                         "significance testing")
    args = ap.parse_args()

    num_runs = 3 if args.quick else args.runs
    n_gen = 100 if args.quick else args.n_gen
    out_path = args.out or f"results_dnsga2_taut{args.tau_t}.csv"

    print(f"DNSGA-II ({args.variant}) baseline: {num_runs} runs, "
          f"pop {args.pop_size} x {n_gen} generations "
          f"= {args.pop_size * n_gen:,} evaluations per run")
    print(f"tau_t = {args.tau_t}, n_t = {args.n_t}, zeta = {args.zeta}")
    print(f"(CCPSO campaign: 100 particles x 1000 iterations = 100,000)\n")

    rows = []
    for problem_name in args.problems:
        if problem_name not in PROBLEMS:
            print(f"unknown problem: {problem_name}")
            continue
        start = time.time()
        migds, migd_bcs, feasibilities, sizes = [], [], [], []
        per_run_records = []

        for run_id in range(1, num_runs + 1):
            seed = experiment_seed(problem_name, "DNSGA2") + run_id
            series, front, evaluations = run_one(
                PROBLEMS[problem_name], args.tau_t, args.n_t,
                args.pop_size, n_gen, seed, args.variant, args.zeta)
            values = [v for _, v in series]
            before = [v for it, v in series if it % args.tau_t == args.tau_t - 1]
            if values:
                migds.append(float(np.mean(values)))
            if before:
                migd_bcs.append(float(np.mean(before)))
            feasibilities.append(
                feasibility_rate(evaluations) if evaluations else 0.0)
            sizes.append(len(front))
            per_run_records.append({
                "run_id": run_id,
                "migd": float(np.mean(values)) if values else float("nan"),
                "migd_bc": float(np.mean(before)) if before else float("nan"),
                "front_feasibility_pct": feasibilities[-1],
                "front_size": len(front),
            })

        migd = np.asarray(migds) if migds else np.asarray([np.nan])
        migd_bc = np.asarray(migd_bcs) if migd_bcs else np.asarray([np.nan])
        feasibility = np.asarray(feasibilities)
        print(f"{problem_name:7} MIGD {migd.mean():8.4f} ± {migd.std():7.4f}   "
              f"MIGD_bc {migd_bc.mean():8.4f}   "
              f"feas {feasibility.mean():6.1f}%   "
              f"|front| {np.mean(sizes):5.1f}   ({time.time() - start:.0f}s)")

        # Per-run values, retained so that significance tests against
        # CCPSO are possible. Aggregate means alone cannot support them.
        os.makedirs(args.per_run_dir, exist_ok=True)
        per_run_path = os.path.join(
            args.per_run_dir,
            f"perrun_dnsga2_{args.variant}_taut{args.tau_t}_"
            f"{problem_name}.csv")
        with open(per_run_path, "w", newline="") as pr:
            writer = csv.writer(pr)
            writer.writerow(["run_id", "migd", "migd_bc",
                             "front_feasibility_pct", "front_size"])
            for record in per_run_records:
                writer.writerow([
                    record["run_id"],
                    "" if np.isnan(record["migd"]) else f"{record['migd']:.8f}",
                    "" if np.isnan(record["migd_bc"])
                    else f"{record['migd_bc']:.8f}",
                    f"{record['front_feasibility_pct']:.4f}",
                    record["front_size"],
                ])

        rows.append([
            problem_name, f"DNSGA2_{args.variant}",
            f"{migd.mean():.6f}", f"{migd.std():.6f}",
            f"{migd_bc.mean():.6f}", f"{migd_bc.std():.6f}",
            f"{feasibility.mean():.4f}", f"{feasibility.std():.4f}",
        ])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "problem", "algorithm", "migd_mean", "migd_std",
            "migd_before_change_mean", "migd_before_change_std",
            "front_feasibility_pct_mean", "front_feasibility_pct_std",
        ])
        writer.writerows(rows)
    print(f"\n-> {out_path}")
    print(f"-> {args.per_run_dir}/perrun_dnsga2_{args.variant}_"
          f"taut{args.tau_t}_<problem>.csv (per-run values)")
    print("Compare against results_dynamic_taut"
          f"{args.tau_t}.csv (identical column meanings).")


if __name__ == "__main__":
    main()
