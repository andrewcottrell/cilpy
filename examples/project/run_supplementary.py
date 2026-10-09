# examples/project/run_supplementary.py
"""Supplementary runs answering the supervisor's feedback on the 8 Sept draft.

These sit beside the main campaign and do not replace it. Two sets:

* ``rho0``    -- CCPSO strict with the explicit penalty switched off
  (``penalty_rho = 0`` for inequality and equality constraints), so all
  constraint pressure comes from the co-evolved multipliers. Compared
  against the existing CCPSO_strict (rho = 1) and MGPSO_feasarch.
* ``sampled`` -- the objective solver's ``w, c1, c2, c3`` are omitted, so
  MGPSO re-samples them per particle each iteration under the stability
  condition. The multiplier swarm is unchanged (plain PSO, fixed
  parameters). Compared against the fixed-parameter results.

Each set has a static part and a dynamic part:

    set      part     problems                         algorithm
    rho0     static   BNH SRN TNK CONSTR OSY            CCPSO_strict_rho0
    rho0     dynamic  DTNK DTNK3 DTNK2                  CCPSO_strict_rho0
    sampled  static   BNH SRN TNK CONSTR OSY            CCPSO_strict_sampled
                      SCH1 ZDT1 ZDT2 ZDT3 ZDT4 ZDT6     MGPSO_sampled
    sampled  dynamic  DTNK DTNK3 DTNK2                  CCPSO_strict_sampled
                      FDA1 FDA3                         MGPSO_sampled

Everything else (swarm sizes, multiplier swarm, max_multiplier, clean
refresh, fixed sentry, seeding rule) matches the main campaign. The new
algorithm names give these runs their own deterministic seeds.

The runner always writes to ./out, so run one invocation at a time and move
out/ aside afterwards (run_supplementary.sh does this).

Usage (from the repository root):
    python examples/project/run_supplementary.py --set rho0 --part static
    python examples/project/run_supplementary.py --set rho0 --part dynamic --tau-t 25
    python examples/project/run_supplementary.py --set sampled --part static --workers 8
    python examples/project/run_supplementary.py --set rho0 --part static --quick
"""

import argparse
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from paths import results_path

import run_static_mo_experiments as static
import run_dynamic_experiements as dynamic

from cilpy.problem.multi_objective import (
    SCH1, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6,
    BNH, SRN, TNK, CONSTR, OSY,
)
from cilpy.problem.dynamic_multi_objective import FDA1, FDA3, DTNK, DTNK2, DTNK3
from cilpy.solver.mgpso import MGPSO
from cilpy.solver.pso import PSO
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver

SWARM_SIZE = static.SWARM_SIZE
FIXED_PARAMS = static.FIXED_PARAMS

UNCONSTRAINED = [SCH1, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6]
CONSTRAINED = [BNH, SRN, TNK, CONSTR, OSY]
DYNAMIC_CONSTRAINED = [DTNK, DTNK3, DTNK2]
DYNAMIC_BOX = [FDA1, FDA3]


def ccpso_strict_config(name, rho, objective_params):
    return {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": name,
            "objective_solver_class": MGPSO,
            "multiplier_solver_class": PSO,
            "objective_solver_params": {
                "swarm_size": SWARM_SIZE, **objective_params,
            },
            "multiplier_solver_params": {
                "swarm_size": 30, "w": 0.40, "c1": 1.20, "c2": 1.20,
            },
            "penalty_rho": rho,
            "penalty_rho_equality": rho,
            "max_multiplier": 1000.0,
            "archive_strategy": "strict",
        },
    }


def mgpso_sampled_config():
    # No w, c1, c2, c3: MGPSO samples them per particle each iteration.
    return {
        "class": MGPSO,
        "params": {"name": "MGPSO_sampled", "swarm_size": SWARM_SIZE},
    }


def experiments(set_name, part):
    """Returns [(problem_factory, config), ...] in priority order."""
    if set_name == "rho0":
        config = ccpso_strict_config("CCPSO_strict_rho0", 0.0, FIXED_PARAMS)
        problems = CONSTRAINED if part == "static" else DYNAMIC_CONSTRAINED
        return [(p, config) for p in problems]

    ccpso = ccpso_strict_config("CCPSO_strict_sampled", 1.0, {})
    mgpso = mgpso_sampled_config()
    if part == "static":
        return ([(p, ccpso) for p in CONSTRAINED]
                + [(p, mgpso) for p in UNCONSTRAINED])
    return ([(p, ccpso) for p in DYNAMIC_CONSTRAINED]
            + [(p, mgpso) for p in DYNAMIC_BOX])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="set_name", required=True,
                        choices=["rho0", "sampled"])
    parser.add_argument("--part", required=True, choices=["static", "dynamic"])
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--tau-t", type=int, default=25,
                        help="change frequency (dynamic part only)")
    parser.add_argument("--n-t", type=int, default=10,
                        help="change severity (dynamic part only)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 2 runs, 100 iterations")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--out-dir", default="out",
                        help="folder of campaign CSVs to aggregate; the "
                             "runner itself always writes to out/")
    args = parser.parse_args()

    num_runs = 2 if args.quick else args.runs
    max_iterations = 100 if args.quick else args.iters
    pairs = experiments(args.set_name, args.part)

    if os.path.exists("out") and not args.aggregate_only:
        sys.exit("out/ already exists. Move it aside first, so these runs "
                 "do not mix with another campaign's files.")

    start = time.time()
    if not args.aggregate_only:
        if args.part == "static":
            tasks = [(p, c, num_runs, max_iterations) for p, c in pairs]
            module = static
        else:
            tasks = [(p, c, num_runs, max_iterations, args.tau_t, args.n_t)
                     for p, c in pairs]
            module = dynamic
        if args.workers <= 1:
            for task in tasks:
                module._run_one_experiment(task)
        else:
            done = 0
            with multiprocessing.Pool(processes=args.workers) as pool:
                for name in pool.imap_unordered(module._run_one_experiment, tasks):
                    done += 1
                    print(f"\n>>> [{done}/{len(tasks)}] finished: {name}")
        print(f"\nAll experiments done in {(time.time() - start) / 60:.1f} min")

    problem_names = list(dict.fromkeys(p().name if args.part == "static"
                                       else p(tau_t=args.tau_t, n_t=args.n_t).name
                                       for p, _ in pairs))
    solver_names = list(dict.fromkeys(c["params"]["name"] for _, c in pairs))
    if args.part == "static":
        static.aggregate(
            problem_names, solver_names, args.out_dir,
            out_path=results_path(f"results_supp_{args.set_name}_static.csv"))
    else:
        dynamic.aggregate(
            problem_names, solver_names, args.tau_t, args.out_dir,
            out_path=results_path(
                f"results_supp_{args.set_name}_dynamic_taut{args.tau_t}.csv"))


if __name__ == "__main__":
    main()
